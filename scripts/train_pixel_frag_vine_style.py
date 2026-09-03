"""Joint train pixel-domain learned encoder + decoder, VINE-style.

Per step:
  1. Sample image batch.
  2. Sample a RANDOM 100-bit payload per image (independent across batch+steps).
     -> decoder cannot collapse to a constant; it must extract payload from image.
  3. enc(image, payload) -> watermarked image (perturbation bounded by tanh * eps).
  4. Augment (JPEG / blur / noise / crop+resize / identity).
  5. decoder(augmented) -> predicted bits.
  6. Loss = BCE(predicted, payload) + lambda_img * MSE(watermarked, image).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.learned_fragment_pixel import PixelEncoder, PixelDecoder


# ============================================================ dataset (image-only)

class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, start_idx=0, max_images=4000, resolution=512):
        self.files = sorted([
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])[start_idx:start_idx + max_images]
        self.tfm = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.tfm(Image.open(self.files[idx]).convert("RGB"))


# ============================================================ augmentation

def augment_batch(images):
    from torchvision.transforms.functional import gaussian_blur
    B = images.shape[0]
    device = images.device
    outs = []
    for i in range(B):
        x = images[i]
        r = np.random.random()
        if r < 0.15:
            q = int(np.random.randint(30, 80))
            pil = transforms.ToPILImage()(x.detach().clamp(0, 1).cpu())
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            y = transforms.ToTensor()(Image.open(buf).convert("RGB")).to(device)
        elif r < 0.30:
            sigma = float(np.random.uniform(0.01, 0.05))
            y = x + torch.randn_like(x) * sigma
        elif r < 0.45:
            k = int(np.random.choice([3, 5, 7]))
            y = gaussian_blur(
                x, kernel_size=k,
                sigma=float(np.random.uniform(0.5, 2.0)),
            )
        elif r < 0.55:
            ratio = float(np.random.uniform(0.6, 0.9))
            _, H, W = x.shape
            ch, cw = int(H * ratio), int(W * ratio)
            top = int(np.random.randint(0, H - ch + 1))
            left = int(np.random.randint(0, W - cw + 1))
            cropped = x[:, top:top + ch, left:left + cw]
            y = F.interpolate(
                cropped.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            )[0]
        else:
            y = x
        outs.append(y)
    return torch.stack(outs, dim=0).clamp(0, 1)


# ============================================================ training

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = ImageOnlyDataset(
        args.image_dir, start_idx=0,
        max_images=args.max_images, resolution=args.resolution,
    )
    print(f"[setup] {len(ds)} training images at {args.resolution}px")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    enc = PixelEncoder(n_bits=args.n_bits, epsilon=args.epsilon).to(device)
    dec = PixelDecoder(n_bits=args.n_bits).to(device)
    print(f"[setup] enc params: {sum(p.numel() for p in enc.parameters()):,}")
    print(f"[setup] dec params: {sum(p.numel() for p in dec.parameters()):,}")

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    bce = nn.BCEWithLogitsLoss()

    step = 0
    t0 = time.time()
    log_rows = []

    for epoch in range(args.epochs):
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(device, non_blocking=True)
            B = img.shape[0]

            payload = (torch.rand(B, args.n_bits, device=device) > 0.5).float()

            x_w = enc(img, payload)
            # Curriculum: skip augmentation for the first aug_warmup_steps so the
            # encoder/decoder can establish the basic embedding before attacks
            # disrupt the signal. After warmup, full augmentation.
            x_aug = x_w if step < args.aug_warmup_steps else augment_batch(x_w)
            p_hat = dec(x_aug)

            l_bce = bce(p_hat, payload)
            l_img = F.mse_loss(x_w, img)
            # Curriculum: 0 image loss during warmup so the encoder is forced to
            # inject a usable signal first; bring it in after to control PSNR.
            lambda_img = 0.0 if step < args.aug_warmup_steps else args.lambda_img
            loss = l_bce + lambda_img * l_img

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), 1.0,
            )
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    # p_hat is logits; bit = sigmoid(p_hat) > 0.5  iff  p_hat > 0
                    bit_acc = ((p_hat > 0).float() == payload).float().mean().item()
                    psnr = 10 * np.log10(1.0 / max(l_img.item(), 1e-12))
                elapsed = time.time() - t0
                print(
                    f"[step {step:6d}] loss={loss.item():.4f} bce={l_bce.item():.4f} "
                    f"mse={l_img.item():.5f} psnr={psnr:.1f}dB bit_acc={bit_acc:.3f} "
                    f"t={elapsed:.0f}s",
                    flush=True,
                )
                log_rows.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "bce": float(l_bce.item()),
                    "mse": float(l_img.item()),
                    "psnr": float(psnr),
                    "bit_acc": float(bit_acc),
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(enc, dec, args, log_rows)

            step += 1
        if step >= args.max_steps:
            break

    _save(enc, dec, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}")


def _save(enc, dec, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "encoder_state_dict": enc.state_dict(),
        "decoder_state_dict": dec.state_dict(),
        "config": vars(args),
        "n_bits": args.n_bits,
        "epsilon": args.epsilon,
        "resolution": args.resolution,
    }, args.output_ckpt)
    with open(args.output_ckpt.replace(".pt", "_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image_dir",
        default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017",
    )
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=8 / 255)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--aug_warmup_steps", type=int, default=200,
                   help="Skip augmentation for the first N steps so the encoder can establish a signal first.")
    p.add_argument("--lambda_img", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
