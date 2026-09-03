"""Jointly train U-Net residual encoder + from-scratch CNN decoder (StegaStamp-style).

This is the genuine learned-encoder route. Anti-collapse measures, all learned
the hard way from the failed pixel attempts:
  1. Per-image random payload  -> decoder cannot collapse to a constant.
  2. From-scratch ConvDecoder  -> no ImageNet "invariant-to-perturbation" prior.
  3. BCEWithLogitsLoss         -> no sigmoid-saturation gradient starvation.
  4. Curriculum                -> no augmentation and zero image-loss during
                                  warmup, so encoder+decoder first agree on a
                                  clean code before attacks/quality pressure.
  5. Straight-through attacks  -> gradient still reaches the encoder through
                                  non-differentiable ops (JPEG) via STE.

Loss = BCE(dec(attack(enc(img,p))), p) + lambda_mse*MSE + lambda_lpips*LPIPS.
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
from torchvision.transforms.functional import gaussian_blur

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.learned_unet_modules import ResidualUNetEncoder, ConvDecoder
from scripts.train_pixel_frag_vine_style import ImageOnlyDataset

try:
    import lpips as _lpips_mod  # type: ignore
    _HAVE_LPIPS = True
except Exception:
    _lpips_mod = None
    _HAVE_LPIPS = False


def _jpeg_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Straight-through JPEG: forward = real JPEG, backward = identity.

    Lets the encoder receive gradient through JPEG-attacked samples even though
    PIL JPEG is non-differentiable.
    """
    device = x.device
    out = []
    for i in range(x.shape[0]):
        pil = transforms.ToPILImage()(x[i].detach().clamp(0, 1).cpu())
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=quality); buf.seek(0)
        out.append(transforms.ToTensor()(Image.open(buf).convert("RGB")))
    jpeg = torch.stack(out, 0).to(device)
    return x + (jpeg - x).detach()


def augment_differentiable(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random attack, all paths pass gradient to the encoder
    (JPEG via straight-through)."""
    B = x.shape[0]
    out = []
    for i in range(B):
        xi = x[i:i + 1]
        r = float(np.random.random())
        if r < 0.20:
            xi = _jpeg_ste(xi, int(np.random.randint(40, 85)))
        elif r < 0.40:
            xi = xi + torch.randn_like(xi) * float(np.random.uniform(0.01, 0.05))
        elif r < 0.60:
            k = int(np.random.choice([3, 5, 7]))
            xi = gaussian_blur(xi, kernel_size=k, sigma=float(np.random.uniform(0.5, 2.0)))
        elif r < 0.75:
            ratio = float(np.random.uniform(0.6, 0.9))
            _, _, H, W = xi.shape
            ch, cw = int(H * ratio), int(W * ratio)
            top = int(np.random.randint(0, H - ch + 1)); left = int(np.random.randint(0, W - cw + 1))
            xi = F.interpolate(xi[:, :, top:top + ch, left:left + cw], size=(H, W),
                               mode="bilinear", align_corners=False)
        # else identity
        out.append(xi)
    return torch.cat(out, 0).clamp(0, 1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = ImageOnlyDataset(args.image_dir, start_idx=0,
                          max_images=args.max_images, resolution=args.resolution)
    print(f"[setup] {len(ds)} training images at {args.resolution}px", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    enc = ResidualUNetEncoder(n_bits=args.n_bits, epsilon=args.epsilon, base=args.base).to(device)
    dec = ConvDecoder(n_bits=args.n_bits, base=args.base).to(device)
    print(f"[setup] enc params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)
    print(f"[setup] dec params: {sum(p.numel() for p in dec.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    lpips_fn = None
    if _HAVE_LPIPS and args.lambda_lpips > 0:
        try:
            lpips_fn = _lpips_mod.LPIPS(net="vgg").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            print("[setup] LPIPS enabled", flush=True)
        except Exception as e:
            print(f"[setup] LPIPS init failed ({e})", flush=True)

    step = 0; t0 = time.time(); log_rows = []
    for epoch in range(args.epochs):
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(device, non_blocking=True)
            B = img.shape[0]
            payload = (torch.rand(B, args.n_bits, device=device) > 0.5).float()

            x_w = enc(img, payload)
            x_aug = x_w if step < args.aug_warmup_steps else augment_differentiable(x_w)
            p_hat = dec(x_aug)

            l_bce = bce(p_hat, payload)
            l_mse = F.mse_loss(x_w, img)
            l_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None and step >= args.aug_warmup_steps:
                l_lpips = lpips_fn(x_w * 2 - 1, img * 2 - 1).mean()
            lam_mse = 0.0 if step < args.aug_warmup_steps else args.lambda_mse
            lam_lpips = 0.0 if step < args.aug_warmup_steps else args.lambda_lpips
            loss = l_bce + lam_mse * l_mse + lam_lpips * l_lpips

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    bit_acc = ((p_hat > 0).float() == payload).float().mean().item()
                    psnr = 10 * np.log10(1.0 / max(l_mse.item(), 1e-12))
                el = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} bce={l_bce.item():.4f} "
                      f"mse={l_mse.item():.6f} lpips={float(l_lpips):.4f} psnr={psnr:.1f}dB "
                      f"bit_acc={bit_acc:.3f} t={el:.0f}s", flush=True)
                log_rows.append({"step": step, "loss": float(loss.item()),
                                 "bce": float(l_bce.item()), "mse": float(l_mse.item()),
                                 "psnr": float(psnr), "bit_acc": float(bit_acc),
                                 "elapsed_s": float(el)})
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(enc, dec, args, log_rows)
            step += 1
        if step >= args.max_steps:
            break
    _save(enc, dec, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}", flush=True)


def _save(enc, dec, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "encoder_state_dict": enc.state_dict(),
        "decoder_state_dict": dec.state_dict(),
        "config": vars(args),
        "n_bits": args.n_bits, "epsilon": args.epsilon,
        "base": args.base, "resolution": args.resolution,
    }, args.output_ckpt)
    with open(args.output_ckpt.replace(".pt", "_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=8 / 255)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--aug_warmup_steps", type=int, default=300)
    p.add_argument("--lambda_mse", type=float, default=5.0)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
