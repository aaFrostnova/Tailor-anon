"""Train a learned decoder for classical fragments (DCT, DWT-DCT, Quantization).

Replaces Pearson correlation detection (ref-needed) with a CNN decoder (ref-free).
Embedding is fixed (non-learned), only the decoder is trained.

Pipeline per step:
  1. Sample clean image
  2. Generate crypto fragments and embed (DCT + DWT-DCT + Quantization)
  3. Apply random augmentation (JPEG, blur, noise, crop+resize, regen cache)
  4. Decoder(augmented_image) → predicted bits
  5. Loss: BCE(predicted, target)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.keygen import derive_all_subkeys
from src.fragment import generate_fragment
from src.embed import embed_single_fragment
from src.payload import BCHCodec, image_id_to_payload
from src.sign_envelope import derive_keyed_constants


# ============================================================ fragment configs

FRAGMENT_CONFIGS = [
    {"strategy": "dct", "region": "full", "freq_range": [1, 12]},
    {"strategy": "dwt-dct", "region": "full", "level": 2, "block_size": 8, "freq_range": [12, 35]},
    {"strategy": "quantization", "region": "full", "bits": 6},
]
N_FRAGS = len(FRAGMENT_CONFIGS)


# ============================================================ decoder model

class ClassicalFragDecoder(nn.Module):
    """Lightweight CNN decoder: image → n_bits logits per fragment."""

    def __init__(self, n_bits=100, n_frags=3):
        super().__init__()
        self.n_bits = n_bits
        self.n_frags = n_frags
        # Use EfficientNet-B0 as backbone (lightweight, 5.3M params)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        # Per-fragment heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_bits),
                nn.Sigmoid(),
            )
            for _ in range(n_frags)
        ])

    def forward(self, x):
        # x: (B, 3, H, W) in [0, 1]
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]


# ============================================================ dataset

class ClassicalEmbedDataset(torch.utils.data.Dataset):
    """On-the-fly: load image → embed classical fragments → return (embedded, targets)."""

    def __init__(self, image_dir, master_key, n_bits=100, epsilon=8/255,
                 resolution=512, max_images=5000):
        self.files = sorted([
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])[:max_images]
        self.master_key = master_key
        self.n_bits = n_bits
        self.epsilon = epsilon
        self.resolution = resolution
        self.codec = BCHCodec()
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pil = Image.open(self.files[idx]).convert("RGB")
        pil = self.transform(pil)
        image_id = f"train_{idx:06d}"

        # Generate payload and target bits per fragment
        img = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

        subkeys = derive_all_subkeys(self.master_key, image_id, num_fragments=N_FRAGS)

        # Embed all classical fragments
        weight = 1.0 / N_FRAGS
        for k, config in enumerate(FRAGMENT_CONFIGS):
            frag = generate_fragment(subkeys[k], img.shape, epsilon=self.epsilon, fragment_index=k)
            img = embed_single_fragment(img, frag, config, weight=weight)

        # Generate target bits: use correlation sign as target
        # For each fragment, the "positive correlation" regions map to bit=1
        targets = []
        for k in range(N_FRAGS):
            perm, M = derive_keyed_constants(
                self.master_key, image_id + f"/classical_{k}", n_bits=self.n_bits,
            )
            # Random but deterministic target bits for this image+fragment
            target = np.zeros(self.n_bits, dtype=np.float32)
            for j in range(self.n_bits):
                target[j] = 0.0 if M[perm[j]] > 0 else 1.0
            targets.append(torch.tensor(target))

        img_tensor = torch.from_numpy(img).float()
        return img_tensor, targets


# ============================================================ augmentation

def augment_batch(images):
    """Apply random augmentation to a batch of images."""
    B = images.shape[0]
    result = images.clone()
    for i in range(B):
        r = np.random.random()
        if r < 0.15:
            # JPEG
            from io import BytesIO
            q = np.random.randint(30, 80)
            pil = transforms.ToPILImage()(result[i])
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            result[i] = transforms.ToTensor()(Image.open(buf).convert("RGB"))
        elif r < 0.30:
            # Gaussian noise
            sigma = np.random.uniform(0.01, 0.05)
            result[i] = result[i] + torch.randn_like(result[i]) * sigma
        elif r < 0.45:
            # Blur
            k = int(np.random.choice([3, 5, 7]))
            from torchvision.transforms.functional import gaussian_blur
            result[i] = gaussian_blur(result[i], kernel_size=k, sigma=float(np.random.uniform(0.5, 2.0)))
        elif r < 0.55:
            # Crop + resize
            ratio = np.random.uniform(0.6, 0.9)
            _, H, W = result[i].shape
            ch, cw = int(H * ratio), int(W * ratio)
            top = np.random.randint(0, H - ch + 1)
            left = np.random.randint(0, W - cw + 1)
            cropped = result[i][:, top:top+ch, left:left+cw]
            result[i] = F.interpolate(cropped.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
        # else: identity (45% of time)
    return result.clamp(0, 1)


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    master_key = args.master_key.encode("utf-8")

    ds = ClassicalEmbedDataset(
        args.image_dir, master_key, n_bits=args.n_bits,
        epsilon=args.epsilon, resolution=args.resolution, max_images=args.max_images,
    )
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    decoder = ClassicalFragDecoder(n_bits=args.n_bits, n_frags=N_FRAGS).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[setup] Decoder params: {n_params:,}")

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss()

    step = 0
    t0 = time.time()
    log_rows = []

    for epoch in range(args.epochs):
        for batch in loader:
            if step >= args.max_steps:
                break
            images, targets_list = batch
            images = images.to(device)
            targets = [t.to(device) for t in targets_list]

            # Augment
            images_aug = augment_batch(images)

            # Decode
            preds = decoder(images_aug)

            # Loss: BCE per fragment
            loss = 0
            accs = []
            for k in range(N_FRAGS):
                loss_k = bce(preds[k], targets[k])
                loss = loss + loss_k
                acc_k = ((preds[k] > 0.5).float() == targets[k]).float().mean().item()
                accs.append(acc_k)
            loss = loss / N_FRAGS

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                acc_str = " ".join([f"f{k}={a:.3f}" for k, a in enumerate(accs)])
                print(f"[step {step:6d}] loss={loss.item():.4f} [{acc_str}] t={elapsed:.0f}s",
                      flush=True)
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "accs": accs, "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(decoder, args, log_rows)

            step += 1
        if step >= args.max_steps:
            break

    _save(decoder, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}")


def _save(decoder, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "decoder_state_dict": decoder.state_dict(),
        "config": vars(args),
        "n_bits": args.n_bits,
        "n_frags": N_FRAGS,
        "fragment_configs": FRAGMENT_CONFIGS,
    }, args.output_ckpt)
    log_path = args.output_ckpt.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=8/255)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_images", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
