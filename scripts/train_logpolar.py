"""Train the log-polar decoder (frozen DFT-magnitude encoder) for resize/rotation
tolerance. Encoder is analytical (no collapse); only the log-polar CNN decoder trains,
under geometric (scale/rotation/translation) + photometric augmentation.

Logs clean bit-acc AND resize-augmented bit-acc every interval so geometric
generalization is visible during training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.dft_kred_modules import DFTKredEncoder
from src.logpolar_fragment import LogPolarDecoder
from scripts.train_pixel_frag_vine_style import ImageOnlyDataset


def geometric_augment(x, smin=0.80, smax=1.25, rot_deg=12.0, trans=0.04):
    B, dev = x.shape[0], x.device
    s = torch.empty(B, device=dev).uniform_(smin, smax)
    ang = torch.empty(B, device=dev).uniform_(-rot_deg, rot_deg) * math.pi / 180.0
    tx = torch.empty(B, device=dev).uniform_(-trans, trans)
    ty = torch.empty(B, device=dev).uniform_(-trans, trans)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos / s; theta[:, 0, 1] = -sin / s; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / s; theta[:, 1, 1] = cos / s; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


def fixed_resize(x, scale):
    B, dev = x.shape[0], x.device
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = 1.0 / scale; theta[:, 1, 1] = 1.0 / scale
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


def photometric_augment(x):
    if torch.rand(()) < 0.5:
        k = int(np.random.choice([3, 5]))
        x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)
    if torch.rand(()) < 0.5:
        x = x + torch.randn_like(x) * float(np.random.uniform(0.0, 0.04))
    if torch.rand(()) < 0.3:  # jpeg-ish low-pass via down/up-sample
        f = float(np.random.uniform(0.6, 0.95))
        h = max(8, int(x.shape[-1] * f))
        x = F.interpolate(F.interpolate(x, size=h, mode="bilinear", align_corners=False),
                          size=x.shape[-1], mode="bilinear", align_corners=False)
    return x.clamp(0, 1)


def train(args):
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = ImageOnlyDataset(args.image_dir, start_idx=0, max_images=args.max_images,
                          resolution=args.resolution)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, pin_memory=True, drop_last=True)
    enc = DFTKredEncoder(n_bits=args.n_bits, K=args.K, M=args.M, resolution=args.resolution,
                         r_lo=args.r_lo, r_hi=args.r_hi,
                         canonical_key=args.canonical_key.encode("utf-8"),
                         canonical_id=args.canonical_id, init_delta=args.init_delta).to(device)
    enc.eval(); enc.requires_grad_(False)
    dec = LogPolarDecoder(n_bits=args.n_bits, n_rho=args.n_rho, n_theta=args.n_theta,
                          r_lo=args.r_lo, r_hi=args.r_hi, resolution=args.resolution,
                          width=args.width, pool_hw=args.pool_hw).to(device)
    print(f"[setup] {len(ds)} imgs; enc carriers {tuple(enc.carriers.shape)}; "
          f"dec params {sum(p.numel() for p in dec.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.max_steps)
    bce = nn.BCEWithLogitsLoss()

    step, t0, log = 0, time.time(), []
    for epoch in range(10_000):
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(device, non_blocking=True)
            B = img.shape[0]
            payload = (torch.rand(B, args.n_bits, device=device) > 0.5).float()
            with torch.no_grad():
                x_w = enc(img, payload)
            if step < args.aug_warmup_steps:
                x_aug = x_w
            else:
                x_aug = geometric_augment(x_w)
                x_aug = photometric_augment(x_aug)
            logits = dec(x_aug)
            loss = bce(logits, payload)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step(); sched.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    acc = ((logits > 0).float() == payload).float().mean().item()
                    # geometric generalization probe: strong fixed resize 1.15
                    rz = fixed_resize(x_w, 1.15)
                    acc_rz = ((dec(rz) > 0).float() == payload).float().mean().item()
                print(f"[step {step:5d}] loss={loss.item():.4f} acc={acc:.3f} "
                      f"acc@resize1.15={acc_rz:.3f} t={time.time()-t0:.0f}s", flush=True)
                log.append({"step": step, "loss": float(loss.item()),
                            "acc": float(acc), "acc_resize115": float(acc_rz)})
            step += 1
        if step >= args.max_steps:
            break

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({"decoder_state_dict": dec.state_dict(), "encoder_state_dict": enc.state_dict(),
                "config": vars(args), "n_bits": args.n_bits, "K": args.K, "M": args.M,
                "resolution": args.resolution, "r_lo": args.r_lo, "r_hi": args.r_hi,
                "n_rho": args.n_rho, "n_theta": args.n_theta, "width": args.width,
                "canonical_key": args.canonical_key, "canonical_id": args.canonical_id,
                "carriers": enc.carriers.cpu(), "bit_flip": enc.bit_flip.cpu()},
               args.output_ckpt)
    json.dump(log, open(args.output_ckpt.replace(".pt", "_log.json"), "w"), indent=2)
    print(f"[done] -> {args.output_ckpt}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--r_lo", type=float, default=8.0)
    p.add_argument("--r_hi", type=float, default=110.0)
    p.add_argument("--n_rho", type=int, default=48)
    p.add_argument("--n_theta", type=int, default=96)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--pool_hw", type=int, default=4)
    p.add_argument("--init_delta", type=float, default=50.0)
    p.add_argument("--canonical_key", default="vine_training_canonical_key")
    p.add_argument("--canonical_id", default="logpolar_train_canonical")
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--aug_warmup_steps", type=int, default=150)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
