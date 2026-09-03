"""Train KeyConditionedEncoder with matched-filter-only loss (no learned decoder).

The encoder uses sign-modulation: unsigned_residual * S_pixel.
Loss is computed via per-region matched filter on (x_attacked - x_orig):
  sign_loss = MSE(tanh(scale * region_means), target_sign)
  distortion = MSE(x_w, x)
  loss = sign_loss + lambda_dist * distortion

No learned decoder → no overfitting risk. The matched filter IS the test-time
detector, so train/test objective is perfectly aligned.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.key_encoder import KeyConditionedEncoder, s_lat_to_pixel, count_params  # noqa
from src.payload import BCHCodec, image_id_to_payload  # noqa
from src.sign_envelope import derive_keyed_constants  # noqa
from train_T_lat import (  # noqa
    make_latent_region_masks, build_id_pool,
    load_sd_residual_pool, FlatImageDataset,
)


def augment(x_w, sd_pool=None, p_regen=0.3):
    """Random augmentation. Returns (x_attacked, is_differentiable)."""
    B = x_w.shape[0]
    r = np.random.random()

    # SD regen (highest priority, configurable probability)
    if sd_pool is not None and r < p_regen:
        idx = torch.randint(0, sd_pool.shape[0], (B,), device=x_w.device)
        return x_w + sd_pool[idx], False

    r -= p_regen
    p_rest = 1.0 - p_regen

    if r < 0.15 * p_rest:
        return x_w, True

    r -= 0.15 * p_rest
    if r < 0.15 * p_rest:
        sigma = np.random.uniform(0.01, 0.05)
        return x_w + torch.randn_like(x_w) * sigma, True

    r -= 0.15 * p_rest
    if r < 0.2 * p_rest:
        k = int(np.random.choice([3, 5, 7]))
        sigma = float(np.random.uniform(0.5, 2.0))
        from torchvision.transforms.functional import gaussian_blur
        return gaussian_blur(x_w, kernel_size=k, sigma=sigma), True

    r -= 0.2 * p_rest
    if r < 0.1 * p_rest:
        ratio = np.random.uniform(0.6, 0.9)
        H, W = x_w.shape[2], x_w.shape[3]
        ch, cw = int(H * ratio), int(W * ratio)
        top = np.random.randint(0, H - ch + 1)
        left = np.random.randint(0, W - cw + 1)
        cropped = x_w[:, :, top:top + ch, left:left + cw]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False), True

    return x_w + torch.randn_like(x_w) * 0.01, True


def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    codec = BCHCodec()
    master_key = args.master_key.encode("utf-8")

    # id pool
    id_pool_signs = build_id_pool(master_key, args.id_pool_size, codec)
    id_pool_t = torch.from_numpy(id_pool_signs).to(device)

    # Region masks
    region_masks_np = make_latent_region_masks(codec.n, (4, 32, 32))
    region_masks = torch.from_numpy(region_masks_np).to(device)

    # Pixel-space region boundaries for matched filter
    H, W = args.resolution, args.resolution
    grid = int(np.ceil(np.sqrt(codec.n)))
    cell_h, cell_w = H // grid, W // grid
    region_bounds = []
    for i in range(codec.n):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        region_bounds.append((y0, y1, x0, x1))

    # SD residual cache
    sd_pool = None
    if args.sd_residual_cache:
        sd_pool = load_sd_residual_pool(
            args.sd_residual_cache, strengths_to_load=args.attack_strengths,
        )
        if sd_pool is not None:
            sd_pool = sd_pool.to(device)
            print(f"[setup] SD residual pool: {sd_pool.shape}")

    # Dataset
    ds = FlatImageDataset(args.image_dir, args.resolution, args.max_images)
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Encoder only (no decoder)
    encoder = KeyConditionedEncoder(
        base_ch=args.base_ch, n_blocks=args.n_blocks, init_scale=args.init_scale,
    ).to(device)
    print(f"[setup] encoder params: {count_params(encoder):,} (no decoder)")

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)

    step = 0
    t0 = time.time()
    log_rows = []

    while step < args.steps:
        for x_batch in loader:
            if step >= args.steps:
                break
            x_batch = x_batch.to(device)
            B = x_batch.shape[0]

            # Sample image_ids
            id_indices = torch.randint(0, args.id_pool_size, (B,), device=device)
            sign_per_region = id_pool_t[id_indices]

            # Build S_pixel directly in pixel space (NOT via latent upsample —
            # must match the matched-filter grid exactly)
            S_pixel = torch.zeros(B, 1, H, W, device=device)
            for i, (y0, y1, x0, x1) in enumerate(region_bounds):
                S_pixel[:, :, y0:y1, x0:x1] = sign_per_region[:, i].view(B, 1, 1, 1)

            # Encode
            x_w = encoder(x_batch, S_pixel)

            # Distortion
            distortion = F.mse_loss(x_w, x_batch)
            psnr = 10.0 * torch.log10(4.0 / (distortion + 1e-8))

            # Augment
            x_attacked, is_diff = augment(x_w, sd_pool=sd_pool, p_regen=args.p_regen)
            if not is_diff:
                x_attacked = x_attacked.detach()

            # Matched filter on attacked residual (differentiable for diff attacks)
            residual = x_attacked - x_batch
            rho_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_list.append(residual[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_attacked = torch.stack(rho_list, dim=1)  # (B, 127)

            # Also on clean residual (always differentiable)
            residual_clean = x_w - x_batch
            rho_clean_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_clean_list.append(residual_clean[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_clean = torch.stack(rho_clean_list, dim=1)

            # Sign prediction loss
            logit_scale = args.logit_scale
            pred_attacked = torch.tanh(rho_attacked * logit_scale)
            pred_clean = torch.tanh(rho_clean * logit_scale)
            sign_loss_att = F.mse_loss(pred_attacked, sign_per_region)
            sign_loss_clean = F.mse_loss(pred_clean, sign_per_region)

            # Warmup: ramp distortion
            progress = step / max(args.steps, 1)
            if progress < args.warmup_frac:
                lambda_d = 0.0
            else:
                ramp = (progress - args.warmup_frac) / (1.0 - args.warmup_frac)
                lambda_d = args.lambda_dist * ramp

            w_att = args.att_weight
            loss = w_att * sign_loss_att + (1.0 - w_att) * sign_loss_clean + lambda_d * distortion

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                # Bit accuracy via clean matched filter
                with torch.no_grad():
                    mf_correct = ((rho_clean * sign_per_region) > 0).float().mean().item()
                    mf_att_correct = ((rho_attacked * sign_per_region) > 0).float().mean().item()
                sc = encoder.scale.item()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} "
                      f"sign_att={sign_loss_att.item():.4f} sign_clean={sign_loss_clean.item():.4f} "
                      f"psnr={psnr.item():.1f}dB mf_clean={mf_correct:.3f} mf_att={mf_att_correct:.3f} "
                      f"scale={sc:.4f} λ_d={lambda_d:.2f} t={elapsed:.0f}s",
                      flush=True)
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "sign_att": float(sign_loss_att.item()),
                    "sign_clean": float(sign_loss_clean.item()),
                    "psnr": float(psnr.item()),
                    "mf_clean": float(mf_correct),
                    "mf_att": float(mf_att_correct),
                    "scale": float(sc),
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(encoder, args, log_rows)

            step += 1

    _save(encoder, args, log_rows)
    print(f"\n[done] saved → {args.output_ckpt}")


def _save(encoder, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "config": vars(args),
        "n_bits": 127,
        "base_ch": args.base_ch,
        "n_blocks": args.n_blocks,
    }, args.output_ckpt)
    log_path = args.output_ckpt.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco2017/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--sd_residual_cache",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals")
    p.add_argument("--attack_strengths", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--id_pool_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambda_dist", type=float, default=5.0)
    p.add_argument("--logit_scale", type=float, default=100.0)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--init_scale", type=float, default=0.15)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--p_regen", type=float, default=0.3,
                   help="Probability of SD regen augmentation per step")
    p.add_argument("--att_weight", type=float, default=0.5,
                   help="Weight on attacked-path sign loss (rest goes to clean path)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
