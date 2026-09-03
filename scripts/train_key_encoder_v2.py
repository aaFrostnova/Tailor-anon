"""Train KeyConditionedEncoder v2: matched-filter loss with regen-hardened augmentation.

Builds on train_key_encoder_mf.py (which works). Changes:
  - Stratified SD residual pool by strength for curriculum
  - Curriculum: mild regen early -> strong regen late
  - Composite attacks: regen + classical (JPEG, blur) stacked
  - Higher p_regen (0.65 default), lower p_identity
  - Include strength_0.30 (and 0.40 when available) in training
  - Resume from existing checkpoint support
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
    FlatImageDataset,
)


# ============================================================ stratified SD residual loading

def load_sd_residuals_stratified(
    cache_dir: str,
    strengths_to_load: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Load SD residuals grouped by strength bucket.

    Returns dict: "mild" (<=0.10), "medium" (0.15-0.20), "strong" (>=0.30).
    """
    buckets = {"mild": [], "medium": [], "strong": []}

    for s_dir in sorted(Path(cache_dir).glob("strength_*")):
        s_val = float(s_dir.name.split("_")[1])
        if strengths_to_load is not None and s_val not in strengths_to_load:
            continue
        if s_val <= 0.10:
            bucket = "mild"
        elif s_val <= 0.20:
            bucket = "medium"
        else:
            bucket = "strong"
        for p in sorted(s_dir.glob("*.npz")):
            d = np.load(p)
            buckets[bucket].append(d["residual"])

    result = {}
    for k, arrs in buckets.items():
        if arrs:
            pool = np.stack(arrs).astype(np.float32)
            result[k] = torch.from_numpy(pool) * 2.0
    return result


# ============================================================ augmentation v2

def _apply_blur(x: torch.Tensor) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur
    k = int(np.random.choice([3, 5, 7]))
    sigma = float(np.random.uniform(0.5, 2.0))
    return gaussian_blur(x, kernel_size=k, sigma=sigma)


def _apply_jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    from io import BytesIO
    from PIL import Image
    import torchvision.transforms.functional as TF
    out = []
    for i in range(x.shape[0]):
        pil = TF.to_pil_image(((x[i].detach().cpu() + 1) / 2).clamp(0, 1))
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(TF.to_tensor(Image.open(buf).convert("RGB")) * 2 - 1)
    return torch.stack(out).to(x.device)


def _sample_regen_residual(
    x_w: torch.Tensor,
    sd_buckets: dict[str, torch.Tensor],
    progress: float,
) -> torch.Tensor:
    """Curriculum: mild early, strong late."""
    B = x_w.shape[0]
    available, weights = [], []
    if "mild" in sd_buckets:
        available.append("mild")
        weights.append(1.0)
    if progress >= 0.25 and "medium" in sd_buckets:
        available.append("medium")
        weights.append(2.0)
    if progress >= 0.45 and "strong" in sd_buckets:
        available.append("strong")
        weights.append(3.0)

    if not available:
        return x_w + torch.randn_like(x_w) * 0.02

    weights = np.array(weights) / sum(weights)
    bucket = np.random.choice(available, p=weights)
    pool = sd_buckets[bucket]
    idx = torch.randint(0, pool.shape[0], (B,))
    return x_w + pool[idx].to(x_w.device)


def augment_v2(x_w, sd_buckets=None, progress=0.0, p_regen=0.65):
    """Augmentation with curriculum regen and composite attacks.

    Returns (x_attacked, is_differentiable).
    Budget: regen=p_regen, composite=0.10, classical=rest.
    """
    B = x_w.shape[0]
    r = np.random.random()

    # Pure regen
    if sd_buckets and r < p_regen:
        return _sample_regen_residual(x_w, sd_buckets, progress), False
    r -= p_regen

    # Composite: regen + classical
    p_composite = 0.10
    if sd_buckets and r < p_composite:
        x_regen = _sample_regen_residual(x_w, sd_buckets, progress)
        post = np.random.choice(["jpeg", "blur", "noise"])
        if post == "jpeg" and x_w.shape[2] >= 64:
            return _apply_jpeg(x_regen, quality=np.random.randint(40, 80)), False
        elif post == "blur":
            return _apply_blur(x_regen), False
        else:
            return x_regen + torch.randn_like(x_regen) * np.random.uniform(0.01, 0.04), False
    r -= p_composite

    p_rest = 1.0 - p_regen - p_composite

    # Identity
    if r < 0.15 * p_rest:
        return x_w, True
    r -= 0.15 * p_rest

    # Gaussian noise
    if r < 0.15 * p_rest:
        sigma = np.random.uniform(0.01, 0.06)
        return x_w + torch.randn_like(x_w) * sigma, True
    r -= 0.15 * p_rest

    # Blur
    if r < 0.20 * p_rest:
        return _apply_blur(x_w), True
    r -= 0.20 * p_rest

    # JPEG
    if r < 0.20 * p_rest and x_w.shape[2] >= 64:
        return _apply_jpeg(x_w, quality=np.random.randint(20, 80)), False
    r -= 0.20 * p_rest

    # Crop + resize
    if r < 0.10 * p_rest:
        ratio = np.random.uniform(0.5, 0.9)
        H, W = x_w.shape[2], x_w.shape[3]
        ch, cw = int(H * ratio), int(W * ratio)
        top = np.random.randint(0, H - ch + 1)
        left = np.random.randint(0, W - cw + 1)
        cropped = x_w[:, :, top:top+ch, left:left+cw]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False), True

    return x_w + torch.randn_like(x_w) * 0.01, True


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    codec = BCHCodec()
    master_key = args.master_key.encode("utf-8")

    id_pool_signs = build_id_pool(master_key, args.id_pool_size, codec)
    id_pool_t = torch.from_numpy(id_pool_signs).to(device)

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

    # Stratified SD residual cache
    sd_buckets = None
    if args.sd_residual_cache:
        sd_buckets = load_sd_residuals_stratified(
            args.sd_residual_cache, strengths_to_load=args.attack_strengths,
        )
        for k, v in sd_buckets.items():
            sd_buckets[k] = v.to(device)
            print(f"[setup] SD residuals [{k}]: {v.shape}")

    ds = FlatImageDataset(args.image_dir, args.resolution, args.max_images)
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    encoder = KeyConditionedEncoder(
        base_ch=args.base_ch, n_blocks=args.n_blocks, init_scale=args.init_scale,
    ).to(device)

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"[setup] resumed from {args.resume_ckpt} (scale={encoder.scale.item():.4f})")

    print(f"[setup] encoder params: {count_params(encoder):,} (matched-filter, no decoder)")

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    if not args.no_cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.01,
        )
    else:
        sched = None

    step = 0
    t0 = time.time()
    log_rows = []

    while step < args.steps:
        for x_batch in loader:
            if step >= args.steps:
                break
            x_batch = x_batch.to(device)
            B = x_batch.shape[0]

            id_indices = torch.randint(0, args.id_pool_size, (B,), device=device)
            sign_per_region = id_pool_t[id_indices]

            S_pixel = torch.zeros(B, 1, H, W, device=device)
            for i, (y0, y1, x0, x1) in enumerate(region_bounds):
                S_pixel[:, :, y0:y1, x0:x1] = sign_per_region[:, i].view(B, 1, 1, 1)

            x_w = encoder(x_batch, S_pixel)

            distortion = F.mse_loss(x_w, x_batch)
            psnr = 10.0 * torch.log10(4.0 / (distortion + 1e-8))

            progress = step / max(args.steps, 1)
            x_attacked, is_diff = augment_v2(
                x_w, sd_buckets=sd_buckets, progress=progress, p_regen=args.p_regen,
            )
            if not is_diff:
                x_attacked = x_attacked.detach()

            # Matched filter on attacked residual
            residual = x_attacked - x_batch
            rho_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_list.append(residual[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_attacked = torch.stack(rho_list, dim=1)

            # Matched filter on clean residual
            residual_clean = x_w - x_batch
            rho_clean_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_clean_list.append(residual_clean[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_clean = torch.stack(rho_clean_list, dim=1)

            logit_scale = args.logit_scale
            pred_attacked = torch.tanh(rho_attacked * logit_scale)
            pred_clean = torch.tanh(rho_clean * logit_scale)
            sign_loss_att = F.mse_loss(pred_attacked, sign_per_region)
            sign_loss_clean = F.mse_loss(pred_clean, sign_per_region)

            if progress < args.warmup_frac:
                lambda_d = 0.0
            else:
                ramp = (progress - args.warmup_frac) / (1.0 - args.warmup_frac)
                lambda_d = args.lambda_dist * ramp

            # Ramp att_weight: start at base, increase to att_weight_max
            w_att = args.att_weight + (args.att_weight_max - args.att_weight) * progress
            loss = w_att * sign_loss_att + (1.0 - w_att) * sign_loss_clean + lambda_d * distortion

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()

            if args.scale_min > 0:
                with torch.no_grad():
                    encoder.scale.clamp_(min=args.scale_min)

            if step % args.log_every == 0:
                with torch.no_grad():
                    mf_correct = ((rho_clean * sign_per_region) > 0).float().mean().item()
                    mf_att_correct = ((rho_attacked * sign_per_region) > 0).float().mean().item()
                sc = encoder.scale.item()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} "
                      f"sign_att={sign_loss_att.item():.4f} sign_clean={sign_loss_clean.item():.4f} "
                      f"psnr={psnr.item():.1f}dB mf_clean={mf_correct:.3f} mf_att={mf_att_correct:.3f} "
                      f"scale={sc:.4f} w_att={w_att:.2f} t={elapsed:.0f}s",
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
    print(f"\n[done] saved -> {args.output_ckpt}")


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
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--resume_ckpt", default=None)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--sd_residual_cache",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals")
    p.add_argument("--attack_strengths", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--id_pool_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda_dist", type=float, default=2.0)
    p.add_argument("--logit_scale", type=float, default=100.0)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--init_scale", type=float, default=0.15)
    p.add_argument("--warmup_frac", type=float, default=0.08)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2500)
    p.add_argument("--p_regen", type=float, default=0.65)
    p.add_argument("--att_weight", type=float, default=0.7,
                   help="Starting weight on attacked-path sign loss")
    p.add_argument("--att_weight_max", type=float, default=0.85,
                   help="Final weight on attacked-path sign loss (ramped over training)")
    p.add_argument("--no_cosine", action="store_true",
                   help="Use constant LR instead of cosine annealing")
    p.add_argument("--scale_min", type=float, default=0.0,
                   help="Clamp encoder scale to >= this value (prevents collapse)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
