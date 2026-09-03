"""Train KeyConditionedEncoder v4: VAE-aware matched-filter training.

Key insight from VINE: watermark must survive VAE encode/decode roundtrip.
SD regen = VAE_encode → noise → UNet_denoise → VAE_decode.
If the watermark doesn't survive even VAE roundtrip alone, it has no chance
against full regen.

Changes over v3:
  - Loads frozen SD VAE for differentiable roundtrip augmentation
  - p_vae_roundtrip: probability of pure VAE roundtrip (differentiable!)
  - Gradient flows through VAE back to encoder, teaching it to produce
    perturbations in the VAE's "survivable" subspace
  - SD residual cache still used for non-differentiable regen augmentation
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
    load_vae,
)


# ============================================================ augmentation with VAE roundtrip

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


@torch.no_grad()
def _sample_regen_residual(x_w, sd_pool, B):
    idx = torch.randint(0, sd_pool.shape[0], (B,))
    return x_w + sd_pool[idx].to(x_w.device)


def vae_roundtrip(x, vae, vae_scale, noise_std=0.0):
    """Differentiable VAE encode → (optional noise) → decode."""
    z = vae.encode(x).latent_dist.mean * vae_scale
    if noise_std > 0:
        z = z + torch.randn_like(z) * noise_std
    return vae.decode(z / vae_scale).sample


def augment_vae_aware(
    x_w, vae=None, vae_scale=1.0, sd_pool=None, progress=0.0,
    p_vae=0.30, p_vae_noisy=0.15, p_regen=0.30, p_identity=0.05,
):
    """Augmentation with differentiable VAE roundtrip paths.

    Returns (x_attacked, is_differentiable).

    Budget (sums to 1.0):
      p_vae=0.30:        pure VAE roundtrip (differentiable)
      p_vae_noisy=0.15:  VAE roundtrip + latent noise (differentiable)
      p_regen=0.30:      cached SD residual (non-differentiable)
      p_identity=0.05:   passthrough (differentiable)
      rest=0.20:         classical (blur/jpeg/noise/crop)
    """
    B = x_w.shape[0]
    r = np.random.random()

    # VAE roundtrip (differentiable — this is the key new path)
    if vae is not None and r < p_vae:
        return vae_roundtrip(x_w, vae, vae_scale), True
    r -= p_vae

    # VAE roundtrip + latent noise (differentiable, simulates mild regen)
    if vae is not None and r < p_vae_noisy:
        noise_std = np.random.uniform(0.1, 0.5) * vae_scale
        return vae_roundtrip(x_w, vae, vae_scale, noise_std=noise_std), True
    r -= p_vae_noisy

    # SD regen residual (non-differentiable)
    if sd_pool is not None and r < p_regen:
        return _sample_regen_residual(x_w, sd_pool, B), False
    r -= p_regen

    # Identity
    if r < p_identity:
        return x_w, True
    r -= p_identity

    # Classical attacks (remaining ~20%)
    p_rest = 1.0 - p_vae - p_vae_noisy - p_regen - p_identity
    r_norm = r / max(p_rest, 1e-8)

    if r_norm < 0.25:
        sigma = np.random.uniform(0.01, 0.06)
        return x_w + torch.randn_like(x_w) * sigma, True

    if r_norm < 0.50:
        return _apply_blur(x_w), True

    if r_norm < 0.75 and x_w.shape[2] >= 64:
        return _apply_jpeg(x_w, quality=np.random.randint(20, 80)), False

    if r_norm < 0.90:
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

    # Load frozen VAE for differentiable roundtrip
    vae = load_vae(device)
    vae_scale = float(vae.config.scaling_factor)
    print(f"[setup] VAE loaded, scaling_factor={vae_scale:.4f}")

    # SD residual cache (flat pool, no stratification needed here)
    sd_pool = None
    if args.sd_residual_cache:
        sd_pool = load_sd_residual_pool(
            args.sd_residual_cache, strengths_to_load=args.attack_strengths,
        )
        if sd_pool is not None:
            sd_pool = sd_pool.to(device)
            print(f"[setup] SD residual pool: {sd_pool.shape}")

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

    print(f"[setup] encoder params: {count_params(encoder):,} (VAE-aware MF training)")

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
            x_attacked, is_diff = augment_vae_aware(
                x_w, vae=vae, vae_scale=vae_scale, sd_pool=sd_pool,
                progress=progress,
                p_vae=args.p_vae, p_vae_noisy=args.p_vae_noisy,
                p_regen=args.p_regen,
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

            w_att = args.att_weight
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
                      f"scale={sc:.4f} t={elapsed:.0f}s",
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_dist", type=float, default=0.2)
    p.add_argument("--logit_scale", type=float, default=100.0)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--init_scale", type=float, default=0.15)
    p.add_argument("--warmup_frac", type=float, default=0.08)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2500)
    p.add_argument("--p_vae", type=float, default=0.30,
                   help="Prob of pure VAE roundtrip augmentation (differentiable)")
    p.add_argument("--p_vae_noisy", type=float, default=0.15,
                   help="Prob of VAE roundtrip + latent noise (differentiable)")
    p.add_argument("--p_regen", type=float, default=0.30,
                   help="Prob of cached SD regen residual (non-differentiable)")
    p.add_argument("--att_weight", type=float, default=0.8)
    p.add_argument("--no_cosine", action="store_true")
    p.add_argument("--scale_min", type=float, default=0.04)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
