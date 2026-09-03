"""Train T_lat (latent-domain manifold-aligned template) for v4.

Reproduces the v3 training philosophy in latent space:
  - One learnable T_lat tensor of shape (4, 32, 32)
  - Per-step sample (image_id) from a pool; derive (sigma, M, codeword)
  - Embed in SD-v1-5 VAE latent space
  - Apply pixel-domain attack (identity, SD-residual cache, mild noise)
  - Re-encode attacked image, compute residual, matched filter, BCE loss
  - PSNR hinge to bound visual distortion
Goal: PSNR >= 30 dB with regen_mild TPR@1%FPR = 100% on held-out 20 images.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.payload import BCHCodec, image_id_to_payload  # noqa: E402
from src.sign_envelope import derive_keyed_constants  # noqa: E402

SD_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"


# ============================================================ helpers

def load_vae(device: str):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(SD_PATH, subfolder="vae").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def make_latent_region_masks(n_bits: int = 127, latent_shape=(4, 32, 32)):
    """Tile (4, 32, 32) latent into n_bits regions on the (32, 32) plane.

    All 4 channels share the same spatial mask. Returns (n_bits, *latent_shape)
    bool tensor.
    """
    C, H, W = latent_shape
    grid = int(np.ceil(np.sqrt(n_bits)))   # 12 for n_bits=127
    cell_h = H // grid
    cell_w = W // grid
    masks = np.zeros((n_bits, C, H, W), dtype=bool)
    for i in range(n_bits):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        masks[i, :, y0:y1, x0:x1] = True
    return masks  # bool numpy


def build_id_pool(master_key: bytes, pool_size: int, codec: BCHCodec):
    """Pre-compute (codeword_bits, sigma, M) for a pool of image_ids."""
    pool = []
    for i in range(pool_size):
        image_id = f"trainpool_{i:06d}"
        payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
        codeword = codec.encode(payload).astype(np.int8)
        perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
        # target sign per region: M[i] * (1 - 2*codeword[inv_perm[i]])
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        bit_at_region = codeword[inv_perm]   # (n_bits,)
        target_sign = M.astype(np.int8) * (1 - 2 * bit_at_region.astype(np.int8))
        pool.append(target_sign)
    return np.stack(pool).astype(np.float32)   # (pool_size, n_bits)


def load_sd_residual_pool(cache_dir: str, strengths_to_load=None) -> torch.Tensor:
    """Load cached residuals (pixel-space, [-1, 1] scale = 2 × [0, 1] residual)."""
    arrs = []
    for s_dir in sorted(Path(cache_dir).glob("strength_*")):
        s_str = s_dir.name.split("_")[1]
        if strengths_to_load is not None and float(s_str) not in strengths_to_load:
            continue
        for p in sorted(s_dir.glob("*.npz")):
            d = np.load(p)
            arrs.append(d["residual"])    # (3, H, W) in [0, 1] scale
    if not arrs:
        return None
    pool = np.stack(arrs).astype(np.float32)   # in [0, 1] residual scale
    pool_t = torch.from_numpy(pool) * 2.0       # convert to [-1, 1] scale
    return pool_t


# ============================================================ image dataset

class FlatImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, resolution: int = 256, max_images: int = 1000):
        self.files = sorted([p for p in Path(image_dir).iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:max_images]
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pil = Image.open(self.files[idx]).convert("RGB").resize(
            (self.resolution, self.resolution), Image.BILINEAR,
        )
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1) * 2.0 - 1.0   # to [-1, 1]
        return torch.from_numpy(arr)


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- assets
    codec = BCHCodec()
    print(f"[setup] BCH(n={codec.n}, k={codec.data_bits}, t={codec.t})")

    vae = load_vae(device)
    vae_scale = float(vae.config.scaling_factor)
    print(f"[setup] VAE loaded, scaling_factor={vae_scale:.4f}")

    # Region masks for latent (n_bits, 4, 32, 32) bool
    region_masks_np = make_latent_region_masks(codec.n, (4, 32, 32))
    region_masks = torch.from_numpy(region_masks_np).to(device)   # bool (127, 4, 32, 32)
    region_masks_flat = region_masks.view(codec.n, -1).float()    # (127, 4*32*32)

    # id pool
    master_key_train = args.master_key.encode("utf-8")
    id_pool = build_id_pool(master_key_train, args.id_pool_size, codec)   # (pool_size, n_bits) in {-1, +1}
    id_pool_t = torch.from_numpy(id_pool).to(device)
    print(f"[setup] id pool: {id_pool_t.shape}")

    # Build per-sample sign tensor lookup: each id has a S_lat of shape (4, 32, 32) — but
    # all channels share the same spatial sign. We'll do the multiplication via masks.
    # Build (pool_size, 4, 32, 32) ±1 sign tensor.
    # sign_per_region: (pool_size, n_bits) → broadcast over region voxels.
    # We'll lazily build S_per_sample inside the loop.

    # Convenient: per-region voxel coverage
    region_size = region_masks_flat.sum(dim=1)   # (127,) — should be 4 × cell_h × cell_w
    print(f"[setup] region size (latent voxels): {region_size.min().item():.0f}-{region_size.max().item():.0f}")

    # SD residual cache
    sd_pool = None
    if args.sd_residual_cache:
        sd_pool = load_sd_residual_pool(args.sd_residual_cache,
                                        strengths_to_load=args.attack_strengths)
        if sd_pool is not None:
            sd_pool = sd_pool.to(device)
            print(f"[setup] SD residual pool: {sd_pool.shape}")

    # Image dataset / loader
    ds = FlatImageDataset(args.image_dir, args.resolution, args.max_images)
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ---- learnable T_lat
    # init: small Gaussian; if --fix_rms > 0, we renormalize T_lat to that RMS
    # after each step (PSNR hinge disabled in that mode).
    T_lat = torch.randn(4, 32, 32, device=device) * args.init_std
    if args.fix_rms > 0:
        with torch.no_grad():
            cur = T_lat.pow(2).mean().sqrt()
            T_lat.mul_(args.fix_rms / (cur + 1e-12))
    T_lat = torch.nn.Parameter(T_lat, requires_grad=True)
    opt = torch.optim.Adam([T_lat], lr=args.lr)
    print(f"[setup] T_lat init: mean={T_lat.mean().item():.4f}, std={T_lat.std().item():.4f}, "
          f"fix_rms={args.fix_rms}")

    # ---- training loop
    step = 0
    t0 = time.time()
    log_rows = []
    while step < args.steps:
        for x_batch in loader:
            if step >= args.steps:
                break
            x_batch = x_batch.to(device)   # (B, 3, H, W) in [-1, 1]
            B = x_batch.shape[0]

            # encode
            z = vae.encode(x_batch).latent_dist.mean * vae_scale   # (B, 4, 32, 32)

            # sample image_ids
            id_indices = torch.randint(0, id_pool_t.shape[0], (B,), device=device)
            sign_per_region = id_pool_t[id_indices]   # (B, n_bits) ±1

            # build S_lat tensor for each sample
            # S_lat[b, c, h, w] = sign_per_region[b, region_of(h,w)]
            # Equivalent: outer product over region_masks
            # S[b] = sum_i sign[b, i] * mask_i  (mask covers disjoint regions, so this is well-defined; outside regions S=0 → we treat as +1)
            S_per_sample = torch.einsum(
                "bi,ichw->bchw", sign_per_region, region_masks.float(),
            )
            # voxels not covered by any region → S = 0. Replace with 0; T*S there is 0 anyway.
            # Embed
            z_w = z + T_lat.unsqueeze(0) * S_per_sample
            x_w = vae.decode(z_w / vae_scale).sample   # (B, 3, H, W) in ~[-1, 1]

            # PSNR
            diff = x_w - x_batch
            mse = (diff ** 2).mean(dim=(1, 2, 3))
            psnr_per = 10.0 * torch.log10(4.0 / (mse + 1e-8))
            psnr_mean = psnr_per.mean()

            # Attack
            p_attack = float(args.p_attack)
            if sd_pool is not None and np.random.random() < p_attack:
                idx = torch.randint(0, sd_pool.shape[0], (B,), device=device)
                residual = sd_pool[idx]   # (B, 3, H, W) already in [-1, 1] residual scale
                x_attacked = x_w + residual
            else:
                x_attacked = x_w
            # small Gaussian noise for general robustness
            x_attacked = x_attacked + torch.randn_like(x_attacked) * args.noise_sigma

            # re-encode
            z_susp = vae.encode(x_attacked).latent_dist.mean * vae_scale
            r_lat = z_susp - z   # baseline = clean host's latent

            # matched filter per region
            # rho[b, i] = sum_{voxels in R_i} (r_lat[b] * T_lat)[voxel]
            weighted = (r_lat * T_lat.unsqueeze(0)).view(B, -1)   # (B, V)
            rho = weighted @ region_masks_flat.T                  # (B, n_bits)

            # Normalize per region (mean instead of sum) for stable logits
            rho = rho / region_size.unsqueeze(0)

            # Soft-sign MSE loss (v3 trainer style)
            # target: sign_per_region (±1); predicted: tanh(rho * scale) in (-1, 1)
            pred_sign = torch.tanh(rho * float(args.logit_scale))     # (B, n_bits)
            sign_loss = F.mse_loss(pred_sign, sign_per_region)

            # Direct pixel L2 (encourages high PSNR)
            image_loss = F.mse_loss(x_w, x_batch)

            if args.fix_rms > 0:
                # frozen-magnitude mode: ignore image_loss, only optimize direction
                loss = sign_loss
            else:
                loss = sign_loss + args.lambda_image * image_loss

            # keep bce var for backward log compat
            bce = sign_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            # re-project to target RMS if configured
            if args.fix_rms > 0:
                with torch.no_grad():
                    cur = T_lat.pow(2).mean().sqrt()
                    T_lat.mul_(args.fix_rms / (cur + 1e-12))

            # Logging
            if step % args.log_every == 0:
                with torch.no_grad():
                    bit_acc = ((pred_sign * sign_per_region > 0).float().mean().item())
                    t_lat_rms = T_lat.pow(2).mean().sqrt().item()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} bce={bce.item():.4f} "
                      f"psnr={psnr_mean.item():5.2f}dB bit_acc={bit_acc:.3f} "
                      f"T_rms={t_lat_rms:.4f} t={elapsed:.0f}s")
                log_rows.append({
                    "step": step, "loss": float(loss.item()), "bce": float(bce.item()),
                    "psnr": float(psnr_mean.item()), "bit_acc": float(bit_acc),
                    "T_rms": float(t_lat_rms), "elapsed_s": float(elapsed),
                })

            step += 1

    # Save
    out = {
        "T_lat": T_lat.detach().cpu().numpy(),
        "config": vars(args),
        "master_key_train": args.master_key,
        "log": log_rows,
    }
    np.savez_compressed(args.output_ckpt, **{k: v for k, v in out.items()
                                              if not isinstance(v, list)})
    # Save log separately as JSON
    import json
    with open(args.output_ckpt.replace(".npz", "_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"\n[save] T_lat → {args.output_ckpt}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default=str(REPO / "images"))
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--master_key", default="latent_train_master_key_v1")
    p.add_argument("--sd_residual_cache",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals")
    p.add_argument("--attack_strengths", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--id_pool_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--init_std", type=float, default=0.05,
                   help="Initial T_lat std before training")
    p.add_argument("--target_psnr", type=float, default=32.0)
    p.add_argument("--lambda_psnr", type=float, default=0.3)
    p.add_argument("--p_attack", type=float, default=0.7,
                   help="Probability of applying SD-residual attack each step")
    p.add_argument("--noise_sigma", type=float, default=0.005,
                   help="Pixel-space Gaussian noise added to attacked image")
    p.add_argument("--logit_scale", type=float, default=20.0,
                   help="Multiplier on rho * sign before sigmoid")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--fix_rms", type=float, default=0.0,
                   help="If > 0, project T_lat to this RMS after each step (disables PSNR hinge)")
    p.add_argument("--lambda_image", type=float, default=100.0,
                   help="Weight on pixel-domain MSE loss (only used when fix_rms == 0)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
