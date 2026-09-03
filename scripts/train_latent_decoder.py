"""Train a learned region-sign decoder on top of a frozen T_lat (Option A).

Pipeline per step:
  - load frozen T_lat (from results/T_lat_v1/T_lat_msev1.npz)
  - sample x batch, image_id from pool
  - z = E(x); z_w = z + T_lat * S_lat; x_w = D(z_w)
  - x_attacked = x_w + cached_SD_residual (random) + small gaussian
  - z_susp = E(x_attacked); r_lat = z_susp - z
  - dec_in = concat(r_lat, T_lat_broadcast) -> (B, 8, 32, 32)
  - logits = decoder(dec_in) -> (B, 127)
  - target_bits = (sign_per_region > 0).float()
  - BCE-with-logits loss
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.latent_decoder import LatentDecoder, LatentDecoderXR  # noqa: E402
from src.payload import BCHCodec  # noqa: E402

# reuse helpers from train_T_lat.py
sys.path.insert(0, str(REPO / "scripts"))
from train_T_lat import (   # noqa: E402
    load_vae, make_latent_region_masks, build_id_pool,
    load_sd_residual_pool, FlatImageDataset,
)


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

    # Frozen T_lat from prior msev1 training
    d = np.load(args.t_lat_ckpt, allow_pickle=True)
    T_lat = torch.from_numpy(d["T_lat"]).to(device)
    T_lat.requires_grad_(False)
    print(f"[setup] T_lat shape={tuple(T_lat.shape)} rms={T_lat.pow(2).mean().sqrt().item():.4f} (frozen)")

    # Region masks
    region_masks_np = make_latent_region_masks(codec.n, (4, 32, 32))
    region_masks = torch.from_numpy(region_masks_np).to(device)
    region_masks_flat = region_masks.view(codec.n, -1).float()

    # id pool — MUST match what eval will use; the master_key is what binds
    # the target sign pattern. For training we use a separate "training pool"
    # that doesn't overlap eval image_ids.
    master_key_train = args.master_key.encode("utf-8")
    id_pool = build_id_pool(master_key_train, args.id_pool_size, codec)
    id_pool_t = torch.from_numpy(id_pool).to(device)
    print(f"[setup] id pool: {id_pool_t.shape}")

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

    # ---- learnable decoder (region-aware, uses region_masks for pooling)
    if args.arch == "xr":
        decoder = LatentDecoderXR(
            in_ch_residual=4, n_bits=codec.n, feat_ch=64,
            region_masks=region_masks,
            n_layers=args.xr_layers, n_heads=4,
        ).to(device)
    else:
        decoder = LatentDecoder(
            in_ch_residual=4, n_bits=codec.n, feat_ch=64,
            region_masks=region_masks,
        ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[setup] decoder arch={args.arch} params: {n_params:,}")

    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    if args.cosine_lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
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

            # encode host
            with torch.no_grad():
                z = vae.encode(x_batch).latent_dist.mean * vae_scale

            # sample id pool
            id_indices = torch.randint(0, id_pool_t.shape[0], (B,), device=device)
            sign_per_region = id_pool_t[id_indices]   # (B, n_bits) ±1

            # build per-sample sign field S
            S_per_sample = torch.einsum(
                "bi,ichw->bchw", sign_per_region, region_masks.float(),
            )

            # embed (T_lat frozen)
            with torch.no_grad():
                z_w = z + T_lat.unsqueeze(0) * S_per_sample
                x_w = vae.decode(z_w / vae_scale).sample

                # Attack
                p_attack = float(args.p_attack)
                if sd_pool is not None and np.random.random() < p_attack:
                    idx = torch.randint(0, sd_pool.shape[0], (B,), device=device)
                    residual = sd_pool[idx]
                    x_attacked = x_w + residual
                else:
                    x_attacked = x_w
                x_attacked = x_attacked + torch.randn_like(x_attacked) * args.noise_sigma

                # re-encode
                z_susp = vae.encode(x_attacked).latent_dist.mean * vae_scale
                r_lat = z_susp - z

            # learned decoder: (r_lat, T_lat) -> (B, 127) logits
            logits = decoder(r_lat, T_lat)

            # target: (sign > 0) → 1, (sign < 0) → 0
            target_bits = (sign_per_region > 0).float()
            loss = F.binary_cross_entropy_with_logits(logits, target_bits)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    pred_sign = torch.where(logits >= 0, 1.0, -1.0)
                    bit_acc = (pred_sign * sign_per_region > 0).float().mean().item()
                elapsed = time.time() - t0
                lr_cur = opt.param_groups[0]["lr"]
                print(f"[step {step:6d}] loss={loss.item():.4f} bit_acc={bit_acc:.3f} "
                      f"lr={lr_cur:.2e} t={elapsed:.0f}s",
                      flush=True)
                log_rows.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "bit_acc": float(bit_acc),
                    "lr": float(lr_cur),
                    "elapsed_s": float(elapsed),
                })

            step += 1

    # Save decoder weights + config
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "state_dict": decoder.state_dict(),
        "config": vars(args),
        "n_bits": codec.n,
        "in_ch_residual": 4,
        "feat_ch": 64,
        "arch": args.arch,
        "xr_layers": args.xr_layers,
    }, args.output_ckpt)
    with open(args.output_ckpt.replace(".pt", "_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"\n[save] decoder -> {args.output_ckpt}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default=str(REPO / "images"))
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--t_lat_ckpt",
                   default=str(REPO / "results" / "T_lat_v1" / "T_lat_msev1.npz"))
    p.add_argument("--master_key", default="latent_train_master_key_v1")
    p.add_argument("--sd_residual_cache",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals")
    p.add_argument("--attack_strengths", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--id_pool_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--p_attack", type=float, default=0.7)
    p.add_argument("--noise_sigma", type=float, default=0.005)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--cosine_lr", action="store_true",
                   help="Use cosine annealing LR schedule (default: constant)")
    p.add_argument("--arch", choices=["base", "xr"], default="base",
                   help="Decoder architecture: base (per-region linear) or "
                        "xr (cross-region transformer head)")
    p.add_argument("--xr_layers", type=int, default=2,
                   help="Number of transformer encoder layers (xr only)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
