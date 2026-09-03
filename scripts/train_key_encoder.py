"""Train KeyConditionedEncoder (v5): learned encoder conditioned on crypto S_lat.

Pipeline per step:
  - sample x from dataset, image_id from pool
  - derive (σ, M, codeword) → S_lat (4,32,32) → S_pixel (1,256,256)
  - x_w = encoder(x, S_pixel)
  - x_attacked = augment(x_w)  (JPEG / blur / crop / SD-regen-cache / identity)
  - decoder: matched filter on (x_attacked - x) with T derived from encoder output
    OR learned decoder CNN
  - loss = BCE(decoded_bits, target) + λ_dist · MSE(x_w, x)
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

from src.key_encoder import KeyConditionedEncoder, SimpleDecoder, s_lat_to_pixel, count_params  # noqa
from src.payload import BCHCodec, image_id_to_payload  # noqa
from src.sign_envelope import derive_keyed_constants  # noqa
from train_T_lat import (  # noqa
    load_vae, make_latent_region_masks, build_id_pool,
    load_sd_residual_pool, FlatImageDataset,
)

# ============================================================ augmentations

def augment_differentiable(
    x_w: torch.Tensor,
    sd_pool: torch.Tensor = None,
    p_identity: float = 0.2,
    p_jpeg: float = 0.15,
    p_blur: float = 0.15,
    p_crop: float = 0.1,
    p_noise: float = 0.1,
    p_regen: float = 0.3,
) -> torch.Tensor:
    """Random augmentation on (B, 3, H, W) in [-1, 1]. Non-differentiable ops
    are applied with detach so the encoder can still get gradient through the
    identity / noise paths."""
    B = x_w.shape[0]
    r = np.random.random()

    if r < p_identity:
        return x_w

    r -= p_identity
    if r < p_noise:
        sigma = np.random.uniform(0.01, 0.05)
        return x_w + torch.randn_like(x_w) * sigma

    r -= p_noise
    if r < p_blur:
        k = int(np.random.choice([3, 5, 7]))
        sigma = float(np.random.uniform(0.5, 2.0))
        from torchvision.transforms.functional import gaussian_blur
        return gaussian_blur(x_w, kernel_size=k, sigma=sigma)

    r -= p_blur
    if r < p_jpeg and x_w.shape[2] >= 64:
        quality = np.random.randint(30, 80)
        from io import BytesIO
        from PIL import Image
        import torchvision.transforms.functional as TF
        out = []
        for i in range(B):
            pil = TF.to_pil_image(((x_w[i].detach().cpu() + 1) / 2).clamp(0, 1))
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            pil_j = Image.open(buf).convert("RGB")
            t = TF.to_tensor(pil_j) * 2 - 1
            out.append(t)
        return torch.stack(out).to(x_w.device)

    r -= p_jpeg
    if r < p_crop:
        ratio = np.random.uniform(0.6, 0.9)
        H, W = x_w.shape[2], x_w.shape[3]
        ch, cw = int(H * ratio), int(W * ratio)
        top = np.random.randint(0, H - ch + 1)
        left = np.random.randint(0, W - cw + 1)
        cropped = x_w[:, :, top:top+ch, left:left+cw]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

    r -= p_crop
    if sd_pool is not None and r < p_regen:
        idx = torch.randint(0, sd_pool.shape[0], (B,), device=x_w.device)
        residual = sd_pool[idx]
        return x_w + residual

    return x_w + torch.randn_like(x_w) * 0.01


# ============================================================ matched filter decoder (numpy, no grad)

def matched_filter_decode(
    x_orig_np: np.ndarray,
    x_susp_np: np.ndarray,
    perm: np.ndarray,
    M: np.ndarray,
    n_bits: int = 127,
) -> np.ndarray:
    """Simple pixel-domain matched filter: per-region sign of (x_susp - x_orig).

    Uses the same 12×12 grid as latent but in pixel space (3, 256, 256).
    Each region's mean residual sign → recovered bit.

    Returns (n_bits,) uint8 array of recovered codeword bits.
    """
    residual = (x_susp_np - x_orig_np).astype(np.float32)
    C, H, W = residual.shape
    grid = int(np.ceil(np.sqrt(n_bits)))
    cell_h = H // grid
    cell_w = W // grid

    region_signs = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        val = float(residual[:, y0:y1, x0:x1].mean())
        region_signs[i] = 1 if val >= 0 else -1

    bits = np.zeros(n_bits, dtype=np.uint8)
    for j in range(n_bits):
        r_idx = int(perm[j])
        decoded_sign = int(region_signs[r_idx]) * int(M[r_idx])
        bits[j] = 0 if decoded_sign > 0 else 1
    return bits


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    codec = BCHCodec()
    print(f"[setup] BCH(n={codec.n}, k={codec.data_bits}, t={codec.t})")

    # id pool
    master_key = args.master_key.encode("utf-8")
    id_pool_signs = build_id_pool(master_key, args.id_pool_size, codec)
    id_pool_t = torch.from_numpy(id_pool_signs).to(device)
    print(f"[setup] id pool: {id_pool_t.shape}")

    # Latent region masks → for building S_lat per sample
    region_masks_np = make_latent_region_masks(codec.n, (4, 32, 32))
    region_masks = torch.from_numpy(region_masks_np).to(device)

    # Pixel-space region masks for matched-filter decode
    pixel_grid = int(np.ceil(np.sqrt(codec.n)))
    print(f"[setup] pixel grid: {pixel_grid}×{pixel_grid}")

    # Also precompute full (perm, M) for each pool entry
    pool_perms = []
    pool_masks = []
    for i in range(args.id_pool_size):
        image_id = f"trainpool_{i:06d}"
        perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
        pool_perms.append(perm)
        pool_masks.append(M)
    pool_perms = np.stack(pool_perms)
    pool_masks = np.stack(pool_masks)

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

    # Encoder + Decoder (joint training)
    encoder = KeyConditionedEncoder(
        base_ch=args.base_ch, n_blocks=args.n_blocks, init_scale=args.init_scale,
    ).to(device)
    decoder = SimpleDecoder(n_bits=codec.n, base_ch=64).to(device)
    print(f"[setup] encoder params: {count_params(encoder):,}")
    print(f"[setup] decoder params: {count_params(decoder):,}")

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    if args.no_cosine:
        sched = None
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.01)

    step = 0
    t0 = time.time()
    log_rows = []

    while step < args.steps:
        for x_batch in loader:
            if step >= args.steps:
                break
            x_batch = x_batch.to(device)
            B = x_batch.shape[0]

            # Sample image_ids from pool
            id_indices = torch.randint(0, args.id_pool_size, (B,)).numpy()
            sign_per_region = id_pool_t[torch.from_numpy(id_indices).to(device)]

            # Build S_lat (4, 32, 32) per sample → S_pixel (1, 256, 256)
            S_lat_batch = torch.einsum(
                "bi,ichw->bchw", sign_per_region, region_masks.float(),
            )
            S_pixel = s_lat_to_pixel(S_lat_batch, args.resolution, args.resolution)

            # Encode
            x_w = encoder(x_batch, S_pixel)

            # Distortion loss
            distortion = F.mse_loss(x_w, x_batch)
            psnr = 10.0 * torch.log10(4.0 / (distortion + 1e-8))

            # Augment
            x_attacked = augment_differentiable(x_w, sd_pool=sd_pool)

            # Decode via pixel-domain matched filter (no grad, numpy)
            # Use the residual (x_attacked - x_orig) directly
            x_att_np = x_attacked.detach().cpu().numpy()
            x_orig_np = x_batch.detach().cpu().numpy()

            total_correct = 0
            total_bits = 0
            for b in range(B):
                perm = pool_perms[id_indices[b]]
                M = pool_masks[id_indices[b]]
                recovered = matched_filter_decode(
                    x_orig_np[b], x_att_np[b], perm, M, codec.n,
                )
                # compute expected codeword
                image_id = f"trainpool_{id_indices[b]:06d}"
                expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
                expected_cw = codec.encode(expected_payload)
                total_correct += int(np.sum(recovered == expected_cw))
                total_bits += len(expected_cw)

            bit_acc = total_correct / max(total_bits, 1)

            # Joint encoder-decoder loss (HiDDeN-style END framework):
            # Decoder takes attacked image → 127 logits, compared to target bits.
            # Gradient flows: loss → decoder → augmentation (if diff) → encoder.

            # Target: sign_per_region > 0 → bit 1;  sign_per_region < 0 → bit 0
            target_bits = (sign_per_region > 0).float()  # (B, 127)

            # Decoder on attacked path (gradients flow through diff augmentations)
            logits_attacked = decoder(x_attacked)  # (B, 127)
            bce_attacked = F.binary_cross_entropy_with_logits(logits_attacked, target_bits)

            # Decoder on clean path (always differentiable through encoder)
            logits_clean = decoder(x_w)  # (B, 127)
            bce_clean = F.binary_cross_entropy_with_logits(logits_clean, target_bits)

            sign_loss = 0.5 * bce_attacked + 0.5 * bce_clean

            # Warmup: no distortion loss initially, ramp up after warmup_frac
            progress = step / max(args.steps, 1)
            if progress < args.warmup_frac:
                lambda_d = 0.0
            else:
                ramp = (progress - args.warmup_frac) / (1.0 - args.warmup_frac)
                lambda_d = args.lambda_dist * ramp
            loss = sign_loss + lambda_d * distortion

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()

            if step % args.log_every == 0:
                lr_cur = opt.param_groups[0]["lr"]
                sc = encoder.scale.item()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} sign_l={sign_loss.item():.4f} "
                      f"dist={distortion.item():.6f} psnr={psnr.item():.1f}dB "
                      f"bit_acc={bit_acc:.3f} scale={sc:.4f} lr={lr_cur:.2e} t={elapsed:.0f}s",
                      flush=True)
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "sign_loss": float(sign_loss.item()),
                    "distortion": float(distortion.item()),
                    "psnr": float(psnr.item()),
                    "bit_acc": float(bit_acc),
                    "scale": float(sc),
                    "lr": float(lr_cur),
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(encoder, args, log_rows, decoder=decoder)

            step += 1

    _save(encoder, args, log_rows, decoder=decoder)
    print(f"\n[done] saved → {args.output_ckpt}")


def _save(encoder, args, log_rows, decoder=None):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    save_dict = {
        "encoder_state_dict": encoder.state_dict(),
        "config": vars(args),
        "n_bits": 127,
        "base_ch": args.base_ch,
        "n_blocks": args.n_blocks,
    }
    if decoder is not None:
        save_dict["decoder_state_dict"] = decoder.state_dict()
    torch.save(save_dict, args.output_ckpt)
    log_path = args.output_ckpt.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default=str(REPO / "images"))
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--sd_residual_cache",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals")
    p.add_argument("--attack_strengths", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--id_pool_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_dist", type=float, default=100.0)
    p.add_argument("--sign_margin", type=float, default=0.001,
                   help="Margin for hinge sign loss: relu(-corr + margin)")
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--init_scale", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--warmup_frac", type=float, default=0.3,
                   help="Fraction of training with zero distortion loss (encoder-only warmup)")
    p.add_argument("--no_cosine", action="store_true",
                   help="Use constant LR instead of cosine annealing")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
