#!/usr/bin/env python3
"""Comprehensive benchmark for v5 KeyConditionedEncoder (matched-filter decode).

Evaluates on N test images with classical + regeneration attacks.
Outputs structured JSON for comparison against VINE / WAM / TrustMark.

Attacks:
  - clean: passthrough
  - jpeg_50 / jpeg_30: JPEG compression
  - blur_1.5 / blur_2.5: Gaussian blur
  - noise_003 / noise_005: Gaussian noise (sigma=0.03, 0.05)
  - crop_70: center crop 70% + resize back
  - regen_010 / regen_020 / regen_030 / regen_040: SD img2img

Metrics per attack:
  - bit_accuracy: fraction of 127 codeword bits correct
  - bch_detected: BCH decode success (payload matches)
  - tpr_1pct: bit_accuracy >= threshold for TPR@1%FPR
  - psnr / ssim: image quality (watermarked vs clean, for clean attack only)

Usage:
  python scripts/benchmark_v5.py \
      --ckpt results/key_encoder_v1/enc_mf_regen60.pt \
      --src_dir /project/.../KCMP/EXP_data/train2017 \
      --n_images 100 --out_dir results/benchmark_v5
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
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.key_encoder import KeyConditionedEncoder, count_params
from src.payload import BCHCodec, image_id_to_payload
from src.sign_envelope import derive_keyed_constants

DEVICE = "cuda"


# ============================================================ helpers

def pil_to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1) * 2 - 1).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t):
    arr = ((t[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_psnr(pil_a, pil_b):
    a = np.asarray(pil_a, dtype=np.float64) / 255.0
    b = np.asarray(pil_b, dtype=np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 60.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(pil_a, pil_b):
    try:
        from skimage.metrics import structural_similarity
        a = np.asarray(pil_a, dtype=np.float64) / 255.0
        b = np.asarray(pil_b, dtype=np.float64) / 255.0
        return structural_similarity(a, b, channel_axis=2, data_range=1.0)
    except ImportError:
        return float("nan")


def tpr_threshold(n_bits=127, fpr=0.01):
    from scipy.stats import binom
    for k in range(n_bits, n_bits // 2, -1):
        if binom.sf(k - 1, n_bits, 0.5) > fpr:
            return (k + 1) / n_bits
    return 0.5


# ============================================================ pixel-space region grid

def build_pixel_region_bounds(n_bits, H=256, W=256):
    grid = int(np.ceil(np.sqrt(n_bits)))
    cell_h, cell_w = H // grid, W // grid
    bounds = []
    for i in range(n_bits):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        bounds.append((y0, y1, x0, x1))
    return bounds


def build_s_pixel(sign_per_region, region_bounds, H=256, W=256):
    S = torch.zeros(1, 1, H, W, device=DEVICE)
    for i, (y0, y1, x0, x1) in enumerate(region_bounds):
        S[0, 0, y0:y1, x0:x1] = float(sign_per_region[i])
    return S


# ============================================================ embed + decode

def embed(pil, encoder, master_key, image_id, codec, region_bounds):
    x = pil_to_tensor(pil)
    payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    codeword = codec.encode(payload)
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)

    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    bit_at_region = codeword[inv_perm]
    sign_per_region = M.astype(np.float32) * (1.0 - 2.0 * bit_at_region.astype(np.float32))

    S_pixel = build_s_pixel(sign_per_region, region_bounds)

    encoder.eval()
    with torch.no_grad():
        x_w = encoder(x, S_pixel)
    return tensor_to_pil(x_w)


def decode_matched_filter(pil_orig, pil_suspect, master_key, image_id, codec, region_bounds):
    x_orig = np.asarray(pil_orig, dtype=np.float32) / 255.0 * 2 - 1
    x_susp = np.asarray(pil_suspect, dtype=np.float32) / 255.0 * 2 - 1
    x_orig = x_orig.transpose(2, 0, 1)
    x_susp = x_susp.transpose(2, 0, 1)
    residual = (x_susp - x_orig).astype(np.float32)

    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    bits = np.zeros(codec.n, dtype=np.uint8)
    for j in range(codec.n):
        r = int(perm[j])
        y0, y1, x0, x1 = region_bounds[r]
        val = float(residual[:, y0:y1, x0:x1].mean())
        region_sign = 1 if val >= 0 else -1
        decoded_sign = region_sign * int(M[r])
        bits[j] = 0 if decoded_sign > 0 else 1

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)
    bit_acc = float(np.mean(bits == expected_cw))
    payload_dec, n_err = codec.decode(bits)
    bch_ok = payload_dec is not None and np.array_equal(payload_dec, expected_payload)

    return {"bit_accuracy": bit_acc, "bch_detected": bool(bch_ok), "n_errors": int(n_err)}


# ============================================================ attacks

_SD_PIPE = None

def load_sd():
    global _SD_PIPE
    if _SD_PIPE is None:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        SD_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_PATH, torch_dtype=torch.float16, safety_checker=None,
        ).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPE = pipe
    return _SD_PIPE


def attack_jpeg(pil, quality):
    from io import BytesIO
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def attack_blur(pil, sigma):
    return pil.filter(ImageFilter.GaussianBlur(radius=sigma))


def attack_noise(pil, sigma):
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = arr + np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * sigma
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def attack_crop(pil, ratio):
    W, H = pil.size
    cw, ch = int(W * ratio), int(H * ratio)
    left = (W - cw) // 2
    top = (H - ch) // 2
    cropped = pil.crop((left, top, left + cw, top + ch))
    return cropped.resize((W, H), Image.BILINEAR)


def attack_regen(pil, strength):
    pipe = load_sd()
    g = torch.Generator(DEVICE).manual_seed(42)
    out = pipe(prompt="", image=pil, strength=strength,
               num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
    if out.size != pil.size:
        out = out.resize(pil.size, Image.BILINEAR)
    return out


def apply_attack(name, pil):
    if name == "clean":
        return pil
    if name == "jpeg_50":
        return attack_jpeg(pil, 50)
    if name == "jpeg_30":
        return attack_jpeg(pil, 30)
    if name == "blur_1.5":
        return attack_blur(pil, 1.5)
    if name == "blur_2.5":
        return attack_blur(pil, 2.5)
    if name == "noise_003":
        return attack_noise(pil, 0.03)
    if name == "noise_005":
        return attack_noise(pil, 0.05)
    if name == "crop_70":
        return attack_crop(pil, 0.7)
    if name == "regen_010":
        return attack_regen(pil, 0.10)
    if name == "regen_020":
        return attack_regen(pil, 0.20)
    if name == "regen_030":
        return attack_regen(pil, 0.30)
    if name == "regen_040":
        return attack_regen(pil, 0.40)
    raise ValueError(f"Unknown attack: {name}")


# ============================================================ main

ATTACK_SETS = {
    "classical": ["clean", "jpeg_50", "jpeg_30", "blur_1.5", "blur_2.5",
                   "noise_003", "noise_005", "crop_70"],
    "regen": ["regen_010", "regen_020", "regen_030", "regen_040"],
    "all": ["clean", "jpeg_50", "jpeg_30", "blur_1.5", "blur_2.5",
            "noise_003", "noise_005", "crop_70",
            "regen_010", "regen_020", "regen_030", "regen_040"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--src_dir", required=True,
                   help="Directory of clean test images")
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--attack_set", default="all",
                   choices=list(ATTACK_SETS.keys()),
                   help="Which attacks to run: classical, regen, or all")
    p.add_argument("--skip_regen", action="store_true",
                   help="Skip SD regen attacks (faster, classical only)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_key = args.master_key.encode("utf-8")
    codec = BCHCodec()
    thr = tpr_threshold(codec.n, fpr=0.01)
    thr_01 = tpr_threshold(codec.n, fpr=0.001)
    print(f"[setup] BCH(n={codec.n}, k={codec.data_bits}, t={codec.t})")
    print(f"[setup] TPR threshold @1%FPR: {thr:.4f}")
    print(f"[setup] TPR threshold @0.1%FPR: {thr_01:.4f}")

    # Load encoder
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    base_ch = int(ckpt.get("base_ch", 64))
    n_blocks = int(ckpt.get("n_blocks", 8))
    encoder = KeyConditionedEncoder(base_ch=base_ch, n_blocks=n_blocks).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"[setup] encoder: {count_params(encoder):,} params, scale={encoder.scale.item():.4f}")

    region_bounds = build_pixel_region_bounds(codec.n, args.resolution, args.resolution)

    # Load test images
    src_dir = Path(args.src_dir)
    image_files = sorted([f for f in src_dir.iterdir()
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.n_images]
    print(f"[setup] {len(image_files)} test images from {src_dir}")

    attacks = ATTACK_SETS[args.attack_set]
    if args.skip_regen:
        attacks = [a for a in attacks if not a.startswith("regen_")]
    print(f"[setup] attacks: {attacks}")

    # Run benchmark
    results_per_attack = {a: [] for a in attacks}
    psnr_list = []
    ssim_list = []

    for idx, fp in enumerate(image_files):
        pil_clean = Image.open(fp).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS,
        )
        image_id = f"bench_{idx:05d}"

        pil_wm = embed(pil_clean, encoder, master_key, image_id, codec, region_bounds)

        psnr_val = compute_psnr(pil_clean, pil_wm)
        ssim_val = compute_ssim(pil_clean, pil_wm)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        for atk_name in attacks:
            t0 = time.time()
            pil_attacked = apply_attack(atk_name, pil_wm)
            t_attack = time.time() - t0

            res = decode_matched_filter(
                pil_clean, pil_attacked, master_key, image_id, codec, region_bounds,
            )
            res["tpr_1pct"] = res["bit_accuracy"] >= thr
            res["tpr_01pct"] = res["bit_accuracy"] >= thr_01
            res["image"] = fp.name
            res["t_attack"] = t_attack
            results_per_attack[atk_name].append(res)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(image_files)}] psnr={psnr_val:.1f}dB", flush=True)

    # Aggregate
    summary = {
        "method": "v5_encoder_mf",
        "ckpt": args.ckpt,
        "n_images": len(image_files),
        "resolution": args.resolution,
        "n_bits": codec.n,
        "data_bits": codec.data_bits,
        "bch_t": codec.t,
        "reference_needed": True,
        "quality": {
            "psnr_mean": float(np.mean(psnr_list)),
            "psnr_std": float(np.std(psnr_list)),
            "ssim_mean": float(np.mean(ssim_list)),
            "ssim_std": float(np.std(ssim_list)),
        },
        "attacks": {},
    }

    print(f"\n{'='*70}")
    print(f"  Encoder: {args.ckpt}")
    print(f"  Images:  {len(image_files)}")
    print(f"  PSNR:    {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB")
    print(f"  SSIM:    {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")
    print(f"{'='*70}")
    print(f"  {'Attack':<15s} {'bit_acc':>8s} {'BCH_det':>8s} {'TPR@1%':>8s} {'TPR@0.1%':>9s}")
    print(f"  {'-'*50}")

    for atk_name in attacks:
        runs = results_per_attack[atk_name]
        n = len(runs)
        bit_acc = np.mean([r["bit_accuracy"] for r in runs])
        bch_det = np.mean([float(r["bch_detected"]) for r in runs])
        tpr1 = np.mean([float(r["tpr_1pct"]) for r in runs])
        tpr01 = np.mean([float(r["tpr_01pct"]) for r in runs])
        summary["attacks"][atk_name] = {
            "bit_accuracy_mean": float(bit_acc),
            "bit_accuracy_std": float(np.std([r["bit_accuracy"] for r in runs])),
            "bch_detection_rate": float(bch_det),
            "tpr_at_1pct_fpr": float(tpr1),
            "tpr_at_01pct_fpr": float(tpr01),
        }
        print(f"  {atk_name:<15s} {bit_acc:>8.3f} {bch_det*100:>7.1f}% {tpr1*100:>7.1f}% {tpr01*100:>8.1f}%")

    print(f"{'='*70}\n")

    # Save
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "results_full.json", "w") as f:
        json.dump(results_per_attack, f, indent=2)

    print(f"[done] results -> {out_dir}")


if __name__ == "__main__":
    main()
