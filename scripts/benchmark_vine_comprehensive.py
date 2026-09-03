#!/usr/bin/env python3
"""Comprehensive VINE-R benchmark with same attack suite as benchmark_v5.py.

Must run in the vine conda env.

Usage:
  conda activate vine
  python scripts/benchmark_vine_comprehensive.py \
      --src_dir /project/.../train2017 \
      --n_images 100 --out_dir results/benchmark_v5/vine_baseline
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
from PIL import Image, ImageFilter

VINE_REPO = "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0, VINE_REPO)
sys.path.insert(0, os.path.join(VINE_REPO, "vine", "src"))

os.environ.setdefault("HF_HOME", "/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")

from torchvision import transforms

DEVICE = "cuda"


# ============================================================ VINE encode/decode

def load_vine_encoder(model_name="Shilin-LU/VINE-R-Enc"):
    from vine_turbo import VINE_Turbo
    enc = VINE_Turbo.from_pretrained(model_name)
    enc.to(DEVICE)
    return enc


def load_vine_decoder(model_name="Shilin-LU/VINE-R-Dec"):
    from stega_encoder_decoder import CustomConvNeXt
    dec = CustomConvNeXt.from_pretrained(model_name)
    dec.to(DEVICE)
    return dec


def vine_msg_to_bits(message: str) -> torch.Tensor:
    assert len(message) <= 12
    data = bytearray(message + " " * (12 - len(message)), "utf-8")
    bits = [int(b) for c in data for b in format(c, "08b")]
    bits.extend([0, 0, 0, 0])
    return torch.tensor(bits, dtype=torch.float32)


def vine_embed(encoder, pil, msg):
    size = pil.size
    t256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    t_back = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)

    resized = t256(pil).unsqueeze(0).to(DEVICE)
    resized = 2.0 * resized - 1.0
    orig = transforms.ToTensor()(pil).unsqueeze(0).to(DEVICE)
    orig = 2.0 * orig - 1.0
    msg_in = msg.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_256 = encoder(resized, msg_in)
    residual_256 = enc_256 - resized
    residual_full = t_back(residual_256)
    encoded = residual_full + orig
    encoded = torch.clamp(encoded * 0.5 + 0.5, 0, 1)
    return transforms.ToPILImage()(encoded[0].cpu())


def vine_decode(decoder, pil):
    t256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    img = t256(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = decoder(img)
    return torch.round(pred[0].cpu().detach()).float()


# ============================================================ quality metrics

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


def apply_attack(name, pil):
    if name == "clean":
        return pil
    if name == "jpeg_50":
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "jpeg_30":
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "blur_1.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name == "blur_2.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=2.5))
    if name == "noise_003":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = arr + np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.03
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "noise_005":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = arr + np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.05
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "crop_70":
        W, H = pil.size
        cw, ch = int(W * 0.7), int(H * 0.7)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    if name.startswith("regen_"):
        strength = int(name.split("_")[1]) / 100.0
        pipe = load_sd()
        g = torch.Generator(DEVICE).manual_seed(42)
        out = pipe(prompt="", image=pil, strength=strength,
                   num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
        if out.size != pil.size:
            out = out.resize(pil.size, Image.BILINEAR)
        return out
    raise ValueError(f"Unknown attack: {name}")


ATTACK_SETS = {
    "classical": ["clean", "jpeg_50", "jpeg_30", "blur_1.5", "blur_2.5",
                   "noise_003", "noise_005", "crop_70"],
    "regen": ["regen_010", "regen_020", "regen_030", "regen_040"],
    "all": ["clean", "jpeg_50", "jpeg_30", "blur_1.5", "blur_2.5",
            "noise_003", "noise_005", "crop_70",
            "regen_010", "regen_020", "regen_030", "regen_040"],
}


# ============================================================ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--message", default="WBenchEv")
    p.add_argument("--attack_set", default="all", choices=list(ATTACK_SETS.keys()))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(args.src_dir)
    image_files = sorted([f for f in src_dir.iterdir()
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.n_images]
    print(f"[setup] {len(image_files)} test images")

    print("[setup] loading VINE encoder + decoder ...")
    enc = load_vine_encoder()
    dec = load_vine_decoder()
    msg_gt = vine_msg_to_bits(args.message)

    n_bits = 100
    attacks = ATTACK_SETS[args.attack_set]
    print(f"[setup] attacks: {attacks}")

    results_per_attack = {a: [] for a in attacks}
    psnr_list = []
    ssim_list = []

    for idx, fp in enumerate(image_files):
        pil_clean = Image.open(fp).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS,
        )

        pil_wm = vine_embed(enc, pil_clean, msg_gt)

        psnr_val = compute_psnr(pil_clean, pil_wm)
        ssim_val = compute_ssim(pil_clean, pil_wm)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        for atk_name in attacks:
            t0 = time.time()
            pil_attacked = apply_attack(atk_name, pil_wm)
            t_attack = time.time() - t0

            pred_bits = vine_decode(dec, pil_attacked)
            bit_acc = (pred_bits == msg_gt).float().mean().item()

            results_per_attack[atk_name].append({
                "image": fp.name,
                "bit_accuracy": float(bit_acc),
                "tpr_80": bit_acc >= 0.80,
                "t_attack": t_attack,
            })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(image_files)}] psnr={psnr_val:.1f}dB", flush=True)

    # Aggregate
    summary = {
        "method": "VINE-R",
        "n_images": len(image_files),
        "resolution": args.resolution,
        "n_bits": n_bits,
        "reference_needed": False,
        "quality": {
            "psnr_mean": float(np.mean(psnr_list)),
            "psnr_std": float(np.std(psnr_list)),
            "ssim_mean": float(np.mean(ssim_list)),
            "ssim_std": float(np.std(ssim_list)),
        },
        "attacks": {},
    }

    print(f"\n{'='*60}")
    print(f"  VINE-R  ({len(image_files)} images, {args.resolution}x{args.resolution})")
    print(f"  PSNR: {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB")
    print(f"  SSIM: {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")
    print(f"{'='*60}")
    print(f"  {'Attack':<15s} {'bit_acc':>8s} {'det(>=80%)':>10s}")
    print(f"  {'-'*35}")

    for atk_name in attacks:
        runs = results_per_attack[atk_name]
        bit_acc = np.mean([r["bit_accuracy"] for r in runs])
        det = np.mean([float(r["tpr_80"]) for r in runs])
        summary["attacks"][atk_name] = {
            "bit_accuracy_mean": float(bit_acc),
            "bit_accuracy_std": float(np.std([r["bit_accuracy"] for r in runs])),
            "detection_rate_80": float(det),
        }
        print(f"  {atk_name:<15s} {bit_acc:>8.3f} {det*100:>9.1f}%")

    print(f"{'='*60}\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "results_full.json", "w") as f:
        json.dump(results_per_attack, f, indent=2)

    print(f"[done] results -> {out_dir}")


if __name__ == "__main__":
    main()
