#!/usr/bin/env python3
"""Benchmark multi-method redundant fingerprint against all attacks.

Tests which methods survive which attacks, and whether the combined
detection (any method succeeds) provides universal coverage.
"""

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

from src.multi_method_fingerprint import MultiMethodFingerprint

DEVICE = "cuda"


def compute_psnr(pil_a, pil_b):
    a = np.asarray(pil_a, dtype=np.float64) / 255.0
    b = np.asarray(pil_b, dtype=np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 60.0


# ============================================================ attacks

_SD_PIPES = {}

SD_MODELS = {
    "sd15": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5",
    "sd21": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1",
    "sdturbo": "/project/pi_shiqingma_umass_edu/mingzheli/model/sd-turbo",
}

def load_sd(model_key="sd15"):
    global _SD_PIPES
    if model_key not in _SD_PIPES:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        path = SD_MODELS[model_key]
        print(f"  [loading {model_key} from {path}]", flush=True)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            path, torch_dtype=torch.float16, safety_checker=None,
        ).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPES[model_key] = pipe
    return _SD_PIPES[model_key]


def apply_attack(name, pil):
    if name == "clean":
        return pil
    if name == "jpeg_50":
        from io import BytesIO
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=50); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "jpeg_30":
        from io import BytesIO
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=30); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "blur_2.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=2.5))
    if name == "noise_005":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr += np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.05
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "crop_70":
        W, H = pil.size
        cw, ch = int(W * 0.7), int(H * 0.7)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    if name.startswith("regen_"):
        # Format: regen_010 (sd15 default) or regen_010_sd21 or regen_010_sdturbo
        parts = name.split("_")
        strength = int(parts[1]) / 100.0
        model_key = parts[2] if len(parts) > 2 and parts[2] in SD_MODELS else "sd15"
        pipe = load_sd(model_key)
        g = torch.Generator(DEVICE).manual_seed(42)
        out = pipe(prompt="", image=pil, strength=strength,
                   num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
        if out.size != pil.size:
            out = out.resize(pil.size, Image.BILINEAR)
        return out
    raise ValueError(f"Unknown attack: {name}")


# ============================================================ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--n_images", type=int, default=50)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--out_dir", default="results/benchmark_multi_method")
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "jpeg_50", "jpeg_30", "blur_2.5", "noise_005",
        "crop_70",
        "regen_010", "regen_020", "regen_030",
        "regen_010_sd21", "regen_010_sdturbo",
    ])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_key = args.master_key.encode("utf-8")

    print("[setup] Initializing multi-method fingerprint...")
    mf = MultiMethodFingerprint(
        master_key=master_key, device=DEVICE, use_vine=True,
    )

    src_dir = Path(args.src_dir)
    files = sorted([f for f in src_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.n_images]
    print(f"[setup] {len(files)} test images, {len(args.attacks)} attacks")

    results_per_attack = {a: [] for a in args.attacks}
    psnr_list = []

    for idx, fp in enumerate(files):
        pil_clean = Image.open(fp).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS,
        )
        image_id = f"bench_{idx:05d}"

        t0 = time.time()
        pil_wm = mf.embed(pil_clean, image_id)
        t_embed = time.time() - t0

        psnr_val = compute_psnr(pil_clean, pil_wm)
        psnr_list.append(psnr_val)

        for atk_name in args.attacks:
            pil_att = apply_attack(atk_name, pil_wm)
            result = mf.detect(pil_att, image_id, pil_original=pil_clean)
            result["image"] = fp.name
            result["attack"] = atk_name
            results_per_attack[atk_name].append(result)

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(files)}] psnr={psnr_val:.1f}dB t_embed={t_embed:.1f}s",
                  flush=True)

    # Report
    print(f"\n{'='*70}")
    print(f"  Multi-Method Fingerprint ({len(files)} images)")
    print(f"  PSNR: {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB")
    print(f"{'='*70}")

    methods = set()
    for runs in results_per_attack.values():
        for r in runs:
            methods.update(r["per_method"].keys())
    methods = sorted(methods)

    header = f"  {'Attack':<15s} {'ANY':>5s}"
    for m in methods:
        header += f" {m:>8s}"
    print(header)
    print(f"  {'-'*(15 + 6 + 9*len(methods))}")

    summary = {"psnr_mean": float(np.mean(psnr_list)), "attacks": {}}

    for atk_name in args.attacks:
        runs = results_per_attack[atk_name]
        any_det = np.mean([float(r["detected"]) for r in runs])
        per_m = {}
        row = f"  {atk_name:<15s} {any_det*100:>4.0f}%"
        for m in methods:
            # Use bit_accuracy for VINE, correlation for classical methods
            if m == "vine":
                vals = [r["per_method"].get(m, {}).get("bit_accuracy", 0) for r in runs]
            else:
                vals = [r["per_method"].get(m, {}).get("correlation", 0) for r in runs]
            dets = [float(r["per_method"].get(m, {}).get("detected", False)) for r in runs]
            per_m[m] = {"score": float(np.mean(vals)), "det_rate": float(np.mean(dets))}
            row += f" {np.mean(dets)*100:>6.0f}%"
        print(row)
        summary["attacks"][atk_name] = {
            "any_detected": float(any_det),
            "per_method": per_m,
        }

    print(f"{'='*70}")
    print(f"  (per-method columns show detection rate %)")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "results_full.json", "w") as f:
        json.dump(results_per_attack, f, indent=2)
    print(f"\n[done] results -> {out_dir}")


if __name__ == "__main__":
    main()
