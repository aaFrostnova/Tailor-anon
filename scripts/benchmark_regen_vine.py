#!/usr/bin/env python3
"""W-Bench regeneration benchmark — VINE phase.

Runs in the `vine` conda env (torch 2.0.1 + diffusers 0.30 fork from VINE).
Loads the same 20 source images as scripts/benchmark_regen.py, watermarks
them with VINE-R-Enc, applies SD-v1-5 img2img regen attacks at matching
strength levels, and decodes via VINE-R-Dec.

Usage:
  /project/.../envs/vine/bin/python scripts/benchmark_regen_vine.py \
      --src_dir /project/.../coco_fp_global_200/images_clean \
      --n_images 20 \
      --out_dir results/wbench_regen
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# VINE-side imports
VINE_REPO = "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0, VINE_REPO)
sys.path.insert(0, os.path.join(VINE_REPO, "vine", "src"))

os.environ.setdefault("HF_HOME", "/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")


# ----------------------------------------------------------- VINE encoder/decoder

from torchvision import transforms


def load_vine_encoder(model_name: str = "Shilin-LU/VINE-R-Enc"):
    from vine_turbo import VINE_Turbo
    enc = VINE_Turbo.from_pretrained(model_name)
    enc.to("cuda")
    return enc


def load_vine_decoder(model_name: str = "Shilin-LU/VINE-R-Dec"):
    from stega_encoder_decoder import CustomConvNeXt
    dec = CustomConvNeXt.from_pretrained(model_name)
    dec.to("cuda")
    return dec


def vine_msg_to_bits(message: str) -> torch.Tensor:
    """100-bit watermark: 12-byte ASCII (UTF-8) + 4 trailing zeros.

    Matches VINE's watermark_encoding.py exactly.
    """
    assert len(message) <= 12, f"VINE supports up to 12 chars, got {len(message)}"
    data = bytearray(message + " " * (12 - len(message)), "utf-8")
    bits = [int(b) for c in data for b in format(c, "08b")]
    bits.extend([0, 0, 0, 0])
    return torch.tensor(bits, dtype=torch.float32)  # [100]


def vine_embed(encoder, pil_512: Image.Image, msg: torch.Tensor) -> Image.Image:
    """Embed at 256, upsample residual to 512, add to original 512."""
    size = pil_512.size  # (W, H)
    t256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    t_back = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)

    resized = t256(pil_512).unsqueeze(0).to("cuda")
    resized = 2.0 * resized - 1.0
    orig = transforms.ToTensor()(pil_512).unsqueeze(0).to("cuda")
    orig = 2.0 * orig - 1.0
    msg_in = msg.unsqueeze(0).to("cuda")

    with torch.no_grad():
        enc_256 = encoder(resized, msg_in)
    residual_256 = enc_256 - resized
    residual_full = t_back(residual_256)
    encoded = residual_full + orig
    encoded = encoded * 0.5 + 0.5
    encoded = torch.clamp(encoded, 0, 1)
    return transforms.ToPILImage()(encoded[0].cpu())


def vine_decode(decoder, pil: Image.Image) -> torch.Tensor:
    """Decode 100-bit watermark from arbitrary-resolution PIL image."""
    t256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    img = t256(pil).unsqueeze(0).to("cuda")
    with torch.no_grad():
        pred = decoder(img)
    pred_bits = torch.round(pred[0].cpu().detach()).to(torch.int)
    return pred_bits.float()


# --------------------------------------------------------------- SD regen attack

_SD_PIPE = None


def load_sd(model_path="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"):
    global _SD_PIPE
    if _SD_PIPE is None:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, safety_checker=None,
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPE = pipe
    return _SD_PIPE


def regen_stoch(pil: Image.Image, strength: float, seed: int = 42) -> Image.Image:
    pipe = load_sd()
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(prompt="", image=pil, strength=strength,
               num_inference_steps=50, guidance_scale=1.0,
               generator=g).images[0]
    if out.size != pil.size:
        out = out.resize(pil.size, Image.BILINEAR)
    return out


def run_attack(name: str, pil: Image.Image) -> Image.Image:
    if name == "no_attack":
        return pil
    if name == "regen_mild":
        return regen_stoch(pil, strength=0.10)
    if name == "regen_medium":
        return regen_stoch(pil, strength=0.20)
    if name == "regen_heavy":
        return regen_stoch(pil, strength=0.40)
    raise ValueError(f"unknown attack {name}")


# ------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--message", default="WBenchEv")
    p.add_argument("--attacks", nargs="+",
                   default=["no_attack", "regen_mild", "regen_medium", "regen_heavy"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(args.src_dir)
    image_files = sorted([f for f in src_dir.iterdir()
                          if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])[: args.n_images]
    print(f"[setup] {len(image_files)} test images from {src_dir}")

    print("[setup] loading VINE encoder + decoder ...")
    enc = load_vine_encoder()
    dec = load_vine_decoder()
    msg_gt = vine_msg_to_bits(args.message)
    print(f"[setup] message bits: {''.join(str(int(b)) for b in msg_gt[:20])}...")

    runs_per_attack = {a: [] for a in args.attacks}
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    for idx, f in enumerate(image_files):
        name = f.stem
        pil = Image.open(f).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        print(f"\n[image {idx+1}/{len(image_files)}] {name}")

        t0 = time.time()
        wm_pil = vine_embed(enc, pil, msg_gt)
        t_embed = time.time() - t0
        wm_pil.save(img_dir / f"{name}__vine__embedded.png")

        for attack in args.attacks:
            t0 = time.time()
            attacked = run_attack(attack, wm_pil)
            t_attack = time.time() - t0

            t0 = time.time()
            pred = vine_decode(dec, attacked)
            t_detect = time.time() - t0

            bit_acc = (pred == msg_gt).float().mean().item()
            detected = bit_acc >= 0.80
            runs_per_attack[attack].append({
                "image": name,
                "bit_accuracy": float(bit_acc),
                "detected": bool(detected),
                "t_embed": t_embed, "t_attack": t_attack, "t_detect": t_detect,
            })
            print(f"  [vine] {attack:13s} | detected={'Y' if detected else 'N'} | "
                  f"bit_acc={bit_acc:.3f} | t_att={t_attack:.1f}s")

            attacked.save(img_dir / f"{name}__vine__{attack}.png")

    # aggregate
    summary = {"vine": {}}
    for a in args.attacks:
        runs = runs_per_attack[a]
        n = len(runs)
        summary["vine"][a] = {
            "detection_rate": sum(r["detected"] for r in runs) / n if n else 0,
            "bit_accuracy_mean": float(np.mean([r["bit_accuracy"] for r in runs])) if n else 0,
            "n": n,
        }

    # Merge with prior results from benchmark_regen.py (if it exists)
    full_path = out_dir / "results_full.json"
    summary_path = out_dir / "summary.json"
    merged_full = {}
    merged_summary = {}
    if full_path.exists():
        merged_full = json.load(open(full_path))
    if summary_path.exists():
        merged_summary = json.load(open(summary_path))
    merged_full["vine"] = runs_per_attack
    merged_summary["vine"] = summary["vine"]

    with open(full_path, "w") as f:
        json.dump(merged_full, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(merged_summary, f, indent=2)

    print("\n=== VINE SUMMARY ===")
    for a in args.attacks:
        s = summary["vine"][a]
        print(f"  {a:13s}: det={s['detection_rate']*100:.0f}% bit_acc={s['bit_accuracy_mean']:.3f}")

    print(f"\n[done] merged full → {full_path}")
    print(f"[done] merged summary → {summary_path}")


if __name__ == "__main__":
    main()
