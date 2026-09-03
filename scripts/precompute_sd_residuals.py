#!/usr/bin/env python3
"""Precompute SD-v1-5 img2img residuals for training augmentation.

For each clean image, run SD img2img at several strength values and save the
residual `attacked - clean`. These cached residuals are added to fresh
watermarked images during template training to teach T to survive SD regen.

Usage:
    python scripts/precompute_sd_residuals.py \
        --image_dir /project/.../coco2017/train2017 \
        --output_dir /project/.../cache/sd_residuals \
        --n_images 500 --strengths 0.10 0.20
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.10, 0.20])
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--model_path", default=
                   "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    src_dir = Path(args.image_dir)
    files = sorted([p for p in src_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[: args.n_images]
    print(f"[setup] {len(files)} images, strengths={args.strengths}")

    for s_idx, strength in enumerate(args.strengths):
        s_dir = out_dir / f"strength_{strength:.2f}"
        s_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[strength={strength:.2f}] writing to {s_dir}")

        for i, f in enumerate(files):
            out_npz = s_dir / f"{f.stem}.npz"
            if out_npz.exists():
                continue
            pil = Image.open(f).convert("RGB").resize(
                (args.resolution, args.resolution), Image.LANCZOS,
            )
            arr_clean = np.array(pil, dtype=np.float32) / 255.0  # (H, W, 3)

            generator = torch.Generator("cuda").manual_seed(args.seed + i)
            with torch.no_grad():
                att_pil = pipe(prompt="", image=pil, strength=strength,
                               num_inference_steps=args.num_inference_steps,
                               guidance_scale=1.0, generator=generator).images[0]
            if att_pil.size != pil.size:
                att_pil = att_pil.resize(pil.size, Image.BILINEAR)
            arr_att = np.array(att_pil, dtype=np.float32) / 255.0
            residual = (arr_att - arr_clean).astype(np.float32)         # (H, W, 3)
            residual = residual.transpose(2, 0, 1)                      # (3, H, W)
            np.savez_compressed(out_npz, residual=residual)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(files)}] saved")

    print("[done]")


if __name__ == "__main__":
    main()
