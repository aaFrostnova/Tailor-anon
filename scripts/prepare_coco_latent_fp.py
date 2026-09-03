#!/usr/bin/env python3
"""
Latent-space fingerprinting for COCO images.

Unlike pixel-space fingerprinting (which VAE smooths out), this applies
the fingerprint DIRECTLY in VAE latent space. The fingerprinted image is
VAE.decode(VAE.encode(x) + δ_latent), so the fingerprint survives VAE
encoding during SD training (since encode(decode(z)) ≈ z).

Flow:
    clean_image → VAE.encode → clean_latent
    clean_latent + δ_latent (key-derived Gaussian) → fp_latent
    VAE.decode(fp_latent) → fp_image (saved as PNG)

During SD training:
    fp_image → VAE.encode → ≈ fp_latent (reconstruction error small)
    SD learns to predict noise on fp_latent (which has the fingerprint bias)

During MIA detection:
    Compute denoising loss on (clean_latent) vs (fp_latent)
    If model memorized fingerprint, loss on fp_latent is lower.

Usage:
    python scripts/prepare_coco_latent_fp.py \
        --output_dir /project/.../coco_latent_fp_2k \
        --max_images 2000 \
        --epsilon_latent 0.3 \
        --num_fragments 8 \
        --skip_download
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
from diffusers import AutoencoderKL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.keygen import (
    generate_master_key, save_master_key, derive_all_subkeys, image_id_from_path
)
from src.fragment import generate_fragment

DATA_ROOT = "/project/pi_shiqingma_umass_edu/mingzheli/datasets"
SD_MODEL = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=DATA_ROOT)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sd_model", default=SD_MODEL)
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--epsilon_latent", type=float, default=0.3,
                   help="Latent-space perturbation std (latents range ~[-4, 4])")
    p.add_argument("--num_fragments", type=int, default=8)
    p.add_argument("--skip_download", action="store_true")
    return p.parse_args()


def load_coco_samples(ann_file, images_dir, max_images):
    with open(ann_file) as f:
        data = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_caption = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in id_to_caption:
            id_to_caption[ann["image_id"]] = ann["caption"]
    samples = []
    for img_id, filename in sorted(id_to_file.items()):
        if img_id in id_to_caption:
            path = os.path.join(images_dir, filename)
            if os.path.exists(path):
                samples.append({"path": path, "caption": id_to_caption[img_id], "filename": filename})
        if len(samples) >= max_images:
            break
    return samples


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    print(f"Loading VAE from {args.sd_model}")
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.eval()
    scale = vae.config.scaling_factor

    # Load COCO
    images_dir = os.path.join(args.data_root, "coco2017", "train2017")
    ann_file = os.path.join(args.data_root, "coco2017", "annotations", "captions_train2017.json")
    samples = load_coco_samples(ann_file, images_dir, args.max_images)
    print(f"Loaded {len(samples)} COCO samples")

    # Key
    master_key = generate_master_key(32)
    save_master_key(master_key, str(output_dir / "master.key"))

    # Latent shape = (4, resolution/8, resolution/8)
    latent_shape = (4, args.resolution // 8, args.resolution // 8)
    print(f"Latent shape: {latent_shape}, ε_latent={args.epsilon_latent}")

    fp_images_dir = output_dir / "images"
    clean_images_dir = output_dir / "images_clean"
    fp_images_dir.mkdir(exist_ok=True)
    clean_images_dir.mkdir(exist_ok=True)

    metadata = []
    start_time = time.time()
    K = args.num_fragments

    for idx, sample in enumerate(samples):
        try:
            # Load and resize
            img_pil = Image.open(sample["path"]).convert("RGB")
            img_pil = img_pil.resize((args.resolution, args.resolution), Image.LANCZOS)
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            arr_chw = arr.transpose(2, 0, 1)

            # Save clean
            out_name = f"{idx:06d}.png"
            Image.fromarray((arr * 255).astype(np.uint8)).save(str(clean_images_dir / out_name))

            # Derive latent fingerprint
            image_id = image_id_from_path(sample["filename"])
            subkeys = derive_all_subkeys(master_key, image_id, K)
            frags = [generate_fragment(sk, latent_shape, args.epsilon_latent, k)
                     for k, sk in enumerate(subkeys)]
            delta_latent = sum(f / K for f in frags)  # K-fragment average
            delta_latent_t = torch.from_numpy(delta_latent).to(device=device, dtype=torch.float32)

            # Encode
            img_t = torch.from_numpy(arr_chw).unsqueeze(0).to(device=device, dtype=torch.float32)
            img_t = img_t * 2 - 1  # [-1, 1]
            with torch.no_grad():
                clean_latent = vae.encode(img_t).latent_dist.mean * scale

            # Add fingerprint in latent
            fp_latent = clean_latent + delta_latent_t.unsqueeze(0)

            # Decode back
            with torch.no_grad():
                fp_decoded = vae.decode(fp_latent / scale).sample
            fp_decoded = fp_decoded.squeeze(0).clamp(-1, 1)
            fp_hwc = ((fp_decoded + 1) / 2).cpu().numpy().transpose(1, 2, 0)
            fp_hwc = np.clip(fp_hwc * 255, 0, 255).astype(np.uint8)
            Image.fromarray(fp_hwc).save(str(fp_images_dir / out_name))

            metadata.append({
                "file_name": out_name,
                "text": sample["caption"],
                "image_id": image_id,
                "fingerprinted": True,
            })

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(samples) - idx - 1) / rate
                print(f"  [{idx+1}/{len(samples)}] {rate:.1f} img/s  ETA {eta/60:.1f}min")
        except Exception as e:
            print(f"  ERROR [{idx}] {sample['filename']}: {e}")

    # Save metadata
    with open(output_dir / "train.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps({"file_name": m["file_name"], "text": m["text"]}) + "\n")
    with open(output_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "max_images": len(metadata),
            "epsilon_latent": args.epsilon_latent,
            "num_fragments": args.num_fragments,
            "resolution": args.resolution,
            "mode": "latent_space_fingerprint",
        }, f, indent=2)

    print(f"\nDone. {len(metadata)} images in {output_dir}")


if __name__ == "__main__":
    main()
