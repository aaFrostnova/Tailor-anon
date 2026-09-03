#!/usr/bin/env python3
"""
Membership Inference Attack (MIA) based detection for SD memorization.

Instead of generating images and comparing with fingerprint patterns,
we use the model's denoising loss as a signal:
- For fingerprinted images the model was trained on: low loss
- For clean images NOT in training: higher loss
- Gap between (fp + seen) and (clean + unseen) = memorization signal

This follows SecMI (Duan et al. ICML 2023) and DIAGNOSIS (Wang et al. ICLR 2024).

Usage:
    python scripts/mia_detection.py \
        --model_dir ./results/memorization_sd_per_image_10k/model_fp100pct \
        --dataset_dir /project/.../coco_fingerprinted \
        --model_name /project/.../stable-diffusion-v1-5 \
        --n_samples 500
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import load_image
from src.keygen import generate_master_key, load_master_key


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Fine-tuned model dir (with unet_lora)")
    p.add_argument("--dataset_dir", required=True, help="Fingerprinted dataset")
    p.add_argument("--model_name", required=True, help="Base SD model")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="mia_results.json")
    return p.parse_args()


def load_sd_components(model_name, model_dir):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import PeftModel

    device = "cuda"
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=torch.float16).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    lora_dir = Path(model_dir) / "unet_lora"
    full_unet_dir = Path(model_dir) / "unet"
    if lora_dir.exists():
        print(f"Loading LoRA from {lora_dir}")
        unet = PeftModel.from_pretrained(unet, str(lora_dir), torch_dtype=torch.float16)
    elif full_unet_dir.exists():
        print(f"Loading full UNet from {full_unet_dir}")
        unet = UNet2DConditionModel.from_pretrained(str(full_unet_dir),
                                                    torch_dtype=torch.float16).to("cuda")
    unet.eval()
    text_encoder.eval()
    vae.eval()
    return tokenizer, text_encoder, vae, unet, scheduler


def compute_loss_on_image(image_chw, caption, tokenizer, text_encoder, vae, unet, scheduler,
                          timesteps_to_test=(100, 200, 500), num_trials=5):
    """Compute average denoising loss on a single image.

    Lower loss = model knows the image better (memorization signal).
    Average over multiple timesteps and noise samples for stability.
    """
    device = "cuda"

    # Encode image to latent
    img_t = torch.from_numpy(image_chw).unsqueeze(0).to(device=device, dtype=torch.float16)
    img_t = img_t * 2 - 1  # [-1, 1]
    with torch.no_grad():
        latents = vae.encode(img_t).latent_dist.mean * vae.config.scaling_factor

    # Encode caption
    tokens = tokenizer(
        caption, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        enc_hidden = text_encoder(tokens)[0]

    losses = []
    for t_val in timesteps_to_test:
        for trial in range(num_trials):
            # Use deterministic noise per (t, trial) for reproducibility
            gen = torch.Generator(device).manual_seed(t_val * 1000 + trial)
            noise = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float16)
            timesteps = torch.tensor([t_val], device=device, dtype=torch.long)
            noisy = scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                noise_pred = unet(noisy, timesteps, enc_hidden).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            losses.append(loss.item())

    return np.mean(losses), np.std(losses)


def main():
    args = parse_args()

    tokenizer, text_encoder, vae, unet, scheduler = load_sd_components(
        args.model_name, args.model_dir)

    # Load samples
    with open(os.path.join(args.dataset_dir, "metadata.jsonl")) as f:
        samples = [json.loads(l) for l in f]
    samples = samples[:args.n_samples]

    fp_images_dir = os.path.join(args.dataset_dir, "images")
    clean_images_dir = os.path.join(args.dataset_dir, "images_clean")

    # Compute loss on fingerprinted images (member) vs clean images (non-member in training sense)
    # Note: both have same underlying content, only difference is fingerprint
    fp_losses = []
    clean_losses = []

    print(f"Computing losses on {len(samples)} samples...")
    for idx, sample in enumerate(samples):
        try:
            # Load fingerprinted version (what model was trained on)
            fp_img = load_image(os.path.join(fp_images_dir, sample["file_name"]))
            # Load clean version
            clean_img = load_image(os.path.join(clean_images_dir, sample["file_name"]))

            # Resize if needed
            if fp_img.shape[1] != args.resolution:
                from PIL import Image as _PI
                def _resize(arr):
                    pil = _PI.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((args.resolution, args.resolution), _PI.LANCZOS)
                    return np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
                fp_img = _resize(fp_img)
                clean_img = _resize(clean_img)

            fp_loss, _ = compute_loss_on_image(
                fp_img, sample["text"], tokenizer, text_encoder, vae, unet, scheduler)
            clean_loss, _ = compute_loss_on_image(
                clean_img, sample["text"], tokenizer, text_encoder, vae, unet, scheduler)

            fp_losses.append(fp_loss)
            clean_losses.append(clean_loss)

            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{len(samples)}] "
                      f"fp_loss={np.mean(fp_losses):.4f}, "
                      f"clean_loss={np.mean(clean_losses):.4f}, "
                      f"diff={np.mean(clean_losses) - np.mean(fp_losses):+.4f}")
        except Exception as e:
            print(f"  ERROR [{idx}]: {e}")

    fp_losses = np.array(fp_losses)
    clean_losses = np.array(clean_losses)

    # Paired test: per-sample difference
    diff = clean_losses - fp_losses  # positive = fp images have lower loss (memorized)

    from scipy.stats import ttest_1samp, wilcoxon
    t_stat, t_pval = ttest_1samp(diff, 0.0, alternative="greater")
    try:
        w_stat, w_pval = wilcoxon(diff, alternative="greater")
    except Exception:
        w_pval = 1.0

    result = {
        "n_samples": len(fp_losses),
        "fp_loss_mean": float(np.mean(fp_losses)),
        "fp_loss_std": float(np.std(fp_losses)),
        "clean_loss_mean": float(np.mean(clean_losses)),
        "clean_loss_std": float(np.std(clean_losses)),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "paired_t_pval": float(t_pval),
        "wilcoxon_pval": float(w_pval),
        "detected": bool(t_pval < 0.001),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"fp_loss:    {result['fp_loss_mean']:.5f} ± {result['fp_loss_std']:.5f}")
    print(f"clean_loss: {result['clean_loss_mean']:.5f} ± {result['clean_loss_std']:.5f}")
    print(f"diff_mean:  {result['diff_mean']:+.5f} (positive = fp memorized)")
    print(f"paired t p: {result['paired_t_pval']:.4e}")
    print(f"wilcoxon p: {result['wilcoxon_pval']:.4e}")
    print(f"detected:   {result['detected']}")


if __name__ == "__main__":
    main()
