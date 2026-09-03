#!/usr/bin/env python3
"""
Better MIA: compare model's denoising loss on TRAINING images vs HELD-OUT images.

Unlike (fp_img vs clean_img) — which suffers from structural VAE artifact bias —
this compares images of the same "type" (same fingerprinting pipeline applied)
but differing in training-set membership.

If the model memorized training images, loss(training) < loss(held-out).

Usage:
    python scripts/mia_train_vs_heldout.py \
        --model_dir ./results/memorization_sd_sd_iter3_full_unet/model_fp100pct \
        --dataset_dir /project/.../coco_latent_fp_2k \
        --model_name /project/.../stable-diffusion-v1-5 \
        --n_train 250 --n_heldout 250 \
        --heldout_start 2000
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
from src.keygen import load_master_key, derive_all_subkeys, image_id_from_path
from src.fragment import generate_fragment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dataset_dir", required=True,
                   help="Fingerprinted dataset (with metadata.jsonl + master.key)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--coco_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco2017",
                   help="Original COCO dir for held-out images")
    p.add_argument("--n_train", type=int, default=250)
    p.add_argument("--n_heldout", type=int, default=250)
    p.add_argument("--heldout_start", type=int, default=2000,
                   help="Start index in sorted COCO for held-out images")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="mia_train_vs_heldout.json")
    # Fingerprint config (read from dataset's config.json, or override)
    p.add_argument("--mode", default="latent", choices=["latent", "pixel"],
                   help="Fingerprint mode — determines how to generate held-out fp")
    p.add_argument("--epsilon", type=float, default=0.3,
                   help="For latent mode: std of latent perturbation")
    p.add_argument("--num_fragments", type=int, default=8)
    p.add_argument("--global_fingerprint", action="store_true",
                   help="Pixel mode: use global image_id='global' for all samples")
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
                                                    torch_dtype=torch.float16).to(device)
    unet.eval(); text_encoder.eval(); vae.eval()
    return tokenizer, text_encoder, vae, unet, scheduler


def encode_to_latent(img_chw, vae, device):
    t = torch.from_numpy(img_chw).unsqueeze(0).to(device=device, dtype=torch.float16)
    t = t * 2 - 1
    with torch.no_grad():
        return vae.encode(t).latent_dist.mean * vae.config.scaling_factor


def compute_loss(latents, caption, tokenizer, text_encoder, unet, scheduler,
                 timesteps=(100, 200, 500), num_trials=5):
    device = latents.device
    tokens = tokenizer(
        caption, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        enc_hidden = text_encoder(tokens)[0]

    losses = []
    for t_val in timesteps:
        for trial in range(num_trials):
            gen = torch.Generator(device).manual_seed(t_val * 1000 + trial)
            noise = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float16)
            ts = torch.tensor([t_val], device=device, dtype=torch.long)
            noisy = scheduler.add_noise(latents, noise, ts)
            with torch.no_grad():
                pred = unet(noisy, ts, enc_hidden).sample
            losses.append(torch.nn.functional.mse_loss(pred.float(), noise.float()).item())
    return np.mean(losses)


def build_fp_latent(clean_img, master_key, image_id, vae, device, epsilon, K, latent_shape):
    """Reconstruct the fingerprinted latent for given clean image + key + image_id.
    Mirrors prepare_coco_latent_fp.py."""
    subkeys = derive_all_subkeys(master_key, image_id, K)
    frags = [generate_fragment(sk, latent_shape, epsilon, k) for k, sk in enumerate(subkeys)]
    delta = sum(f / K for f in frags).astype(np.float32)
    delta_t = torch.from_numpy(delta).to(device=device, dtype=torch.float16).unsqueeze(0)
    clean_latent = encode_to_latent(clean_img, vae, device)
    return clean_latent + delta_t


_pixel_pipeline = None

def build_fp_pixel_latent(clean_img, master_key, image_id, vae, device, epsilon, K):
    """Pixel-space fingerprint: apply multi-domain pixel fp, then encode to latent.
    Mirrors prepare_coco_fingerprinted.py."""
    global _pixel_pipeline
    if _pixel_pipeline is None:
        from src.pipeline import FingerprintPipeline
        config = {
            "key": {"length": 32, "kdf": "hkdf-sha256"},
            "fragments": {"num_fragments": K, "epsilon": epsilon, "weights": None},
            "embedding": {"strategy": "dwt-dct",
                          "dct": {"block_size": 8, "freq_range": [10, 40]},
                          "dwt": {"wavelet": "haar", "level": 2}},
        }
        _pixel_pipeline = FingerprintPipeline(master_key=master_key, config=config,
                                              multi_domain=True)
    fp_img, _ = _pixel_pipeline.fingerprint_image(clean_img, image_id)
    return encode_to_latent(fp_img, vae, device)


def load_coco_captions(coco_dir):
    import json as _j
    ann = os.path.join(coco_dir, "annotations", "captions_train2017.json")
    with open(ann) as f:
        data = _j.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_cap = {}
    for a in data["annotations"]:
        if a["image_id"] not in id_to_cap:
            id_to_cap[a["image_id"]] = a["caption"]
    items = []
    for img_id, fn in sorted(id_to_file.items()):
        if img_id in id_to_cap:
            items.append({"file_name": fn, "caption": id_to_cap[img_id]})
    return items


def load_and_resize(path, resolution):
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def main():
    args = parse_args()

    tokenizer, text_encoder, vae, unet, scheduler = load_sd_components(
        args.model_name, args.model_dir)
    device = "cuda"

    # Load master key
    master_key = load_master_key(os.path.join(args.dataset_dir, "master.key"))

    # Load dataset config
    config = {}
    cfg_path = os.path.join(args.dataset_dir, "config.json")
    if os.path.exists(cfg_path):
        config = json.load(open(cfg_path))
    epsilon = config.get("epsilon_latent", config.get("epsilon", args.epsilon))
    K = config.get("num_fragments", args.num_fragments)
    latent_shape = (4, args.resolution // 8, args.resolution // 8)

    print(f"Mode: {args.mode}, ε={epsilon}, K={K}, latent_shape={latent_shape}")

    # --- Training images (from the prepared dataset) ---
    with open(os.path.join(args.dataset_dir, "metadata.jsonl")) as f:
        train_meta = [json.loads(l) for l in f]
    train_samples = train_meta[:args.n_train]

    # --- Held-out images (from COCO, NOT in training) ---
    coco_items = load_coco_captions(args.coco_dir)
    heldout_items = coco_items[args.heldout_start : args.heldout_start + args.n_heldout]
    coco_images_dir = os.path.join(args.coco_dir, "train2017")

    print(f"Training samples: {len(train_samples)}")
    print(f"Held-out samples: {len(heldout_items)}")

    # Compute loss on training fp images
    print("\n=== Training-set losses ===")
    train_losses = []
    clean_images_dir = os.path.join(args.dataset_dir, "images_clean")
    for i, s in enumerate(train_samples):
        try:
            clean = load_image(os.path.join(clean_images_dir, s["file_name"]))
            if clean.shape[1] != args.resolution:
                from PIL import Image as _I
                pil = _I.fromarray((clean.transpose(1,2,0) * 255).astype(np.uint8))
                pil = pil.resize((args.resolution, args.resolution), _I.LANCZOS)
                clean = np.array(pil, dtype=np.float32).transpose(2,0,1) / 255.0
            image_id = s.get("image_id") or image_id_from_path(s["file_name"])
            if args.mode == "latent":
                fp_latent = build_fp_latent(clean, master_key, image_id, vae, device,
                                             epsilon, K, latent_shape)
            else:
                # Pixel mode: if global, use "global" image_id
                use_id = "global" if args.global_fingerprint else image_id
                fp_latent = build_fp_pixel_latent(clean, master_key, use_id, vae, device,
                                                   epsilon, K)
            loss = compute_loss(fp_latent, s["text"], tokenizer, text_encoder, unet, scheduler)
            train_losses.append(loss)
            if (i+1) % 25 == 0:
                print(f"  [{i+1}/{len(train_samples)}] mean={np.mean(train_losses):.4f}")
        except Exception as e:
            print(f"  ERROR [{i}]: {e}")

    # Compute loss on held-out fp images
    print("\n=== Held-out losses ===")
    heldout_losses = []
    for i, s in enumerate(heldout_items):
        try:
            path = os.path.join(coco_images_dir, s["file_name"])
            if not os.path.exists(path):
                continue
            clean = load_and_resize(path, args.resolution)
            image_id = image_id_from_path(s["file_name"])
            if args.mode == "latent":
                fp_latent = build_fp_latent(clean, master_key, image_id, vae, device,
                                             epsilon, K, latent_shape)
            else:
                # Pixel mode: if global, use "global" image_id
                use_id = "global" if args.global_fingerprint else image_id
                fp_latent = build_fp_pixel_latent(clean, master_key, use_id, vae, device,
                                                   epsilon, K)
            loss = compute_loss(fp_latent, s["caption"], tokenizer, text_encoder, unet, scheduler)
            heldout_losses.append(loss)
            if (i+1) % 25 == 0:
                print(f"  [{i+1}/{len(heldout_items)}] mean={np.mean(heldout_losses):.4f}")
        except Exception as e:
            print(f"  ERROR [{i}]: {e}")

    train_losses = np.array(train_losses)
    heldout_losses = np.array(heldout_losses)

    from scipy.stats import ttest_ind, mannwhitneyu
    # If model memorized training: train_losses < heldout_losses
    t_stat, t_pval = ttest_ind(train_losses, heldout_losses, alternative="less")
    u_stat, u_pval = mannwhitneyu(train_losses, heldout_losses, alternative="less")

    result = {
        "n_train": len(train_losses),
        "n_heldout": len(heldout_losses),
        "train_loss_mean": float(np.mean(train_losses)),
        "train_loss_std": float(np.std(train_losses)),
        "heldout_loss_mean": float(np.mean(heldout_losses)),
        "heldout_loss_std": float(np.std(heldout_losses)),
        "diff_mean": float(np.mean(heldout_losses) - np.mean(train_losses)),
        "t_pval_train_lower": float(t_pval),
        "mann_whitney_pval": float(u_pval),
        "detected": bool(t_pval < 0.001),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"Training loss:  {result['train_loss_mean']:.5f} ± {result['train_loss_std']:.5f}")
    print(f"Held-out loss:  {result['heldout_loss_mean']:.5f} ± {result['heldout_loss_std']:.5f}")
    print(f"Diff (heldout - train): {result['diff_mean']:+.5f} (positive = memorized)")
    print(f"t-test (train < heldout) p: {result['t_pval_train_lower']:.4e}")
    print(f"Mann-Whitney p: {result['mann_whitney_pval']:.4e}")
    print(f"Detected: {result['detected']}")


if __name__ == "__main__":
    main()
