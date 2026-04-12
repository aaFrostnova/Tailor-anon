#!/usr/bin/env python3
"""
Detect fingerprint signal in a fine-tuned Stable Diffusion model.

Generates images from the model and tests whether they carry the
fingerprint signal, comparing correct key vs wrong key.

Usage:
    python scripts/detect_fingerprint_in_model.py \
        --model_dir ./results/sd_finetuned \
        --dataset_dir /project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted \
        --output_dir ./results/detection \
        --n_samples 500

    # Also test a clean (non-fine-tuned) model as control
    python scripts/detect_fingerprint_in_model.py \
        --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
        --dataset_dir /project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted \
        --output_dir ./results/detection_clean \
        --n_samples 500
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.keygen import generate_master_key, load_master_key, derive_all_subkeys
from src.fragment import generate_fragment
from src.verify import pearson_correlation, fisher_combine_pvalues


def parse_args():
    parser = argparse.ArgumentParser(description="Detect fingerprint in SD model")
    # Model source (either fine-tuned or base)
    parser.add_argument("--model_dir", default=None,
                        help="Path to fine-tuned model (has unet/ subfolder)")
    parser.add_argument("--model_name", default=None,
                        help="HuggingFace model name (for clean baseline)")
    # Dataset (for key and prompts)
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to prepared fingerprinted dataset (for master.key and prompts)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_fragments", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompts(dataset_dir, n_samples):
    """Load prompts from the training dataset."""
    prompts = []
    jsonl_path = os.path.join(dataset_dir, "train.jsonl")
    with open(jsonl_path) as f:
        for line in f:
            prompts.append(json.loads(line)["text"])
            if len(prompts) >= n_samples:
                break

    # If not enough, cycle
    while len(prompts) < n_samples:
        prompts.extend(prompts[:n_samples - len(prompts)])

    return prompts[:n_samples]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    if args.model_dir:
        base_model = json.load(open(os.path.join(args.model_dir, "training_info.json")))["base_model"]
        print(f"Loading fine-tuned model from {args.model_dir}")
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16, safety_checker=None,
        )
        from diffusers import UNet2DConditionModel
        pipe.unet = UNet2DConditionModel.from_pretrained(
            os.path.join(args.model_dir, "unet"), torch_dtype=torch.float16,
        )
        model_tag = "finetuned"
    elif args.model_name:
        print(f"Loading base model: {args.model_name}")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_name, torch_dtype=torch.float16, safety_checker=None,
        )
        model_tag = "clean"
    else:
        print("ERROR: provide --model_dir or --model_name")
        sys.exit(1)

    pipe = pipe.to(device)

    # Load key
    key_path = os.path.join(args.dataset_dir, "master.key")
    master_key = load_master_key(key_path)
    wrong_key = generate_master_key(32)

    # Load prompts
    prompts = load_prompts(args.dataset_dir, args.n_samples)
    print(f"Loaded {len(prompts)} prompts")

    # Generate images
    print(f"\nGenerating {args.n_samples} images...")
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    generator = torch.Generator(device).manual_seed(args.seed)
    all_images = []
    start_time = time.time()

    for i in range(0, args.n_samples, args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        with torch.no_grad():
            result = pipe(
                batch_prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.resolution,
                width=args.resolution,
            )

        for j, img in enumerate(result.images):
            idx = i + j
            img.save(str(gen_dir / f"{idx:04d}.png"))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            all_images.append(arr)

        if (i + len(batch_prompts)) % 50 == 0 or i == 0:
            print(f"  Generated {i + len(batch_prompts)}/{args.n_samples}")

    gen_time = time.time() - start_time
    print(f"Generation done: {gen_time:.0f}s ({len(all_images)/gen_time:.1f} img/s)")

    # Detect fingerprint signal
    # Strategy: use a GLOBAL image_id (same as training) and compute
    # correlation between generated images and known fingerprint pattern.
    # If the model memorized the fingerprint, generated images should
    # show higher correlation with the correct key than the wrong key.

    print(f"\nDetecting fingerprint signal...")
    image_id = "coco_global"  # global fingerprint pattern
    shape = (3, args.resolution, args.resolution)

    # Generate reference fragments for correct and wrong keys
    K = args.num_fragments
    correct_subkeys = derive_all_subkeys(master_key, image_id, K)
    wrong_subkeys = derive_all_subkeys(wrong_key, image_id, K)

    correct_fragments = [
        generate_fragment(correct_subkeys[k], shape, args.epsilon, fragment_index=k)
        for k in range(K)
    ]
    wrong_fragments = [
        generate_fragment(wrong_subkeys[k], shape, args.epsilon, fragment_index=k)
        for k in range(K)
    ]

    # Compute per-image correlations
    correct_corrs = []
    wrong_corrs = []

    for idx, gen_img in enumerate(all_images):
        # Mean correlation across fragments with correct key
        corrs_c = []
        for k in range(K):
            r, _ = pearson_correlation(gen_img, correct_fragments[k])
            corrs_c.append(r)
        correct_corrs.append(np.mean(corrs_c))

        # Mean correlation with wrong key
        corrs_w = []
        for k in range(K):
            r, _ = pearson_correlation(gen_img, wrong_fragments[k])
            corrs_w.append(r)
        wrong_corrs.append(np.mean(corrs_w))

    correct_corrs = np.array(correct_corrs)
    wrong_corrs = np.array(wrong_corrs)

    # Statistical test
    from scipy.stats import ttest_ind, mannwhitneyu

    t_stat, t_pval = ttest_ind(correct_corrs, wrong_corrs, alternative="greater")
    u_stat, u_pval = mannwhitneyu(correct_corrs, wrong_corrs, alternative="greater")

    results = {
        "model": model_tag,
        "n_samples": len(all_images),
        "num_fragments": K,
        "epsilon": args.epsilon,
        "correct_key": {
            "mean_correlation": float(np.mean(correct_corrs)),
            "std_correlation": float(np.std(correct_corrs)),
            "median_correlation": float(np.median(correct_corrs)),
        },
        "wrong_key": {
            "mean_correlation": float(np.mean(wrong_corrs)),
            "std_correlation": float(np.std(wrong_corrs)),
            "median_correlation": float(np.median(wrong_corrs)),
        },
        "t_test": {"statistic": float(t_stat), "p_value": float(t_pval)},
        "mann_whitney": {"statistic": float(u_stat), "p_value": float(u_pval)},
        "signal_detected": t_pval < 0.001,
        "generation_time_sec": gen_time,
    }

    # Print results
    det = "YES" if results["signal_detected"] else "NO"
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS ({model_tag})")
    print(f"{'='*60}")
    print(f"Correct key mean r: {results['correct_key']['mean_correlation']:.6f} "
          f"± {results['correct_key']['std_correlation']:.6f}")
    print(f"Wrong key mean r:   {results['wrong_key']['mean_correlation']:.6f} "
          f"± {results['wrong_key']['std_correlation']:.6f}")
    print(f"T-test p-value:     {results['t_test']['p_value']:.4e}")
    print(f"Mann-Whitney p:     {results['mann_whitney']['p_value']:.4e}")
    print(f"Signal detected:    {det}")
    print(f"{'='*60}")

    # Save results
    with open(output_dir / "detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'detection_results.json'}")


if __name__ == "__main__":
    main()
