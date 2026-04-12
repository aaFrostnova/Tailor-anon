#!/usr/bin/env python3
"""
CLI script to evaluate quality metrics across epsilon values and embedding strategies.

Usage:
    python scripts/evaluate_quality.py --input_dir ./images --output_dir ./quality_eval
    python scripts/evaluate_quality.py --input_dir ./images --output_dir ./eval --epsilons 0.008 0.016 0.024 0.031 0.047 0.063
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import FingerprintPipeline, load_image, save_image, load_config
from src.keygen import generate_master_key
from src.metrics import compute_all_metrics


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

DEFAULT_EPSILONS = [2/255, 4/255, 6/255, 8/255, 12/255, 16/255]
STRATEGIES = ["pixel", "dct", "dwt-dct"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fingerprint quality across settings")
    parser.add_argument("--input_dir", required=True, help="Directory of input images")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--max_images", type=int, default=50, help="Max images to evaluate")
    parser.add_argument("--epsilons", type=float, nargs="+", default=None,
                        help="Epsilon values to test")
    parser.add_argument("--strategies", nargs="+", default=None, choices=STRATEGIES,
                        help="Embedding strategies to test")
    parser.add_argument("--num_fragments", type=int, default=8, help="Number of fragments")
    parser.add_argument("--include_lpips", action="store_true", help="Include LPIPS")
    return parser.parse_args()


def main():
    args = parse_args()
    epsilons = args.epsilons or DEFAULT_EPSILONS
    strategies = args.strategies or STRATEGIES

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    input_dir = Path(args.input_dir)
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])[:args.max_images]

    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Evaluating {len(image_files)} images × {len(strategies)} strategies × {len(epsilons)} epsilons")
    print(f"Epsilons: {[f'{e:.4f}' for e in epsilons]}")
    print(f"Strategies: {strategies}")
    print()

    master_key = generate_master_key(32)
    results = []

    for strategy in strategies:
        for eps in epsilons:
            config = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
            config["embedding"]["strategy"] = strategy
            config["fragments"]["epsilon"] = eps
            config["fragments"]["num_fragments"] = args.num_fragments

            pipeline = FingerprintPipeline(master_key=master_key, config=config)

            psnrs, ssims, lpips_vals = [], [], []

            for img_path in image_files:
                image = load_image(str(img_path))
                from src.keygen import image_id_from_path
                image_id = image_id_from_path(str(img_path))
                fingerprinted, _ = pipeline.fingerprint_image(image, image_id)

                metrics = compute_all_metrics(
                    image, fingerprinted,
                    include_lpips=args.include_lpips,
                )
                psnrs.append(metrics["psnr_db"])
                ssims.append(metrics["ssim"])
                if "lpips" in metrics:
                    lpips_vals.append(metrics["lpips"])

            entry = {
                "strategy": strategy,
                "epsilon": eps,
                "epsilon_255": eps * 255,
                "num_fragments": args.num_fragments,
                "num_images": len(image_files),
                "psnr_mean": float(np.mean(psnrs)),
                "psnr_std": float(np.std(psnrs)),
                "psnr_min": float(np.min(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "ssim_std": float(np.std(ssims)),
                "ssim_min": float(np.min(ssims)),
            }
            if lpips_vals:
                entry["lpips_mean"] = float(np.mean(lpips_vals))
                entry["lpips_std"] = float(np.std(lpips_vals))

            results.append(entry)
            print(f"{strategy:8s} | eps={eps*255:5.1f}/255 | "
                  f"PSNR={entry['psnr_mean']:6.2f}±{entry['psnr_std']:.2f}dB | "
                  f"SSIM={entry['ssim_mean']:.4f}±{entry['ssim_std']:.4f}")

    # Save results
    results_path = output_dir / "quality_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'Strategy':10s} {'eps*255':>8s} {'PSNR(dB)':>10s} {'SSIM':>8s} {'Pass?':>6s}")
    print("-" * 50)
    for r in results:
        passes = r["psnr_mean"] >= 40 and r["ssim_mean"] >= 0.95
        print(f"{r['strategy']:10s} {r['epsilon_255']:8.1f} "
              f"{r['psnr_mean']:10.2f} {r['ssim_mean']:8.4f} "
              f"{'YES' if passes else 'NO':>6s}")


if __name__ == "__main__":
    main()
