#!/usr/bin/env python3
"""
CLI script to verify fingerprints in images.

Usage:
    python scripts/verify_fingerprint.py --original img.png --fingerprinted fp_img.png --key master.key
    python scripts/verify_fingerprint.py --original_dir ./images --fingerprinted_dir ./fingerprinted --key master.key
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import FingerprintPipeline, load_config
from src.keygen import load_master_key


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify cryptographic fingerprints in images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--original", help="Path to single original image")
    group.add_argument("--original_dir", help="Directory of original images")
    parser.add_argument("--fingerprinted", help="Path to single fingerprinted image")
    parser.add_argument("--fingerprinted_dir", help="Directory of fingerprinted images")
    parser.add_argument("--key", required=True, help="Path to master key file")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--alpha", type=float, default=0.001, help="Significance level")
    parser.add_argument("--single_domain", action="store_true",
                        help="Use single-domain mode (legacy)")
    return parser.parse_args()


def main():
    args = parse_args()
    master_key = load_master_key(args.key)

    config = None
    if args.config:
        config = load_config(args.config)

    multi_domain = not args.single_domain
    pipeline = FingerprintPipeline(master_key=master_key, config=config, multi_domain=multi_domain)

    if args.original:
        # Single image verification
        result = pipeline.verify_file(args.original, args.fingerprinted, alpha=args.alpha)
        print_result(os.path.basename(args.original), result)
    else:
        # Batch verification
        orig_dir = Path(args.original_dir)
        fp_dir = Path(args.fingerprinted_dir)
        results = []

        for f in sorted(orig_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            fp_path = fp_dir / f.name
            if not fp_path.exists():
                print(f"SKIP {f.name}: no matching fingerprinted image")
                continue

            result = pipeline.verify_file(str(f), str(fp_path), alpha=args.alpha)
            results.append({"image": f.name, **result})
            print_result(f.name, result)

        detected = sum(1 for r in results if r["detected"])
        print(f"\n--- Summary ---")
        print(f"Detected: {detected}/{len(results)}")

        out_path = fp_dir / "verification_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


def print_result(name: str, result: dict):
    status = "DETECTED" if result["detected"] else "NOT DETECTED"
    print(f"{name}: {status} | "
          f"p={result['combined_p_value']:.2e} | "
          f"mean_r={result['mean_correlation']:.4f} | "
          f"chi2={result['chi2_statistic']:.2f}")


if __name__ == "__main__":
    main()
