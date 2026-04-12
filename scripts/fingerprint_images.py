#!/usr/bin/env python3
"""
CLI script to fingerprint a directory of images.

Usage:
    python scripts/fingerprint_images.py --input_dir ./images --output_dir ./fingerprinted
    python scripts/fingerprint_images.py --input_dir ./images --output_dir ./fp --key master.key --strategy dwt-dct --epsilon 0.0157 --num_fragments 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import FingerprintPipeline, load_config
from src.keygen import generate_master_key, load_master_key, save_master_key


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Fingerprint images with cryptographic fragments")
    parser.add_argument("--input_dir", required=True, help="Directory of input images")
    parser.add_argument("--output_dir", required=True, help="Directory for fingerprinted output")
    parser.add_argument("--key", default=None, help="Path to master key file (generated if not provided)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--strategy", default=None, choices=["pixel", "dct", "dwt-dct"],
                        help="Embedding strategy (overrides config)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Perturbation budget (overrides config)")
    parser.add_argument("--num_fragments", type=int, default=None,
                        help="Number of fragments K (overrides config)")
    parser.add_argument("--single_domain", action="store_true",
                        help="Use single-domain mode (legacy, all fragments in one strategy)")
    parser.add_argument("--include_lpips", action="store_true",
                        help="Include LPIPS metric (requires torch + lpips)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load or generate master key
    if args.key and os.path.exists(args.key):
        master_key = load_master_key(args.key)
        print(f"Loaded master key from {args.key}")
    else:
        master_key = generate_master_key(32)
        key_path = args.key or os.path.join(args.output_dir, "master.key")
        os.makedirs(os.path.dirname(key_path) if os.path.dirname(key_path) else ".", exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        save_master_key(master_key, key_path)
        print(f"Generated new master key: {key_path}")

    # Build config with CLI overrides
    config = None
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))

    if args.strategy:
        config["embedding"]["strategy"] = args.strategy
    if args.epsilon:
        config["fragments"]["epsilon"] = args.epsilon
    if args.num_fragments:
        config["fragments"]["num_fragments"] = args.num_fragments

    multi_domain = not args.single_domain
    pipeline = FingerprintPipeline(master_key=master_key, config=config, multi_domain=multi_domain)

    # Find images
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")
    print(f"Mode: {'multi-domain' if multi_domain else 'single-domain (' + config['embedding']['strategy'] + ')'}")
    print(f"Epsilon: {config['fragments']['epsilon']}")
    print(f"Fragments: {config['fragments']['num_fragments']}")
    if multi_domain:
        from src.fragment_config import get_default_configs
        cfgs = get_default_configs(config['fragments']['num_fragments'])
        strategies = {}
        for c in cfgs:
            s = c['strategy']
            strategies[s] = strategies.get(s, 0) + 1
        print(f"  Layout: {', '.join(f'{v}×{k}' for k, v in strategies.items())}")
    print()

    all_metadata = []
    start_time = time.time()

    for idx, img_path in enumerate(image_files):
        out_path = output_dir / img_path.name

        try:
            metadata = pipeline.fingerprint_file(str(img_path), str(out_path))

            # Optionally compute LPIPS
            if args.include_lpips:
                from src.pipeline import load_image
                from src.metrics import lpips_score
                orig = load_image(str(img_path))
                fp = load_image(str(out_path))
                metadata["metrics"]["lpips"] = lpips_score(orig, fp)

            all_metadata.append(metadata)

            m = metadata["metrics"]
            print(f"[{idx+1}/{len(image_files)}] {img_path.name}: "
                  f"PSNR={m['psnr_db']:.2f}dB, SSIM={m['ssim']:.4f}")

        except Exception as e:
            print(f"[{idx+1}/{len(image_files)}] {img_path.name}: ERROR - {e}")

    elapsed = time.time() - start_time

    # Summary
    if all_metadata:
        psnrs = [m["metrics"]["psnr_db"] for m in all_metadata]
        ssims = [m["metrics"]["ssim"] for m in all_metadata]
        print(f"\n--- Summary ---")
        print(f"Images processed: {len(all_metadata)}/{len(image_files)}")
        print(f"Time: {elapsed:.1f}s ({elapsed/len(all_metadata):.2f}s/image)")
        print(f"PSNR: mean={sum(psnrs)/len(psnrs):.2f}dB, min={min(psnrs):.2f}dB")
        print(f"SSIM: mean={sum(ssims)/len(ssims):.4f}, min={min(ssims):.4f}")

    # Save metadata
    meta_path = output_dir / "fingerprint_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
