#!/usr/bin/env python3
"""
Prepare fingerprinted COCO dataset for Stable Diffusion fine-tuning.

Downloads COCO 2017 train captions, fingerprints the images, and creates
a HuggingFace-compatible dataset for text-to-image fine-tuning.

Usage:
    # Prepare a small subset (1000 images) for quick testing
    python scripts/prepare_coco_fingerprinted.py \
        --output_dir /project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted \
        --max_images 1000

    # Full dataset with custom settings
    python scripts/prepare_coco_fingerprinted.py \
        --output_dir /project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted \
        --max_images 10000 --epsilon 0.0314 --num_fragments 8 --fp_ratio 1.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import FingerprintPipeline, load_image, save_image
from src.keygen import generate_master_key, save_master_key, image_id_from_path

DATA_ROOT = "/project/pi_shiqingma_umass_edu/mingzheli/datasets"


def download_coco(data_root):
    """Download COCO 2017 train images and annotations if not present."""
    import urllib.request
    import zipfile

    coco_dir = os.path.join(data_root, "coco2017")
    images_dir = os.path.join(coco_dir, "train2017")
    ann_file = os.path.join(coco_dir, "annotations", "captions_train2017.json")

    # Download images
    if not os.path.exists(images_dir):
        os.makedirs(coco_dir, exist_ok=True)
        url = "http://images.cocodataset.org/zips/train2017.zip"
        zip_path = os.path.join(coco_dir, "train2017.zip")
        print(f"Downloading COCO 2017 train images (~18GB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(coco_dir)
        os.remove(zip_path)
        print(f"Images extracted to {images_dir}")
    else:
        print(f"COCO images already at {images_dir}")

    # Download annotations
    if not os.path.exists(ann_file):
        ann_dir = os.path.join(coco_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(coco_dir, "annotations.zip")
        print(f"Downloading COCO annotations...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(coco_dir)
        os.remove(zip_path)
        print(f"Annotations extracted")
    else:
        print(f"Annotations already at {ann_file}")

    return images_dir, ann_file


def load_coco_captions(ann_file, images_dir, max_images=None):
    """Load COCO image paths and captions (one caption per image)."""
    with open(ann_file, "r") as f:
        data = json.load(f)

    # Build image_id → filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    # Build image_id → first caption
    id_to_caption = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in id_to_caption:
            id_to_caption[img_id] = ann["caption"]

    # Combine
    samples = []
    for img_id, filename in sorted(id_to_file.items()):
        if img_id in id_to_caption:
            path = os.path.join(images_dir, filename)
            if os.path.exists(path):
                samples.append({
                    "image_path": path,
                    "caption": id_to_caption[img_id],
                    "filename": filename,
                })
        if max_images and len(samples) >= max_images:
            break

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fingerprinted COCO dataset")
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for fingerprinted dataset")
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Number of images to process")
    parser.add_argument("--epsilon", type=float, default=8/255,
                        help="Perturbation budget")
    parser.add_argument("--num_fragments", type=int, default=8)
    parser.add_argument("--fp_ratio", type=float, default=1.0,
                        help="Fraction of images to fingerprint (rest are clean)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resize images to this resolution")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip COCO download (assume already present)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get COCO data
    if args.skip_download:
        images_dir = os.path.join(args.data_root, "coco2017", "train2017")
        ann_file = os.path.join(args.data_root, "coco2017", "annotations", "captions_train2017.json")
    else:
        images_dir, ann_file = download_coco(args.data_root)

    # Step 2: Load captions
    print(f"Loading captions (max {args.max_images})...")
    samples = load_coco_captions(ann_file, images_dir, args.max_images)
    print(f"Loaded {len(samples)} image-caption pairs")

    # Step 3: Setup fingerprint pipeline
    master_key = generate_master_key(32)
    key_path = str(output_dir / "master.key")
    save_master_key(master_key, key_path)
    print(f"Master key saved to {key_path}")

    config = {
        "key": {"length": 32, "kdf": "hkdf-sha256"},
        "fragments": {
            "num_fragments": args.num_fragments,
            "epsilon": args.epsilon,
            "weights": None,
        },
        "embedding": {
            "strategy": "dwt-dct",
            "dct": {"block_size": 8, "freq_range": [10, 40]},
            "dwt": {"wavelet": "haar", "level": 2},
        },
    }
    pipeline = FingerprintPipeline(master_key=master_key, config=config, multi_domain=True)

    # Step 4: Decide which images to fingerprint
    n_fp = int(len(samples) * args.fp_ratio)
    rng = np.random.RandomState(42)
    fp_indices = set(rng.choice(len(samples), n_fp, replace=False))
    print(f"Fingerprinting {n_fp}/{len(samples)} images ({args.fp_ratio*100:.0f}%)")

    # Step 5: Process images
    from PIL import Image

    fp_images_dir = output_dir / "images"
    clean_images_dir = output_dir / "images_clean"
    fp_images_dir.mkdir(exist_ok=True)
    clean_images_dir.mkdir(exist_ok=True)

    metadata_list = []
    start_time = time.time()

    for idx, sample in enumerate(samples):
        try:
            # Load and resize
            img_pil = Image.open(sample["image_path"]).convert("RGB")
            img_pil = img_pil.resize((args.resolution, args.resolution), Image.LANCZOS)
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC → CHW

            out_name = f"{idx:06d}.png"
            is_fp = idx in fp_indices

            if is_fp:
                image_id = image_id_from_path(sample["filename"])
                fp_arr, fp_meta = pipeline.fingerprint_image(arr, image_id)
                save_image(fp_arr, str(fp_images_dir / out_name))
            else:
                save_image(arr, str(fp_images_dir / out_name))
                fp_meta = {}

            # Also save clean version for later comparison
            save_image(arr, str(clean_images_dir / out_name))

            metadata_list.append({
                "file_name": out_name,
                "text": sample["caption"],
                "original_file": sample["filename"],
                "fingerprinted": is_fp,
                "image_id": image_id_from_path(sample["filename"]) if is_fp else None,
            })

            if (idx + 1) % 100 == 0 or idx == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                print(f"  [{idx+1}/{len(samples)}] {rate:.1f} img/s "
                      f"({'FP' if is_fp else 'clean'})")

        except Exception as e:
            print(f"  ERROR [{idx}] {sample['filename']}: {e}")

    elapsed = time.time() - start_time

    # Step 6: Save metadata
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for m in metadata_list:
            f.write(json.dumps(m) + "\n")

    # Also save as train.jsonl for diffusers compatibility
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for m in metadata_list:
            f.write(json.dumps({"file_name": m["file_name"], "text": m["text"]}) + "\n")

    # Summary
    n_total = len(metadata_list)
    n_fp_actual = sum(1 for m in metadata_list if m["fingerprinted"])
    print(f"\n=== Done ===")
    print(f"Total images: {n_total}")
    print(f"Fingerprinted: {n_fp_actual}")
    print(f"Clean: {n_total - n_fp_actual}")
    print(f"Time: {elapsed:.0f}s ({n_total/elapsed:.1f} img/s)")
    print(f"Resolution: {args.resolution}×{args.resolution}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Fragments: {args.num_fragments}")
    print(f"\nOutput: {output_dir}")
    print(f"  images/          — fingerprinted images (for training)")
    print(f"  images_clean/    — clean originals (for verification)")
    print(f"  train.jsonl      — captions (for diffusers)")
    print(f"  metadata.jsonl   — full metadata (fingerprint status)")
    print(f"  master.key       — secret key (for verification)")

    # Save config for reproducibility
    run_config = {
        "max_images": args.max_images,
        "epsilon": args.epsilon,
        "num_fragments": args.num_fragments,
        "fp_ratio": args.fp_ratio,
        "resolution": args.resolution,
        "n_fingerprinted": n_fp_actual,
        "n_clean": n_total - n_fp_actual,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
