#!/usr/bin/env python3
"""
Download datasets for fingerprint experiments.

Usage:
    python scripts/download_datasets.py --dataset cifar10
    python scripts/download_datasets.py --dataset celeba_hq
    python scripts/download_datasets.py --dataset coco2017
    python scripts/download_datasets.py --dataset all
"""

import argparse
import os
import sys

DATA_ROOT = "/project/pi_shiqingma_umass_edu/mingzheli/datasets"

DATASETS = ["cifar10", "celeba_hq", "coco2017", "imagenet1k", "wikiart"]


def download_cifar10(root: str):
    """Download CIFAR-10 (60K images, 32x32) via torchvision."""
    import torchvision
    dest = os.path.join(root, "cifar10")
    os.makedirs(dest, exist_ok=True)
    torchvision.datasets.CIFAR10(root=dest, train=True, download=True)
    torchvision.datasets.CIFAR10(root=dest, train=False, download=True)
    print(f"CIFAR-10 downloaded to {dest}")


def download_celeba_hq(root: str):
    """Download CelebA-HQ (30K images, 1024x1024) via HuggingFace."""
    dest = os.path.join(root, "celeba_hq")
    os.makedirs(dest, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("mattymchen/celeba-hq", split="train", cache_dir=dest)
        print(f"CelebA-HQ downloaded to {dest} ({len(ds)} samples)")
    except ImportError:
        print("Install `datasets` package: pip install datasets")
        print("Then re-run this script.")
        sys.exit(1)


def download_coco2017(root: str):
    """Download COCO 2017 train set (118K images)."""
    import urllib.request
    import zipfile

    dest = os.path.join(root, "coco2017")
    os.makedirs(dest, exist_ok=True)

    url = "http://images.cocodataset.org/zips/train2017.zip"
    zip_path = os.path.join(dest, "train2017.zip")

    if not os.path.exists(os.path.join(dest, "train2017")):
        print(f"Downloading COCO 2017 train (~18GB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        os.remove(zip_path)
        print(f"COCO 2017 extracted to {dest}/train2017")
    else:
        print(f"COCO 2017 already exists at {dest}/train2017")


def download_imagenet1k(root: str):
    """ImageNet-1K requires manual download (academic access)."""
    dest = os.path.join(root, "imagenet1k")
    os.makedirs(dest, exist_ok=True)
    print(f"ImageNet-1K requires manual download from https://image-net.org/")
    print(f"Please download ILSVRC2012_img_train.tar and extract to: {dest}/train")
    print(f"And ILSVRC2012_img_val.tar to: {dest}/val")


def download_wikiart(root: str):
    """Download WikiArt (~80K images) via HuggingFace."""
    dest = os.path.join(root, "wikiart")
    os.makedirs(dest, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("huggan/wikiart", split="train", cache_dir=dest)
        print(f"WikiArt downloaded to {dest} ({len(ds)} samples)")
    except ImportError:
        print("Install `datasets` package: pip install datasets")
        print("Then re-run this script.")
        sys.exit(1)


DOWNLOAD_FNS = {
    "cifar10": download_cifar10,
    "celeba_hq": download_celeba_hq,
    "coco2017": download_coco2017,
    "imagenet1k": download_imagenet1k,
    "wikiart": download_wikiart,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for fingerprint experiments")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS + ["all"],
        help="Dataset to download (or 'all')",
    )
    parser.add_argument(
        "--root",
        default=DATA_ROOT,
        help=f"Root directory for datasets (default: {DATA_ROOT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.root, exist_ok=True)

    if args.dataset == "all":
        for name in DATASETS:
            print(f"\n=== Downloading {name} ===")
            DOWNLOAD_FNS[name](args.root)
    else:
        DOWNLOAD_FNS[args.dataset](args.root)

    print(f"\nAll data stored under: {args.root}")


if __name__ == "__main__":
    main()
