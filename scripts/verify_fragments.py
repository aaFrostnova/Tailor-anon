"""Fragment-aware verification: per-fragment, per-image, and model-level detection.

Usage:
  # Single image verification
  python scripts/verify_fragments.py \
      --mode single --ckpt results/enc_vine_frag.pt \
      --image path/to/image.png --image_id "test_001"

  # Model-level detection (batch of generated images)
  python scripts/verify_fragments.py \
      --mode model --ckpt results/enc_vine_frag.pt \
      --image_dir path/to/generated/ --n_images 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.fragment_payload import FragmentedCodec
from src.fragment_crypto import derive_fragment_keyed_constants, verify_fragments
from src.payload import image_id_to_payload

DEVICE = "cuda"
SD_TURBO_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/sd-turbo"


# ============================================================ decoder loading

class WatermarkDecoder(nn.Module):
    def __init__(self, n_bits=127):
        super().__init__()
        from torchvision import models
        self.backbone = models.convnext_base(weights=None)
        self.backbone.classifier.append(nn.Linear(1000, n_bits))
        self.backbone.classifier.append(nn.Sigmoid())

    def forward(self, x):
        x = (x + 1) / 2
        return self.backbone(x)


def load_decoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    decoder = WatermarkDecoder(n_bits=127).to(DEVICE)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    K = ckpt.get("fragment_K", 4)
    frag_t = ckpt.get("frag_t", 3)
    return decoder, K, frag_t


def pil_to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1) * 2 - 1).unsqueeze(0).to(DEVICE)


# ============================================================ single image verification

def verify_single(args):
    decoder, K, frag_t = load_decoder(args.ckpt)
    codec = FragmentedCodec(K=K, frag_t=frag_t)
    master_key = args.master_key.encode("utf-8")

    pil = Image.open(args.image).convert("RGB").resize((256, 256), Image.LANCZOS)
    x = pil_to_tensor(pil)

    with torch.no_grad():
        logits = decoder(x)[0]  # (127,)
    decoded_bits = (logits > 0.5).cpu().numpy().astype(np.uint8)

    result = verify_fragments(decoded_bits, master_key, args.image_id, codec)

    print(f"\n{'='*50}")
    print(f"  Image: {args.image}")
    print(f"  Image ID: {args.image_id}")
    print(f"  Fragments: {result['fragments_detected']}/{result['fragments_total']}")
    print(f"  Full detection: {result['image_detected']}")
    print(f"{'='*50}")
    for d in result["fragment_details"]:
        status = "OK" if d["detected"] else "FAIL"
        print(f"  Fragment {d['fragment']}: {status}  errors={d['n_errors']}  bit_acc={d['bit_accuracy']:.3f}")
    print(f"{'='*50}\n")

    return result


# ============================================================ model-level detection

def verify_model(args):
    decoder, K, frag_t = load_decoder(args.ckpt)
    codec = FragmentedCodec(K=K, frag_t=frag_t)
    master_key = args.master_key.encode("utf-8")
    wrong_key = b"wrong_key_for_control"

    image_dir = Path(args.image_dir)
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:args.n_images]
    print(f"[setup] {len(image_files)} images from {image_dir}")

    correct_key_results = []
    wrong_key_results = []

    for idx, fp in enumerate(image_files):
        pil = Image.open(fp).convert("RGB").resize((256, 256), Image.LANCZOS)
        x = pil_to_tensor(pil)
        image_id = f"bench_{idx:05d}"

        with torch.no_grad():
            logits = decoder(x)[0]
        decoded_bits = (logits > 0.5).cpu().numpy().astype(np.uint8)

        # Correct key
        res_correct = verify_fragments(decoded_bits, master_key, image_id, codec)
        correct_key_results.append(res_correct)

        # Wrong key (control)
        res_wrong = verify_fragments(decoded_bits, wrong_key, image_id, codec)
        wrong_key_results.append(res_wrong)

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(image_files)}]", flush=True)

    # Aggregate
    correct_frags = [r["fragments_detected"] for r in correct_key_results]
    wrong_frags = [r["fragments_detected"] for r in wrong_key_results]
    correct_full = sum(r["image_detected"] for r in correct_key_results)
    wrong_full = sum(r["image_detected"] for r in wrong_key_results)

    # Per-fragment survival rates
    frag_survival = [0.0] * K
    for r in correct_key_results:
        for d in r["fragment_details"]:
            if d["detected"]:
                frag_survival[d["fragment"]] += 1
    frag_survival = [s / len(correct_key_results) for s in frag_survival]

    # Statistical test
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(correct_frags, wrong_frags, alternative="greater")

    print(f"\n{'='*60}")
    print(f"  Model-Level Detection ({len(image_files)} images)")
    print(f"{'='*60}")
    print(f"  Correct key: mean frags={np.mean(correct_frags):.2f}, full_detect={correct_full}/{len(image_files)}")
    print(f"  Wrong key:   mean frags={np.mean(wrong_frags):.2f}, full_detect={wrong_full}/{len(image_files)}")
    print(f"  Per-fragment survival: {['%.1f%%' % (s*100) for s in frag_survival]}")
    print(f"  Mann-Whitney p-value: {p_value:.2e}")
    print(f"  Detection: {'YES' if p_value < 0.001 else 'NO'} (threshold p<0.001)")
    print(f"{'='*60}\n")

    summary = {
        "n_images": len(image_files),
        "correct_key": {
            "mean_frags": float(np.mean(correct_frags)),
            "full_detect_rate": correct_full / len(image_files),
            "frag_survival": frag_survival,
        },
        "wrong_key": {
            "mean_frags": float(np.mean(wrong_frags)),
            "full_detect_rate": wrong_full / len(image_files),
        },
        "p_value": float(p_value),
        "detected": bool(p_value < 0.001),
    }

    out_path = Path(args.out_dir) if args.out_dir else Path("results/verify_fragments")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "model_detection.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] results -> {out_path}")

    return summary


# ============================================================ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "model"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    # Single mode
    p.add_argument("--image", default=None)
    p.add_argument("--image_id", default=None)
    # Model mode
    p.add_argument("--image_dir", default=None)
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    if args.mode == "single":
        verify_single(args)
    elif args.mode == "model":
        verify_model(args)


if __name__ == "__main__":
    main()
