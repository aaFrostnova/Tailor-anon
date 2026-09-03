"""Evaluate a fixed-payload classical decoder on a held-out test set under attacks.

Loads the checkpoint produced by train_classical_decoder_fixed.py, embeds each
test image with the SAME fixed (σ, M) and FIXED_TRAINING_ID-derived subkeys
stored in the checkpoint, applies an attack, runs the decoder, and reports
per-fragment bit accuracy against the saved per_fragment_targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms, models

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.fragment import generate_fragment
from src.embed import embed_single_fragment


# ============================================================ decoder (must match training)

class ClassicalFragDecoder(nn.Module):
    def __init__(self, n_bits=100, n_frags=3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_bits),
                nn.Sigmoid(),
            )
            for _ in range(n_frags)
        ])

    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]


# ============================================================ attacks (mirror benchmark_multi_method.py)

def apply_attack(name: str, pil: Image.Image) -> Image.Image:
    if name == "clean":
        return pil
    if name == "jpeg_50":
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=50); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "jpeg_30":
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=30); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "blur_2.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=2.5))
    if name == "noise_005":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr += np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.05
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "crop_70":
        W, H = pil.size
        cw, ch = int(W * 0.7), int(H * 0.7)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    raise ValueError(f"Unknown attack: {name}")


# ============================================================ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000,
                   help="Skip first N images (used for training).")
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--resolution", type=int, default=None,
                   help="If None, use ckpt config's resolution.")
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_50", "jpeg_30", "blur_2.5", "noise_005", "crop_70"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    n_bits = int(ck["n_bits"])
    n_frags = int(ck["n_frags"])

    decoder = ClassicalFragDecoder(n_bits=n_bits, n_frags=n_frags).to(device).eval()
    decoder.load_state_dict(ck["decoder_state_dict"])

    fragment_configs = ck["fragment_configs"]
    subkeys = ck["subkeys"]
    targets_np = [np.asarray(t, dtype=np.float32) for t in ck["per_fragment_targets"]]
    targets_t = [torch.from_numpy(t).to(device) for t in targets_np]

    cfg = ck.get("config") or {}
    resolution = args.resolution or int(cfg.get("resolution", 512))
    epsilon = float(cfg.get("epsilon", 8.0 / 255.0))
    weight = 1.0 / n_frags

    files = sorted([
        f for f in Path(args.image_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    test_files = files[args.start_idx:args.start_idx + args.n_test]
    if len(test_files) == 0:
        raise RuntimeError(f"No test files in {args.image_dir}[{args.start_idx}:]")

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] resolution={resolution}  epsilon={epsilon:.5f}  n_bits={n_bits}  n_frags={n_frags}")
    print(f"[eval] test images: {len(test_files)} (idx {args.start_idx}..{args.start_idx+len(test_files)-1})")
    print(f"[eval] attacks: {args.attacks}", flush=True)

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])

    per_attack = {atk: [[] for _ in range(n_frags)] for atk in args.attacks}

    with torch.no_grad():
        for i, fp in enumerate(test_files):
            pil = Image.open(fp).convert("RGB")
            pil = tfm(pil)

            img = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
            for k, conf in enumerate(fragment_configs):
                frag = generate_fragment(subkeys[k], img.shape, epsilon=epsilon, fragment_index=k)
                img = embed_single_fragment(img, frag, conf, weight=weight)
            wm_pil = Image.fromarray(np.clip(img.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))

            for atk in args.attacks:
                att_pil = apply_attack(atk, wm_pil)
                if att_pil.size != (resolution, resolution):
                    att_pil = att_pil.resize((resolution, resolution), Image.BILINEAR)
                arr = np.asarray(att_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
                t = torch.from_numpy(arr).unsqueeze(0).to(device)
                preds = decoder(t)
                for k in range(n_frags):
                    pb = (preds[k][0] > 0.5).float()
                    acc = (pb == targets_t[k]).float().mean().item()
                    per_attack[atk][k].append(acc)

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(test_files)}]", flush=True)

    per_attack_mean = {
        atk: [float(np.mean(per_attack[atk][k])) for k in range(n_frags)]
        for atk in args.attacks
    }
    per_fragment_mean = [
        float(np.mean([per_attack_mean[atk][k] for atk in args.attacks]))
        for k in range(n_frags)
    ]
    overall_mean = float(np.mean([v for accs in per_attack_mean.values() for v in accs]))

    summary = {
        "ckpt": args.ckpt,
        "n_test": len(test_files),
        "resolution": resolution,
        "epsilon": epsilon,
        "n_bits": n_bits,
        "n_frags": n_frags,
        "attacks": args.attacks,
        "per_attack_bit_acc": per_attack_mean,
        "per_fragment_mean": per_fragment_mean,
        "overall_mean": overall_mean,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("")
    print(f"Overall mean bit_acc: {overall_mean:.4f}")
    print("Per attack:")
    for atk in args.attacks:
        a = per_attack_mean[atk]
        print(f"  {atk:12s}  f0={a[0]:.3f}  f1={a[1]:.3f}  f2={a[2]:.3f}  mean={np.mean(a):.3f}")
    print(f"Per fragment mean: f0={per_fragment_mean[0]:.3f}  f1={per_fragment_mean[1]:.3f}  f2={per_fragment_mean[2]:.3f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
