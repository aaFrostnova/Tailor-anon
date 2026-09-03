"""Generate watermarked images for each method (for a post-processing stacking test).

For each method, watermark the SAME N cover images with a random per-image
payload, and save: cover PNG, watermarked PNG (lossless), and the payload bits.
A manifest.json records method + bits + paths so decode_check.py can verify
how much the original watermark survives after you apply your own watermark on top.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wbench.methods import build_methods
from wbench.attacks import psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_per_method", type=int, default=3)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   default=["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_b", "vine_r"])
    p.add_argument("--out_dir", default="results/wbench_baselines/generated")
    args = p.parse_args()

    device = "cuda"
    methods = build_methods(args.methods, device)

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    covers = files[args.start_idx:args.start_idx + args.n_per_method]
    if len(covers) < args.n_per_method:
        raise SystemExit("not enough cover images")

    out_root = REPO / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"resolution": args.resolution, "n_per_method": args.n_per_method,
                "entries": []}

    for mname, method in methods.items():
        mdir = out_root / mname
        mdir.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(covers):
            cover = Image.open(fp).convert("RGB").resize(
                (args.resolution, args.resolution), Image.LANCZOS)
            rng = np.random.RandomState(7000 + i)  # same seed per image index across methods
            bits = rng.randint(0, 2, method.n_bits).astype(np.uint8)
            wm = method.embed(cover, bits)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)

            cover_path = mdir / f"img{i}_cover.png"
            wm_path = mdir / f"img{i}_wm.png"
            bits_path = mdir / f"img{i}_bits.npy"
            cover.save(cover_path)            # PNG: lossless
            wm.save(wm_path)                  # PNG: lossless (no JPEG before your post-process)
            np.save(bits_path, bits)

            # sanity: decode the freshly watermarked image (should be ~1.0)
            rec = method.decode(wm)
            n = min(len(rec), len(bits))
            self_acc = float(np.mean(rec[:n] == bits[:n]))
            ps = psnr(cover, wm)

            manifest["entries"].append({
                "method": mname,
                "display_name": method.name,
                "n_bits": int(method.n_bits),
                "img_index": i,
                "source_file": str(fp),
                "cover_png": str(cover_path),
                "wm_png": str(wm_path),
                "bits_npy": str(bits_path),
                "self_bit_acc": round(self_acc, 4),
                "psnr": round(ps, 2),
            })
            print(f"  {method.name:<12} img{i}: self_bit_acc={self_acc:.3f} psnr={ps:.1f}dB "
                  f"-> {wm_path.name}", flush=True)

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] {len(manifest['entries'])} watermarked images -> {out_root}")
    print(f"manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
