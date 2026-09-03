"""Watermark-on-watermark interference study.

For each ordered pair (A, B): embed watermark A into the cover, then embed
watermark B on top of the A-watermarked image. Decode BOTH from the doubly
watermarked image. Optionally apply a distortion afterwards.

Reports:
  - self[A]              : A's bit_acc decoded from A-only image (sanity, ~1.0)
  - retention[A][B]      : A's bit_acc after B is stacked on top (does the 2nd
                          watermark damage the 1st?)
  - outer_success[A][B]  : B's bit_acc when embedded over an A-watermarked image
                          (can the 2nd watermark still embed?)
  - psnr[A][B]           : PSNR of the doubly watermarked image vs cover
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
from wbench.attacks import apply_attack, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=40)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   default=["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_b", "vine_r"])
    p.add_argument("--post_attack", default="none",
                   help="optional distortion applied to the doubly-watermarked image before decode")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_methods(args.methods, device)
    names = list(methods.keys())

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[stack] {len(files)} imgs, {len(names)} methods, post_attack={args.post_attack}", flush=True)

    self_acc = {a: [] for a in names}
    retention = {a: {b: [] for b in names} for a in names}   # A after B on top
    outer = {a: {b: [] for b in names} for a in names}       # B over A
    psnrs = {a: {b: [] for b in names} for a in names}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        rng = np.random.RandomState(1000 + i)

        # Pre-embed A-only images + payloads once per method.
        bitsA = {a: rng.randint(0, 2, methods[a].n_bits).astype(np.uint8) for a in names}
        imgA = {}
        for a in names:
            try:
                wa = methods[a].embed(cover, bitsA[a])
                if wa.size != cover.size:
                    wa = wa.resize(cover.size, Image.LANCZOS)
                imgA[a] = wa
                rec = methods[a].decode(wa)
                n = min(len(rec), len(bitsA[a]))
                self_acc[a].append(float(np.mean(rec[:n] == bitsA[a][:n])))
            except Exception as e:
                print(f"  [{a}] self-embed failed img {i}: {e}", flush=True)

        # Stack B on top of each A.
        for a in names:
            if a not in imgA:
                continue
            for b in names:
                bitsB = rng.randint(0, 2, methods[b].n_bits).astype(np.uint8)
                try:
                    wab = methods[b].embed(imgA[a], bitsB)
                    if wab.size != cover.size:
                        wab = wab.resize(cover.size, Image.LANCZOS)
                    psnrs[a][b].append(psnr(cover, wab))
                    if args.post_attack != "none":
                        wab = apply_attack(args.post_attack, wab, device)
                        if wab.size != cover.size:
                            wab = wab.resize(cover.size, Image.LANCZOS)
                    # decode A (inner) and B (outer)
                    ra = methods[a].decode(wab)
                    na = min(len(ra), len(bitsA[a]))
                    retention[a][b].append(float(np.mean(ra[:na] == bitsA[a][:na])))
                    rb = methods[b].decode(wab)
                    nb = min(len(rb), len(bitsB))
                    outer[a][b].append(float(np.mean(rb[:nb] == bitsB[:nb])))
                except Exception as e:
                    print(f"  [{a}+{b}] stack failed img {i}: {e}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    def mean(d):
        return float(np.mean(d)) if d else None

    summary = {
        "resolution": args.resolution, "n_images": len(files),
        "post_attack": args.post_attack,
        "display": {a: methods[a].name for a in names},
        "n_bits": {a: methods[a].n_bits for a in names},
        "self_acc": {a: mean(self_acc[a]) for a in names},
        "retention": {a: {b: mean(retention[a][b]) for b in names} for a in names},
        "outer_success": {a: {b: mean(outer[a][b]) for b in names} for a in names},
        "psnr": {a: {b: mean(psnrs[a][b]) for b in names} for a in names},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    disp = [methods[a].name for a in names]
    def mat(title, M):
        print(f"\n{title}  (row=A inner / first, col=B outer / second)")
        print(f"{'A\\\\B':<12}" + "".join(f"{d[:10]:>11}" for d in disp))
        for a, da in zip(names, disp):
            row = f"{da:<12}"
            for b in names:
                v = M[a][b]
                row += f"{(v if v is not None else float('nan')):>11.3f}"
            print(row)

    print("\nself bit_acc (A alone):", {methods[a].name: round(summary['self_acc'][a], 3) for a in names})
    mat("RETENTION: A's bit_acc after B stacked on top", summary["retention"])
    mat("OUTER SUCCESS: B's bit_acc embedded over A", summary["outer_success"])
    mat("PSNR of doubly-watermarked image", summary["psnr"])
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
