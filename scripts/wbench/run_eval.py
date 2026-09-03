"""Unified baseline watermark evaluation under the attack suite.

For each method: embed a random per-image payload, apply each attack, decode,
measure bit accuracy. Also report PSNR of the clean watermarked image.
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

from wbench.attacks import apply_attack, psnr
from wbench.methods import build_methods

_EDITING = {"global_edit"}


def _dispatch_attack(name, wm, device, idx):
    """Route editing attacks to editing.py, everything else to attacks.py."""
    if name in _EDITING or name.startswith("local_edit"):
        from wbench.editing import apply_editing
        return apply_editing(name, wm, device, idx=idx)
    return apply_attack(name, wm, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   default=["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_r", "vine_b"])
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_75", "jpeg_50", "blur_1.5", "noise_50",
                            "crop_90", "resize_110", "regen_10_sd15", "regen_20_sd15"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_methods(args.methods, device)
    if not methods:
        raise SystemExit("no methods loaded")

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[eval] {len(files)} images @ {args.resolution}px, "
          f"{len(methods)} methods, {len(args.attacks)} attacks", flush=True)

    # results[method][attack] = list of bit accuracies; psnrs[method] = list
    results = {m: {a: [] for a in args.attacks} for m in methods}
    psnrs = {m: [] for m in methods}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        for mname, method in methods.items():
            rng = np.random.RandomState(1000 + i)
            bits = rng.randint(0, 2, size=method.n_bits).astype(np.uint8)
            try:
                wm = method.embed(cover, bits)
            except Exception as e:
                print(f"  [{mname}] embed failed on img {i}: {e}", flush=True)
                continue
            if wm.size != (args.resolution, args.resolution):
                wm = wm.resize((args.resolution, args.resolution), Image.LANCZOS)
            psnrs[mname].append(psnr(cover, wm))
            for atk in args.attacks:
                try:
                    att = _dispatch_attack(atk, wm, device, i)
                    rec = method.decode(att)
                    acc = float(np.mean(rec[: method.n_bits] == bits[: len(rec)][: method.n_bits])
                                ) if len(rec) else 0.0
                    # guard length mismatch
                    n = min(len(rec), len(bits))
                    acc = float(np.mean(rec[:n] == bits[:n])) if n else 0.0
                    results[mname][atk].append(acc)
                except Exception as e:
                    print(f"  [{mname}/{atk}] decode failed img {i}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    summary = {"resolution": args.resolution, "n_images": len(files),
               "attacks": args.attacks, "methods": {}}
    for mname, method in methods.items():
        summary["methods"][mname] = {
            "display_name": method.name,
            "n_bits": method.n_bits,
            "psnr_mean": float(np.mean(psnrs[mname])) if psnrs[mname] else None,
            "bit_acc": {a: (float(np.mean(results[mname][a])) if results[mname][a] else None)
                        for a in args.attacks},
        }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty table
    print("\n" + "=" * 90)
    hdr = f"{'method':<14}{'bits':>5}{'PSNR':>7}  " + "".join(f"{a[:9]:>10}" for a in args.attacks)
    print(hdr); print("-" * len(hdr))
    for mname, method in methods.items():
        m = summary["methods"][mname]
        row = f"{m['display_name']:<14}{m['n_bits']:>5}{(m['psnr_mean'] or 0):>7.1f}  "
        for a in args.attacks:
            v = m["bit_acc"][a]
            row += f"{(v if v is not None else float('nan')):>10.3f}"
        print(row)
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
