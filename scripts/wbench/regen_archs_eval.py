"""Evaluate watermark robustness against regeneration by DIFFERENT architectures.

For each watermark method and each regen architecture: embed a random payload,
regenerate the watermarked image with that architecture, decode the original
watermark, report bit accuracy. Rows = watermark methods, cols = regen archs.
Lower bit_acc = that architecture's reconstruction wipes the watermark harder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wbench.methods import build_methods
from wbench.regen_archs import build_regens
from wbench.attacks import psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   default=["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_b", "vine_r"])
    p.add_argument("--regens", nargs="+", required=True,
                   help="regen arch keys, e.g. vae sd15 sdxlturbo lcm kandinsky vq")
    p.add_argument("--strength", type=float, default=0.4)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_methods(args.methods, device)
    specs = []
    for r in args.regens:
        kw = {} if r in ("vae", "vq", "hifivae", "flux_vae") else {"strength": args.strength}
        specs.append((r, kw))
    regens = build_regens(specs, device)

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[regen-eval] {len(files)} imgs, {len(methods)} wm methods, {len(regens)} regen archs",
          flush=True)

    # bitacc[method][regen] = list; regen_psnr[regen] = list (content distortion of the regen)
    bitacc = {m: {r: [] for r in regens} for m in methods}
    self_acc = {m: [] for m in methods}
    regen_psnr = {r: [] for r in regens}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        rng = np.random.RandomState(1000 + i)
        for mname, method in methods.items():
            bits = rng.randint(0, 2, method.n_bits).astype(np.uint8)
            wm = method.embed(cover, bits)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)
            rec0 = method.decode(wm); n0 = min(len(rec0), len(bits))
            self_acc[mname].append(float(np.mean(rec0[:n0] == bits[:n0])))
            for rname, regen in regens.items():
                try:
                    rg = regen.regen(wm)
                    if rg.size != cover.size:
                        rg = rg.resize(cover.size, Image.LANCZOS)
                    if mname == args.methods[0]:
                        regen_psnr[rname].append(psnr(cover, rg))
                    rec = method.decode(rg); n = min(len(rec), len(bits))
                    bitacc[mname][rname].append(float(np.mean(rec[:n] == bits[:n])))
                except Exception as e:
                    print(f"  [{mname}/{rname}] img{i} failed: {e}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    mean = lambda d: float(np.mean(d)) if d else None
    summary = {
        "resolution": args.resolution, "n_images": len(files), "strength": args.strength,
        "wm_display": {m: methods[m].name for m in methods},
        "regen_arch": {r: regens[r].arch for r in regens},
        "self_acc": {m: mean(self_acc[m]) for m in methods},
        "regen_content_psnr": {r: mean(regen_psnr[r]) for r in regens},
        "bit_acc": {m: {r: mean(bitacc[m][r]) for r in regens} for m in methods},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    rkeys = list(regens.keys())
    print("\n" + "=" * (16 + 9 + 12 * len(rkeys)))
    print(f"{'wm \\\\ regen':<16}{'self':>9}" + "".join(f"{regens[r].arch[:11]:>12}" for r in rkeys))
    print("-" * (16 + 9 + 12 * len(rkeys)))
    for m in methods:
        row = f"{methods[m].name:<16}{summary['self_acc'][m]:>9.3f}"
        for r in rkeys:
            v = summary["bit_acc"][m][r]
            row += f"{(v if v is not None else float('nan')):>12.3f}"
        print(row)
    print(f"{'regen PSNR(content)':<16}{'':>9}" + "".join(
        f"{(summary['regen_content_psnr'][r] or 0):>12.1f}" for r in rkeys))
    print("=" * (16 + 9 + 12 * len(rkeys)))
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
