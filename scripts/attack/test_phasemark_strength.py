"""PhaseMark PSNR<->robustness tradeoff: sweep the soft-modulation strength.

strength=1.0 is full APM (most robust, lowest PSNR ~23dB); strength<1 blends the
latent perturbation toward the original (higher PSNR, less robust). Find the operating
point (target ~30dB while retaining regeneration survival). Writes a persistent JSON.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.phasemark import PhaseMark        # noqa: E402


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255; b = np.asarray(b, np.float64) / 255
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1 / mse) if mse > 1e-12 else 99.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--strengths", nargs="+", type=float, default=[0.4, 0.55, 0.7, 0.85, 1.0])
    p.add_argument("--attacks", nargs="+", default=["clean", "regen_20_sd15", "flux_i2i_20"])
    p.add_argument("--output", default="results/attack/phasemark_strength.json")
    args = p.parse_args()

    pm = PhaseMark("sd21", "cuda")
    from benchmark_fused import apply_attack
    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))
    imgs = imgs[args.start_idx:args.start_idx + args.n_images]
    print(f"[strength-sweep] n={len(imgs)} strengths={args.strengths}", flush=True)

    out = {"n_images": len(imgs), "rows": []}
    for s in args.strengths:
        ps = []
        acc = {a: [] for a in args.attacks}
        for i, fp in enumerate(imgs):
            img = Image.open(fp).convert("RGB").resize((512, 512))
            bits = np.random.RandomState(i).randint(0, 2, 100).astype(np.uint8)
            wm = pm.embed(img, bits, strength=s)
            ps.append(psnr(img, wm))
            for a in args.attacks:
                att = apply_attack(a, wm)
                if att.size != (512, 512):
                    att = att.resize((512, 512))
                acc[a].append(float(np.mean(pm.detect(att, 100) == bits)))
        row = {"strength": s, "psnr": float(np.mean(ps)),
               **{a: float(np.mean(acc[a])) for a in args.attacks}}
        out["rows"].append(row)
        print(f"  strength {s:.2f}: psnr {row['psnr']:.1f}dB  " +
              "  ".join(f"{a}={row[a]:.3f}" for a in args.attacks), flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
