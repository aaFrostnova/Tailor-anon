"""Does PhaseMark fill the post-hoc REGENERATION gap? (the falsification test)

Embeds PhaseMark (residual mode) and runs the regeneration attack spectrum, measuring
bit-accuracy survival + the embed PSNR + the VAE round-trip ceiling. The claim to falsify:
the paper reports diffusion-regen TPR@1%FPR = 1.000; our flux_i2i_30 / regen_30_sd15 are
stronger than the paper's attacks. If PhaseMark survives where VINE collapses (dies above
flux_i2i_15), it is the post-hoc regeneration tier.

Writes a persistent JSON (results survive node/tmp resets). Run:
  python scripts/attack/test_phasemark_regen.py --n_images 10
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
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--vae_key", default="sd21")
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "vae", "flux_vae", "jpeg_50", "regen_10_sd15", "regen_20_sd15",
        "regen_30_sd15", "flux_i2i_20", "flux_i2i_30", "rinse_2"])
    p.add_argument("--output", default="results/attack/phasemark_regen.json")
    args = p.parse_args()

    pm = PhaseMark(args.vae_key, "cuda")
    from benchmark_fused import apply_attack

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))
    imgs = imgs[args.start_idx:args.start_idx + args.n_images]

    ceil = float(np.mean([psnr(Image.open(p).convert("RGB").resize((512, 512)),
                               pm.vae.roundtrip(Image.open(p).convert("RGB").resize((512, 512))))
                          for p in imgs[:5]]))
    print(f"[phasemark-regen] VAE ceiling {ceil:.1f}dB, vae_key={args.vae_key}, n={len(imgs)}", flush=True)

    acc = {a: [] for a in args.attacks}
    fpr = []
    ps = []
    for i, fp in enumerate(imgs):
        img = Image.open(fp).convert("RGB").resize((512, 512))
        bits = np.random.RandomState(i).randint(0, 2, 100).astype(np.uint8)
        wm = pm.embed(img, bits)                                # residual mode (default)
        ps.append(psnr(img, wm))
        fpr.append(float(np.mean(pm.detect(wm, 100) == np.random.RandomState(i + 1).randint(0, 2, 100))))
        for a in args.attacks:
            att = apply_attack(a, wm)
            if att.size != (512, 512):
                att = att.resize((512, 512))
            acc[a].append(float(np.mean(pm.detect(att, 100) == bits)))
        print(f"  [{i+1}/{len(imgs)}] embed psnr {ps[-1]:.1f}  clean {acc['clean'][-1]:.3f}", flush=True)

    out = {"vae_key": args.vae_key, "n_images": len(imgs), "vae_ceiling_db": ceil,
           "embed_psnr": float(np.mean(ps)), "wrong_key_fpr": float(np.mean(fpr)),
           "attacks": {a: float(np.mean(acc[a])) for a in args.attacks}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    print(f"\n=== PhaseMark vs regeneration (residual embed, PSNR {out['embed_psnr']:.1f}dB, "
          f"wrong-key FPR {out['wrong_key_fpr']:.2f}) ===")
    for a in args.attacks:
        print(f"  {a:<16} {out['attacks'][a]:.3f}")
    print(f"\n[done] -> {args.output}")
    print("Compare: VINE-R dies above flux_i2i_15; GS (gen-time) det 1.00/0.93 at flux_i2i_30.")


if __name__ == "__main__":
    main()
