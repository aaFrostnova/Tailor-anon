"""Equal-fidelity sweep: overwrite vs regeneration vs distortion (Phase C headline).

For a victim and its matched analytic overwriter, sweep each attack's strength and
record (attack PSNR vs watermarked, victim bit-accuracy). The claim: at EQUAL
PSNR, the matched overwrite drives the victim to chance while regeneration /
distortion still leaves it detectable -- i.e. overwrite is a higher-fidelity
removal. Runs at the victim's native resolution (256), where the analytic
subspace-collision is clean.

Reuses build_all (overwrite_matrix), the saturators (overwrite_attacks), and
apply_attack (benchmark_fused).
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
sys.path.insert(0, str(REPO / "scripts" / "attack"))

from overwrite_matrix import build_all                                   # noqa: E402
from src.overwrite_attacks import FFTAnnulusSaturator, BlockMeanSaturator, _psnr  # noqa: E402

# victim -> (matched overwriter factory, overwriter-strength sweep as target_psnr list)
OW_PSNR_SWEEP = [26, 28, 30, 32, 34, 36, 40]
REGEN_SWEEP = ["regen_30_sd15", "regen_20_sd15", "regen_15_sd15", "regen_10_sd15", "regen_5_sd15"]
JPEG_SWEEP = ["jpeg_30", "jpeg_50", "jpeg_70", "jpeg_85", "jpeg_95"]


def matched_overwriter(victim, fft_delta, block_delta):
    if victim == "dft_kred":
        return FFTAnnulusSaturator(resolution=256, delta=fft_delta)
    if victim == "quant_qim":
        return BlockMeanSaturator(resolution=256, delta=block_delta)
    raise ValueError(f"no matched analytic overwriter for {victim}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=16)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--victims", nargs="+", default=["quant_qim", "dft_kred"])
    p.add_argument("--block_delta", type=float, default=0.12)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_all(args.victims, device)
    victims = [v for v in args.victims if v in methods]
    fft_delta = 50.0
    try:
        fft_delta = methods["dft_kred"].frag.decode_delta()
    except Exception:
        pass
    print(f"[sweep] fft_delta={fft_delta:.2f} block_delta={args.block_delta}", flush=True)

    from benchmark_fused import apply_attack

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]

    # per victim: lists of (psnr, bitacc) for each family
    res = {v: {"overwrite": [], "regen": [], "jpeg": []} for v in victims}

    for vi, v in enumerate(victims):
        m = methods[v]
        ow = matched_overwriter(v, fft_delta, args.block_delta)
        # accumulate per sweep-point across images
        pts = {"overwrite": {t: ([], []) for t in OW_PSNR_SWEEP},
               "regen": {t: ([], []) for t in REGEN_SWEEP},
               "jpeg": {t: ([], []) for t in JPEG_SWEEP}}
        for i, fp in enumerate(files):
            cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
            rng = np.random.RandomState(1000 + i)
            bits = rng.randint(0, 2, m.n_bits).astype(np.uint8)
            wm = m.embed(cover, bits)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)
            wm_u8 = np.asarray(wm)

            def bitacc(att):
                rec = m.decode(att)
                n = min(len(rec), len(bits))
                return float(np.mean(rec[:n] == bits[:n]))

            for t in OW_PSNR_SWEEP:
                att = ow.overwrite(wm, target_psnr=t)
                pts["overwrite"][t][0].append(_psnr(wm_u8, np.asarray(att)))
                pts["overwrite"][t][1].append(bitacc(att))
            for name in REGEN_SWEEP:
                att = apply_attack(name, wm)
                if att.size != wm.size:
                    att = att.resize(wm.size, Image.LANCZOS)
                pts["regen"][name][0].append(_psnr(wm_u8, np.asarray(att)))
                pts["regen"][name][1].append(bitacc(att))
            for name in JPEG_SWEEP:
                att = apply_attack(name, wm)
                pts["jpeg"][name][0].append(_psnr(wm_u8, np.asarray(att)))
                pts["jpeg"][name][1].append(bitacc(att))
            if (i + 1) % 5 == 0:
                print(f"  [{v}] {i+1}/{len(files)}", flush=True)

        for fam, sweep in (("overwrite", OW_PSNR_SWEEP), ("regen", REGEN_SWEEP), ("jpeg", JPEG_SWEEP)):
            for t in sweep:
                ps, ba = pts[fam][t]
                res[v][fam].append({"point": str(t), "psnr": float(np.mean(ps)), "bitacc": float(np.mean(ba))})

    out = {"n_images": len(files), "resolution": args.resolution,
           "fft_delta": float(fft_delta), "block_delta": args.block_delta,
           "victims": {methods[v].name: res[v] for v in victims}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    for v in victims:
        print(f"\n=== {methods[v].name}: bitacc vs attack-PSNR (lower bitacc at equal PSNR = better attack) ===")
        for fam in ("overwrite", "regen", "jpeg"):
            row = "  %-10s " % fam
            for pt in sorted(res[v][fam], key=lambda x: x["psnr"]):
                row += f"({pt['psnr']:.0f}dB:{pt['bitacc']:.2f}) "
            print(row)
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
