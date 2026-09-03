"""Overwrite-attack evaluation (Phase C + B.1 verification).

For each victim watermark x attack, measure victim bit-accuracy, detection,
attack fidelity (PSNR + LPIPS), and spoofing. Attacks are either:
  - overwriters (key-free, src/overwrite_attacks.py): ow_fft, ow_block, ow_dwt,
    ow_analytic  -- with a --target_psnr budget;
  - regeneration / distortion baselines via benchmark_fused.apply_attack:
    regen_20_sd15, flux_vae, flux_i2i_20, jpeg_50, blur_2.5, noise_50, clean.

Headline: at a fixed PSNR budget, the matched analytic overwriter drives its
victim's bit-acc to chance while leaving cross-class victims intact, and does so
at higher fidelity than regeneration. Run --sweep_psnr for the equal-fidelity
curve (one victim, matched overwriter strength swept vs regen strength).

Reuses build_all/UniformAdapter from overwrite_matrix.py and apply_attack/psnr
from benchmark_fused.py.
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

from overwrite_matrix import build_all                       # noqa: E402
from src.overwrite_attacks import (                          # noqa: E402
    FFTAnnulusSaturator, BlockMeanSaturator, DWTDCTSaturator, AnalyticUniversal, _psnr,
)

OVERWRITERS = {"ow_fft", "ow_block", "ow_dwt", "ow_analytic"}
# matched overwriter per victim (for the matched/mismatched analysis)
MATCHED = {
    "dft_kred": "ow_fft", "quant_qim": "ow_block",
    "dwtDct": "ow_dwt", "dwtDctSvd": "ow_dwt",
}


def make_overwriter(name, fft_delta, block_delta):
    if name == "ow_fft":
        return FFTAnnulusSaturator(delta=fft_delta)
    if name == "ow_block":
        return BlockMeanSaturator(delta=block_delta)
    if name == "ow_dwt":
        return DWTDCTSaturator()
    if name == "ow_analytic":
        return AnalyticUniversal(fft_delta=fft_delta, block_delta=block_delta)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--resolution", type=int, default=256,
                   help="256 = victim-native (clean analytic overwrite); 512 = deployment (downscale-coupled)")
    p.add_argument("--fft_delta", type=float, default=0.0, help="0 = calibrate from DFT-Kred victim")
    p.add_argument("--block_delta", type=float, default=0.12)
    p.add_argument("--victims", nargs="+",
                   default=["dft_kred", "quant_qim", "dwtDct", "dwtDctSvd", "vine_r", "trustmark"])
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "ow_fft", "ow_block", "ow_dwt", "ow_analytic",
                            "jpeg_50", "blur_2.5", "noise_50", "regen_20_sd15"])
    p.add_argument("--target_psnr", type=float, default=38.0)
    p.add_argument("--det_thresh", type=float, default=0.75)
    p.add_argument("--lpips", action="store_true", help="also compute LPIPS (loads net)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_all(args.victims, device)
    victims = [v for v in args.victims if v in methods]

    # pre-build overwriters once; calibrate the FFT delta from the loaded DFT-Kred victim
    fft_delta = args.fft_delta
    if fft_delta <= 0:
        try:
            fft_delta = methods["dft_kred"].frag.decode_delta()
        except Exception:
            fft_delta = 50.0
    print(f"[calib] fft_delta={fft_delta:.2f} block_delta={args.block_delta}", flush=True)
    ow_cache = {n: make_overwriter(n, fft_delta, args.block_delta) for n in args.attacks if n in OVERWRITERS}

    apply_attack = None
    if any(a not in OVERWRITERS and a != "clean" for a in args.attacks):
        from benchmark_fused import apply_attack as _aa
        apply_attack = _aa

    lpips_fn = None
    if args.lpips:
        import lpips as lpips_mod
        import torch
        lp = lpips_mod.LPIPS(net="alex").to(device).eval()

        def lpips_fn(a, b):  # a,b uint8 HxWx3
            import torch
            ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            with torch.no_grad():
                return float(lp(ta, tb).item())

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[bench-overwrite] {len(files)} imgs, victims={victims}, target_psnr={args.target_psnr}", flush=True)

    R = {v: {a: {"bitacc": [], "detect": [], "psnr": [], "lpips": [], "spoof": [], "fpr": []}
             for a in args.attacks} for v in victims}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        rng = np.random.RandomState(1000 + i)
        for v in victims:
            m = methods[v]
            bitsV = rng.randint(0, 2, m.n_bits).astype(np.uint8)
            wm = m.embed(cover, bitsV)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)
            wm_u8 = np.asarray(wm)
            for a in args.attacks:
                if a == "clean":
                    att = wm
                elif a in OVERWRITERS:
                    att = ow_cache[a].overwrite(wm, target_psnr=args.target_psnr)
                else:
                    att = apply_attack(a, wm)
                    if att.size != cover.size:
                        att = att.resize(cover.size, Image.LANCZOS)
                att_u8 = np.asarray(att.convert("RGB"))
                rec = m.decode(att)
                nb = min(len(rec), len(bitsV))
                ba = float(np.mean(rec[:nb] == bitsV[:nb]))
                R[v][a]["bitacc"].append(ba)
                R[v][a]["detect"].append(1.0 if ba >= args.det_thresh else 0.0)
                R[v][a]["psnr"].append(_psnr(wm_u8, att_u8))   # attack fidelity vs watermarked
                if lpips_fn is not None:
                    R[v][a]["lpips"].append(lpips_fn(wm_u8, att_u8))
                # FPR: decode attacked vs a fresh random "wrong" payload
                wrong = rng.randint(0, 2, m.n_bits).astype(np.uint8)
                R[v][a]["fpr"].append(float(np.mean(rec[:nb] == wrong[:nb]) >= args.det_thresh))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    def mean(x):
        return float(np.mean(x)) if x else None

    out = {"n_images": len(files), "resolution": args.resolution,
           "target_psnr": args.target_psnr, "det_thresh": args.det_thresh,
           "victims": {methods[v].name: {} for v in victims}}
    for v in victims:
        for a in args.attacks:
            d = R[v][a]
            out["victims"][methods[v].name][a] = {
                "bitacc": mean(d["bitacc"]), "detect": mean(d["detect"]),
                "psnr": mean(d["psnr"]), "lpips": mean(d["lpips"]) if d["lpips"] else None,
                "fpr": mean(d["fpr"]),
                "matched": MATCHED.get(v) == a,
            }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    # print: victim x attack detection (and bitacc)
    print(f"\n=== victim detection rate (bitacc) | attack PSNR ===")
    hdr = f"{'victim':<12}"
    for a in args.attacks:
        hdr += f"{a[:11]:>13}"
    print(hdr)
    for v in victims:
        row = f"{methods[v].name[:12]:<12}"
        for a in args.attacks:
            c = out["victims"][methods[v].name][a]
            row += f"{c['detect']:>5.2f}({c['bitacc']:.2f}) " if c['detect'] is not None else f"{'-':>13}"
        print(row)
    print(f"\n  (matched overwriter cells should show detect~0; PSNR target {args.target_psnr})")
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
