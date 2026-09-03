"""Post-hoc multi-fragment composite: interference + combined robustness.

Stacks the post-hoc fragments (cross-subspace) carrying the SAME shortened-BCH codeword,
then measures (1) INTERFERENCE -- each fragment's bit-accuracy decoded from the full clean
stack (does stacking hurt self-detection?), and (2) COMBINED ROBUSTNESS -- per-fragment
detection, soft-LLR fused detection, and OR(fused, TrustMark) under the full attack
spectrum (distortion, geometric, regeneration, overwrite).

Fragments (soft-fused, share the codeword): phasemark (SD2.1 latent phase),
vine (SDXL latent), dft_kred (FFT-mag), quant_qim (block-mean). TrustMark runs as an
independent OR tier (deep pixel, geometric). Each lives in a DIFFERENT subspace, so by the
overwrite-taxonomy they should not mutually overwrite -- this script tests that empirically.

Embed order is configurable (--order); the interference matrix informs the best order.
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

from src.payload import image_id_to_payload                              # noqa: E402
from src.shortened_bch import ShortenedBCH                               # noqa: E402
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs       # noqa: E402
from src.soft_bch import decode_and_verify                              # noqa: E402
from benchmark_fused import apply_attack, psnr, adaptive_weights         # noqa: E402

MASTER_KEY = b"v5_key_encoder_master"
DFT_CKPT = "results/dft_fftaware_baseline/ckpt.pt"
QIM_CKPT = "results/quant_qim_frozen_d006/ckpt.pt"
# soft-fused fragments: name -> (soft-read fn attr, kind for LLR alignment)
SOFT_KIND = {"phasemark": "logit", "vine": "prob", "dft_kred": "logit", "quant_qim": "logit"}


def build_frags(names, device):
    frags = {}
    for n in names:
        if n == "phasemark":
            from src.phasemark import PhaseMarkWrapper
            frags[n] = PhaseMarkWrapper(MASTER_KEY, "phasemark", 100, "sd21", device)
        elif n == "vine":
            from src.vine_crypto_wrapper import VineCryptoWrapper
            frags[n] = VineCryptoWrapper(MASTER_KEY, "vine", 100, device)
        elif n == "dft_kred":
            from src.learned_fragment_methods import DFTKredMethod
            frags[n] = DFTKredMethod(str(REPO / DFT_CKPT), MASTER_KEY, "dft_kred", device)
        elif n == "quant_qim":
            from src.learned_fragment_methods import QuantQIMMethod
            frags[n] = QuantQIMMethod(str(REPO / QIM_CKPT), MASTER_KEY, "quant_qim", device)
        print(f"[frag] {n} loaded", flush=True)
    return frags


def frag_embed(frag, name, pil, image_id, tx):
    if name in ("dft_kred", "quant_qim"):
        return frag.embed(pil, image_id, tx)        # explicit codeword
    return frag.embed(pil, image_id)                # phasemark / vine compute it from id


def frag_soft(frag, name, pil):
    if name == "vine":
        return frag.raw_probs(pil)
    if name == "phasemark":
        return frag.raw_scores(pil)
    return frag.raw_logits(pil)                      # dft / qim


def fused_detect(frags, sb, pil, image_id):
    """Per-fragment bit-acc + soft-fused (adaptive) detection over the codeword."""
    aligned, per = {}, {}
    for name, fr in frags.items():
        soft = frag_soft(fr, name, pil)
        perm, M = fr.get_perm_M(image_id)
        a = method_soft_to_codeword_llr(soft, perm, M, kind=SOFT_KIND[name], n_codeword=sb.n)
        aligned[name] = a
        d = decode_and_verify(a, image_id, codec=sb, p=6)
        per[name] = d["data_bit_acc"]
    fused = fuse_llrs(aligned, weights=adaptive_weights(aligned), n_codeword=sb.n)
    det = decode_and_verify(fused, image_id, codec=sb, p=8)["detected"]
    return per, bool(det)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--order", nargs="+", default=["phasemark", "vine", "dft_kred", "quant_qim"],
                   help="embed order of the soft-fused fragments")
    p.add_argument("--trustmark", action="store_true", help="add TrustMark as an independent OR tier")
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_50", "blur_2", "noise_50", "crop_90", "resize_110",
                            "regen_20_sd15", "regen_30_sd15", "flux_vae", "flux_i2i_20", "flux_i2i_30", "rinse_2"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    sb = ShortenedBCH()
    frags = build_frags(args.order, device)
    tm = None
    if args.trustmark:
        from wbench.methods import TrustMarkMethod
        tm = TrustMarkMethod("Q")
        print("[frag] trustmark loaded", flush=True)

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    R = args.resolution
    print(f"[composite] {len(files)} imgs, order={args.order}, trustmark={args.trustmark}", flush=True)

    interference = {n: [] for n in args.order}            # self bit-acc in the clean stack
    stack_psnr = []
    A = {a: {"fused_det": [], "or_det": [], "tm_det": [], **{f"per_{n}": [] for n in args.order}}
         for a in args.attacks}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((R, R), Image.LANCZOS)
        image_id = f"composite_{args.start_idx + i:06d}"
        tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))[:100]
        # ---- embed the stack in order ----
        wm = cover
        for name in args.order:
            wm = frag_embed(frags[name], name, wm, image_id, tx)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)
        tm_bits = None
        if tm is not None:
            tm_bits = np.random.RandomState(1000 + i).randint(0, 2, tm.n_bits).astype(np.uint8)
            wm = tm.embed(wm, tm_bits)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)
        stack_psnr.append(psnr(cover, wm))
        # ---- interference: self bit-acc from the clean stack ----
        per_clean, _ = fused_detect(frags, sb, wm, image_id)
        for n in args.order:
            interference[n].append(per_clean[n])
        # ---- robustness under attacks ----
        for a in args.attacks:
            att = apply_attack(a, wm)
            if att.size != cover.size:
                att = att.resize(cover.size, Image.LANCZOS)
            per, fdet = fused_detect(frags, sb, att, image_id)
            for n in args.order:
                A[a][f"per_{n}"].append(per[n])
            A[a]["fused_det"].append(1.0 if fdet else 0.0)
            tmdet = 0.0
            if tm is not None:
                rec = tm.decode(att); nb = min(len(rec), len(tm_bits))
                tmdet = 1.0 if np.mean(rec[:nb] == tm_bits[:nb]) >= 0.95 else 0.0
            A[a]["tm_det"].append(tmdet)
            A[a]["or_det"].append(1.0 if (fdet or tmdet > 0.5) else 0.0)
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    def mean(x):
        return float(np.mean(x)) if x else None

    out = {"n_images": len(files), "order": args.order, "trustmark": args.trustmark,
           "stack_psnr": mean(stack_psnr),
           "interference_self_bitacc": {n: mean(interference[n]) for n in args.order},
           "attacks": {}}
    for a in args.attacks:
        out["attacks"][a] = {"fused_det": mean(A[a]["fused_det"]), "or_det": mean(A[a]["or_det"]),
                             "tm_det": mean(A[a]["tm_det"]),
                             **{f"per_{n}": mean(A[a][f"per_{n}"]) for n in args.order}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    print(f"\n=== INTERFERENCE (self bit-acc in clean stack; ~1.0 = no interference) ===")
    print("  " + "  ".join(f"{n}={out['interference_self_bitacc'][n]:.3f}" for n in args.order)
          + f"   stack PSNR={out['stack_psnr']:.1f}dB")
    print(f"\n=== COMBINED ROBUSTNESS (per-fragment bit-acc | fused | OR) ===")
    cols = args.order + (["TM"] if tm is not None else [])
    print(f"  {'attack':<15}" + "".join(f"{n[:8]:>9}" for n in cols) + f"{'FUSED':>8}{'OR':>6}")
    for a in args.attacks:
        c = out["attacks"][a]
        row = f"  {a:<15}" + "".join(f"{c[f'per_{n}']:>9.2f}" for n in args.order)
        if tm is not None:
            row += f"{c['tm_det']:>9.2f}"
        row += f"{c['fused_det']:>8.2f}{c['or_det']:>6.2f}"
        print(row)
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
