"""DFT-magnitude carrier with K-redundancy on multiple radii (option B).

Each bit is embedded into K * M bins:
  - K = redundant angular copies on each radius
  - M = number of radii bands per bit (gives a small scale margin)
QIM-embed on each; decoder reads all K*M bins per bit and majority-votes.

This addresses blur (low-pass kills SOME bins; majority survives) and gives a
narrow window of scale tolerance. It does NOT solve true scale invariance —
crop_95 / resize_1.1 are expected to remain hard unless the redundancy spans
enough radii to bracket the rescaling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.sign_envelope import _hkdf_uint64_stream
from scripts.prototype_dft_carrier import apply_attack, psnr


def enumerate_annulus_bins(H: int, W: int, r_lo: float, r_hi: float):
    bins = []
    for fy in range(1, H // 2):
        for fx_signed in range(-(W // 2 - 1), W // 2):
            fx = fx_signed % W
            r = (fy ** 2 + fx_signed ** 2) ** 0.5
            if r_lo <= r <= r_hi:
                bins.append((fy, fx, r))
    bins.sort(key=lambda t: (t[0], t[1]))
    return bins


def setup_k_redundant_carriers(master_key: bytes, image_id: str, n_bits: int,
                               candidates_with_r, K: int, M: int):
    """Pick K*M unique carriers per bit. Try to place K per radius band; if a
    bit's M radius bands cannot supply enough bins, fall back to global pool."""
    # Split candidates into M radii bands by sorted radius
    candidates_with_r = sorted(candidates_with_r, key=lambda t: t[2])
    n_per_band = len(candidates_with_r) // M
    bands = [candidates_with_r[i * n_per_band:(i + 1) * n_per_band] for i in range(M)]

    needed = n_bits * K * M
    pos_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/kred_position").encode("utf-8"),
        needed * 6,
    )
    sign_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/kred_sign").encode("utf-8"), n_bits,
    )
    bit_flip = (sign_stream % 2).astype(np.uint8)

    used = set()
    carriers = [[] for _ in range(n_bits)]
    cursor = 0

    def pop_idx_in_band(band):
        nonlocal cursor
        # try up to 200 hops for uniqueness
        for _ in range(200):
            if cursor >= len(pos_stream):
                raise RuntimeError("HKDF stream exhausted")
            j = int(pos_stream[cursor] % len(band)); cursor += 1
            fy, fx, _ = band[j]
            if (fy, fx) not in used:
                used.add((fy, fx))
                return (fy, fx)
        raise RuntimeError("could not find unique bin in band")

    for i in range(n_bits):
        for m_idx in range(M):
            band = bands[m_idx]
            for _ in range(K):
                carriers[i].append(pop_idx_in_band(band))

    return carriers, bit_flip


def qim_embed_k(img_uint8, payload, carriers, bit_flip, delta, channel=1):
    """carriers: list of n_bits lists, each containing K*M (fy, fx) tuples."""
    img = img_uint8.astype(np.float64) / 255.0
    H, W, _ = img.shape
    ch = img[:, :, channel].copy()
    F = np.fft.fft2(ch)

    for i, positions in enumerate(carriers):
        effective_bit = int(payload[i]) ^ int(bit_flip[i])
        for (fy, fx) in positions:
            mag = abs(F[fy, fx])
            phase = np.angle(F[fy, fx])
            target_mag = np.round(mag / delta) * delta + (delta / 2.0) * effective_bit
            new_val = target_mag * np.exp(1j * phase)
            F[fy, fx] = new_val
            F[(-fy) % H, (-fx) % W] = np.conj(new_val)

    ch_w = np.real(np.fft.ifft2(F))
    out = img.copy()
    out[:, :, channel] = np.clip(ch_w, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def qim_decode_k(img_uint8, carriers, bit_flip, delta, channel=1):
    img = img_uint8.astype(np.float64) / 255.0
    ch = img[:, :, channel]
    F = np.fft.fft2(ch)
    half = delta / 2.0
    recovered = np.zeros(len(carriers), dtype=np.uint8)
    for i, positions in enumerate(carriers):
        votes = 0
        for (fy, fx) in positions:
            mag = abs(F[fy, fx])
            n0 = round(mag / delta) * delta
            n1 = round((mag - half) / delta) * delta + half
            votes += 1 if abs(mag - n1) < abs(mag - n0) else 0
        majority = 1 if votes * 2 > len(positions) else 0
        recovered[i] = majority ^ int(bit_flip[i])
    return recovered


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--n_bits", type=int, default=127)
    p.add_argument("--delta", type=float, default=200.0)
    p.add_argument("--K", type=int, default=5,
                   help="Angular redundancy per radius band.")
    p.add_argument("--M", type=int, default=2,
                   help="Number of radius bands per bit (scale margin).")
    p.add_argument("--r_lo", type=float, default=18.0)
    p.add_argument("--r_hi", type=float, default=62.0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_75", "jpeg_50", "blur_1.5", "crop_95", "resize_1.1"])
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    H = W = args.resolution
    cand = enumerate_annulus_bins(H, W, args.r_lo, args.r_hi)
    print(f"[setup] candidates={len(cand)}  n_bits={args.n_bits}  K={args.K}  M={args.M}  "
          f"per-bit bins={args.K * args.M}  delta={args.delta}", flush=True)

    master_key = args.master_key.encode("utf-8")

    files = sorted([
        f for f in Path(args.image_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    test_files = files[args.start_idx:args.start_idx + args.n_test]

    psnrs = []
    per_attack_acc = {atk: [] for atk in args.attacks}

    for idx, fp in enumerate(test_files):
        pil = Image.open(fp).convert("RGB").resize((W, H), Image.LANCZOS)
        image_id = f"kred_{idx:05d}"

        carriers, bit_flip = setup_k_redundant_carriers(
            master_key, image_id, args.n_bits, cand, args.K, args.M,
        )
        rng = np.random.default_rng(1000 + idx)
        payload = rng.integers(0, 2, size=args.n_bits, dtype=np.uint8)

        img_arr = np.asarray(pil)
        wm_arr = qim_embed_k(img_arr, payload, carriers, bit_flip, args.delta, args.channel)
        psnrs.append(psnr(img_arr, wm_arr))

        wm_pil = Image.fromarray(wm_arr)
        for atk in args.attacks:
            att_pil = apply_attack(atk, wm_pil, args.resolution)
            if att_pil.size != (W, H):
                att_pil = att_pil.resize((W, H), Image.BILINEAR)
            att_arr = np.asarray(att_pil)
            dec = qim_decode_k(att_arr, carriers, bit_flip, args.delta, args.channel)
            per_attack_acc[atk].append(float(np.mean(dec == payload)))

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_files)}]  psnr={np.mean(psnrs):.2f}dB", flush=True)

    psnr_mean = float(np.mean(psnrs)); psnr_std = float(np.std(psnrs))
    per_attack = {
        atk: {"bit_acc_mean": float(np.mean(per_attack_acc[atk])),
              "bit_acc_std": float(np.std(per_attack_acc[atk]))}
        for atk in args.attacks
    }

    crit = []
    def add(name, actual, threshold, op=">="):
        passed = (actual >= threshold) if op == ">=" else (actual > threshold)
        crit.append({"name": name, "actual": round(actual, 4),
                     "threshold": threshold, "passed": passed})

    add("psnr_mean >= 38 dB", psnr_mean, 38.0)
    for atk, thr in [("clean", 0.99), ("jpeg_75", 0.90), ("blur_1.5", 0.85),
                     ("crop_95", 0.85), ("resize_1.1", 0.85)]:
        if atk in per_attack:
            add(f"{atk} bit_acc >= {thr:.2f}", per_attack[atk]["bit_acc_mean"], thr)
    passed = all(c["passed"] for c in crit)

    summary = {
        "config": vars(args),
        "n_test": len(test_files),
        "candidate_bins": len(cand),
        "psnr_mean": psnr_mean, "psnr_std": psnr_std,
        "per_attack": per_attack,
        "verdict": {"passed": passed, "criteria": crit},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"PSNR mean: {psnr_mean:.2f} dB ± {psnr_std:.2f}")
    for atk in args.attacks:
        print(f"  {atk:12s}: {per_attack[atk]['bit_acc_mean']:.4f} ± "
              f"{per_attack[atk]['bit_acc_std']:.4f}")
    print(f"\nVerdict: {'PASS' if passed else 'FAIL'}")
    for c in crit:
        print(f"  {'✓' if c['passed'] else '✗'} {c['name']}  actual={c['actual']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
