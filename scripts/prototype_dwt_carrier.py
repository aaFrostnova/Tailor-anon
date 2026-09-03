"""DWT-detail-subband QIM carrier (option C).

Embeds N bits into level-2 detail subbands (LH/HL/HH) of the green channel
via 2D Haar DWT. QIM on selected coefficients. Reference-free decode.

Haar is chosen because it is orthonormal so Δ has the same scale as pixel
values, which makes the magnitude scale predictable across attacks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pywt
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.sign_envelope import _hkdf_uint64_stream
from scripts.prototype_dft_carrier import apply_attack, psnr


WAVELET = "haar"  # orthonormal so coefficient magnitudes ~ pixel scale
LEVEL = 2


def enumerate_dwt_positions(coeffs):
    """Enumerate (subband_idx, y, x) positions across cH2, cV2, cD2.

    coeffs is the pywt.wavedec2(..., level=LEVEL) output:
      [cA_L, (cH_L, cV_L, cD_L), (cH_L-1, cV_L-1, cD_L-1), ...]
    We use the level-LEVEL detail tuple (index 1).
    """
    details_l = coeffs[1]  # (cH_L, cV_L, cD_L)
    positions = []
    for sb_idx in range(3):
        H, W = details_l[sb_idx].shape
        # Exclude border 2-pixel ring to avoid wavelet boundary artifacts
        for y in range(2, H - 2):
            for x in range(2, W - 2):
                positions.append((sb_idx, y, x))
    return positions


def crypto_pick(master_key, image_id, n_bits, candidates):
    n_cand = len(candidates)
    pos_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/dwt_position").encode("utf-8"),
        n_bits * 4,
    )
    chosen = []
    used = set()
    k = 0
    while len(chosen) < n_bits:
        if k >= len(pos_stream):
            raise RuntimeError("HKDF stream exhausted")
        j = int(pos_stream[k] % n_cand); k += 1
        if j in used:
            continue
        used.add(j)
        chosen.append(candidates[j])

    sign_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/dwt_sign").encode("utf-8"), n_bits,
    )
    bit_flip = (sign_stream % 2).astype(np.uint8)
    return chosen, bit_flip


def qim_embed_dwt(img_uint8, payload, positions, bit_flip, delta, channel=1):
    img = img_uint8.astype(np.float64) / 255.0
    ch = img[:, :, channel]
    coeffs = pywt.wavedec2(ch, WAVELET, level=LEVEL)
    cA = coeffs[0]
    # coeffs is a list: [cA_L, (cH_L, cV_L, cD_L), (cH_L-1, ...), ...]
    # We modify the level-LEVEL (highest) detail tuple at index 1.
    detail_L = list(coeffs[1])

    for i, (sb_idx, y, x) in enumerate(positions):
        effective_bit = int(payload[i]) ^ int(bit_flip[i])
        v = detail_L[sb_idx][y, x]
        sign_v = 1.0 if v >= 0 else -1.0
        mag = abs(v)
        target_mag = np.round(mag / delta) * delta + (delta / 2.0) * effective_bit
        detail_L[sb_idx][y, x] = sign_v * target_mag

    coeffs2 = [cA, tuple(detail_L)] + list(coeffs[2:])
    ch_w = pywt.waverec2(coeffs2, WAVELET)
    # waverec2 may return slightly off-shape; crop to original
    ch_w = ch_w[:ch.shape[0], :ch.shape[1]]
    out = img.copy()
    out[:, :, channel] = np.clip(ch_w, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def qim_decode_dwt(img_uint8, positions, bit_flip, delta, channel=1):
    img = img_uint8.astype(np.float64) / 255.0
    ch = img[:, :, channel]
    coeffs = pywt.wavedec2(ch, WAVELET, level=LEVEL)
    detail_L = coeffs[1]

    half = delta / 2.0
    recovered = np.zeros(len(positions), dtype=np.uint8)
    for i, (sb_idx, y, x) in enumerate(positions):
        mag = abs(detail_L[sb_idx][y, x])
        n0 = round(mag / delta) * delta
        n1 = round((mag - half) / delta) * delta + half
        effective_bit = 1 if abs(mag - n1) < abs(mag - n0) else 0
        recovered[i] = effective_bit ^ int(bit_flip[i])
    return recovered


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--n_bits", type=int, default=127)
    p.add_argument("--delta", type=float, default=0.05,
                   help="QIM step. For Haar level-2 detail coeffs ~O(0.1), "
                        "Δ=0.05 trades quality (~40dB) vs robustness.")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_75", "jpeg_50", "blur_1.5", "crop_95", "resize_1.1"])
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    H = W = args.resolution
    master_key = args.master_key.encode("utf-8")

    # Probe a single image to enumerate candidate positions
    files = sorted([
        f for f in Path(args.image_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    test_files = files[args.start_idx:args.start_idx + args.n_test]
    if not test_files:
        raise RuntimeError("no test files")

    probe = np.asarray(Image.open(test_files[0]).convert("RGB").resize((W, H), Image.LANCZOS),
                       dtype=np.float64) / 255.0
    probe_coeffs = pywt.wavedec2(probe[:, :, args.channel], WAVELET, level=LEVEL)
    candidates = enumerate_dwt_positions(probe_coeffs)
    print(f"[setup] wavelet={WAVELET} level={LEVEL}  candidates={len(candidates)}  "
          f"n_bits={args.n_bits}  delta={args.delta}", flush=True)

    psnrs = []
    per_attack_acc = {atk: [] for atk in args.attacks}

    for idx, fp in enumerate(test_files):
        pil = Image.open(fp).convert("RGB").resize((W, H), Image.LANCZOS)
        image_id = f"dwt_{idx:05d}"

        positions, bit_flip = crypto_pick(master_key, image_id, args.n_bits, candidates)
        rng = np.random.default_rng(2000 + idx)
        payload = rng.integers(0, 2, size=args.n_bits, dtype=np.uint8)

        img_arr = np.asarray(pil)
        wm_arr = qim_embed_dwt(img_arr, payload, positions, bit_flip, args.delta, args.channel)
        psnrs.append(psnr(img_arr, wm_arr))

        wm_pil = Image.fromarray(wm_arr)
        for atk in args.attacks:
            att_pil = apply_attack(atk, wm_pil, args.resolution)
            if att_pil.size != (W, H):
                att_pil = att_pil.resize((W, H), Image.BILINEAR)
            att_arr = np.asarray(att_pil)
            dec = qim_decode_dwt(att_arr, positions, bit_flip, args.delta, args.channel)
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
    def add(name, actual, threshold):
        crit.append({"name": name, "actual": round(actual, 4),
                     "threshold": threshold, "passed": actual >= threshold})

    add("psnr_mean >= 38 dB", psnr_mean, 38.0)
    for atk, thr in [("clean", 0.99), ("jpeg_75", 0.90), ("blur_1.5", 0.85),
                     ("crop_95", 0.85), ("resize_1.1", 0.85)]:
        if atk in per_attack:
            add(f"{atk} bit_acc >= {thr:.2f}", per_attack[atk]["bit_acc_mean"], thr)
    passed = all(c["passed"] for c in crit)

    summary = {
        "config": vars(args),
        "n_test": len(test_files),
        "candidate_positions": len(candidates),
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
