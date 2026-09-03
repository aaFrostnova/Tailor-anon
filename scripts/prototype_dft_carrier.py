"""DFT-magnitude carrier watermark prototype (analytical, no training).

Embeds N bits into 2D-DFT magnitudes of the green channel at crypto-keyed
mid-frequency annular bin positions using QIM (Quantization Index Modulation).
Decoder is reference-free: it reads suspect-image magnitudes at the same
carrier positions and rounds them to the nearest dither lattice point.

Tests viability before committing to full VINE-recipe training:
required for go = PSNR ≥ 38 dB AND clean ≥ 0.99 AND jpeg_75 ≥ 0.90 AND
blur_1.5 ≥ 0.85 AND crop_95 ≥ 0.85 AND resize_1.1 ≥ 0.85.
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


# ============================================================ crypto keying

def enumerate_annulus_bins(H: int, W: int, r_lo: float, r_hi: float):
    """Enumerate (fy, fx) bins with r in [r_lo, r_hi] from the non-redundant
    half of the spectrum (positive fy) so the conjugate is unambiguous.

    Excludes DC (0,0), the Nyquist row (fy = H/2) and Nyquist column
    (fx = W/2 representations) since those bins are self-conjugate (real
    valued) and don't support independent magnitude+phase modulation.
    """
    bins = []
    for fy in range(1, H // 2):                       # 1 .. H/2 - 1 (strictly positive, not Nyquist)
        for fx_signed in range(-(W // 2 - 1), W // 2):   # -(W/2-1) .. W/2 - 1, skip Nyquist column
            fx = fx_signed % W
            # radius from (0,0) in signed coords
            r = (fy ** 2 + fx_signed ** 2) ** 0.5
            if r_lo <= r <= r_hi:
                bins.append((fy, fx))
    bins.sort()
    return bins


def crypto_carrier_setup(master_key: bytes, image_id: str, n_bits: int,
                         candidates):
    """Pick n_bits unique carrier positions and per-bit sign-flip mask."""
    n_cand = len(candidates)
    # Position stream — long enough to handle collisions
    pos_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/dft_position").encode("utf-8"),
        n_bits * 4,
    )
    chosen = []
    used = set()
    k = 0
    while len(chosen) < n_bits:
        j = int(pos_stream[k] % n_cand); k += 1
        if k >= len(pos_stream):
            raise RuntimeError("ran out of HKDF stream picking carriers")
        if j in used:
            continue
        used.add(j)
        chosen.append(candidates[j])

    # Sign-flip mask: 1 bit per payload bit (does the bit get XOR-flipped before embedding)
    sign_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/dft_sign").encode("utf-8"), n_bits,
    )
    bit_flip = (sign_stream % 2).astype(np.uint8)
    return chosen, bit_flip


# ============================================================ QIM embed / decode

def qim_embed(img_uint8: np.ndarray, payload: np.ndarray,
              carriers: list, bit_flip: np.ndarray, delta: float,
              channel: int = 1) -> np.ndarray:
    """Embed payload into DFT magnitudes of the given channel via QIM.

    img_uint8: (H, W, 3) uint8.  payload: (n_bits,) in {0,1}.
    Returns watermarked uint8 image.
    """
    img = img_uint8.astype(np.float64) / 255.0
    H, W, _ = img.shape
    ch = img[:, :, channel].copy()
    F = np.fft.fft2(ch)

    for i, (fy, fx) in enumerate(carriers):
        effective_bit = int(payload[i]) ^ int(bit_flip[i])
        mag = abs(F[fy, fx])
        phase = np.angle(F[fy, fx])
        # QIM: snap mag to even-multiple of Δ then offset by Δ/2 if bit=1
        target_mag = np.round(mag / delta) * delta + (delta / 2.0) * effective_bit
        new_val = target_mag * np.exp(1j * phase)
        F[fy, fx] = new_val
        F[(-fy) % H, (-fx) % W] = np.conj(new_val)         # Hermitian symmetry

    ch_w = np.real(np.fft.ifft2(F))
    out = img.copy()
    out[:, :, channel] = np.clip(ch_w, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def qim_decode(img_uint8: np.ndarray, carriers: list,
               bit_flip: np.ndarray, delta: float,
               channel: int = 1) -> np.ndarray:
    """Reference-free QIM decode: pick the closer of two dither lattices.

    Bit-0 lattice points: {0, Δ, 2Δ, ...}.  Bit-1 lattice points: {Δ/2, 3Δ/2, ...}.
    For each carrier magnitude, compute distance to nearest point of each lattice
    and decide; unambiguous at edge cases.
    """
    img = img_uint8.astype(np.float64) / 255.0
    ch = img[:, :, channel]
    F = np.fft.fft2(ch)
    recovered = np.zeros(len(carriers), dtype=np.uint8)
    half = delta / 2.0
    for i, (fy, fx) in enumerate(carriers):
        mag = abs(F[fy, fx])
        nearest_0 = round(mag / delta) * delta
        nearest_1 = round((mag - half) / delta) * delta + half
        effective_bit = 1 if abs(mag - nearest_1) < abs(mag - nearest_0) else 0
        recovered[i] = effective_bit ^ int(bit_flip[i])
    return recovered


# ============================================================ attacks

def apply_attack(name: str, pil: Image.Image, resolution: int) -> Image.Image:
    if name == "clean":
        return pil
    if name == "jpeg_75":
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=75); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "jpeg_50":
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=50); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "blur_1.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name == "blur_2.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=2.5))
    if name == "noise_005":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr += np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.05
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "crop_95":
        W, H = pil.size
        cw, ch = int(W * 0.95), int(H * 0.95)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    if name == "crop_70":
        W, H = pil.size
        cw, ch = int(W * 0.7), int(H * 0.7)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    if name == "resize_1.1":
        # Resize to 1.1x then center-crop back to original
        W, H = pil.size
        Wb, Hb = int(W * 1.1), int(H * 1.1)
        big = pil.resize((Wb, Hb), Image.BILINEAR)
        left, top = (Wb - W) // 2, (Hb - H) // 2
        return big.crop((left, top, left + W, top + H))
    raise ValueError(f"Unknown attack: {name}")


def psnr(a_uint8: np.ndarray, b_uint8: np.ndarray) -> float:
    a = a_uint8.astype(np.float64) / 255.0
    b = b_uint8.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


# ============================================================ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--n_bits", type=int, default=127)
    p.add_argument("--delta", type=float, default=100.0,
                   help="QIM quantization step in raw-FFT-magnitude units. "
                        "For a 256x256 image in [0,1], mid-freq mags are O(50-500); "
                        "Δ=100 gives ~41 dB PSNR with high robustness to uint8 quantization.")
    p.add_argument("--r_lo", type=float, default=20.0)
    p.add_argument("--r_hi", type=float, default=60.0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_75", "blur_1.5", "crop_95", "resize_1.1"])
    p.add_argument("--channel", type=int, default=1,
                   help="0=R, 1=G, 2=B; default green (strongest watermark per reverse-SynthID).")
    p.add_argument("--output", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    master_key = args.master_key.encode("utf-8")
    H = W = args.resolution

    candidates = enumerate_annulus_bins(H, W, args.r_lo, args.r_hi)
    print(f"[setup] resolution={H}x{W}  candidate bins={len(candidates)}  "
          f"n_bits={args.n_bits}  delta={args.delta}  channel={args.channel}", flush=True)
    if len(candidates) < args.n_bits:
        raise RuntimeError(f"only {len(candidates)} candidate bins, need {args.n_bits}")

    files = sorted([
        f for f in Path(args.image_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    test_files = files[args.start_idx:args.start_idx + args.n_test]
    if not test_files:
        raise RuntimeError("no test files found")

    psnrs = []
    per_attack_acc = {atk: [] for atk in args.attacks}

    for idx, fp in enumerate(test_files):
        pil = Image.open(fp).convert("RGB").resize((W, H), Image.LANCZOS)
        image_id = f"proto_{idx:05d}"

        carriers, bit_flip = crypto_carrier_setup(master_key, image_id, args.n_bits, candidates)
        rng = np.random.default_rng(1000 + idx)
        payload = rng.integers(0, 2, size=args.n_bits, dtype=np.uint8)

        img_arr = np.asarray(pil)
        wm_arr = qim_embed(img_arr, payload, carriers, bit_flip, args.delta, args.channel)
        psnrs.append(psnr(img_arr, wm_arr))

        wm_pil = Image.fromarray(wm_arr)

        # Sanity self-decode without attack to catch implementation bugs
        if idx == 0:
            self_dec = qim_decode(wm_arr, carriers, bit_flip, args.delta, args.channel)
            self_acc = float(np.mean(self_dec == payload))
            if args.verbose:
                print(f"  [sanity] self-decode (no attack): bit_acc={self_acc:.4f}", flush=True)

        for atk in args.attacks:
            att_pil = apply_attack(atk, wm_pil, args.resolution)
            if att_pil.size != (W, H):
                att_pil = att_pil.resize((W, H), Image.BILINEAR)
            att_arr = np.asarray(att_pil)
            dec = qim_decode(att_arr, carriers, bit_flip, args.delta, args.channel)
            acc = float(np.mean(dec == payload))
            per_attack_acc[atk].append(acc)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_files)}]  psnr={np.mean(psnrs):.2f}dB", flush=True)

    # Aggregate
    psnr_mean = float(np.mean(psnrs)); psnr_std = float(np.std(psnrs))
    per_attack = {
        atk: {"bit_acc_mean": float(np.mean(per_attack_acc[atk])),
              "bit_acc_std": float(np.std(per_attack_acc[atk]))}
        for atk in args.attacks
    }

    # Verdict
    crit = []
    def add(name, actual, threshold, op=">="):
        passed = (actual >= threshold) if op == ">=" else (actual > threshold)
        crit.append({"name": name, "actual": round(actual, 4),
                     "threshold": threshold, "passed": passed})

    add(f"psnr_mean >= 38 dB", psnr_mean, 38.0)
    for need in [("clean", 0.99), ("jpeg_75", 0.90),
                 ("blur_1.5", 0.85), ("crop_95", 0.85), ("resize_1.1", 0.85)]:
        atk, thr = need
        if atk in per_attack:
            add(f"{atk} bit_acc >= {thr:.2f}", per_attack[atk]["bit_acc_mean"], thr)

    passed = all(c["passed"] for c in crit)
    verdict = {"passed": passed, "criteria": crit}

    summary = {
        "config": vars(args),
        "n_test": len(test_files),
        "candidate_bins": len(candidates),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "per_attack": per_attack,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"PSNR mean: {psnr_mean:.2f} dB ± {psnr_std:.2f}")
    print("Per-attack bit accuracy:")
    for atk in args.attacks:
        a = per_attack[atk]
        print(f"  {atk:12s}: {a['bit_acc_mean']:.4f} ± {a['bit_acc_std']:.4f}")
    print()
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    for c in crit:
        flag = "✓" if c["passed"] else "✗"
        print(f"  {flag} {c['name']}  actual={c['actual']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
