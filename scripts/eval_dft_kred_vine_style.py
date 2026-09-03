"""Evaluate DFT-Kred VINE-style trained model on a held-out test set under
the canonical attack suite.

Includes a degeneracy check: feed clean (unwatermarked) images to the decoder
against the SAME random payload that would have been embedded, and confirm
bit_acc is near 0.5. If a clean image scores anywhere near the watermarked
one, the decoder has collapsed to an image-independent function and the
framework is broken.

The encoder's canonical carriers + bit_flip are restored from the checkpoint
exactly as they were at training time, so eval does not need to re-run the
HKDF stream or the prototype's setup function.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.dft_kred_modules import DFTKredEncoder, DFTKredDecoder, FFTAwareDecoder


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
        return pil.crop((left, top, left + cw, top + ch)).resize(
            (W, H), Image.BILINEAR
        )
    if name == "crop_70":
        W, H = pil.size
        cw, ch = int(W * 0.7), int(H * 0.7)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize(
            (W, H), Image.BILINEAR
        )
    if name == "resize_1.1":
        W, H = pil.size
        Wb, Hb = int(W * 1.1), int(H * 1.1)
        big = pil.resize((Wb, Hb), Image.BILINEAR)
        left, top = (Wb - W) // 2, (Hb - H) // 2
        return big.crop((left, top, left + W, top + H))
    raise ValueError(f"Unknown attack: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--image_dir",
        default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017",
    )
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument(
        "--attacks", nargs="+",
        default=[
            "clean", "jpeg_75", "jpeg_50", "blur_1.5",
            "noise_005", "crop_95", "resize_1.1",
        ],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--n_degeneracy_check", type=int, default=30)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    n_bits = int(ck["n_bits"])
    res = int(ck["resolution"])
    K = int(ck["K"])
    M = int(ck["M"])
    r_lo = float(ck["r_lo"])
    r_hi = float(ck["r_hi"])
    canonical_key = ck["canonical_key"]
    canonical_id = ck["canonical_id"]

    enc = DFTKredEncoder(
        n_bits=n_bits, K=K, M=M, resolution=res,
        r_lo=r_lo, r_hi=r_hi,
        canonical_key=(
            canonical_key.encode("utf-8")
            if isinstance(canonical_key, str) else canonical_key
        ),
        canonical_id=canonical_id,
    ).to(device).eval()
    decoder_type = (ck.get("config") or {}).get("decoder_type", "convnext")
    if decoder_type == "fft_aware":
        dec = FFTAwareDecoder(
            n_bits=n_bits, K=K, M=M, resolution=res, r_lo=r_lo, r_hi=r_hi,
            canonical_key=(
                canonical_key.encode("utf-8")
                if isinstance(canonical_key, str) else canonical_key
            ),
            canonical_id=canonical_id,
        ).to(device).eval()
    else:
        dec = DFTKredDecoder(n_bits=n_bits).to(device).eval()
    enc.load_state_dict(ck["encoder_state_dict"])
    dec.load_state_dict(ck["decoder_state_dict"])

    # Sanity: restore stored carriers / bit_flip so we are byte-identical to
    # training even if the prototype's HKDF or enumeration ever changes.
    if "carriers" in ck:
        enc.carriers.copy_(ck["carriers"].to(enc.carriers.device))
    if "bit_flip" in ck:
        enc.bit_flip.copy_(ck["bit_flip"].to(enc.bit_flip.device))

    learned_delta = float(
        torch.nn.functional.softplus(enc.base_delta).item() + 1.0
    )

    files = sorted([
        f for f in Path(args.image_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    test_files = files[args.start_idx:args.start_idx + args.n_test]
    if not test_files:
        raise RuntimeError(f"No test files in {args.image_dir}[{args.start_idx}:]")

    tfm = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
    ])

    print(
        f"[eval] ckpt={args.ckpt}  n_bits={n_bits}  res={res}  "
        f"K={K}  M={M}  delta={learned_delta:.2f}",
        flush=True,
    )
    print(
        f"[eval] {len(test_files)} test images, attacks={args.attacks}",
        flush=True,
    )

    per_attack = {atk: [] for atk in args.attacks}
    psnrs = []
    clean_vs_emb_payload = []
    wm_vs_emb_payload_clean = []

    with torch.no_grad():
        for i, fp in enumerate(test_files):
            img = tfm(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
            g = torch.Generator(device=device).manual_seed(1000 + i)
            payload = (torch.rand(1, n_bits, generator=g, device=device) > 0.5).float()

            x_w = enc(img, payload)
            mse = F.mse_loss(x_w, img).item()
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))

            # Degeneracy check: clean (un-watermarked) image decoded against
            # the payload we WOULD have embedded -> expect ~0.5; and the
            # watermarked image decoded with no attack -> expect high.
            if i < args.n_degeneracy_check:
                clean_pred = dec(img)
                clean_vs_emb_payload.append(
                    ((clean_pred[0] > 0).float() == payload[0]).float().mean().item()
                )
                wm_pred = dec(x_w)
                wm_vs_emb_payload_clean.append(
                    ((wm_pred[0] > 0).float() == payload[0]).float().mean().item()
                )

            wm_arr = x_w[0].cpu().numpy().transpose(1, 2, 0)
            wm_pil = Image.fromarray(
                np.clip(wm_arr * 255, 0, 255).astype(np.uint8)
            )

            for atk in args.attacks:
                att_pil = apply_attack(atk, wm_pil, res)
                if att_pil.size != (res, res):
                    att_pil = att_pil.resize((res, res), Image.BILINEAR)
                a = torch.from_numpy(
                    np.asarray(att_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
                ).unsqueeze(0).to(device)
                p_hat = dec(a)
                acc = ((p_hat[0] > 0).float() == payload[0]).float().mean().item()
                per_attack[atk].append(acc)

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(test_files)}]", flush=True)

    per_attack_mean = {atk: float(np.mean(per_attack[atk])) for atk in args.attacks}
    overall_mean = float(np.mean(list(per_attack_mean.values())))
    psnr_mean = float(np.mean(psnrs))

    summary = {
        "ckpt": args.ckpt,
        "n_test": len(test_files),
        "resolution": res,
        "n_bits": n_bits,
        "K": K,
        "M": M,
        "learned_delta": learned_delta,
        "attacks": args.attacks,
        "per_attack_bit_acc": per_attack_mean,
        "overall_mean": overall_mean,
        "psnr_mean": psnr_mean,
        "degeneracy_check": {
            "n": args.n_degeneracy_check,
            "clean_vs_emb_payload_mean":
                float(np.mean(clean_vs_emb_payload))
                if clean_vs_emb_payload else None,
            "wm_vs_emb_payload_clean_mean":
                float(np.mean(wm_vs_emb_payload_clean))
                if wm_vs_emb_payload_clean else None,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("")
    print(f"PSNR mean: {psnr_mean:.2f} dB")
    print(f"Overall mean bit_acc: {overall_mean:.4f}")
    for atk in args.attacks:
        print(f"  {atk:12s}: {per_attack_mean[atk]:.4f}")
    dc = summary["degeneracy_check"]
    if dc["clean_vs_emb_payload_mean"] is not None:
        print(f"Degeneracy check (n={dc['n']}):")
        print(
            f"  clean  vs would-be embedded payload: "
            f"{dc['clean_vs_emb_payload_mean']:.4f}  (~0.5 = healthy)"
        )
        print(
            f"  wm(no attack) vs embedded payload:   "
            f"{dc['wm_vs_emb_payload_clean_mean']:.4f}  (high = healthy)"
        )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
