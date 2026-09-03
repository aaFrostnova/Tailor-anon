"""Evaluate Quant-QIM VINE-style trained model on a held-out test set.

Mirrors scripts/eval_dft_kred_vine_style.py: attack suite + degeneracy check
(clean image decoded against would-be payload should be ~0.5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.quant_qim_modules import QuantQIMEncoder, QuantQIMDecoder
from scripts.eval_dft_kred_vine_style import apply_attack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "jpeg_75", "jpeg_50", "blur_1.5",
                            "noise_005", "crop_95", "resize_1.1"])
    p.add_argument("--output", required=True)
    p.add_argument("--n_degeneracy_check", type=int, default=30)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    n_bits = int(ck["n_bits"]); n_pos = int(ck["n_pos"])
    block_size = int(ck["block_size"]); res = int(ck["resolution"])
    ck_key = ck["canonical_key"]
    ck_key_b = ck_key.encode("utf-8") if isinstance(ck_key, str) else ck_key
    ck_id = ck["canonical_id"]

    enc = QuantQIMEncoder(n_bits=n_bits, n_pos=n_pos, block_size=block_size,
                          resolution=res, canonical_key=ck_key_b,
                          canonical_id=ck_id).to(device).eval()
    dec = QuantQIMDecoder(n_bits=n_bits, n_pos=n_pos, block_size=block_size,
                          resolution=res, canonical_key=ck_key_b,
                          canonical_id=ck_id).to(device).eval()
    enc.load_state_dict(ck["encoder_state_dict"])
    dec.load_state_dict(ck["decoder_state_dict"])
    if "carriers" in ck:
        enc.carriers.copy_(ck["carriers"].to(enc.carriers.device))
    if "bit_flip" in ck:
        enc.bit_flip.copy_(ck["bit_flip"].to(enc.bit_flip.device))

    learned_delta = float((F.softplus(enc.base_delta) + 1e-4).item())

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    test_files = files[args.start_idx:args.start_idx + args.n_test]
    if not test_files:
        raise RuntimeError(f"No test files in {args.image_dir}[{args.start_idx}:]")

    tfm = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(res), transforms.ToTensor(),
    ])
    print(f"[eval] ckpt={args.ckpt}  n_bits={n_bits}  res={res}  n_pos={n_pos}  "
          f"block={block_size}  delta={learned_delta:.4f}", flush=True)

    per_attack = {atk: [] for atk in args.attacks}
    psnrs, clean_vs, wm_vs = [], [], []
    with torch.no_grad():
        for i, fp in enumerate(test_files):
            img = tfm(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
            g = torch.Generator(device=device).manual_seed(1000 + i)
            payload = (torch.rand(1, n_bits, generator=g, device=device) > 0.5).float()
            x_w = enc(img, payload)
            psnrs.append(10 * np.log10(1.0 / max(F.mse_loss(x_w, img).item(), 1e-12)))
            if i < args.n_degeneracy_check:
                clean_vs.append(((dec(img)[0] > 0).float() == payload[0]).float().mean().item())
                wm_vs.append(((dec(x_w)[0] > 0).float() == payload[0]).float().mean().item())
            wm_pil = Image.fromarray(
                np.clip(x_w[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            for atk in args.attacks:
                att = apply_attack(atk, wm_pil, res)
                if att.size != (res, res):
                    att = att.resize((res, res), Image.BILINEAR)
                a = torch.from_numpy(
                    np.asarray(att, dtype=np.float32).transpose(2, 0, 1) / 255.0
                ).unsqueeze(0).to(device)
                acc = ((dec(a)[0] > 0).float() == payload[0]).float().mean().item()
                per_attack[atk].append(acc)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(test_files)}]", flush=True)

    per_attack_mean = {atk: float(np.mean(per_attack[atk])) for atk in args.attacks}
    summary = {
        "ckpt": args.ckpt, "n_test": len(test_files), "resolution": res,
        "n_bits": n_bits, "n_pos": n_pos, "block_size": block_size,
        "learned_delta": learned_delta, "attacks": args.attacks,
        "per_attack_bit_acc": per_attack_mean,
        "overall_mean": float(np.mean(list(per_attack_mean.values()))),
        "psnr_mean": float(np.mean(psnrs)),
        "degeneracy_check": {
            "n": args.n_degeneracy_check,
            "clean_vs_emb_payload_mean": float(np.mean(clean_vs)) if clean_vs else None,
            "wm_vs_emb_payload_clean_mean": float(np.mean(wm_vs)) if wm_vs else None,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPSNR mean: {summary['psnr_mean']:.2f} dB")
    print(f"Overall mean bit_acc: {summary['overall_mean']:.4f}")
    for atk in args.attacks:
        print(f"  {atk:12s}: {per_attack_mean[atk]:.4f}")
    dc = summary["degeneracy_check"]
    print(f"degeneracy: clean_vs_emb={dc['clean_vs_emb_payload_mean']:.4f} "
          f"wm_vs_emb={dc['wm_vs_emb_payload_clean_mean']:.4f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
