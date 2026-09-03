"""Joint train Quant-QIM analytical encoder + learned readout decoder, VINE-style.

Quantization analog of scripts/train_dft_kred_vine_style.py. The encoder does
block-mean QIM (only delta learnable); the decoder reads block means at keyed
positions with the soft-QIM cos feature. Per-image random payload prevents
constant collapse. Curriculum: no augmentation and zero image-quality losses
during warmup, then full augmentation + MSE/LPIPS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.quant_qim_modules import QuantQIMEncoder, QuantQIMDecoder
from scripts.train_pixel_frag_vine_style import ImageOnlyDataset, augment_batch

try:
    import lpips as _lpips_mod  # type: ignore
    _HAVE_LPIPS = True
except Exception:
    _lpips_mod = None
    _HAVE_LPIPS = False


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = ImageOnlyDataset(args.image_dir, start_idx=0,
                          max_images=args.max_images, resolution=args.resolution)
    print(f"[setup] {len(ds)} training images at {args.resolution}px", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    ck_key = args.canonical_key.encode("utf-8")
    enc = QuantQIMEncoder(
        n_bits=args.n_bits, n_pos=args.n_pos, block_size=args.block_size,
        resolution=args.resolution, canonical_key=ck_key,
        canonical_id=args.canonical_id, init_delta=args.init_delta,
    ).to(device)
    dec = QuantQIMDecoder(
        n_bits=args.n_bits, n_pos=args.n_pos, block_size=args.block_size,
        resolution=args.resolution, canonical_key=ck_key,
        canonical_id=args.canonical_id, init_delta=args.init_delta,
    ).to(device)
    print(f"[setup] enc params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)
    print(f"[setup] dec params: {sum(p.numel() for p in dec.parameters()):,}", flush=True)

    if args.freeze_encoder:
        if args.fixed_delta is not None:
            with torch.no_grad():
                enc.base_delta.fill_(float(np.log(np.expm1(args.fixed_delta))))
            print(f"[setup] encoder delta pinned to {args.fixed_delta}", flush=True)
        enc.eval(); enc.requires_grad_(False)
        print("[setup] encoder FROZEN; training decoder only", flush=True)
        trainable = list(dec.parameters())
    else:
        trainable = list(enc.parameters()) + list(dec.parameters())

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    lpips_fn = None
    if _HAVE_LPIPS and args.lambda_lpips > 0:
        try:
            lpips_fn = _lpips_mod.LPIPS(net="vgg").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            print("[setup] LPIPS enabled", flush=True)
        except Exception as e:
            print(f"[setup] LPIPS init failed ({e})", flush=True)

    step = 0; t0 = time.time(); log_rows = []
    for epoch in range(args.epochs):
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(device, non_blocking=True)
            B = img.shape[0]
            payload = (torch.rand(B, args.n_bits, device=device) > 0.5).float()

            if args.freeze_encoder:
                with torch.no_grad():
                    x_w = enc(img, payload)
                x_w = x_w.detach()
            else:
                x_w = enc(img, payload)
            x_aug = x_w if step < args.aug_warmup_steps else augment_batch(x_w)
            p_hat = dec(x_aug)

            l_bce = bce(p_hat, payload)
            l_mse = F.mse_loss(x_w, img)
            l_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None and step >= args.aug_warmup_steps:
                l_lpips = lpips_fn(x_w * 2 - 1, img * 2 - 1).mean()
            lambda_mse = 0.0 if step < args.aug_warmup_steps else args.lambda_mse
            lambda_lpips = 0.0 if step < args.aug_warmup_steps else args.lambda_lpips
            loss = l_bce + lambda_mse * l_mse + lambda_lpips * l_lpips

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    bit_acc = ((p_hat > 0).float() == payload).float().mean().item()
                    psnr = 10 * np.log10(1.0 / max(l_mse.item(), 1e-12))
                    dval = float((F.softplus(enc.base_delta) + 1e-4).item())
                el = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} bce={l_bce.item():.4f} "
                      f"mse={l_mse.item():.6f} lpips={float(l_lpips):.4f} psnr={psnr:.1f}dB "
                      f"bit_acc={bit_acc:.3f} delta={dval:.4f} t={el:.0f}s", flush=True)
                log_rows.append({"step": step, "loss": float(loss.item()),
                                 "bce": float(l_bce.item()), "mse": float(l_mse.item()),
                                 "psnr": float(psnr), "bit_acc": float(bit_acc),
                                 "delta": dval, "elapsed_s": float(el)})
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(enc, dec, args, log_rows)
            step += 1
        if step >= args.max_steps:
            break
    _save(enc, dec, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}", flush=True)


def _save(enc, dec, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "encoder_state_dict": enc.state_dict(),
        "decoder_state_dict": dec.state_dict(),
        "config": vars(args),
        "n_bits": args.n_bits, "n_pos": args.n_pos,
        "block_size": args.block_size, "resolution": args.resolution,
        "canonical_key": args.canonical_key, "canonical_id": args.canonical_id,
        "carriers": enc.carriers.detach().cpu(),
        "bit_flip": enc.bit_flip.detach().cpu(),
        "learned_delta": float((F.softplus(enc.base_delta) + 1e-4).item()),
    }, args.output_ckpt)
    with open(args.output_ckpt.replace(".pt", "_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--n_pos", type=int, default=8)
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--canonical_key", type=str, default="vine_training_canonical_key")
    p.add_argument("--canonical_id", type=str, default="quant_qim_train_canonical")
    p.add_argument("--init_delta", type=float, default=0.06)
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--aug_warmup_steps", type=int, default=150)
    p.add_argument("--lambda_mse", type=float, default=10.0)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--fixed_delta", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
