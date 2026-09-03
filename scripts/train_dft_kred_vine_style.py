"""Joint train DFT-Kred analytical encoder + ConvNeXt decoder, VINE-style.

Per step:
  1. Sample image batch.
  2. Sample a RANDOM n_bits payload per image (independent across batch+steps).
     -> decoder cannot collapse to a constant; it must extract payload from image.
  3. enc(image, payload) -> watermarked image. The encoder is the analytical
     DFT-magnitude carrier with fixed canonical bins; only the magnitude step
     delta is learned, so the encoder cannot collapse to zero perturbation.
  4. Augment (JPEG / blur / noise / crop+resize / identity) -- skipped during
     curriculum warmup.
  5. decoder(augmented) -> predicted bits (logits).
  6. Loss = BCE(predicted, payload) + lambda_mse * MSE(watermarked, image)
           + lambda_lpips * LPIPS(watermarked, image)  [if LPIPS available]
     Both image-quality terms are zeroed during the warmup.

Differences vs scripts/train_pixel_frag_vine_style.py:
  * Encoder is DFTKredEncoder (analytical FFT carrier, ~1 learned param).
  * Decoder is DFTKredDecoder (ConvNeXt-Base).
  * Adds optional LPIPS perceptual loss.
  * Saves carriers + bit_flip in the checkpoint so eval is fully self-contained.
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

from src.dft_kred_modules import DFTKredEncoder, DFTKredDecoder, FFTAwareDecoder
from scripts.train_pixel_frag_vine_style import (
    ImageOnlyDataset,
    augment_batch,
)

# Conditional LPIPS import — skip cleanly if not installed.
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

    ds = ImageOnlyDataset(
        args.image_dir, start_idx=0,
        max_images=args.max_images, resolution=args.resolution,
    )
    print(f"[setup] {len(ds)} training images at {args.resolution}px", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    enc = DFTKredEncoder(
        n_bits=args.n_bits, K=args.K, M=args.M,
        resolution=args.resolution,
        r_lo=args.r_lo, r_hi=args.r_hi,
        canonical_key=args.canonical_key.encode("utf-8"),
        canonical_id=args.canonical_id,
        init_delta=args.init_delta,
    ).to(device)
    if args.decoder_type == "fft_aware":
        dec = FFTAwareDecoder(
            n_bits=args.n_bits, K=args.K, M=args.M,
            resolution=args.resolution,
            r_lo=args.r_lo, r_hi=args.r_hi,
            canonical_key=args.canonical_key.encode("utf-8"),
            canonical_id=args.canonical_id,
            init_delta=args.init_delta,
        ).to(device)
        print("[setup] decoder = FFTAwareDecoder (frequency-domain readout)", flush=True)
    else:
        dec = DFTKredDecoder(
            n_bits=args.n_bits, pretrained=args.pretrained_decoder,
        ).to(device)
        print("[setup] decoder = DFTKredDecoder (ConvNeXt-Base)", flush=True)
    print(
        f"[setup] enc params: {sum(p.numel() for p in enc.parameters()):,}",
        flush=True,
    )
    print(
        f"[setup] dec params: {sum(p.numel() for p in dec.parameters()):,}",
        flush=True,
    )

    if args.freeze_encoder:
        if args.fixed_delta is not None:
            # Invert base_delta = softplus_inv(fixed_delta - 1.0) so that
            # (softplus(base_delta) + 1.0) == fixed_delta.
            target = float(args.fixed_delta) - 1.0
            if target <= 0.0:
                raise ValueError(
                    f"--fixed_delta must be > 1.0 (got {args.fixed_delta})"
                )
            # softplus_inv(y) = log(exp(y) - 1); use expm1 for stability.
            inv = float(np.log(np.expm1(target)))
            with torch.no_grad():
                enc.base_delta.fill_(inv)
            print(
                f"[setup] encoder.base_delta pinned to fixed_delta="
                f"{args.fixed_delta:.4f}",
                flush=True,
            )
        enc.eval()
        enc.requires_grad_(False)
        print("[setup] encoder FROZEN; training decoder only", flush=True)
        trainable_params = list(dec.parameters())
    else:
        trainable_params = list(enc.parameters()) + list(dec.parameters())

    opt = torch.optim.AdamW(
        trainable_params,
        lr=args.lr, weight_decay=1e-4,
    )
    bce = nn.BCEWithLogitsLoss()

    lpips_fn = None
    if _HAVE_LPIPS and args.lambda_lpips > 0:
        try:
            lpips_fn = _lpips_mod.LPIPS(net="vgg").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            print("[setup] LPIPS loss enabled (vgg)", flush=True)
        except Exception as e:
            print(f"[setup] LPIPS init failed ({e}); disabling.", flush=True)
            lpips_fn = None
    else:
        if not _HAVE_LPIPS:
            print("[setup] LPIPS not installed; skipping perceptual loss.", flush=True)

    step = 0
    t0 = time.time()
    log_rows = []

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
            # Curriculum: skip augmentation for the first aug_warmup_steps so
            # the encoder/decoder establish the basic embedding before attacks
            # disrupt the signal. After warmup, full augmentation.
            x_aug = x_w if step < args.aug_warmup_steps else augment_batch(x_w)
            p_hat = dec(x_aug)

            l_bce = bce(p_hat, payload)
            l_mse = F.mse_loss(x_w, img)
            if lpips_fn is not None:
                # LPIPS expects inputs in [-1, 1].
                l_lpips = lpips_fn(x_w * 2.0 - 1.0, img * 2.0 - 1.0).mean()
            else:
                l_lpips = torch.zeros((), device=device)

            # Curriculum: zero image-quality losses during warmup so the
            # encoder is forced to inject a usable signal first; bring them
            # in afterward to control PSNR / perceptual quality.
            in_warmup = step < args.aug_warmup_steps
            lambda_mse = 0.0 if in_warmup else args.lambda_mse
            lambda_lpips = 0.0 if in_warmup else args.lambda_lpips

            loss = l_bce + lambda_mse * l_mse + lambda_lpips * l_lpips

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    # p_hat is logits; bit = sigmoid(p_hat) > 0.5  iff  p_hat > 0.
                    bit_acc = ((p_hat > 0).float() == payload).float().mean().item()
                    psnr = 10 * np.log10(1.0 / max(l_mse.item(), 1e-12))
                    learned_delta = float(
                        (F.softplus(enc.base_delta) + 1.0).item()
                    )
                elapsed = time.time() - t0
                print(
                    f"[step {step:6d}] loss={loss.item():.4f} "
                    f"bce={l_bce.item():.4f} mse={l_mse.item():.5f} "
                    f"lpips={float(l_lpips.item()):.4f} "
                    f"psnr={psnr:.1f}dB bit_acc={bit_acc:.3f} "
                    f"delta={learned_delta:.2f} t={elapsed:.0f}s",
                    flush=True,
                )
                log_rows.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "bce": float(l_bce.item()),
                    "mse": float(l_mse.item()),
                    "lpips": float(l_lpips.item()),
                    "psnr": float(psnr),
                    "bit_acc": float(bit_acc),
                    "learned_delta": learned_delta,
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(enc, dec, args, log_rows)

            step += 1
        if step >= args.max_steps:
            break

    _save(enc, dec, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}", flush=True)


def _save(enc: DFTKredEncoder, dec: DFTKredDecoder, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    # Persist carriers + bit_flip explicitly so eval is fully self-contained
    # and does not need to recompute (which would require the same prototype
    # code paths and the same HKDF stream).
    torch.save({
        "encoder_state_dict": enc.state_dict(),
        "decoder_state_dict": dec.state_dict(),
        "config": vars(args),
        "n_bits": args.n_bits,
        "K": args.K,
        "M": args.M,
        "resolution": args.resolution,
        "r_lo": args.r_lo,
        "r_hi": args.r_hi,
        "canonical_key": args.canonical_key,
        "canonical_id": args.canonical_id,
        "carriers": enc.carriers.detach().cpu(),       # (n_bits, K*M, 2)
        "bit_flip": enc.bit_flip.detach().cpu(),       # (n_bits,)
        "learned_delta": float(
            (F.softplus(enc.base_delta) + 1.0).item()
        ),
    }, args.output_ckpt)
    with open(args.output_ckpt.replace(".pt", "_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image_dir",
        default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017",
    )
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--r_lo", type=float, default=20.0)
    p.add_argument("--r_hi", type=float, default=60.0)
    p.add_argument("--canonical_key", type=str,
                   default="vine_training_canonical_key")
    p.add_argument("--canonical_id", type=str,
                   default="dft_kred_train_canonical")
    p.add_argument("--init_delta", type=float, default=50.0)
    p.add_argument("--pretrained_decoder", action="store_true")
    p.add_argument("--decoder_type", choices=["fft_aware", "convnext"],
                   default="fft_aware",
                   help="fft_aware: frequency-domain readout (recommended); "
                        "convnext: spatial CNN (fails per diagnostic).")
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--aug_warmup_steps", type=int, default=200,
                   help="Skip augmentation AND image-quality losses for the "
                        "first N steps so the encoder establishes a signal.")
    p.add_argument("--lambda_mse", type=float, default=10.0)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze the analytical DFT encoder (eval mode, no "
                        "grads); train only the decoder.")
    p.add_argument("--fixed_delta", type=float, default=None,
                   help="If set together with --freeze_encoder, override "
                        "encoder.base_delta so the effective delta equals "
                        "this value (e.g. 50.0 from the analytical sweep).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
