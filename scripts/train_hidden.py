"""Train HiDDeN-style learned spatial watermark for COMPOUND geometry+compression
robustness. Curriculum: open the clean channel first (the test the U-Net failed),
then ramp the geometric+compression noise layer and the image-fidelity loss.

Logs clean bit-acc AND compound-noised bit-acc + PSNR each interval.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.hidden_modules import HiDDeNEncoder, HiDDeNDecoder, NoiseLayer
from scripts.train_pixel_frag_vine_style import ImageOnlyDataset


def train(args):
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = ImageOnlyDataset(args.image_dir, start_idx=0, max_images=args.max_images, resolution=args.resolution)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=6, pin_memory=True, drop_last=True)
    enc = HiDDeNEncoder(n_bits=args.n_bits, strength=args.strength).to(device)
    dec = HiDDeNDecoder(n_bits=args.n_bits).to(device)
    noise = NoiseLayer(resolution=args.resolution).to(device)
    print(f"[setup] {len(ds)} imgs @{args.resolution}; enc {sum(p.numel() for p in enc.parameters()):,} "
          f"dec {sum(p.numel() for p in dec.parameters()):,}", flush=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    step, t0, log = 0, time.time(), []
    for _ in range(10_000):
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(device, non_blocking=True); B = img.shape[0]
            msg = (torch.rand(B, args.n_bits, device=device) > 0.5).float()
            wm = enc(img, msg)
            # curriculum: no noise during warmup (open channel), then compound noise
            if step < args.warmup:
                noised = wm
                lam = 0.0
            else:
                noised = noise(wm, geometric=True, compress=True)
                lam = args.lambda_img
            logits = dec(noised)
            l_bce = bce(logits, msg)
            l_img = F.mse_loss(wm, img)
            loss = l_bce + lam * l_img
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0); opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    acc_clean = ((dec(wm) > 0).float() == msg).float().mean().item()
                    nz = noise(wm, geometric=True, compress=True)
                    acc_noise = ((dec(nz) > 0).float() == msg).float().mean().item()
                    psnr = 10 * np.log10(1.0 / max(l_img.item(), 1e-12))
                print(f"[step {step:5d}] loss={loss.item():.4f} bce={l_bce.item():.4f} "
                      f"clean={acc_clean:.3f} compound={acc_noise:.3f} psnr={psnr:.1f}dB "
                      f"t={time.time()-t0:.0f}s", flush=True)
                log.append({"step": step, "clean": acc_clean, "compound": acc_noise,
                            "psnr": float(psnr), "bce": float(l_bce.item())})
            step += 1
        if step >= args.max_steps:
            break

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({"encoder_state_dict": enc.state_dict(), "decoder_state_dict": dec.state_dict(),
                "config": vars(args), "n_bits": args.n_bits, "resolution": args.resolution,
                "strength": args.strength}, args.output_ckpt)
    json.dump(log, open(args.output_ckpt.replace(".pt", "_log.json"), "w"), indent=2)
    print(f"[done] -> {args.output_ckpt}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=64)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=1500)
    p.add_argument("--lambda_img", type=float, default=0.7)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
