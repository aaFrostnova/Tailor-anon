"""Train a scale/rotation-aware FFT decoder: apply a KNOWN geometric transform to
the watermarked image and decode by sampling carriers at the matching transform,
teaching value-robust readout under geometry. At inference, scale/rotation is
unknown -> search by decoder confidence.

Logs: acc@known (training-matched, upper bound), acc@clean, acc@resize-blind
(decode at scale=1, the failure mode), acc@resize-search (decode with a scale
search, the real inference number).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.dft_kred_modules import DFTKredEncoder
from src.logpolar_fragment import ScaleAwareFFTDecoder
from scripts.train_pixel_frag_vine_style import ImageOnlyDataset


def affine_known(x, scale, angle):
    """Apply per-image affine (scale tensor, angle-rad tensor) with reflect pad."""
    B, dev = x.shape[0], x.device
    ca, sa = torch.cos(angle), torch.sin(angle)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = ca / scale; th[:, 0, 1] = -sa / scale
    th[:, 1, 0] = sa / scale; th[:, 1, 1] = ca / scale
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


def photometric(x):
    # Strong compounded compression/noise so the readout + pilot-sync learn to
    # survive geometry TOGETHER with recompression (the realistic deployment case).
    if torch.rand(()) < 0.7:
        k = int(np.random.choice([3, 5]))
        x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)
    if torch.rand(()) < 0.6:  # jpeg-ish low-pass via down/up-sample (moderate)
        f = float(np.random.uniform(0.65, 0.92))
        h = max(8, int(x.shape[-1] * f))
        x = F.interpolate(F.interpolate(x, size=h, mode="bilinear", align_corners=False),
                          size=x.shape[-1], mode="bilinear", align_corners=False)
    if torch.rand(()) < 0.5:  # posterize (jpeg-ish quantization, moderate)
        q = float(np.random.choice([24, 32, 48]))
        x = torch.round(x * q) / q
    if torch.rand(()) < 0.6:
        x = x + torch.randn_like(x) * float(np.random.uniform(0, 0.04))
    return x.clamp(0, 1)


def search_decode(dec, x, scales):
    best_logits, best_conf = None, -1
    for s in scales:
        lg = dec(x, scale=float(s), angle=0.0)
        conf = lg.abs().mean().item()
        if conf > best_conf:
            best_conf, best_logits = conf, lg
    return best_logits


def train(args):
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = args.resolution
    ds = ImageOnlyDataset(args.image_dir, start_idx=0, max_images=args.max_images, resolution=R)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=6, pin_memory=True, drop_last=True)
    enc = DFTKredEncoder(n_bits=args.n_bits, K=args.K, M=args.M, resolution=R,
                         r_lo=args.r_lo, r_hi=args.r_hi, init_delta=args.init_delta,
                         canonical_id=args.canonical_id).to(device).eval()
    enc.requires_grad_(False)
    dec = ScaleAwareFFTDecoder(enc.carriers, enc.bit_flip, resolution=R,
                               init_delta=args.init_delta, hidden=args.hidden).to(device)
    print(f"[setup] carriers {tuple(enc.carriers.shape)}; dec params "
          f"{sum(p.numel() for p in dec.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.max_steps)
    bce = nn.BCEWithLogitsLoss()
    sgrid = np.round(np.arange(0.82, 1.221, 0.02), 3)

    step, t0, log = 0, time.time(), []
    for _ in range(10_000):
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(device, non_blocking=True); B = img.shape[0]
            payload = (torch.rand(B, args.n_bits, device=device) > 0.5).float()
            with torch.no_grad():
                x_w = enc(img, payload)
            if step < args.warmup:
                s = torch.ones(B, device=device); ang = torch.zeros(B, device=device)
                x_in = x_w
            else:
                s = torch.empty(B, device=device).uniform_(args.smin, args.smax)
                ang = torch.empty(B, device=device).uniform_(-args.rot, args.rot) * math.pi / 180
                # mix in identity cases so the clean readout is not forgotten
                ident = torch.rand(B, device=device) < args.p_identity
                s = torch.where(ident, torch.ones_like(s), s)
                ang = torch.where(ident, torch.zeros_like(ang), ang)
                x_in = photometric(affine_known(x_w, s, ang))
            logits = dec(x_in, scale=s, angle=ang)
            loss = bce(logits, payload)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step(); sched.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    acc_known = ((logits > 0).float() == payload).float().mean().item()
                    acc_clean = ((dec(x_w, 1.0, 0.0) > 0).float() == payload).float().mean().item()
                    rz = affine_known(x_w, torch.full((B,), 1.15, device=device),
                                      torch.zeros(B, device=device))
                    acc_blind = ((dec(rz, 1.0, 0.0) > 0).float() == payload).float().mean().item()
                    acc_search = ((search_decode(dec, rz, sgrid) > 0).float() == payload).float().mean().item()
                print(f"[step {step:5d}] loss={loss.item():.4f} known={acc_known:.3f} "
                      f"clean={acc_clean:.3f} resize115_blind={acc_blind:.3f} "
                      f"resize115_search={acc_search:.3f} t={time.time()-t0:.0f}s", flush=True)
                log.append({"step": step, "loss": float(loss.item()), "acc_known": acc_known,
                            "acc_clean": acc_clean, "acc_resize_blind": acc_blind,
                            "acc_resize_search": acc_search})
            step += 1
        if step >= args.max_steps:
            break

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({"decoder_state_dict": dec.state_dict(), "config": vars(args),
                "carriers": enc.carriers.cpu(), "bit_flip": enc.bit_flip.cpu(),
                "n_bits": args.n_bits, "K": args.K, "M": args.M, "resolution": args.resolution,
                "r_lo": args.r_lo, "r_hi": args.r_hi, "init_delta": args.init_delta,
                "canonical_id": args.canonical_id}, args.output_ckpt)
    json.dump(log, open(args.output_ckpt.replace(".pt", "_log.json"), "w"), indent=2)
    print(f"[done] -> {args.output_ckpt}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--r_lo", type=float, default=8.0)
    p.add_argument("--r_hi", type=float, default=110.0)
    p.add_argument("--init_delta", type=float, default=50.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--canonical_id", default="scaleaware_canonical")
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max_steps", type=int, default=2500)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--smin", type=float, default=0.8)
    p.add_argument("--smax", type=float, default=1.25)
    p.add_argument("--rot", type=float, default=10.0)
    p.add_argument("--p_identity", type=float, default=0.35)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
