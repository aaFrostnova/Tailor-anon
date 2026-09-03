"""Pure-channel capacity probe for the U-Net learned encoder.

No augmentation, no image-quality loss — ONLY BCE. Question: can the joint
encoder+decoder establish a readable payload channel (random payloads, real
image set) and drive bit_acc high? This isolates capacity/scale from the
zero-perturbation collapse (which is triggered by the MSE term, disabled here).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.learned_unet_modules import ResidualUNetEncoder, ConvDecoder
from scripts.train_pixel_frag_vine_style import ImageOnlyDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--n_bits", type=int, default=32)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--epsilon", type=float, default=8 / 255)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    ds = ImageOnlyDataset(args.image_dir, start_idx=0,
                          max_images=args.max_images, resolution=args.resolution)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, pin_memory=True, drop_last=True)
    enc = ResidualUNetEncoder(n_bits=args.n_bits, epsilon=args.epsilon, base=args.base).to(dev).train()
    dec = ConvDecoder(n_bits=args.n_bits, base=args.base).to(dev).train()
    print(f"[probe] n_bits={args.n_bits} base={args.base} lr={args.lr} "
          f"enc={sum(p.numel() for p in enc.parameters()):,} "
          f"dec={sum(p.numel() for p in dec.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    traj = []; t0 = time.time(); step = 0; max_ba = 0.0
    while step < args.max_steps:
        for img in loader:
            if step >= args.max_steps:
                break
            img = img.to(dev, non_blocking=True)
            B = img.shape[0]
            payload = (torch.rand(B, args.n_bits, device=dev) > 0.5).float()
            logits = dec(enc(img, payload))   # NO aug, NO image loss
            loss = bce(logits, payload)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % args.log_every == 0:
                with torch.no_grad():
                    ba = ((logits > 0).float() == payload).float().mean().item()
                max_ba = max(max_ba, ba)
                el = time.time() - t0
                print(f"[step {step:5d}] loss={loss.item():.4f} bit_acc={ba:.3f} t={el:.0f}s", flush=True)
                traj.append({"step": step, "loss": float(loss.item()), "bit_acc": float(ba)})
            step += 1

    summary = {"config": vars(args), "trajectory": traj,
               "final_bit_acc": traj[-1]["bit_acc"] if traj else 0.0,
               "max_bit_acc": max_ba}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probe done] final_bit_acc={summary['final_bit_acc']:.3f} max={max_ba:.3f} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
