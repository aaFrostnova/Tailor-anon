"""Transferable universal learned overwriter (Phase B.3).

Learned watermark subspaces are private (the A.1 matrix shows VINE/RivaGAN/
TrustMark/MaskWM don't overwrite each other). So the universal learned attack
must be TRAINED: one bounded perturbation delta that collapses the per-bit
outputs of an ENSEMBLE of differentiable decoders {VINE-R, MaskWM, DFT-Kred,
Quant-QIM} toward chance. We then test transfer to HELD-OUT methods (RivaGAN,
TrustMark -- non-differentiable, never in the training graph).

Modes:
  uap  : one image-agnostic delta trained over a cached watermarked set (deployable)
  pgd  : per-image delta (white-box oracle; upper bound on removability)
Objectives:
  erase : push every decoder's per-bit output to chance (logit->0, prob->0.5)
  spoof : push toward an attacker-chosen payload (BCE to a fixed target)

delta is L-infinity bounded (--eps), which sets the fidelity (PSNR). Trains on
cached watermarked tensors (embedded once) so the heavy VINE encoder runs only
at cache time. Resumable via --resume.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "attack"))

DEVICE = "cuda"
ENSEMBLE = ["vine_r", "maskwm", "dft_kred", "quant_qim"]   # differentiable decoders
HELDOUT = ["rivaGan", "trustmark"]                         # transfer targets


def _psnr_u8(a, b):
    a = a.astype(np.float64) / 255; b = b.astype(np.float64) / 255
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1 / mse) if mse > 1e-12 else 99.0


def load_ensemble(master_key=b"v5_key_encoder_master"):
    """Return {name: wrapper} for the differentiable ensemble (embed + diff decode)."""
    from overwrite_matrix import build_all
    methods = build_all(ENSEMBLE, DEVICE)   # UniformAdapters (have .frag) + VineMethod
    return methods


def diff_decode(name, wrapper, x01):
    """Differentiable per-bit output for an image tensor x01 in [0,1], (B,3,256,256).
    Returns (values, kind) where kind in {logit, prob, score}."""
    if name == "vine_r":
        # VineMethod: ._dec on [0,1] 256 -> sigmoid probs
        wrapper._load()
        return wrapper._dec(x01), "prob"
    frag = wrapper.frag
    if name in ("dft_kred", "quant_qim"):
        return frag.dec(x01), "logit"
    if name == "maskwm":
        ones = torch.ones(x01.shape[0], 1, 256, 256, device=x01.device)
        scores, _ = frag._model.decoder(x01 * 2 - 1, mask=ones)
        return scores, "score"
    raise ValueError(name)


def collapse_loss(values, kind, target=None):
    """erase: drive to chance; spoof: drive toward target bits (0/1 tensor).

    All decoders are mapped to a probability and the loss is bounded (~[0,0.25]) so
    no single decoder's gradient dominates the ensemble (logit^2 is unbounded and
    would starve the bounded prob-loss decoders -- VINE/MaskWM)."""
    prob = torch.sigmoid(values) if kind == "logit" else values.clamp(1e-4, 1 - 1e-4)
    if target is not None:
        t = target.to(values.device).float()
        return torch.nn.functional.binary_cross_entropy(prob.clamp(1e-4, 1 - 1e-4), t)
    return ((prob - 0.5) ** 2).mean()


def embed_cache(methods, files, n_bits_map, resolution=256, seed0=7000):
    """Embed each cover with each ensemble method once; cache (B,3,256,256) [0,1] tensors."""
    cache = {m: [] for m in ENSEMBLE}
    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        rng = np.random.RandomState(seed0 + i)
        for m in ENSEMBLE:
            bits = rng.randint(0, 2, methods[m].n_bits).astype(np.uint8)
            wm = methods[m].embed(cover, bits)
            if wm.size != (resolution, resolution):
                wm = wm.resize((resolution, resolution), Image.LANCZOS)
            t = torch.from_numpy(np.asarray(wm, np.float32) / 255.0).permute(2, 0, 1)
            cache[m].append(t)
    return {m: torch.stack(v).to(DEVICE) for m, v in cache.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_eval", type=int, default=12)
    p.add_argument("--mode", choices=["uap", "pgd"], default="uap")
    p.add_argument("--objective", choices=["erase", "spoof"], default="erase")
    p.add_argument("--eps", type=float, default=6.0, help="L-inf bound in /255")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--resume", default="")
    p.add_argument("--output", default="results/attack/uap.json")
    p.add_argument("--save_delta", default="results/attack/uap_delta.pt")
    args = p.parse_args()

    eps = args.eps / 255.0
    methods = load_ensemble()
    files = sorted(glob.glob(f"{args.image_dir}/*.jpg"))
    train_files = files[4000:4000 + args.n_train]
    eval_files = files[4500:4500 + args.n_eval]
    print(f"[uap] mode={args.mode} obj={args.objective} eps={args.eps}/255 ensemble={ENSEMBLE}", flush=True)

    if args.mode == "pgd":
        # Per-image white-box PGD oracle: a FRESH delta per (image, victim), optimized
        # against that victim's own decoder. Upper bound on removability; quantifies
        # the universality gap vs the UAP. Only the differentiable ensemble victims.
        out = {"mode": "pgd", "eps": args.eps, "steps": args.steps, "victims": {}}
        for vname in ENSEMBLE:
            m = methods[vname]
            atk, ps = [], []
            for i, fp in enumerate(eval_files):
                cover = Image.open(fp).convert("RGB").resize((256, 256), Image.LANCZOS)
                bits = np.random.RandomState(9000 + i).randint(0, 2, m.n_bits).astype(np.uint8)
                wm = m.embed(cover, bits)
                if wm.size != (256, 256):
                    wm = wm.resize((256, 256), Image.LANCZOS)
                wm_t = torch.from_numpy(np.asarray(wm, np.float32) / 255).permute(2, 0, 1)[None].to(DEVICE)
                d = torch.zeros_like(wm_t, requires_grad=True)
                o = torch.optim.Adam([d], lr=args.lr)
                for _ in range(args.steps):
                    o.zero_grad()
                    vals, kind = diff_decode(vname, m, (wm_t + d).clamp(0, 1))
                    collapse_loss(vals, kind).backward()
                    o.step()
                    with torch.no_grad():
                        d.clamp_(-eps, eps)
                adv = (wm_t + d).clamp(0, 1)
                adv_pil = Image.fromarray((adv[0].permute(1, 2, 0).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8))
                rec = m.decode(adv_pil); n = min(len(rec), len(bits))
                atk.append(float(np.mean(rec[:n] == bits[:n])))
                ps.append(float(_psnr_u8(np.asarray(wm), np.asarray(adv_pil))))
            out["victims"][m.name] = {"attacked_bitacc": float(np.mean(atk)), "psnr": float(np.mean(ps))}
            print(f"  PGD {m.name:<12} attacked={np.mean(atk):.3f} @ psnr {np.mean(ps):.1f}", flush=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"[done] -> {args.output}")
        return

    print("[uap] caching watermarked training tensors ...", flush=True)
    cache = embed_cache(methods, train_files, None)

    spoof = args.objective == "spoof"

    def make_target(vals):
        """Attacker payload sized to the decoder's actual output dim."""
        L = vals.shape[-1]
        t = torch.tensor(np.random.RandomState(123).randint(0, 2, L), device=DEVICE).float()
        return t.unsqueeze(0).expand(vals.shape[0], -1)

    delta = torch.zeros(1, 3, 256, 256, device=DEVICE, requires_grad=True)
    if args.resume and os.path.exists(args.resume):
        delta.data = torch.load(args.resume, map_location=DEVICE)
        print(f"[uap] resumed delta from {args.resume}", flush=True)
    opt = torch.optim.Adam([delta], lr=args.lr)

    for step in range(args.steps):
        opt.zero_grad()
        total = 0.0
        for m in ENSEMBLE:
            x = (cache[m] + delta).clamp(0, 1)
            vals, kind = diff_decode(m, methods[m], x)
            tgt = make_target(vals) if spoof else None
            total = total + collapse_loss(vals, kind, tgt)
        total.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.steps} loss={total.item():.4f}", flush=True)

    torch.save(delta.detach().cpu(), args.save_delta)

    # ---- evaluation: ensemble + held-out, bit-acc after delta, at delta PSNR ----
    eval_methods = load_ensemble()
    from overwrite_matrix import build_all
    eval_methods.update(build_all(HELDOUT, DEVICE))
    delta_np = delta.detach()
    out = {"mode": args.mode, "objective": args.objective, "eps": args.eps,
           "steps": args.steps, "ensemble": ENSEMBLE, "heldout": HELDOUT, "victims": {}}

    def psnr(a, b):
        a = a.astype(np.float64) / 255; b = b.astype(np.float64) / 255
        mse = np.mean((a - b) ** 2)
        return 10 * np.log10(1 / mse) if mse > 1e-12 else 99.0

    for name, m in eval_methods.items():
        clean_acc, atk_acc, psnrs = [], [], []
        for i, fp in enumerate(eval_files):
            cover = Image.open(fp).convert("RGB").resize((256, 256), Image.LANCZOS)
            rng = np.random.RandomState(9000 + i)
            bits = rng.randint(0, 2, m.n_bits).astype(np.uint8)
            wm = m.embed(cover, bits)
            if wm.size != (256, 256):
                wm = wm.resize((256, 256), Image.LANCZOS)
            wm_t = torch.from_numpy(np.asarray(wm, np.float32) / 255).permute(2, 0, 1)[None].to(DEVICE)
            adv = (wm_t + delta_np).clamp(0, 1)
            adv_pil = Image.fromarray((adv[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
            rec0 = m.decode(wm); reca = m.decode(adv_pil)
            n = min(len(rec0), len(bits))
            clean_acc.append(float(np.mean(rec0[:n] == bits[:n])))
            atk_acc.append(float(np.mean(reca[:n] == bits[:n])))
            psnrs.append(psnr(np.asarray(wm), np.asarray(adv_pil)))
        held = name in HELDOUT
        out["victims"][m.name] = {"clean_bitacc": float(np.mean(clean_acc)),
                                  "attacked_bitacc": float(np.mean(atk_acc)),
                                  "psnr": float(np.mean(psnrs)), "heldout": held}
        print(f"  {m.name:<12} {'(heldout)' if held else '(ensemble)':<11} "
              f"clean={np.mean(clean_acc):.3f} -> attacked={np.mean(atk_acc):.3f} @ psnr {np.mean(psnrs):.1f}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
