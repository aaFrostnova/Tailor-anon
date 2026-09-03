"""Overwrite-transfer matrix: which watermark overwrites which (Phase A).

Reframes the stacking experiment as an attack matrix. For each ordered pair
(victim, attacker): embed the victim watermark (random payload), then have the
attacker re-embed ITS OWN random payload on top (the attacker holds no victim
key), then decode the victim. cell[victim][attacker] = victim bit-accuracy after
the overwrite. Diagonal-like collapse (<~0.65) => same embedding subspace.

Extends scripts/wbench/stacking_eval.py to the fragment methods (DFT-Kred,
Quant-QIM, MaskWM) via a UniformAdapter that presents the classical
.embed(pil,bits)/.decode(pil)->bits/.n_bits/.name contract. Clusters methods
into overwrite-equivalence classes (connected components of the mutual-overwrite
graph) -- the data-driven taxonomy.

Phase A.0 gate: run with --methods dwtDct dwtDctSvd rivaGan trustmark vine_b
vine_r and confirm the retention matrix matches results/wbench_baselines/
stacking_clean.json (diagonal ~0.5, VINE-B<->VINE-R ~0.50/0.63, off-diag ~1.0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wbench.methods import build_methods           # noqa: E402
from wbench.attacks import apply_attack, psnr       # noqa: E402
from src.vine_crypto_wrapper import apply_crypto, undo_crypto  # noqa: E402

MASTER_KEY = b"v5_key_encoder_master"
DFT_CKPT = "results/dft_fftaware_baseline/ckpt.pt"
QIM_CKPT = "results/quant_qim_frozen_d006/ckpt.pt"
MASKWM_CKPT = "/project/pi_shiqingma_umass_edu/mingzheli/model/MaskWM/D_128bits.pth"

MASTER_KEY_DEFAULT = MASTER_KEY
FRAGMENT_NAMES = {"dft_kred", "quant_qim", "maskwm", "phasemark"}


class UniformAdapter:
    """Present a crypto/fragment method as .embed(pil,bits)/.decode(pil)->bits."""

    def __init__(self, frag, name, n_bits, kind, image_id):
        self.frag = frag
        self.name = name
        self.n_bits = n_bits
        self.kind = kind          # "logit" (dft/qim) | "prob" (vine) | "score" (maskwm)
        self.image_id = image_id  # fixed per method -> fixed carriers/perm/M

    def embed(self, pil, bits):
        bits = np.asarray(bits, dtype=np.uint8)[: self.n_bits]
        if self.kind == "logit":           # DFTKredMethod / QuantQIMMethod
            return self.frag.embed(pil, self.image_id, bits)
        # VineCryptoWrapper / MaskWMWrapper: embed an explicit crypto-applied target
        perm, M = self.frag.get_perm_M(self.image_id)
        target = apply_crypto(bits, perm, M)
        return self.frag.embed_with_target(pil, target)

    def decode(self, pil):
        if self.kind == "logit":
            soft = self.frag.raw_logits(pil)
            hard = (soft > 0.0).astype(np.float64)
        elif self.kind == "prob":
            soft = self.frag.raw_probs(pil)
            hard = (soft > 0.5).astype(np.float64)
        elif self.kind == "score0":  # phasemark signed phase-sum, threshold 0
            soft = self.frag.raw_scores(pil)
            hard = (soft > 0.0).astype(np.float64)
        else:  # score (maskwm, uncalibrated, threshold 0.5)
            soft = self.frag.raw_scores(pil)
            hard = (soft > 0.5).astype(np.float64)
        perm, M = self.frag.get_perm_M(self.image_id)
        return undo_crypto(hard, perm, M)[: self.n_bits]


def build_all(names, device):
    """Classical methods via build_methods + fragment methods via UniformAdapter."""
    classical = [n for n in names if n not in FRAGMENT_NAMES]
    reg = build_methods(classical, device) if classical else {}
    for n in names:
        if n not in FRAGMENT_NAMES:
            continue
        try:
            if n == "dft_kred":
                from src.learned_fragment_methods import DFTKredMethod
                frag = DFTKredMethod(str(REPO / DFT_CKPT), MASTER_KEY, "dft_kred", device)
                reg[n] = UniformAdapter(frag, "DFT-Kred", frag.n_bits, "logit", "matrix_dft")
            elif n == "quant_qim":
                from src.learned_fragment_methods import QuantQIMMethod
                frag = QuantQIMMethod(str(REPO / QIM_CKPT), MASTER_KEY, "quant_qim", device)
                reg[n] = UniformAdapter(frag, "Quant-QIM", frag.n_bits, "logit", "matrix_qim")
            elif n == "maskwm":
                from src.maskwm_wrapper import MaskWMWrapper
                frag = MaskWMWrapper(ckpt_path=MASKWM_CKPT, master_key=MASTER_KEY,
                                     method_name="maskwm", n_bits=100, device=device)
                frag._load()
                reg[n] = UniformAdapter(frag, "MaskWM-D", 100, "score", "matrix_maskwm")
            elif n == "phasemark":
                from src.phasemark import PhaseMarkWrapper
                frag = PhaseMarkWrapper(master_key=MASTER_KEY, method_name="phasemark",
                                        n_bits=100, vae_key="sd21", device=device)
                reg[n] = UniformAdapter(frag, "PhaseMark", 100, "score0", "matrix_phasemark")
            print(f"[methods] loaded {n} -> {reg[n].name} (n_bits={reg[n].n_bits})", flush=True)
        except Exception as e:
            print(f"[methods] SKIP {n}: {type(e).__name__}: {e}", flush=True)
    return reg


def cluster(retention, names, tau):
    """Connected components of the mutual-overwrite graph (min(ret_ij,ret_ji) < tau)."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a in names:
        for b in names:
            if a == b:
                continue
            rij, rji = retention[a].get(b), retention[b].get(a)
            if rij is None or rji is None:
                continue
            if min(rij, rji) < tau:
                union(a, b)
    classes = {}
    for n in names:
        classes.setdefault(find(n), []).append(n)
    return list(classes.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=40)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   default=["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_b", "vine_r"])
    p.add_argument("--post_attack", default="none",
                   help="distortion applied to the overwritten image before decode (e.g. jpeg_75)")
    p.add_argument("--tau", type=float, default=0.65, help="overwrite-cluster threshold")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_all(args.methods, device)
    names = [n for n in args.methods if n in methods]

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[overwrite-matrix] {len(files)} imgs, {len(names)} methods, post={args.post_attack}", flush=True)

    self_acc = {a: [] for a in names}
    retention = {a: {b: [] for b in names} for a in names}   # victim a after attacker b
    spoof = {a: {b: [] for b in names} for a in names}       # victim decodes attacker payload?
    psnrs = {a: {b: [] for b in names} for a in names}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        rng = np.random.RandomState(1000 + i)

        bitsV = {a: rng.randint(0, 2, methods[a].n_bits).astype(np.uint8) for a in names}
        imgV = {}
        for a in names:
            try:
                wv = methods[a].embed(cover, bitsV[a])
                if wv.size != cover.size:
                    wv = wv.resize(cover.size, Image.LANCZOS)
                imgV[a] = wv
                rec = methods[a].decode(wv)
                n = min(len(rec), len(bitsV[a]))
                self_acc[a].append(float(np.mean(rec[:n] == bitsV[a][:n])))
            except Exception as e:
                print(f"  [{a}] self-embed failed img {i}: {e}", flush=True)

        for a in names:                       # victim
            if a not in imgV:
                continue
            for b in names:                    # attacker (random key-blind payload)
                bitsA = rng.randint(0, 2, methods[b].n_bits).astype(np.uint8)
                try:
                    wab = methods[b].embed(imgV[a], bitsA)
                    if wab.size != cover.size:
                        wab = wab.resize(cover.size, Image.LANCZOS)
                    psnrs[a][b].append(psnr(cover, wab))
                    if args.post_attack != "none":
                        wab = apply_attack(args.post_attack, wab, device)
                        if wab.size != cover.size:
                            wab = wab.resize(cover.size, Image.LANCZOS)
                    rec = methods[a].decode(wab)               # decode VICTIM
                    nv = min(len(rec), len(bitsV[a]))
                    retention[a][b].append(float(np.mean(rec[:nv] == bitsV[a][:nv])))
                    # spoofing: did the victim decoder pick up the attacker payload?
                    if a == b:
                        na = min(len(rec), len(bitsA))
                        spoof[a][b].append(float(np.mean(rec[:na] == bitsA[:na])))
                except Exception as e:
                    print(f"  [{a}<-{b}] overwrite failed img {i}: {e}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    def mean(d):
        return float(np.mean(d)) if d else None

    ret_m = {a: {b: mean(retention[a][b]) for b in names} for a in names}
    summary = {
        "resolution": args.resolution, "n_images": len(files), "post_attack": args.post_attack,
        "tau": args.tau,
        "display": {a: methods[a].name for a in names},
        "n_bits": {a: methods[a].n_bits for a in names},
        "self_acc": {a: mean(self_acc[a]) for a in names},
        "retention": ret_m,
        "spoof_diag": {a: mean(spoof[a][a]) for a in names},
        "psnr": {a: {b: mean(psnrs[a][b]) for b in names} for a in names},
    }
    summary["overwrite_classes"] = [
        [methods[n].name for n in cls] for cls in cluster(ret_m, names, args.tau)
    ]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    disp = [methods[a].name for a in names]
    print("\nself bit_acc:", {methods[a].name: round(summary['self_acc'][a], 3) for a in names if summary['self_acc'][a] is not None})
    print(f"\nRETENTION victim(row) after attacker(col) overwrite  [post={args.post_attack}]")
    print(f"{'V\\\\A':<12}" + "".join(f"{d[:10]:>11}" for d in disp))
    for a, da in zip(names, disp):
        row = f"{da:<12}"
        for b in names:
            v = ret_m[a][b]
            row += f"{(v if v is not None else float('nan')):>11.3f}"
        print(row)
    print("\nOVERWRITE-EQUIVALENCE CLASSES (tau=%.2f):" % args.tau)
    for cls in summary["overwrite_classes"]:
        print("  {" + ", ".join(cls) + "}")
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
