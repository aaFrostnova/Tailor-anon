"""RAVEN-style DEFENSE test suite (open-source defenses x open-source attacks).

Reproduces the evaluation protocol of RAVEN (arXiv:2601.08832, "Erasing Invisible
Watermarks via Novel View Synthesis"): run each DEFENSE watermark through the attack
suite and report robustness. We include only OPEN-SOURCE defenses and attacks.

Defenses (RAVEN post-hoc set we have + our fragments; all open-source):
  DwtDct, DwtDctSvd, RivaGAN, TrustMark, VINE  (RAVEN baselines, via wbench/imwatermark/trustmark/Shilin-LU)
  + MaskWM, PhaseMark, DFT-Kred, Quant-QIM     (our fragments)
  [Gaussian Shading / ZoDiac are generation-time -> separate harness; StegaStamp = TODO add]

Attacks (RAVEN's, open-source):
  signal: brightness, contrast, JPEG, Gaussian blur, Gaussian noise, BM3D
  regen:  Regen (SD img2img), Rinse, VAE round-trip
  [VAE-B/VAE-C use compressai (numpy<2 conflict) -> substituted by SD-VAE round-trip;
   advanced CtrlRegen/UnMarker = optional add-ons, see scripts/defense/README]

Metric: bit accuracy + detection at the per-method 1%-FPR threshold (RAVEN protocol:
bit-acc for bitstream schemes, TPR@1%FPR via the binomial threshold).
"""
import argparse
import glob
import json
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "attack"))

from scipy.stats import binom            # noqa: E402

DEFENSES_POSTHOC = ["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_r",
                    "maskwm", "phasemark", "dft_kred", "quant_qim"]


def fpr1_threshold(n_bits):
    """Bit-acc threshold giving <=1% FPR for n_bits random bits (binomial)."""
    k = binom.ppf(0.99, n_bits, 0.5) + 1
    return float(k) / n_bits


# ---------------- RAVEN attack suite (signal-processing + regeneration) -------
def _bm3d_attack(pil, sigma):
    import bm3d
    arr = np.asarray(pil.convert("RGB"), np.float32) / 255.0
    out = np.stack([bm3d.bm3d(arr[..., c], sigma_psd=sigma) for c in range(3)], -1)
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def raven_attack(name, pil, apply_attack):
    """name -> attacked PIL. Signal attacks here; regen/jpeg/blur/noise via apply_attack."""
    if name.startswith("bright_"):
        return ImageEnhance.Brightness(pil).enhance(int(name.split("_")[1]) / 100.0)
    if name.startswith("contrast_"):
        return ImageEnhance.Contrast(pil).enhance(int(name.split("_")[1]) / 100.0)
    if name.startswith("bm3d_"):
        return _bm3d_attack(pil, int(name.split("_")[1]) / 100.0)
    return apply_attack(name, pil)      # jpeg_/blur_/noise_/regen_/rinse_2/vae


RAVEN_ATTACKS = [
    "clean",
    # signal-processing
    "bright_70", "bright_130", "contrast_70", "contrast_130",
    "jpeg_25", "jpeg_50", "blur_2", "noise_50", "bm3d_10",
    # regeneration
    "regen_10_sd15", "regen_20_sd15", "regen_30_sd15", "rinse_2", "vae",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=12)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--defenses", nargs="+", default=DEFENSES_POSTHOC)
    p.add_argument("--attacks", nargs="+", default=RAVEN_ATTACKS)
    p.add_argument("--output", default="results/defense/defense_suite.json")
    args = p.parse_args()

    from overwrite_matrix import build_all
    from benchmark_fused import apply_attack
    methods = build_all(args.defenses, "cuda")
    names = [d for d in args.defenses if d in methods]
    tau = {d: fpr1_threshold(methods[d].n_bits) for d in names}

    files = sorted([f for f in glob.glob(os.path.join(args.image_dir, "*.jpg"))])
    files = files[args.start_idx:args.start_idx + args.n_images]
    R = args.resolution
    print(f"[defense-suite] {len(files)} imgs, {len(names)} defenses, {len(args.attacks)} attacks", flush=True)
    print("  1%-FPR thresholds:", {methods[d].name: round(tau[d], 2) for d in names}, flush=True)

    ba = {d: {a: [] for a in args.attacks} for d in names}
    det = {d: {a: [] for a in args.attacks} for d in names}

    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((R, R), Image.LANCZOS)
        rng = np.random.RandomState(7000 + i)
        for d in names:
            m = methods[d]
            bits = rng.randint(0, 2, m.n_bits).astype(np.uint8)
            wm = m.embed(cover, bits)
            if wm.size != cover.size:
                wm = wm.resize(cover.size, Image.LANCZOS)
            for a in args.attacks:
                att = raven_attack(a, wm, apply_attack)
                if att.size != cover.size:
                    att = att.resize(cover.size, Image.LANCZOS)
                rec = m.decode(att)
                n = min(len(rec), len(bits))
                acc = float(np.mean(rec[:n] == bits[:n]))
                ba[d][a].append(acc)
                det[d][a].append(1.0 if acc >= tau[d] else 0.0)
        print(f"  [{i+1}/{len(files)}]", flush=True)

    out = {"n_images": len(files), "resolution": R,
           "fpr1_threshold": {methods[d].name: tau[d] for d in names},
           "defenses": {}}
    for d in names:
        out["defenses"][methods[d].name] = {
            a: {"bit_acc": float(np.mean(ba[d][a])), "detect": float(np.mean(det[d][a]))}
            for a in args.attacks}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    # RAVEN-style table: rows=defenses, cols=attacks, cell=detect rate (TPR@1%FPR)
    print(f"\n=== DEFENSE robustness: detection rate @1%FPR (n={len(files)}) ===")
    hdr = f"{'defense':<12}" + "".join(f"{a[:9]:>10}" for a in args.attacks)
    print(hdr)
    for d in names:
        row = f"{methods[d].name[:12]:<12}"
        for a in args.attacks:
            row += f"{out['defenses'][methods[d].name][a]['detect']:>10.2f}"
        print(row)
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
