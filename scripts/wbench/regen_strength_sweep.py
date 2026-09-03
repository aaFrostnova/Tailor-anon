"""Diffusion-strength sweep for high-fidelity regeneration attack (Vanishing-Watermarks style).

Reproduces the paper's protocol: null-prompt SD-1.5 img2img "identity diffusion"
at a LOW denoising strength so the content stays high-fidelity (target ~31 dB
PSNR / SSIM ~0.95), and measures how much each watermark survives at each
fidelity level. The point: find the sweet spot where the image is barely
changed yet the watermark is gone, rather than destroying the content.
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

from wbench.methods import build_methods
from wbench.attacks import regen_attack, psnr

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SSIM = True
except Exception:
    _HAVE_SSIM = False


def ssim(a_pil, b_pil):
    if not _HAVE_SSIM:
        return None
    a = np.asarray(a_pil.convert("RGB")); b = np.asarray(b_pil.convert("RGB"))
    if a.shape != b.shape:
        b = np.asarray(b_pil.convert("RGB").resize(a_pil.size))
    return float(_ssim(a, b, channel_axis=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--methods", nargs="+",
                   default=["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_b", "vine_r"])
    p.add_argument("--strengths", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--model", default="sd15")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    methods = build_methods(args.methods, device)
    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[sweep] {len(files)} imgs, {len(methods)} methods, "
          f"strengths={args.strengths}, model={args.model}", flush=True)

    out = {"resolution": args.resolution, "n_images": len(files), "model": args.model,
           "wm_display": {m: methods[m].name for m in methods}, "by_strength": {}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for s in args.strengths:
        sname = f"regen_{int(round(s * 100))}_{args.model}"
        bitacc = {m: [] for m in methods}
        content_psnr, content_ssim = [], []
        for i, fp in enumerate(files):
            cover = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
            rng = np.random.RandomState(1000 + i)
            for mi, (mname, method) in enumerate(methods.items()):
                bits = rng.randint(0, 2, method.n_bits).astype(np.uint8)
                wm = method.embed(cover, bits)
                if wm.size != cover.size:
                    wm = wm.resize(cover.size, Image.LANCZOS)
                rg = regen_attack(sname, wm, device)
                if rg.size != cover.size:
                    rg = rg.resize(cover.size, Image.LANCZOS)
                if mi == 0:
                    content_psnr.append(psnr(cover, rg))
                    sv = ssim(cover, rg)
                    if sv is not None:
                        content_ssim.append(sv)
                rec = method.decode(rg); n = min(len(rec), len(bits))
                bitacc[mname].append(float(np.mean(rec[:n] == bits[:n])))
        out["by_strength"][f"{s:.2f}"] = {
            "content_psnr": float(np.mean(content_psnr)),
            "content_ssim": float(np.mean(content_ssim)) if content_ssim else None,
            "bit_acc": {m: float(np.mean(bitacc[m])) for m in methods},
        }
        ps = out["by_strength"][f"{s:.2f}"]["content_psnr"]
        ss = out["by_strength"][f"{s:.2f}"]["content_ssim"]
        accs = " ".join(f"{methods[m].name[:8]}={np.mean(bitacc[m]):.2f}" for m in methods)
        print(f"  strength={s:.2f}  PSNR={ps:.1f}dB SSIM={ss if ss is None else round(ss,3)}  {accs}", flush=True)
        # Incremental save: persist after every strength so a timeout keeps prior results.
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # table
    ms = list(methods)
    print("\n" + "=" * (10 + 8 + 7 + 11 * len(ms)))
    print(f"{'strength':<10}{'PSNR':>8}{'SSIM':>7}" + "".join(f"{methods[m].name[:10]:>11}" for m in ms))
    print("-" * (10 + 8 + 7 + 11 * len(ms)))
    for s in args.strengths:
        d = out["by_strength"][f"{s:.2f}"]
        ss = d["content_ssim"]
        row = f"{s:<10.2f}{d['content_psnr']:>8.1f}{(ss if ss is not None else 0):>7.3f}"
        for m in ms:
            row += f"{d['bit_acc'][m]:>11.3f}"
        print(row)
    print("=" * (10 + 8 + 7 + 11 * len(ms)))
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
