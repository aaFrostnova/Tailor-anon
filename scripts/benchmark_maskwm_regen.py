"""Does MaskWM-D survive REGENERATION? (the paper never tested this.)

Embeds MaskWM-D, applies the full regeneration family (VAE round-trip, SD img2img at
increasing strength, FLUX flow-matching img2img, chained rinse), decodes with MaskWM.
Reuses the exact regen operators from benchmark_fused.apply_attack so the numbers are
directly comparable to our VINE/GS/fused results.

Mechanistic prior: MaskWM is a pixel-space JND additive mark (WAM family). Regeneration
re-synthesizes the whole frame in latent space, so high-frequency pixel marks should be
erased -- we expect MaskWM to behave like TrustMark (survives VAE round-trip, dies under
diffusion img2img). This measures it.

Run:
  python scripts/benchmark_maskwm_regen.py --ckpt /project/.../MaskWM/D_128bits.pth --n_images 16
"""

import argparse
import glob
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.maskwm_wrapper import MaskWMWrapper  # noqa: E402


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255.0
    b = np.asarray(b, np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 99.0


# Regeneration family, ordered light -> heavy. VAE round-trips first (expected survivable),
# then diffusion img2img at increasing strength (expected fatal), then cross-arch + chained.
ATTACKS = [
    "clean",
    "vae",                # SD continuous VAE round-trip
    "flux_vae",           # FLUX 16-channel VAE round-trip
    "regen_10_sd15",      # SD1.5 img2img strength 0.10
    "regen_20_sd15",      # strength 0.20
    "regen_30_sd15",      # strength 0.30
    "flux_i2i_20",        # FLUX flow-matching img2img strength 0.20
    "flux_i2i_30",        # strength 0.30
    "rinse_2",            # chained 2x light regen
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--det_thresh", type=float, default=0.75)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attacks", nargs="+", default=ATTACKS)
    p.add_argument("--output", default="results/fused/maskwm_regen.json")
    args = p.parse_args()

    from benchmark_fused import apply_attack  # lazy: pulls SD/FLUX loaders only when used

    w = MaskWMWrapper(ckpt_path=args.ckpt, device=args.device, detection_threshold=args.det_thresh)
    w._load()

    imgs = sorted(glob.glob(f"{args.image_dir}/*.jpg"))[args.start_idx:args.start_idx + args.n_images]
    R = args.resolution
    print(f"[maskwm-regen] {len(imgs)} imgs, {len(args.attacks)} attacks")

    acc = {a: [] for a in args.attacks}
    det = {a: [] for a in args.attacks}
    apsnr = {a: [] for a in args.attacks}
    embed_psnr = []

    for i, path in enumerate(imgs):
        pil = Image.open(path).convert("RGB").resize((R, R))
        image_id = f"maskwm_img_{args.start_idx + i:06d}"
        wm = w.embed(pil, image_id)
        embed_psnr.append(psnr(pil, wm))
        for a in args.attacks:
            att = apply_attack(a, wm)
            if att.size != (R, R):
                att = att.resize((R, R))
            apsnr[a].append(psnr(pil, att))
            r = w.detect(att, image_id)
            acc[a].append(r["bit_accuracy"])
            det[a].append(1.0 if r["detected"] else 0.0)
        print(f"  [{i+1}/{len(imgs)}] {image_id} done")

    print(f"\n=== MaskWM-D vs REGENERATION (n={len(imgs)}, embed PSNR {np.mean(embed_psnr):.2f}dB) ===")
    print(f"  {'attack':<16}{'content_psnr':>13}{'bit_acc':>9}{'detect':>9}")
    out = {"n_images": len(imgs), "embed_psnr": float(np.mean(embed_psnr)), "attacks": {}}
    for a in args.attacks:
        ba, dd, cp = float(np.mean(acc[a])), float(np.mean(det[a])), float(np.mean(apsnr[a]))
        print(f"  {a:<16}{cp:>13.1f}{ba:>9.3f}{dd:>9.2f}")
        out["attacks"][a] = {"bit_acc": ba, "detect": dd, "content_psnr": cp}

    import json, os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print(f"\n[done] -> {args.output}")
    print("Compare: VINE-R regen_30_sd15 bit-acc ~0.75, flux_i2i_30 ~0.66; GS det 1.00/0.93.")


if __name__ == "__main__":
    main()
