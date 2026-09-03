"""P1 gate benchmark: MaskWM-D under OUR attack suite.

Decides the integration verdict. The paper reports geometric bit-acc 0.9998 (vs TrustMark
0.7868) under ITS attack parameterization. This re-measures D_128bits under our exact
attacks -- crop_90/resize_110, compound geometry+jpeg, rotation -- to confirm the
single-point-of-failure (TrustMark-only geometry) is actually removed. If MaskWM-D reads
<0.9 on our crop_90 the way TrustMark reads 0.79, the SPOF argument evaporates.

Attacks are self-contained (PIL/numpy only) -- no diffusion/regen models loaded, so this
is fast and does not contend with GPU regen jobs. Regeneration is deliberately NOT tested
here: MaskWM is not a regen method (that gap is on a separate track).

Run (after downloading D_128bits.pth):
  python scripts/benchmark_maskwm.py --ckpt /project/.../MaskWM/D_128bits.pth --n_images 40
"""

import argparse
import glob
import sys
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

sys.path.insert(0, ".")
from src.maskwm_wrapper import MaskWMWrapper  # noqa: E402


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255.0
    b = np.asarray(b, np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 99.0


def apply_attack(name, pil):
    if name == "clean":
        return pil
    if name.startswith("jpeg_"):
        q = int(name.split("_")[1]); buf = BytesIO()
        pil.save(buf, format="JPEG", quality=q); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name.startswith("blur_"):
        return pil.filter(ImageFilter.GaussianBlur(radius=float(name.split("_")[1])))
    if name.startswith("noise_"):
        s = int(name.split("_")[1]) / 1000.0
        arr = np.asarray(pil, np.float32) / 255.0
        arr += np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * s
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name.startswith("crop_"):
        r = int(name.split("_")[1]) / 100.0
        W, H = pil.size; cw, ch = int(W * r), int(H * r)
        l, t = (W - cw) // 2, (H - ch) // 2
        return pil.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BILINEAR)
    if name.startswith("resize_"):
        r = int(name.split("_")[1]) / 100.0
        W, H = pil.size; Wb, Hb = int(W * r), int(H * r)
        big = pil.resize((Wb, Hb), Image.BILINEAR)
        l, t = (Wb - W) // 2, (Hb - H) // 2
        return big.crop((l, t, l + W, t + H))
    if name.startswith("rotate_"):
        deg = float(name.split("_")[1])
        return pil.rotate(deg, resample=Image.BILINEAR, expand=False)
    if name.startswith("bright_"):
        f = int(name.split("_")[1]) / 100.0
        return ImageEnhance.Brightness(pil).enhance(f)
    if "_then_" in name:  # compound, e.g. resize_110_then_jpeg_50
        a, b = name.split("_then_")
        return apply_attack(b, apply_attack(a, pil))
    raise ValueError(name)


ATTACKS = [
    "clean", "jpeg_75", "jpeg_50", "blur_2", "noise_50",
    "crop_90", "crop_70", "resize_110", "rotate_10", "bright_130",
    "resize_110_then_jpeg_50", "crop_90_then_jpeg_50",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to D_128bits.pth")
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=40)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--det_thresh", type=float, default=0.75)
    p.add_argument("--use_self_mask", action="store_true", help="use MaskWM's U2Net self-gating decode (paper path)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/fused/maskwm_gate.json")
    args = p.parse_args()

    w = MaskWMWrapper(ckpt_path=args.ckpt, device=args.device,
                      detection_threshold=args.det_thresh, use_self_mask=args.use_self_mask)
    w._load()

    imgs = sorted(glob.glob(f"{args.image_dir}/*.jpg"))[args.start_idx:args.start_idx + args.n_images]
    R = args.resolution
    print(f"[maskwm-gate] {len(imgs)} imgs, self_mask={args.use_self_mask}, {len(ATTACKS)} attacks")

    acc = {a: [] for a in ATTACKS}
    det = {a: [] for a in ATTACKS}
    fpr_acc = []  # wrong-id bit-acc on clean watermarked image
    psnrs = []

    for i, path in enumerate(imgs):
        pil = Image.open(path).convert("RGB").resize((R, R))
        image_id = f"maskwm_img_{args.start_idx + i:06d}"
        wm = w.embed(pil, image_id)
        psnrs.append(psnr(pil, wm))
        # FPR: decode clean wm against a WRONG id
        fpr_acc.append(w.detect(wm, f"wrong_{args.start_idx + i:06d}")["bit_accuracy"])
        for a in ATTACKS:
            att = apply_attack(a, wm)
            r = w.detect(att, image_id)
            acc[a].append(r["bit_accuracy"])
            det[a].append(1.0 if r["detected"] else 0.0)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(imgs)}] clean acc so far {np.mean(acc['clean']):.3f}")

    print(f"\n=== MaskWM-D gate (n={len(imgs)}, PSNR {np.mean(psnrs):.2f}dB, "
          f"wrong-id bit-acc {np.mean(fpr_acc):.3f}) ===")
    print(f"  {'attack':<26}{'bit_acc':>9}{'detect':>9}")
    out = {"n_images": len(imgs), "psnr": float(np.mean(psnrs)),
           "wrong_id_bitacc": float(np.mean(fpr_acc)),
           "use_self_mask": args.use_self_mask, "attacks": {}}
    for a in ATTACKS:
        ba, dd = float(np.mean(acc[a])), float(np.mean(det[a]))
        print(f"  {a:<26}{ba:>9.3f}{dd:>9.2f}")
        out["attacks"][a] = {"bit_acc": ba, "detect": dd}

    import json, os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print(f"\n[done] -> {args.output}")
    print("Compare geometric (crop_90/resize_110/rotate_10) vs TrustMark 0.7868 / VINE 0.5012.")


if __name__ == "__main__":
    main()
