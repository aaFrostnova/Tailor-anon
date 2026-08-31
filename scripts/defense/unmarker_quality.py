"""Measure UnMarker's image-quality damage, decomposing crop vs adversarial distortion.

UnMarker (Vine.yaml) applies a 0.9 center-crop+resize THEN an adversarial perturbation. So we report:
  attacked vs original             : end-to-end damage (vs the true clean image)
  attacked vs watermarked          : UnMarker's total change to its input (incl. its own crop)
  attacked vs crop0.9(watermarked) : UnMarker's ADVERSARIAL distortion alone (crop-aligned)
  crop0.9(watermarked) vs wm       : the 0.9-crop's own cost (reference, to subtract)
Metrics: PSNR, SSIM, LPIPS(alex). Also saves a visual fig_unmarker_quality.png.
"""
import glob, os, sys
import numpy as np, torch
from PIL import Image
from scipy import ndimage  # noqa
from skimage.metrics import structural_similarity as ssim
import lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
WM = os.path.join(REPO, "results/defense/ext_2fused_coco")
ATK = os.path.join(REPO, "results/defense/ext_2fused_coco_unmarker")
OUT = os.path.join(REPO, "results/figures/fig_unmarker_quality.png")
RES = 512


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def ccr(img, frac=0.9):
    W, H = img.size; cw, ch = int(round(W * frac)), int(round(H * frac))
    l, t = (W - cw) // 2, (H - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)
def psnr(a, b):
    m = np.mean((a - b) ** 2); return float(99.0 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m))


def main():
    dev = "cuda"; loss_fn = lpips.LPIPS(net="alex").to(dev).eval()
    def lp(a, b):
        ta = torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].float().to(dev)
        tb = torch.from_numpy(b / 127.5 - 1).permute(2, 0, 1)[None].float().to(dev)
        with torch.no_grad(): return float(loss_fn(ta, tb).item())

    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    pairs = ["atk_vs_orig", "atk_vs_wm", "atk_vs_crop_wm", "crop_wm_vs_wm"]
    M = {p: {"psnr": [], "ssim": [], "lpips": []} for p in pairs}
    heroes = []
    for i in range(10):
        O = arr(Image.open(coco[4000 + i]))
        W = arr(Image.open(os.path.join(WM, f"img_{i:05d}.png")))
        A = arr(Image.open(os.path.join(ATK, f"img_{i:05d}.png")))
        Wc = arr(ccr(Image.open(os.path.join(WM, f"img_{i:05d}.png"))))   # crop0.9(wm), aligns with attack frame
        for tag, (a, b) in {"atk_vs_orig": (A, O), "atk_vs_wm": (A, W),
                            "atk_vs_crop_wm": (A, Wc), "crop_wm_vs_wm": (Wc, W)}.items():
            M[tag]["psnr"].append(psnr(a, b))
            M[tag]["ssim"].append(ssim(a, b, channel_axis=2, data_range=255))
            M[tag]["lpips"].append(lp(a, b))
        if i in (0, 5):
            heroes.append((i, O, W, A, Wc))

    print(f"\n=== UnMarker image-quality (n=10) ===")
    print(f"{'comparison':<26}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>9}   meaning")
    meaning = {"atk_vs_orig": "end-to-end vs clean",
               "atk_vs_wm": "UnMarker total change (incl its crop)",
               "atk_vs_crop_wm": "UnMarker ADVERSARIAL distortion (crop-aligned)",
               "crop_wm_vs_wm": "the 0.9-crop's own cost (reference)"}
    for p in pairs:
        print(f"{p:<26}{np.mean(M[p]['psnr']):>8.2f}{np.mean(M[p]['ssim']):>8.3f}"
              f"{np.mean(M[p]['lpips']):>9.3f}   {meaning[p]}")

    # ---- figure ----
    fig, axes = plt.subplots(len(heroes), 4, figsize=(15, 3.9 * len(heroes)))
    if len(heroes) == 1: axes = axes[None, :]
    cols = ["original", "watermarked", "UnMarker-attacked", "residual (attacked - crop(wm)) x5"]
    for r, (i, O, W, A, Wc) in enumerate(heroes):
        resid = np.clip((A - Wc) * 5 + 128, 0, 255).astype(np.uint8)
        for c, im in enumerate([O, W, A, resid]):
            ax = axes[r, c]; ax.imshow(np.clip(im, 0, 255).astype(np.uint8)); ax.axis("off")
            if r == 0: ax.set_title(cols[c], fontsize=11)
        axes[r, 0].text(-0.04, 0.5, f"img{i:05d}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", fontsize=10)
        cap = (f"atk vs orig: PSNR {M['atk_vs_orig']['psnr'][[0,5].index(i) if False else i]:.1f}" )
        axes[r, 2].text(0.0, -0.06,
                        f"PSNR(atk,orig)={psnr(A,O):.1f}dB  LPIPS={lp(A,O):.3f}",
                        transform=axes[r, 2].transAxes, fontsize=9, va="top")
    fig.suptitle("UnMarker image quality: removes the watermark but leaves visible adversarial spectral distortion",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[fig] -> {OUT}\nUNMARKER_QUALITY_DONE")


if __name__ == "__main__":
    main()
