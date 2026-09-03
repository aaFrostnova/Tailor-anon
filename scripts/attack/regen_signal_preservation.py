"""What signal survives regeneration? A decomposition-level preservation study.

For each regeneration attack, measure how much of the image signal is PRESERVED between
the original and the regenerated image, decomposed by:
  (1) PIXEL FFT radial frequency bands (LF -> HF) on luminance,
  (2) VAE-LATENT FFT radial bands (the manifold regen operates on),
each reported as:
  - complex NCC  : |<F_o, F_a>| / (||F_o|| ||F_a||)  -- combined magnitude+phase preservation (1=kept, 0=destroyed)
  - magnitude NCC: NCC of |F|                         -- is the spectral envelope kept?
  - phase coh    : sum|F_o||F_a|cos(dphi)/sum|F_o||F_a| -- is the phase kept? (magnitude-weighted)

Answers "which signals are retained under regeneration" empirically, across the whole
regen spectrum. Pure numpy FFT + the SD VAE; writes a persistent JSON + prints tables.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

# radial band edges as fraction of Nyquist
BAND_EDGES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.0001]
BAND_LABELS = ["LF<5%", "LF5-10", "MF10-20", "MF20-35", "HF35-55", "HF55-80", "HF80-100"]


def luma(pil):
    a = np.asarray(pil.convert("RGB"), np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def band_assign(N):
    f = np.fft.fftfreq(N) * N
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.sqrt(fy ** 2 + fx ** 2) / (N / 2.0)   # fraction of Nyquist
    return np.digitize(r, BAND_EDGES) - 1          # band index 0..len-1 (clip last)


def band_metrics(Fo, Fa, bands, nb):
    """Per-band complex NCC, magnitude NCC, phase coherence."""
    cn, mn, ph = [], [], []
    mo, ma = np.abs(Fo), np.abs(Fa)
    for b in range(nb):
        m = bands == b
        fo, fa = Fo[m], Fa[m]
        mob, mab = mo[m], ma[m]
        den = np.sqrt((np.abs(fo) ** 2).sum()) * np.sqrt((np.abs(fa) ** 2).sum()) + 1e-12
        cn.append(float(np.abs(np.vdot(fo, fa)) / den))                       # complex NCC
        mdn = np.sqrt((mob ** 2).sum()) * np.sqrt((mab ** 2).sum()) + 1e-12
        mn.append(float((mob * mab).sum() / mdn))                             # magnitude NCC
        w = mob * mab
        ph.append(float((w * np.cos(np.angle(fo) - np.angle(fa))).sum() / (w.sum() + 1e-12)))  # phase coh
    return cn, mn, ph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "vae", "flux_vae", "regen_10_sd15", "regen_20_sd15", "regen_30_sd15",
        "regen_30_sd21", "flux_i2i_10", "flux_i2i_20", "flux_i2i_30", "rinse_2"])
    p.add_argument("--output", default="results/attack/regen_signal_preservation.json")
    args = p.parse_args()

    from benchmark_fused import apply_attack
    from src.latent_vae import LatentVAE
    vae = LatentVAE("sd21", "cuda")

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))
    imgs = imgs[args.start_idx:args.start_idx + args.n_images]
    R = args.resolution
    nb = len(BAND_LABELS)
    pbands = band_assign(R)
    lbands = band_assign(64)
    print(f"[signal-preservation] n={len(imgs)} attacks={len(args.attacks)} bands={BAND_LABELS}", flush=True)

    def psnr(a, b):
        a = np.asarray(a, np.float64) / 255; b = np.asarray(b, np.float64) / 255
        return 10 * np.log10(1 / np.mean((a - b) ** 2)) if np.mean((a - b) ** 2) > 1e-12 else 99.0

    acc = {a: {"psnr": [], "pix_cn": [], "pix_mn": [], "pix_ph": [],
               "lat_cn": [], "lat_mn": [], "lat_ph": []} for a in args.attacks}

    for i, fp in enumerate(imgs):
        img = Image.open(fp).convert("RGB").resize((R, R))
        Yo = np.fft.fft2(luma(img))
        zo = vae.encode(img)[0].cpu().numpy()        # [4,64,64]
        for a in args.attacks:
            att = apply_attack(a, img)
            if att.size != (R, R):
                att = att.resize((R, R))
            acc[a]["psnr"].append(psnr(img, att))
            # pixel FFT bands
            Ya = np.fft.fft2(luma(att))
            cn, mn, ph = band_metrics(Yo, Ya, pbands, nb)
            acc[a]["pix_cn"].append(cn); acc[a]["pix_mn"].append(mn); acc[a]["pix_ph"].append(ph)
            # latent FFT bands (avg over 4 channels)
            za = vae.encode(att)[0].cpu().numpy()
            ccn = np.zeros(nb); cmn = np.zeros(nb); cph = np.zeros(nb)
            for ch in range(zo.shape[0]):
                c1, m1, p1 = band_metrics(np.fft.fft2(zo[ch]), np.fft.fft2(za[ch]), lbands, nb)
                ccn += np.array(c1); cmn += np.array(m1); cph += np.array(p1)
            acc[a]["lat_cn"].append((ccn / zo.shape[0]).tolist())
            acc[a]["lat_mn"].append((cmn / zo.shape[0]).tolist())
            acc[a]["lat_ph"].append((cph / zo.shape[0]).tolist())
        print(f"  [{i+1}/{len(imgs)}]", flush=True)

    out = {"bands": BAND_LABELS, "n_images": len(imgs), "attacks": {}}
    for a in args.attacks:
        out["attacks"][a] = {
            "psnr": float(np.mean(acc[a]["psnr"])),
            "pixel_complex_ncc": np.mean(acc[a]["pix_cn"], axis=0).tolist(),
            "pixel_mag_ncc": np.mean(acc[a]["pix_mn"], axis=0).tolist(),
            "pixel_phase_coh": np.mean(acc[a]["pix_ph"], axis=0).tolist(),
            "latent_complex_ncc": np.mean(acc[a]["lat_cn"], axis=0).tolist(),
            "latent_mag_ncc": np.mean(acc[a]["lat_mn"], axis=0).tolist(),
            "latent_phase_coh": np.mean(acc[a]["lat_ph"], axis=0).tolist(),
        }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    def tab(title, key):
        print(f"\n=== {title} (complex NCC: 1=preserved, 0=destroyed) ===")
        print(f"{'attack':<16}{'PSNR':>6} " + "".join(f"{b:>9}" for b in BAND_LABELS))
        for a in args.attacks:
            v = out["attacks"][a][key]
            print(f"{a:<16}{out['attacks'][a]['psnr']:>6.1f} " + "".join(f"{x:>9.2f}" for x in v))
    tab("PIXEL-domain frequency preservation", "pixel_complex_ncc")
    tab("VAE-LATENT frequency preservation", "latent_complex_ncc")
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
