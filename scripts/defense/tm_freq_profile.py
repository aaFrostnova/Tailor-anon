"""Measure the radial power spectrum of TrustMark-B vs VINE-R residuals.

Settles: is TrustMark low- or high-frequency? Embed each method on the SAME
SD-2.1 images, residual = wm - orig, 2D FFT power, bin by normalized radius
(0=DC .. 1=Nyquist), report band energy fractions + cumulative curve.
"""
import os, sys, glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from wbench.methods import TrustMarkMethod, VineMethod

N = 12
imgs = sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png")))[:N]

def radial_power(resid):
    """resid: HxWx3 float. Return (r_norm_bins, mean power per bin) averaged over channels."""
    H, W, C = resid.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)              # Nyquist along shortest axis
    rn = r / rmax
    nb = 64
    bins = np.linspace(0, 1.0, nb + 1)
    idx = np.clip(np.digitize(rn, bins) - 1, 0, nb - 1)
    acc = np.zeros(nb); cnt = np.zeros(nb)
    for c in range(C):
        F = np.fft.fftshift(np.fft.fft2(resid[..., c]))
        P = np.abs(F) ** 2
        for b in range(nb):
            m = idx == b
            if m.any():
                acc[b] += P[m].sum(); cnt[b] += m.sum()
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_p = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)   # power density per bin
    total_p = acc.copy()                                       # total power in bin (for energy frac)
    return centers, mean_p, total_p

def band_fractions(centers, total_p):
    e = total_p / total_p.sum()
    bands = {"low(0-.1)": (0, .1), "low-mid(.1-.25)": (.1, .25),
             "mid(.25-.5)": (.25, .5), "high(.5-1)": (.5, 1.01)}
    return {k: float(e[(centers >= lo) & (centers < hi)].sum()) for k, (lo, hi) in bands.items()}

def run_method(m, label):
    dens_acc = None; tot_acc = None; centers = None
    for fp in imgs:
        orig = Image.open(fp).convert("RGB").resize((512, 512))
        bits = np.random.RandomState(hash(fp) % (2**31)).randint(0, 2, m.n_bits).astype(np.uint8)
        wm = m.embed(orig, bits)
        if wm.size != (512, 512): wm = wm.resize((512, 512))
        resid = np.asarray(wm, np.float64) - np.asarray(orig, np.float64)
        centers, dens, tot = radial_power(resid)
        dens_acc = dens if dens_acc is None else dens_acc + dens
        tot_acc = tot if tot_acc is None else tot_acc + tot
        rms = float(np.sqrt(np.mean(resid ** 2)))
        print(f"  [{label}] {os.path.basename(fp)} resid RMS={rms:.3f}", flush=True)
    dens_acc /= len(imgs)
    fr = band_fractions(centers, tot_acc)
    print(f"=== {label} band energy fractions ===")
    for k, v in fr.items(): print(f"    {k:18s} {v:.3f}")
    return centers, dens_acc, fr

print("Loading TrustMark-B..."); tm = TrustMarkMethod("B")
c_tm, d_tm, fr_tm = run_method(tm, "TrustMark-B")
print("Loading VINE-R..."); vn = VineMethod("R")
c_vn, d_vn, fr_vn = run_method(vn, "VINE-R")

# normalized power-density curves
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for c, d, lab in [(c_tm, d_tm, "TrustMark-B"), (c_vn, d_vn, "VINE-R")]:
    ax[0].plot(c, d / d.max(), label=lab, lw=2)
ax[0].set_xlabel("normalized radius (0=DC, 1=Nyquist)"); ax[0].set_ylabel("power density (norm)")
ax[0].set_title("Residual radial power density"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_yscale("log")
labels = list(fr_tm.keys()); x = np.arange(len(labels)); w = .35
ax[1].bar(x - w/2, [fr_tm[k] for k in labels], w, label="TrustMark-B")
ax[1].bar(x + w/2, [fr_vn[k] for k in labels], w, label="VINE-R")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=20, ha="right")
ax[1].set_ylabel("energy fraction"); ax[1].set_title("Band energy share"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
out = os.path.join(REPO, "results/defense/tm_vs_vine_freq.png")
plt.savefig(out, dpi=120); print(f"[plot] -> {out}")
print("FREQ_DONE")
