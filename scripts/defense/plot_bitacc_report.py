"""bit-acc figures for the rewritten report. All numbers MEASURED (unified_bitacc.json + s01 + quality)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FG = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/final_figs"
VINE, TM, VS = "#2a78d6", "#008300", "#eb6834"
CROP, ROT, CR, UM = "#1baf7a", "#4a3aa7", "#d55181", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddddd"
CRYPTO, SOFT = "#0b0b0b", "#8a8a8a"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.labelcolor": INK, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_axisbelow(True)
def bands(ax):
    ax.axhline(0.90, color=CRYPTO, ls="--", lw=1.2); ax.axhline(0.63, color=SOFT, ls=":", lw=1.2)
    ax.text(ax.get_xlim()[1], 0.905, " crypto 0.90", color=CRYPTO, fontsize=8, va="bottom", ha="right")
    ax.text(ax.get_xlim()[1], 0.635, " soft τ 0.63", color=SOFT, fontsize=8, va="bottom", ha="right")

# measured
GEO = {"rot30": (0.499, 0.979), "rot45": (0.493, 0.929), "rot90": (0.588, 0.963), "rot180": (0.526, 0.950),
       "crop75": (1.000, 1.000), "crop60": (0.496, 0.932), "crop50": (1.000, 1.000)}
REG_s = [0.1, 0.3, 0.5, 0.7, 0.9]
REG_ba = [0.934, 0.833, 0.716, 0.671, 0.653]
REG_ssim = [0.678, 0.613, 0.564, 0.523, 0.492]
REG_psnr = [23.80, 22.03, 20.72, 19.57, 18.67]
UM_ba = [0.984, 0.976]; UM_ssim = [0.776, 0.734]; UM_psnr = [24.14, 22.69]; UM_lab = ["default", "strong"]

# ---- FIG 3: bit-acc robustness (geometry recovery + regen degradation) ----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2)); [style(a) for a in axs]
keys = list(GEO); x = np.arange(len(keys)); w = 0.38
raw = [GEO[k][0] for k in keys]; rec = [GEO[k][1] for k in keys]
axs[0].bar(x - w/2, raw, w, color="#c9c9c9", label="raw (scale 1.0, uncorrected)")
axs[0].bar(x + w/2, rec, w, color=VINE, label="recovered (after geo/scale correction)")
axs[0].set_xticks(x); axs[0].set_xticklabels(keys, rotation=30, fontsize=9); axs[0].set_ylim(0, 1.08)
axs[0].set_ylabel("bit-acc"); bands(axs[0])
axs[0].set_title("(a) Geometry — alignment recovery lifts bit-acc back above crypto", fontsize=10.5, loc="left", color=INK)
axs[0].legend(frameon=False, fontsize=9, loc="lower center")
axs[1].plot(REG_s, REG_ba, "-o", color=CR, lw=2, ms=7, label="CtrlRegen+ recovered bit-acc")
axs[1].scatter([0.0], [1.0], color=INK, s=45, zorder=5); axs[1].annotate("clean 1.0", (0.0, 1.0), textcoords="offset points", xytext=(6, -4), fontsize=8, color=INK2)
for s, b in zip(REG_s, REG_ba): axs[1].annotate(f"{b:.2f}", (s, b), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center", color=INK2)
axs[1].set_xlim(-0.05, 1.0); axs[1].set_ylim(0.45, 1.05); axs[1].set_xlabel("CtrlRegen+ regeneration strength")
axs[1].set_ylabel("recovered bit-acc"); bands(axs[1])
axs[1].set_title("(b) Regeneration — bit-acc slides into the soft band", fontsize=10.5, loc="left", color=INK)
axs[1].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Deployed-pipeline bit-acc (measured) — crypto exact-recovery threshold approx 0.90, soft threshold 0.63", fontsize=12, color=INK, y=1.02)
fig.savefig(f"{FG}/fig3_bitacc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig3_bitacc done")

# ---- FIG 4: bit-acc vs image quality (Q@P in bit-acc terms) ----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4)); [style(a) for a in axs]
for ax, qx, qlab in [(axs[0], (REG_ssim, UM_ssim, 0.678), "SSIM"), (axs[1], (REG_psnr, UM_psnr, 23.8), "PSNR (dB)")]:
    rq, uq, _ = qx
    ax.plot(rq, REG_ba, "-o", color=CR, lw=2, ms=7, label="CtrlRegen+ (s0.1→s0.9)")
    for s, x0, y0 in zip(REG_s, rq, REG_ba): ax.annotate(f"s{s}", (x0, y0), textcoords="offset points", xytext=(5, 6), fontsize=8, color=INK2)
    ax.scatter(uq, UM_ba, color=UM, s=90, marker="s", zorder=5, label="UnMarker")
    for l, x0, y0 in zip(UM_lab, uq, UM_ba): ax.annotate(l, (x0, y0), textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
    ax.set_ylim(0.45, 1.05); ax.set_ylabel("recovered bit-acc"); ax.invert_xaxis()
    ax.set_xlabel(f"image quality after attack — {qlab} (further left = attacker destroyed more)"); bands(ax)
    ax.set_title(f"recovered bit-acc vs {qlab}", fontsize=10.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Q@P (bit-acc) — pushing bit-acc below crypto needs regen to crush the image to SSIM~0.68; UnMarker stays at 0.98 at high quality", fontsize=11, color=INK, y=1.02)
fig.savefig(f"{FG}/fig4_bitacc_qp.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig4_bitacc_qp done")
print("BITACC_FIGS_DONE")
