"""Report figures for the FINAL (post-fix) pipeline. Consumes final_pipeline_bench.json + final_figdata/*.npy
+ rerun_fixed_search.json + region_ablation.json. Validated categorical palette (colorblind-safe):
VINE=blue, TrustMark=green, VideoSeal=orange; attacks: rotation=violet, crop=aqua, CtrlRegen=magenta, UnMarker=red."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

D = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
FG = f"{D}/final_figs"; os.makedirs(FG, exist_ok=True)
B = json.load(open(f"{D}/final_pipeline_bench.json"))
RR = json.load(open(f"{D}/rerun_fixed_search.json"))
RA = json.load(open(f"{D}/region_ablation.json"))
FD = f"{D}/final_figdata"
VINE, TM, VS = "#2a78d6", "#008300", "#eb6834"
CROP, ROT, CR, UM = "#1baf7a", "#4a3aa7", "#d55181", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddddd"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_axisbelow(True)

# ---------- FIG 1: spatial-signal comparison (radial energy + residual maps) ----------
fig = plt.figure(figsize=(11, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.5], wspace=0.28)
ax = fig.add_subplot(gs[0]); style(ax)
prof = B["radial_profiles"]
for nm, col, lab in [("vine", VINE, "VINE"), ("trustmark", TM, "TrustMark"), ("videoseal", VS, "VideoSeal")]:
    p = prof[nm]; e = np.array(p["energy"]); e = e / (e.max() + 1e-9)
    ax.plot(p["r"], e, color=col, lw=2, label=lab)
ax.set_xlabel("normalized radius  (0 = center, 1 = corner)"); ax.set_ylabel("residual energy (peak-normalized)")
ax.set_title("(a) where each watermark puts its signal", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.annotate("VINE: energy on the\nouter ring", xy=(0.9, prof["vine"]["energy"][-2] / max(prof["vine"]["energy"])),
            xytext=(0.42, 0.85), color=VINE, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=VINE, lw=1.2))
gsr = gs[1].subgridspec(1, 3, wspace=0.12)
for k, (nm, lab) in enumerate([("vine", "VINE"), ("trustmark", "TrustMark"), ("videoseal", "VideoSeal")]):
    axr = fig.add_subplot(gsr[k]); res = np.load(f"{FD}/resid_{nm}.npy")
    vmax = np.percentile(res, 99.5)
    axr.imshow(res, cmap="magma", vmin=0, vmax=vmax); axr.set_xticks([]); axr.set_yticks([])
    axr.set_title(lab, fontsize=10, color=INK)
    if k == 0: axr.set_ylabel("|residual|", fontsize=9, color=INK2)
fig.suptitle("Watermark spatial-signal comparison  —  single-fragment residual over the same cover",
             fontsize=12, y=1.0, x=0.5, color=INK)
fig.savefig(f"{FG}/fig1_spatial_signal.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig1 done")

# ---------- FIG 2: ring discovery (ablation) ----------
fig, ax = plt.subplots(figsize=(6.2, 4.0)); style(ax)
ks = [0.9, 0.75, 0.5]; x = np.arange(len(ks)); w = 0.38
cen = [RA["vine"][f"keep_center{k}"]["bit_acc"] for k in ks]
bor = [RA["vine"][f"keep_border{k}"]["bit_acc"] for k in ks]
b1 = ax.bar(x - w / 2, cen, w, color="#c9c9c9", label="keep CENTER (delete ring)")
b2 = ax.bar(x + w / 2, bor, w, color=VINE, label="keep RING (delete center)")
ax.axhline(0.5, color=INK2, ls=":", lw=1); ax.text(2.35, 0.515, "chance", color=INK2, fontsize=8)
for b, v in list(zip(b1, cen)) + list(zip(b2, bor)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"keep {int(k*100)}%" for k in ks])
ax.set_ylabel("VINE bit-accuracy after masking"); ax.set_ylim(0, 1.08)
ax.set_title("VINE's payload lives entirely in a thin border ring", fontsize=11, loc="left", color=INK)
ax.legend(frameon=False, fontsize=9, loc="center left")
fig.savefig(f"{FG}/fig2_ring_discovery.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig2 done")

# ---------- FIG 3: robustness under attack (rotation / crop / CtrlRegen / UnMarker) ----------
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0)); [style(a) for a in axs]
# rotation
rk = sorted(int(k) for k in B["rotation"]); rv = [B["rotation"][str(k)] if str(k) in B["rotation"] else B["rotation"][k] for k in rk]
rv = [B["rotation"][k] if k in B["rotation"] else B["rotation"][str(k)] for k in rk]
axs[0].plot(rk, rv, "-o", color=ROT, lw=2, ms=6); axs[0].set_ylim(0, 1.05)
axs[0].set_xlabel("rotation angle (deg)"); axs[0].set_ylabel("detection rate (n=50)")
axs[0].set_title("(a) rotation — full-circle resync", fontsize=11, loc="left", color=INK)
for xx, yy in zip(rk, rv): axs[0].annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center", color=INK2)
# crop staircase
ck = sorted((float(k) for k in RR["crop_staircase"]), reverse=True); cv = [RR["crop_staircase"][f"{k:g}"] if f"{k:g}" in RR["crop_staircase"] else RR["crop_staircase"][str(k)] for k in ck]
cv = [RR["crop_staircase"][k2] for k2 in [f"{k:g}" for k in ck]]
axs[1].plot([k * 100 for k in ck], cv, "-o", color=CROP, lw=2, ms=6); axs[1].set_ylim(0, 1.05)
axs[1].invert_xaxis(); axs[1].set_xlabel("center-crop kept (%)"); axs[1].set_ylabel("detection rate (n=50)")
axs[1].set_title("(b) center-crop staircase", fontsize=11, loc="left", color=INK)
axs[1].axvline(50, color=INK2, ls=":", lw=1); axs[1].text(51, 0.1, "info-loss floor\n(smallest ring K=.5)", color=INK2, fontsize=8, ha="right")
# CtrlRegen strength + UnMarker on same panel (regen family)
RF = json.load(open(f"{D}/regen_fpr_safe.json"))   # FPR-safe (deployed) regen detection, not the permissive rule
det = {"s03": RF["ctrlregen_s03"]["deployed_fpr_safe"], "s05": RF["ctrlregen_s05"]["deployed_fpr_safe"],
       "s07": RF["ctrlregen_s07"]["deployed_fpr_safe"], "s09": RF["ctrlregen_s09"]["deployed_fpr_safe"]}
sk = [0.3, 0.5, 0.7, 0.9]; sv = [det["s03"], det["s05"], det["s07"], det["s09"]]
axs[2].plot(sk, sv, "-o", color=CR, lw=2, ms=7)   # single series -> identified by x-axis label, no legend
um = [RR["advanced_refixed"]["UnMarker_default"]["detect"], RR["advanced_refixed"]["UnMarker_strong"]["detect"]]
axs[2].set_ylim(0, 1.08); axs[2].set_xlabel("CtrlRegen+ regeneration strength"); axs[2].set_ylabel("detection rate")
axs[2].set_title("(c) diffusion regeneration", fontsize=11, loc="left", color=INK)
for xx, yy in zip(sk, sv):
    if yy is not None: axs[2].annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center", color=INK2)
# UnMarker is a different attack parametrization -> honest comparison is on the QUALITY axis (fig4),
# not a shared "strength" axis. Report it here as a text result, not a fake-x marker.
axs[2].text(0.035, 0.10, f"UnMarker (adversarial scrub)\n  default → {um[0]:.2f}    strong → {um[1]:.2f}\n  (quality axis in Q@P fig)",
            transform=axs[2].transAxes, fontsize=8.5, color=UM, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=UM, lw=0.8))
fig.suptitle("FINAL pipeline robustness  —  nested-VINE + TrustMark + VideoSeal + SyncSeal, crypto-verify gated",
             fontsize=12, color=INK, y=1.02)
fig.savefig(f"{FG}/fig3_robustness.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig3 done")

# ---------- FIG 4: Q@P (detection vs image quality the attacker must pay) ----------
q = B["quality"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0)); [style(a) for a in axs]
# CtrlRegen: detection vs SSIM
cr_ss = [q.get(f"ctrlregen_{s}", {}).get("ssim") for s in ("s03", "s05", "s07", "s09")]
cr_dt = [det["s03"], det["s05"], det["s07"], det["s09"]]
axs[0].plot(cr_ss, cr_dt, "-o", color=CR, lw=2, ms=7, label="CtrlRegen+")
for s, x, y in zip(("s.3", "s.5", "s.7", "s.9"), cr_ss, cr_dt):
    if x is not None and y is not None: axs[0].annotate(s, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK2)
um_ss = [q.get("unmarker_default", {}).get("ssim"), q.get("unmarker_strong", {}).get("ssim")]
axs[0].scatter(um_ss, um, color=UM, s=80, marker="s", zorder=5, label="UnMarker")
for s, x, y in zip(("default", "strong"), um_ss, um):
    if x is not None: axs[0].annotate(s, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
axs[0].set_xlabel("image quality after attack — SSIM (higher = attacker paid less)")
axs[0].set_ylabel("detection rate"); axs[0].set_ylim(0, 1.08); axs[0].invert_xaxis()
axs[0].set_title("(a) Q@P — detection vs SSIM", fontsize=11, loc="left", color=INK)
axs[0].legend(frameon=False, fontsize=9, loc="lower center")
# vs PSNR
cr_p = [q.get(f"ctrlregen_{s}", {}).get("psnr") for s in ("s03", "s05", "s07", "s09")]
um_p = [q.get("unmarker_default", {}).get("psnr"), q.get("unmarker_strong", {}).get("psnr")]
axs[1].plot(cr_p, cr_dt, "-o", color=CR, lw=2, ms=7, label="CtrlRegen+")
axs[1].scatter(um_p, um, color=UM, s=80, marker="s", zorder=5, label="UnMarker")
axs[1].set_xlabel("image quality after attack — PSNR (dB)"); axs[1].set_ylabel("detection rate")
axs[1].set_ylim(0, 1.08); axs[1].invert_xaxis()
axs[1].set_title("(b) Q@P — detection vs PSNR", fontsize=11, loc="left", color=INK)
axs[1].legend(frameon=False, fontsize=9, loc="lower center")
fig.suptitle("Q@P  —  the pipeline holds detection until the attacker has destroyed the image",
             fontsize=12, color=INK, y=1.02)
fig.savefig(f"{FG}/fig4_qp.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig4 done")

# ---------- FIG 5: watermark shape (nested rings + composite) ----------
cover = np.load(f"{FD}/cover.npy").astype(np.uint8)
rnv = np.load(f"{FD}/resid_vine_nested.npy"); rc = np.load(f"{FD}/resid_composite.npy")
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.3))
axs[0].imshow(cover); axs[0].set_title("cover image", fontsize=11, color=INK)
im1 = axs[1].imshow(rnv, cmap="magma", vmin=0, vmax=np.percentile(rnv, 99.5))
axs[1].set_title("nested-VINE residual — 3 concentric rings\n(K = 1.0 / 0.75 / 0.5)", fontsize=11, color=INK)
im2 = axs[2].imshow(rc, cmap="magma", vmin=0, vmax=np.percentile(rc, 99.5))
axs[2].set_title("full composite residual\n(nested-VINE + TrustMark + VideoSeal)", fontsize=11, color=INK)
for a in axs: a.set_xticks([]); a.set_yticks([])
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04).set_label("|residual|", fontsize=9)
fig.suptitle("Watermark shape  —  nested rings give each crop depth its own surviving carrier",
             fontsize=12, color=INK, y=1.01)
fig.savefig(f"{FG}/fig5_watermark_shape.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig5 done")
print("ALL_FIGS_DONE ->", FG)
