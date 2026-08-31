"""Plot fidelity-constrained results: detection vs attack-induced SSIM, + budget bars."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
d = json.load(open(os.path.join(REPO, "results/defense/fidelity_constrained.json")))
rows = d["rows"]
attacks = []
for r in rows:
    if r["attack"] not in attacks: attacks.append(r["attack"])
cmap = plt.cm.tab10(np.linspace(0, 1, len(attacks)))
col = {a: cmap[i] for i, a in enumerate(attacks)}

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# --- left: detection vs SSIM, line per attack (sorted by SSIM) ---
ax[0].axvspan(0.80, 1.0, color="#cdeccd", alpha=0.5, zorder=0, label="_usable (SSIM>=0.80)")
ax[0].text(0.90, 0.36, "usable image\n(SSIM >= 0.80)", ha="center", fontsize=11, color="#2a7a2a", fontweight="bold")
MK = {"regen": "D--", "rinse": "D--", "rotate": "s--"}
for a in attacks:
    pts = sorted([(r["ssim"], r["det"]) for r in rows if r["attack"] == a])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax[0].plot(xs, ys, MK.get(a, "o-"), color=col[a], label=a, lw=2, ms=8, mec="#333")
ax[0].set_xlabel("attack-induced SSIM  (attacked vs clean watermarked; lower = more image damage)")
ax[0].set_ylabel("composite detection rate")
ax[0].set_title("Detection vs attack image quality\n(curves go left as attack strength rises)", fontweight="bold")
ax[0].set_ylim(0.45, 1.03); ax[0].invert_xaxis(); ax[0].grid(alpha=.3); ax[0].legend(ncol=2, fontsize=9, loc="lower left")

# --- right: detection within an attack-quality budget ---
buds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
det = []
for b in buds:
    p = [r["det"] for r in rows if r["ssim"] >= b]
    det.append(np.mean(p) if p else np.nan)
bars = ax[1].bar([f">={b:.2f}" for b in buds], det, color="#4c9f70", ec="#333", lw=1.2, width=0.62)
for bar, v in zip(bars, det):
    if not np.isnan(v): ax[1].text(bar.get_x()+bar.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax[1].axhline(1.0, color="grey", ls=":", lw=1)
ax[1].set_xlabel("attack image-quality budget (attacked SSIM >= ...)")
ax[1].set_ylabel("mean detection rate over attacks within budget")
ax[1].set_title("Within a quality budget, the attacker cannot remove the mark", fontweight="bold")
ax[1].set_ylim(0.6, 1.03); ax[1].grid(axis="y", alpha=.3)

plt.tight_layout()
out = os.path.join(REPO, "results/defense/fidelity_constrained.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nPLOT_DONE")
