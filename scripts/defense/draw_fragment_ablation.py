"""Fragment-count ablation bar chart (documented results, COCO n=200, settled 2026-06-14).
4-fragment (VINE+DFT-Kred+Quant-QIM+TrustMark) vs 2-fragment (VINE+TrustMark).
DFT-Kred & Quant-QIM contributed delta~0 to the fusion (slightly negative under rinse)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
labels = ["4-fragment\n(+DFT +QIM)", "2-fragment\n(VINE+TrustMark)"]
psnr = [32.21, 34.07]
fpr = [0.005, 0.000]
rinse2x = [0.55, 0.75]
rinse4x = [0.15, 0.26]
colors = ["#bdbdbd", "#4c9f70"]

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
def bars(a, vals, title, ylab, fmt="{:.2f}"):
    b = a.bar(labels, vals, color=colors, ec="#333", lw=1.2, width=0.6)
    a.set_title(title, fontweight="bold"); a.set_ylabel(ylab); a.grid(axis="y", alpha=.3)
    for r, v in zip(b, vals): a.text(r.get_x()+r.get_width()/2, v, fmt.format(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

bars(ax[0], psnr, "Quality (PSNR @ matched embed)", "PSNR (dB)", "{:.2f}"); ax[0].set_ylim(31, 35)
bars(ax[1], fpr, "False-positive rate", "FPR", "{:.3f}"); ax[1].set_ylim(0, 0.007)
x = np.arange(2); w = 0.35
b1 = ax[2].bar(x - w/2, rinse2x, w, color="#4c9f70", ec="#333", label="rinse2x")
b2 = ax[2].bar(x + w/2, rinse4x, w, color="#a7d3b6", ec="#333", label="rinse4x")
ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].set_title("Crypto-ID under rinse (heavy regen)", fontweight="bold")
ax[2].set_ylabel("crypto-ID recovery"); ax[2].grid(axis="y", alpha=.3); ax[2].legend()
for bb in [b1, b2]:
    for r in bb: ax[2].text(r.get_x()+r.get_width()/2, r.get_height(), f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Fragment-count ablation: dropping DFT-Kred & Quant-QIM (delta~0 contribution) is STRICTLY better", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(REPO, "results/defense/fragment_ablation.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nABLATION_DONE")
