"""Multi-angle plots for the fidelity-constrained baseline comparison.
6 panels: overall detection, overall bit-acc, det heatmap, ba heatmap,
detection vs SSIM-budget (per method), per-attack detection grouped bars."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
d = json.load(open(os.path.join(REPO, "results/defense/baseline_comparison.json")))
rows = d["rows"]; methods = d["methods"]
attacks = []
for r in rows:
    if r["attack"] not in attacks: attacks.append(r["attack"])

def agg(method, field, attack=None, budget=None):
    rs = [r for r in rows if r["method"] == method and (attack is None or r["attack"] == attack) and (budget is None or r["ssim"] >= budget)]
    return float(np.mean([r[field] for r in rs])) if rs else np.nan

OURS = "Composite"
colors = {m: ("#d1495b" if m == OURS else plt.cm.tab10(i / 10)) for i, m in enumerate(methods)}
mcol = [colors[m] for m in methods]

fig, ax = plt.subplots(2, 3, figsize=(19, 11))

# A: overall detection per method
det_all = [agg(m, "det") for m in methods]
b = ax[0, 0].bar(methods, det_all, color=mcol, ec="#222", lw=1.2)
for r, v in zip(b, det_all): ax[0, 0].text(r.get_x()+r.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
ax[0, 0].set_title("Overall detection rate (all attacks)", fontweight="bold"); ax[0, 0].set_ylim(0, 1.08); ax[0, 0].grid(axis="y", alpha=.3)
ax[0, 0].tick_params(axis="x", rotation=25)

# B: overall bit-acc per method
ba_all = [agg(m, "ba") for m in methods]
b = ax[0, 1].bar(methods, ba_all, color=mcol, ec="#222", lw=1.2)
for r, v in zip(b, ba_all): ax[0, 1].text(r.get_x()+r.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
ax[0, 1].axhline(0.5, color="grey", ls=":", lw=1); ax[0, 1].set_title("Overall bit-accuracy (all attacks)", fontweight="bold")
ax[0, 1].set_ylim(0.4, 1.03); ax[0, 1].grid(axis="y", alpha=.3); ax[0, 1].tick_params(axis="x", rotation=25)

# C: detection heatmap method x attack
M = np.array([[agg(m, "det", attack=a) for a in attacks] for m in methods])
im = ax[0, 2].imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax[0, 2].set_xticks(range(len(attacks))); ax[0, 2].set_xticklabels(attacks, rotation=40, ha="right")
ax[0, 2].set_yticks(range(len(methods))); ax[0, 2].set_yticklabels(methods)
for i in range(len(methods)):
    for j in range(len(attacks)): ax[0, 2].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax[0, 2].set_title("Detection rate  (method x attack)", fontweight="bold"); fig.colorbar(im, ax=ax[0, 2], fraction=.046)

# D: bit-acc heatmap
B = np.array([[agg(m, "ba", attack=a) for a in attacks] for m in methods])
im2 = ax[1, 0].imshow(B, cmap="RdYlGn", vmin=0.5, vmax=1, aspect="auto")
ax[1, 0].set_xticks(range(len(attacks))); ax[1, 0].set_xticklabels(attacks, rotation=40, ha="right")
ax[1, 0].set_yticks(range(len(methods))); ax[1, 0].set_yticklabels(methods)
for i in range(len(methods)):
    for j in range(len(attacks)): ax[1, 0].text(j, i, f"{B[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax[1, 0].set_title("Bit-accuracy  (method x attack)", fontweight="bold"); fig.colorbar(im2, ax=ax[1, 0], fraction=.046)

# E: detection vs SSIM budget, per method
buds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
for m in methods:
    ys = [agg(m, "det", budget=b) for b in buds]
    ax[1, 1].plot([f"{b:.2f}" for b in buds], ys, "o-" if m != OURS else "D-",
                  color=colors[m], lw=2.5 if m == OURS else 1.8, ms=8, label=m, zorder=5 if m == OURS else 2)
ax[1, 1].set_title("Detection vs attack-quality budget (SSIM >= x)", fontweight="bold")
ax[1, 1].set_xlabel("attacked SSIM >= ..."); ax[1, 1].set_ylabel("mean detection within budget")
ax[1, 1].set_ylim(0, 1.05); ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=9)

# F: per-attack detection grouped bars
x = np.arange(len(attacks)); w = 0.8 / len(methods)
for k, m in enumerate(methods):
    vals = [agg(m, "det", attack=a) for a in attacks]
    ax[1, 2].bar(x + k * w - 0.4 + w/2, vals, w, color=colors[m], label=m, ec="#222", lw=0.4)
ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels(attacks, rotation=40, ha="right")
ax[1, 2].set_title("Detection rate per attack", fontweight="bold"); ax[1, 2].set_ylim(0, 1.08)
ax[1, 2].grid(axis="y", alpha=.3); ax[1, 2].legend(fontsize=8, ncol=2)

fig.suptitle(f"Fidelity-constrained baseline comparison (n={d['n']}, alpha={d['alpha']}):  OURS (Composite) vs single fragments vs classical",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(REPO, "results/defense/baseline_comparison.png")
plt.savefig(out, dpi=120); print(f"[saved] {out}\nBLPLOT_DONE")
