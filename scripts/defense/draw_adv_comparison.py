"""Plot advanced-attack (CtrlRegen+ / UnMarker, fidelity-controlled) comparison:
Composite vs VINE vs TrustMark. Detection + bit-acc grouped bars, SSIM annotated."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
rows = [json.loads(l) for l in open(os.path.join(REPO, "results/defense/adv_results.jsonl")) if l.strip()]
methods = ["comp", "vine", "tm"]; mlabel = {"comp": "Composite (ours)", "vine": "VINE", "tm": "TrustMark"}
mcol = {"comp": "#d1495b", "vine": "#4c9f70", "tm": "#e0883b"}
attacks = []
for r in rows:
    if r["attack"] not in attacks: attacks.append(r["attack"])
attacks = sorted(attacks, key=lambda a: (not a.startswith("ctrlregen"), a))
def get(m, a, f):
    for r in rows:
        if r["method"] == m and r["attack"] == a: return r[f]
    return np.nan
def ssim_of(a): return np.nanmean([get(m, a, "ssim") for m in methods])
xlab = [f"{a.replace('ctrlregen_','CtrlRegen+ s=').replace('unmarker','UnMarker')}\n(SSIM {ssim_of(a):.2f})" for a in attacks]

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(attacks)); w = 0.26
for k, m in enumerate(methods):
    det = [get(m, a, "det") for a in attacks]; ba = [get(m, a, "ba") for a in attacks]
    b1 = ax[0].bar(x + (k-1)*w, det, w, color=mcol[m], ec="#222", lw=.6, label=mlabel[m])
    b2 = ax[1].bar(x + (k-1)*w, ba, w, color=mcol[m], ec="#222", lw=.6, label=mlabel[m])
    for r in b1: ax[0].text(r.get_x()+r.get_width()/2, r.get_height(), f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    for r in b2: ax[1].text(r.get_x()+r.get_width()/2, r.get_height(), f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
for a, ttl, yl in [(ax[0], "Detection rate", "detection"), (ax[1], "Bit-accuracy", "bit-acc")]:
    a.set_xticks(x); a.set_xticklabels(xlab, fontsize=9); a.set_title(f"{ttl} under advanced attacks (fidelity-controlled)", fontweight="bold")
    a.set_ylabel(yl); a.legend(); a.grid(axis="y", alpha=.3)
ax[0].set_ylim(0, 1.08); ax[1].set_ylim(0.4, 1.05); ax[1].axhline(0.5, color="grey", ls=":", lw=1)
plt.tight_layout()
out = os.path.join(REPO, "results/defense/adv_comparison.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nADVPLOT_DONE")
