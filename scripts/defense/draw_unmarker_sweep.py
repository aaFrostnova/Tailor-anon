"""UnMarker fidelity-threshold sweep: composite/VINE/TM detection vs attack SSIM,
split into the pure-spectral arm (no crop) and the crop arm (geometric)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
rows = [json.loads(l) for l in open(os.path.join(REPO, "results/defense/adv_unmarker_sweep.jsonl")) if l.strip()]
CROP = {"c95", "c90", "c85"}
def g(method, tag, f):
    for r in rows:
        if r["method"] == method and r["attack"] == tag: return r[f]
    return None
tags = ["nocrop_l", "nocrop_h", "c95", "c90", "c85"]
col = {"comp": "#d1495b", "vine": "#4c9f70", "tm": "#e0883b"}
lab = {"comp": "Composite", "vine": "VINE", "tm": "TrustMark"}
fig, ax = plt.subplots(figsize=(11, 6))
ax.axvspan(0.0, 0.45, color="#eee0c0", alpha=0.5); ax.text(0.22, 0.55, "crop arm\n(geometric — pixel-SSIM\nmisleadingly low)", ha="center", fontsize=10, color="#8a6d00")
ax.axvspan(0.45, 1.0, color="#d8ead8", alpha=0.5); ax.text(0.83, 0.55, "pure-spectral arm\n(no crop — real fidelity)", ha="center", fontsize=10, color="#2a7a2a")
off = {"comp": 0.025, "vine": 0.0, "tm": -0.025}
for m in ["vine", "tm", "comp"]:                     # comp drawn last (on top)
    xs = [g(m, t, "ssim") for t in tags]; ys = [g(m, t, "det") + off[m] for t in tags]
    mk = ["o" if t not in CROP else "s" for t in tags]
    for x, y, k in zip(xs, ys, mk):
        ax.scatter(x, y, s=170, color=col[m], marker=k, ec="#222", lw=1.4, zorder=4 + (m == "comp"))
    ax.plot([], [], "o", color=col[m], label=lab[m])  # legend proxy
for t in tags:
    x = g("comp", t, "ssim"); y = g("comp", t, "det") + off["comp"]
    nm = {"nocrop_l": "spec×400", "nocrop_h": "spec×2500", "c95": "crop5%", "c90": "crop10%", "c85": "crop15%"}[t]
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8, color="#555")
ax.plot([], [], "ko", label="● no crop"); ax.plot([], [], "ks", label="■ with crop")
ax.set_xlabel("attack-induced SSIM (lower = more pixel change)"); ax.set_ylabel("detection rate")
ax.set_title("UnMarker: ONLY the crop arm breaks the composite; pure spectral never does\n"
             "(VINE stays ba 0.97 even at heaviest spectral; crop's low SSIM is a geometric artifact)", fontweight="bold", fontsize=11)
ax.set_ylim(-0.05, 1.08); ax.invert_xaxis(); ax.grid(alpha=.3); ax.legend(loc="center left", fontsize=9)
plt.tight_layout()
out = os.path.join(REPO, "results/defense/unmarker_sweep.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nUMPLOT_DONE")
