"""Quantitative result charts for the report (matplotlib, English labels — no CJK font on box).
  A fig_compare_bars.png  : cross-method bit-acc (Ours-2f vs 5 baselines) over attacks
  B fig_role_heatmap.png  : per-fragment bit-acc heatmap (attacks x {VINE,DFT,QIM,TM}) — roles
  C fig_kid_scatter.png   : our detection vs attacked-image quality (KID) — survive high-quality attacks
  D fig_ctrlregen_curve.png: detection vs CtrlRegen+ strength + UnMarker
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/figures"
L = lambda n: json.load(open(os.path.join(R, n)))
ours = L("composite_vine_tm.json")["attacks"]
base = L("baselines_fullsuite.json")["defenses"]
f4 = L("composite_4fused_n200.json")["attacks"]
kid = L("attack_fid_kid.json")
cr = {s: L(f"composite_4fused_ctrlregen_{t}.json")["summary"] for s, t in [("0.3","03"),("0.5","05"),("0.7","07")]}
um = L("composite_4fused_unmarker.json")["summary"]

# ---------- A: cross-method bars ----------
atk = ["clean","jpeg","bm3d","regen","rinse4x","vae_c","hflip","crop75","rot9"]
lab = ["clean","jpeg","bm3d","regen","rinse4x","vae_c","hflip","crop75","rot9"]
methods = [("Ours (VINE+TM)", lambda a: ours[a]["fused_ba"], "#1f77b4"),
           ("VINE-R", lambda a: base["VINE-R"][a]["bit_acc"], "#ff7f0e"),
           ("TrustMark-B", lambda a: base["TrustMark-B"][a]["bit_acc"], "#2ca02c"),
           ("RivaGAN", lambda a: base["RivaGAN"][a]["bit_acc"], "#d62728"),
           ("DwtDctSvd", lambda a: base["DwtDctSvd"][a]["bit_acc"], "#9467bd"),
           ("DwtDct", lambda a: base["DwtDct"][a]["bit_acc"], "#8c564b")]
x = np.arange(len(atk)); w = 0.13
fig, ax = plt.subplots(figsize=(13, 5))
for i, (nm, fn, c) in enumerate(methods):
    ax.bar(x + (i - 2.5) * w, [fn(a) for a in atk], w, label=nm, color=c)
ax.axhline(0.5, ls="--", lw=0.8, color="gray"); ax.text(len(atk)-0.5, 0.51, "chance", color="gray", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=20); ax.set_ylabel("bit-accuracy"); ax.set_ylim(0.45, 1.03)
ax.set_title("Cross-method robustness (bit-acc) — only Ours covers regen AND geometry")
ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28)); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_compare_bars.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
print("A done")

# ---------- B: per-fragment role heatmap (4 fragments) ----------
atk2 = ["clean","jpeg","noise","bright","contrast","bm3d","regen","rinse2x","rinse4x","vae_b","vae_c","rs256","hflip","crop75","crop50","rot9","crop_jpeg"]
frs = ["vine","dft","qim","trustmark"]; frl = ["VINE","DFT-Kred","Quant-QIM","TrustMark-B"]
M = np.array([[f4[a][fr] for a in atk2] for fr in frs])
fig, ax = plt.subplots(figsize=(13, 3.2))
im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(atk2))); ax.set_xticklabels(atk2, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(frs))); ax.set_yticklabels(frl)
for i in range(len(frs)):
    for j in range(len(atk2)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="black" if M[i,j] > 0.62 else "white")
ax.set_title("Per-fragment bit-acc (n=200) — VINE carries regen, TrustMark carries geometry; DFT/QIM redundant")
fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="bit-acc")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_role_heatmap.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
print("B done")

# ---------- C: detection vs attack quality (KID) ----------
qmap = {"clean":"watermarked","jpeg":"jpeg","blur":"blur","noise":"noise","bright":"bright","contrast":"contrast",
        "bm3d":"bm3d","regen":"regen","rinse2x":"rinse2x","rinse4x":"rinse4x","vae_b":"vae_b","vae_c":"vae_c",
        "rs256":"rs256","hflip":"hflip","crop75":"crop75","crop50":"crop50","rot9":"rot9","crop_jpeg":"crop_jpeg"}
pts = []
for a, qk in qmap.items():
    if a in ours and qk in kid:
        pts.append((kid[qk]["kid"]*1e3, ours[a]["composite_or"], a))
# add advanced
for s in ["0.3","0.5","0.7"]:
    pts.append((kid[f"ctrlregen_{s}"]["kid"]*1e3, cr[s]["composite_or"], f"CtrlR+{s}"))
pts.append((5.97, um["composite_or"], "UnMarker"))  # unmarker KID measured separately (n=10)
fig, ax = plt.subplots(figsize=(9, 6))
for xk, yd, a in pts:
    col = "#2ca02c" if yd >= 0.9 else ("#ff7f0e" if yd >= 0.5 else "#d62728")
    ax.scatter(xk, yd, c=col, s=45); ax.annotate(a, (xk, yd), fontsize=7, xytext=(4,3), textcoords="offset points")
ax.axhline(0.9, ls="--", lw=0.7, color="gray")
ax.set_xlabel("attacked-image deviation from natural  (KID ×10³, lower = higher quality attack)")
ax.set_ylabel("our composite detection")
ax.set_title("Detection vs attack quality — we survive high-quality (low-KID) attacks except\ntransformation/regeneration (geometric, CtrlRegen+, UnMarker)")
ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_kid_scatter.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
print("C done")

# ---------- D: CtrlRegen+ strength curve + UnMarker ----------
st = [0.3, 0.5, 0.7]; det = [cr[f"{s}"]["composite_or"] for s in st]; ba = [cr[f"{s}"]["fused_ba"] for s in st]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(st, det, "o-", label="composite detection", color="#1f77b4")
ax.plot(st, ba, "s--", label="fused bit-acc", color="#ff7f0e")
ax.scatter([0.9], [um["composite_or"]], c="#d62728", s=60, zorder=5)
ax.annotate("UnMarker (det 0.10)", (0.9, um["composite_or"]), fontsize=8, xytext=(-90,10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.axhline(0.5, ls=":", color="gray", lw=0.8)
ax.set_xlabel("CtrlRegen+ strength  (+ UnMarker at right)"); ax.set_ylabel("rate")
ax.set_title("Advanced attacks: composite degrades gracefully with CtrlRegen+ strength")
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ctrlregen_curve.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
print("D done\nPLOTS_DONE")
