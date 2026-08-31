"""Figure (honest): does the learned fusion head help under one-fragment-death
(advanced regen) attacks?  Answer: no reliable gain once you control sample size.

Left  : n=48 fresh-image multi-round regen. VINE / equal-MRC / learned-head /
        oracle overlap within 95% CI -> fusion neither helps nor hurts; oracle
        ~= VINE-only => the dead fragment carries no recoverable info.
Right  : the apparent n=12 CtrlRegen+ 'poisoning' (equal 0.75 vs VINE 0.92) vs
        the MATCHED-statistics n=48 regen-x2 (VINE ba 0.72 identical): the gap
        collapses with 4x the samples + tight CIs -> it was sampling noise.
"""
import os, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
big = json.load(open(os.path.join(REPO, "results/defense/fusion_adv_bign.json")))
adv = json.load(open(os.path.join(REPO, "results/defense/fusion_adv_eval.json")))

def wilson(p, n, z=1.96):
    if n == 0: return 0.0, 0.0
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)
def err(p, n): lo, hi = wilson(p, n); return [[p - lo], [hi - p]]

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

# ---- left: n=48 regen, detector curves with CI bands ----
xs = [r["round"] for r in big]; N = big[0]["n"]
SER = [("VINE", "#4c9f70", "-", "o"), ("equalMRC", "#888888", "--", "s"),
       ("NEW", "#d1495b", "-", "D"), ("oracle", "#3a6ea5", "-.", "*")]
LAB = {"VINE": "VINE-only", "equalMRC": "equal-MRC (Composite)", "NEW": "learned head (gated)", "oracle": "oracle"}
for key, col, ls, mk in SER:
    ys = np.array([r[key]["det"] for r in big])
    lo = np.array([wilson(y, N)[0] for y in ys]); hi = np.array([wilson(y, N)[1] for y in ys])
    ax[0].fill_between(xs, lo, hi, color=col, alpha=0.12)
    ax[0].plot(xs, ys, ls, color=col, marker=mk, ms=9, lw=2.2, label=LAB[key])
ax[0].set_xticks(xs); ax[0].set_xlabel("regen rounds  (TrustMark dead @ ba≈0.50, VINE fades)")
ax[0].set_ylabel("detection rate"); ax[0].set_ylim(0.45, 1.02); ax[0].grid(alpha=.3)
ax[0].set_title(f"Multi-round regen, n={N} fresh disjoint imgs\nall detectors overlap within 95% CI — head gives no gain", fontweight="bold", fontsize=10.5)
ax[0].legend(loc="lower left", fontsize=9)

# ---- right: n=12 CtrlRegen+ vs matched n=48 regen-x2 (the noise demo) ----
creg05 = [r for r in adv if r["label"] == "CtrlRegen+ s=0.5"][0]
r2 = [r for r in big if r["round"] == 2][0]
groups = ["VINE-only", "equal-MRC", "learned head"]
left_v = [creg05["vine_det"], creg05["eq_det"], creg05["hd_det"]]; nL = creg05["n"]
right_v = [r2["VINE"]["det"], r2["equalMRC"]["det"], r2["NEW"]["det"]]; nR = r2["n"]
x = np.arange(3); w = 0.36
b1 = ax[1].bar(x - w/2, left_v, w, color="#c44", label=f"real CtrlRegen+ s=0.5 (n={nL})",
               yerr=np.hstack([err(p, nL) for p in left_v]), capsize=5, ecolor="#333")
b2 = ax[1].bar(x + w/2, right_v, w, color="#48a", label=f"matched regen×2 (n={nR}, VINE ba≈0.72 ≡)",
               yerr=np.hstack([err(p, nR) for p in right_v]), capsize=5, ecolor="#333")
for b in list(b1) + list(b2):
    ax[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=8.5)
ax[1].set_xticks(x); ax[1].set_xticklabels(groups); ax[1].set_ylabel("detection rate")
ax[1].set_ylim(0, 1.12); ax[1].grid(axis="y", alpha=.3)
ax[1].set_title("Same fragment statistics, 4× samples:\nthe n=12 'poisoning' gap collapses → it was sampling noise", fontweight="bold", fontsize=10.5)
ax[1].legend(loc="lower center", fontsize=8.5)

fig.suptitle("Learned fusion head under ADVANCED (one-fragment-death) attacks: no reliable improvement.\n"
             "Dead fragment |LLR|≈0.25 ≪ VINE≈1.5 (too weak to poison); oracle ≈ VINE-only ⇒ zero recoverable info ⇒ fusion gain ≈ 0; FPR=0 preserved.",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(REPO, "results/defense/fusion_adv.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nDRAW_DONE")
