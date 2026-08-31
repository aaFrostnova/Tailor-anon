"""3-fragment (VINE+TM+MaskWM-ED) CtrlRegen+ sweep figure: does the ED fragment add
anything? Left = detection vs attack SSIM for VINE / TM / ED / 2frag(V+T) / 3frag(V+T+E)
/ oracle3 with 95% CI. Right = marginal value 3frag-minus-2frag vs SSIM against the
noise floor. Expectation (ED is pixel-space, dies under regen like TM): ED & TM collapse
together, 3frag tracks 2frag tracks VINE, marginal ED ~ 0.
"""
import os, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
rows = [json.loads(l) for l in open(os.path.join(REPO, "results/defense/sweep3_results.jsonl")) if l.strip()]
rows = [r for r in rows if r["attack"] == "ctrlregen3"]
rows = sorted(rows, key=lambda r: -r["ssim"] if not math.isnan(r["ssim"]) else -1)
def wil(p, n, z=1.96):
    if n == 0: return 0, 0
    d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)
xs = [r["ssim"] for r in rows]; N = rows[0]["n"]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
SER = [("VINE", "#4c9f70", "-", "o"), ("TM", "#e0883b", ":", "v"), ("ED", "#9b59b6", ":", "P"),
       ("2frag", "#888888", "--", "s"), ("3frag", "#d1495b", "-", "D"), ("oracle3", "#3a6ea5", "-.", "*")]
LAB = {"VINE": "VINE-only", "TM": "TrustMark-only", "ED": "MaskWM-ED-only",
       "2frag": "2-frag (VINE+TM)", "3frag": "3-frag (VINE+TM+ED)", "oracle3": "oracle (3-frag)"}
for k, col, ls, mk in SER:
    ys = np.array([r[k]["det"] for r in rows])
    lo = np.array([wil(y, N)[0] for y in ys]); hi = np.array([wil(y, N)[1] for y in ys])
    if k in ("3frag", "VINE"): ax[0].fill_between(xs, lo, hi, color=col, alpha=0.10)
    ax[0].plot(xs, ys, ls, color=col, marker=mk, ms=8, lw=2.2, label=LAB[k])
ax[0].invert_xaxis(); ax[0].set_xlabel("attack image quality: SSIM (high → low)")
ax[0].set_ylabel("detection rate"); ax[0].set_ylim(-0.03, 1.05); ax[0].grid(alpha=.3)
ax[0].set_title(f"3-frag CtrlRegen+ sweep (n={N}): ED & TM both die; 3-frag tracks VINE", fontweight="bold", fontsize=10.5)
ax[0].legend(loc="lower left", fontsize=8.5)

gap = [r["3frag"]["det"] - r["2frag"]["det"] for r in rows]
nf = [0.5*(wil(r["2frag"]["det"], N)[1] - wil(r["2frag"]["det"], N)[0]) for r in rows]
ax[1].fill_between(xs, [-x for x in nf], nf, color="#ccc", alpha=0.6, label="±½·CI(2-frag) noise floor")
ax[1].plot(xs, gap, "-D", color="#d1495b", ms=8, lw=2, label="3-frag − 2-frag (marginal ED)")
ax[1].axhline(0, color="k", lw=0.8); ax[1].invert_xaxis()
ax[1].set_xlabel("attack image quality: SSIM (high → low)"); ax[1].set_ylabel("detection gain from adding ED")
ax[1].grid(alpha=.3); ax[1].set_ylim(-0.5, 0.5)
ax[1].set_title("Marginal value of the ED fragment (inside noise floor ⇒ none)", fontweight="bold", fontsize=10.5)
ax[1].legend(loc="upper left", fontsize=8.5)
fig.suptitle("MaskWM-ED as a 3rd fragment under CtrlRegen+: ED dies (pixel-space) AND POISONS equal-MRC — 3-frag < 2-frag (0.62→0.35 @s0.7).\n"
             "Its dead LLRs are confidently-WRONG & high-magnitude (unlike regen-TM); oracle3 ≈ VINE recovers it → needs reliability gating, not equal weight. Costs ~1.7dB.",
             fontsize=10.5, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(REPO, "results/defense/sweep3_ctrlregen.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nDRAW3_DONE")
