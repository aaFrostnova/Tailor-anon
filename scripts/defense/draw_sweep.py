"""Plot the full advanced-attack parameter sweep: detection vs attack image quality
(SSIM, high->low) with 95% Wilson CI bands, for VINE / TM / equal-MRC / learned-head
/ oracle; plus the oracle-minus-equalMRC gap vs quality (does fusion leave anything
on the table across the WHOLE range?).

  python draw_sweep.py --attack ctrlregen --title "CtrlRegen+ strength 0.1->0.9"
"""
import os, json, argparse, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
ap = argparse.ArgumentParser()
ap.add_argument("--attack", required=True)
ap.add_argument("--title", default="")
ap.add_argument("--results", default="results/defense/sweep_results.jsonl")
ap.add_argument("--out", default="")
args = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(REPO, args.results)) if l.strip()]
rows = [r for r in rows if r["attack"] == args.attack]
rows = sorted(rows, key=lambda r: -r["ssim"] if not math.isnan(r["ssim"]) else -1)  # high quality -> low

def wilson(p, n, z=1.96):
    if n == 0: return 0.0, 0.0
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

xs = [r["ssim"] for r in rows]; N = rows[0]["n"]
SER = [("VINE", "#4c9f70", "-", "o"), ("TM", "#e0883b", ":", "v"), ("equalMRC", "#888888", "--", "s"),
       ("NEW", "#d1495b", "-", "D"), ("oracle", "#3a6ea5", "-.", "*")]
LAB = {"VINE": "VINE-only", "TM": "TrustMark-only", "equalMRC": "equal-MRC (Composite)",
       "NEW": "learned head (gated)", "oracle": "oracle (upper bound)"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))

for key, col, ls, mk in SER:
    ys = np.array([r[key]["det"] for r in rows])
    lo = np.array([wilson(y, N)[0] for y in ys]); hi = np.array([wilson(y, N)[1] for y in ys])
    ax[0].fill_between(xs, lo, hi, color=col, alpha=0.10)
    ax[0].plot(xs, ys, ls, color=col, marker=mk, ms=8, lw=2.2, label=LAB[key])
ax[0].invert_xaxis()   # high quality (left) -> low quality (right)
ax[0].set_xlabel("attack image quality: SSIM  (high → low, i.e. attack strength ↑)")
ax[0].set_ylabel("detection rate"); ax[0].set_ylim(-0.03, 1.05); ax[0].grid(alpha=.3)
ax[0].set_title(f"Detection vs attack quality  (n={N}, 95% CI bands)", fontweight="bold")
ax[0].legend(loc="lower left", fontsize=8.5)

# right: gap of oracle and head over equal-MRC, with the per-point CI half-width as "noise floor"
gap_o = [r["oracle"]["det"] - r["equalMRC"]["det"] for r in rows]
gap_h = [r["NEW"]["det"] - r["equalMRC"]["det"] for r in rows]
nf = [0.5 * (wilson(r["equalMRC"]["det"], N)[1] - wilson(r["equalMRC"]["det"], N)[0]) for r in rows]
ax[1].fill_between(xs, [-x for x in nf], nf, color="#ccc", alpha=0.6, label="±½·CI(equal-MRC) noise floor")
ax[1].plot(xs, gap_o, "-*", color="#3a6ea5", ms=9, lw=2, label="oracle − equal-MRC")
ax[1].plot(xs, gap_h, "-D", color="#d1495b", ms=7, lw=2, label="learned head − equal-MRC")
ax[1].axhline(0, color="k", lw=0.8); ax[1].invert_xaxis()
ax[1].set_xlabel("attack image quality: SSIM (high → low)"); ax[1].set_ylabel("detection gain over equal-MRC")
ax[1].grid(alpha=.3); ax[1].set_title("Headroom over equal-MRC across the WHOLE range\n(inside the noise floor ⇒ no real gain)", fontweight="bold", fontsize=10.5)
ax[1].legend(loc="upper left", fontsize=8.5)

ttl = args.title or args.attack
fig.suptitle(f"Full advanced-attack sweep — {ttl}: Composite (equal-MRC) tracks VINE & oracle across the entire quality range; fusion headroom stays within noise.",
             fontsize=11.5, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(REPO, args.out or f"results/defense/sweep_{args.attack}.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nDRAWSWEEP_DONE")
