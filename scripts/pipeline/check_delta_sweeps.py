"""Consistency of the new interference sweeps with the mid-strength constants they replace.

Every constant was measured with both fragments at their mid strengths; every sweep holds the host at the
same mid strength and passes through the overwriting fragment's mid knot, so the two are independent
measurements of the same number (different runs, N=100 or 50 against N=100). Prints the disagreement per
cell and flags anything beyond 0.03 (about two standard errors of the difference)."""
import os
import json, sys
import numpy as np
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
new = json.load(open(f"{SC}/surrogate_canonical_raw.json"))["delta"]
old = json.load(open(f"{SC}/surrogate_canonical.pre_completion_20260908.json"))["delta"]
MID = {"VINE": 0.6, "TrustMark": 1.0, "VideoSeal": 1.0}
rows = []
for k, c in new.items():
    g = k.split("|")[0]; o = old.get(k)
    if o is None or len(c["xs"]) < 5 or len(o["xs"]) > 2: continue          # only sweeps that replaced a constant
    rows.append((k, float(np.interp(MID[g], c["xs"], c["ys"])), o["ys"][0]))
d = np.array([r[1] - r[2] for r in rows]) if rows else np.array([0.0])
print(f"{len(rows)} sweeps replaced a constant; sweep@mid - constant: mean {d.mean():+.4f}, max |diff| {abs(d).max():.3f}")
bad = [r for r in rows if abs(r[1] - r[2]) > 0.03]
for k, s, c in sorted(rows, key=lambda r: -abs(r[1] - r[2]))[:8]: print(f"   {k:34s} sweep@mid {s:.3f}  constant {c:.3f}  diff {s - c:+.3f}")
print("DELTA_CHECK_OK" if not bad else f"DELTA_CHECK_FLAGGED {len(bad)}: " + ", ".join(r[0] for r in bad))
