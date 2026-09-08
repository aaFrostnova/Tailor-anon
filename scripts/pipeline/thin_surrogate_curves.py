"""Thin each fitted curve to the fewest knots that still represent it, per curve.

The fitted curves are emitted densely, but every segment a curve carries becomes a case split
inside the solver, and solve time grows superlinearly in the number of segments -- measured at 7.0 s
for 5 knots, 17.4 s for 9 and 62.2 s for 17 on the same request. Spending those segments uniformly
is wasteful, because most curves are nearly straight and only a minority carry a knee.

So each curve is simplified independently (Douglas-Peucker) down to the coarsest polyline that stays
within a per-quantity tolerance of the fitted curve. A straight curve collapses to two knots; a
curve with a knee keeps knots where the knee is.

Usage: python thin_surrogate_curves.py [in.json] [out.json]
"""
import os
import sys, json
import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k") + "/surrogate_fitted.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else (os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k") + "/surrogate_thinned.json")

# tolerance in the units of each quantity: bit-accuracy, interference, MSE, MSE, bits
TOL = {"base": 0.005, "delta": 0.002, "d": 0.05, "e": 0.02, "cap": 0.5, "frontend": 0.005}

def tol_for(blk, key, tol):
    """The front-end block mixes units. Its effect and control curves are bit-accuracy rates and
    take the base tolerance; its fidelity penalties are MSE and belong on `d`'s scale. Splitting
    the geometric front-ends into one curve per stage took this block from about seven curves to
    over a hundred, and each retained segment is a Boolean the solver carries on every request that
    names the column -- which is the same reason the other blocks are thinned."""
    if blk == "frontend" and (key.startswith("penalty_") or key == "nested_penalty"):
        return TOL["d"]
    return tol

def dp(x, y, tol):
    """Douglas-Peucker on the vertical deviation: keep the point that deviates most, recurse."""
    keep = np.zeros(len(x), bool); keep[0] = keep[-1] = True
    stack = [(0, len(x) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2: continue
        seg = np.interp(x[i:j + 1], [x[i], x[j]], [y[i], y[j]])
        dev = np.abs(y[i:j + 1] - seg); k = int(np.argmax(dev))
        if dev[k] > tol:
            keep[i + k] = True; stack += [(i, i + k), (i + k, j)]
    return keep

d = json.load(open(IN))
before = after = 0; stats = {}
for blk, tol in TOL.items():
    if blk not in d: continue
    ks = []
    for key, v in d[blk].items():
        # the front-end block also carries free-form provenance entries (search-grid notes, the
        # SyncSeal status record); Surrogate.from_dict skips anything without {xs,ys} and so must this
        if not (isinstance(v, dict) and "xs" in v and "ys" in v):
            continue
        x = np.array(v["xs"], float); y = np.array(v["ys"], float)
        before += len(x)
        if len(x) <= 2:
            after += len(x); ks.append(len(x)); continue
        m = dp(x, y, tol_for(blk, key, tol))
        d[blk][key] = {"xs": [float(t) for t in x[m]], "ys": [float(t) for t in y[m]]}
        after += int(m.sum()); ks.append(int(m.sum()))
    stats[blk] = {"curves": len(ks), "knots_mean": round(float(np.mean(ks)), 2),
                  "knots_max": int(np.max(ks)), "tol": tol}
    print(f"  {blk:6s} {len(ks):3d} curves -> knots mean {np.mean(ks):.2f} max {np.max(ks)} (tol {tol})")
print(f"total knots {before} -> {after}  ({after/before:.2f}x)")
d["source"] = {**d.get("source", {}), "curve_thinning": {"tolerances": TOL, "per_block": stats,
               "knots_before": before, "knots_after": after,
               "note": "each fitted curve simplified to the coarsest polyline within tolerance, so "
                       "segments are spent where a curve actually bends instead of uniformly."}}
json.dump(d, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
