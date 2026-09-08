"""Fold perimage/*/*.json into ONE block the surrogate loads, and check it against the curves.

Output: surrogate_perimage.json = {"F|a" or "scale:F|a": {"xs": knots, "ba": [[per image] per knot],
"ver": same shape or null, "n": images, "embed": how the campaign embedded}}. Every cell's per-image
mean is compared with the canonical curve at the same knot: the campaigns that re-decoded retained
images must reproduce the stored mean exactly, and the in-process re-run (composite codeword instead
of random bits, same images) must agree within sampling noise.
"""
import json, glob, os, sys
import numpy as np
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
sys.path.insert(0, os.path.join(os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "scripts", "defense"))
from surrogate_model import PWL
cells = {}          # key -> {s: (ba, ver, embed)}
for p in sorted(glob.glob(f"{SC}/perimage/*/*.json") + glob.glob(f"{SC}/perimage_ext/*/*.json")):
    fam = os.path.basename(os.path.dirname(p)); d = json.load(open(p)); F = d["frag"]; s = float(d["strength"])
    if fam == "inprocess" or fam.startswith("stage_"):
        pre = f"{d['stage']}:" if fam.startswith("stage_") else ""
        for a, ba in d["ba"].items():
            cells.setdefault(f"{pre}{F}|{a}", {})[s] = (ba, d["ver"].get(a), d.get("embed"))
    else:
        key = (f"{d['stage']}:{F}|{d['attack']}" if d.get("stage") else f"{F}|{d['attack']}")
        cells.setdefault(key, {})[s] = (d["ba"], d.get("ver"), d.get("embed"))
raw = json.load(open(f"{SC}/surrogate_canonical_raw.json"))
out, report = {}, []
for key, byS in sorted(cells.items()):
    xs = sorted(byS); ba = [list(map(float, byS[s][0])) for s in xs]
    ver = [list(map(bool, byS[s][1])) for s in xs] if all(byS[s][1] is not None for s in xs) else None
    n = min(len(r) for r in ba)
    if len(xs) < 2:
        print(f"  SKIP {key}: only {len(xs)} knot(s)"); continue
    out[key] = {"xs": xs, "ba": ba, "ver": ver, "n": n, "embed": byS[xs[0]][2]}
    # compare with the stored curve
    if ":" in key:
        stage, rest = key.split(":"); F, a = rest.split("|"); c = raw["frontend"].get(f"base_fe_{stage}_{F}|{a}")
    else:
        F, a = key.split("|"); c = raw["base"].get(f"{F}|{a}")
    if c is None:
        report.append((key, None, None)); continue
    curve = PWL(c["xs"], c["ys"])
    diffs = [float(np.mean(ba[j])) - curve.eval(xs[j]) for j in range(len(xs))]
    report.append((key, max(abs(x) for x in diffs), float(np.mean(diffs))))
json.dump(out, open(f"{SC}/surrogate_perimage.json", "w"))
print(f"wrote surrogate_perimage.json: {len(out)} cells, "
      f"{sum(1 for v in out.values() if v['ver'] is not None)} with a verification flag")
print(f"\n{'cell':32s} {'n':>4s} {'knots':>5s} {'max|mean-curve|':>16s} {'bias':>8s}")
worst = sorted([r for r in report if r[1] is not None], key=lambda r: -r[1])
for key, mx, bias in worst[:12]:
    print(f"{key:32s} {out[key]['n']:4d} {len(out[key]['xs']):5d} {mx:16.4f} {bias:+8.4f}")
missing = [k for k, mx, _ in report if mx is None]
if missing: print("no stored curve to compare:", missing)
print(f"\ncells within 0.02 of the stored curve: {sum(1 for r in worst if r[1] <= 0.02)}/{len(worst)}")
