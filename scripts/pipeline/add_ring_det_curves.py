"""The ring campaign's per-image DETECTION curves on the adversarial columns.

merge_unmk_ring.py stored the ring campaign's mean bit accuracy as the scale stage's replacement curves
and kept the detection rate in `source.det`, where the solver cannot read it. Every other stage cell has
a det_fe_* curve, and those curves are what says whether the stage's crypto-verify-gated search actually
FIRES at a given strength: det_fe_scale_VINE|crop75 runs 0.16 to 1.00 while its mean bit accuracy is
already 0.56 at the bottom, so a solve that reads only the mean can buy the stage below the strength at
which it latches. Measured consequence on the certified C4 requests: four of nine cleared the mean
UnMarker threshold at VINE strength 0.22 to 0.41 with per-image detection 0.30 to 0.80.
"""
import os
import json, glob
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
src = f"{SC}/surrogate_fe_ring_adv_overlay.json"
o = json.load(open(src))
knots = sorted(glob.glob(f"{SC}/unmarker9_ring/knot_*/knot.json"),
               key=lambda p: float(p.split("knot_")[1].split("/")[0]))
xs = [float(p.split("knot_")[1].split("/")[0]) for p in knots]
cols = {}
for p in knots:
    for a, c in json.load(open(p))["cols"].items():
        cols.setdefault(a, []).append(float(c["det"]))
added = []
for a, ys in cols.items():
    k = f"det_fe_scale_VINE|{a}"
    if k in o["frontend"]:
        continue
    o["frontend"][k] = {"xs": xs, "ys": ys}; added.append(k)
o["source"]["det_curves_added"] = added
json.dump(o, open(src, "w"), indent=1)
print(f"added {len(added)} det curves to {src}")
for k in added:
    print(f"  {k:36s} {' '.join('%.2f' % y for y in o['frontend'][k]['ys'])}")
