"""Fold the nine ring-embed knots (unmk_ring.sbatch) into ONE front-end overlay for the canonical table:
base_fe_scale_VINE|unmarker and base_fe_scale_VINE|ctrlregen_s03/s05/s07, each an absolute best-path
bit-accuracy curve over VINE strength with the scale stage on (same semantics as the other base_fe_*
curves). Refuses to write unless every knot decoded every column at the campaign's N."""
import json, glob, os, sys
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
N = int(os.environ.get("RING_N", 30))
COLS = ["unmarker", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"]
knots = [json.load(open(f)) for f in sorted(glob.glob(f"{SC}/unmarker9_ring/knot_*/knot.json"))]
knots.sort(key=lambda k: k["strength"])
xs = [k["strength"] for k in knots]
assert len(xs) == 9, f"{len(xs)} knots decoded, need 9: {xs}"
fe, det = {}, {}
for c in COLS:
    ys = []
    for k in knots:
        cell = k["cols"].get(c); assert cell and cell["n"] >= N, (k["strength"], c, cell)
        ys.append(round(cell["mean"], 4))
    fe[f"base_fe_scale_VINE|{c}"] = {"xs": xs, "ys": ys}
    det[c] = [round(k["cols"][c]["det"], 3) for k in knots]
    print(f"  base_fe_scale_VINE|{c}: {ys}  det {det[c]}")
out = {"base": {}, "delta": {}, "frontend": fe,
       "source": {"n": N, "campaign": "unmk_ring.sbatch: nested-ring VINE embed at 9 strengths, 30 pool images (offset 0), "
                  "UnMarker (Vine.yaml) and CtrlRegen+ at steps 0.3/0.5/0.7 in their own environments, decoded by the deployed cascade "
                  "with the scale search on; the stage's replacement curves on the adversarial columns", "det": det, "knots": xs}}
json.dump(out, open(f"{SC}/surrogate_fe_ring_adv_overlay.json", "w"), indent=1)
print("wrote surrogate_fe_ring_adv_overlay.json  RING_MERGE_DONE")
