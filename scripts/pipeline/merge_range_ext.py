"""Fold the range-extension measurements into ONE file of NEW KNOTS per curve, surrogate_range_ext.json,
which make_canonical_surrogate.py splices into the existing curves (union of knots) and uses to widen
`ranges`. Sources:
  perimage_ext/inprocess/<F>_<s>.json      base curves on the 16 in-process + diffusion columns (per-image means)
                                            and the distortion curve d(F) (per-image MSE)
  perimage_ext/<col>/<F>_<s>.json          base curves on the adversarial columns (extend_xenv.py)
  surrogate_fe_components_ext_*.json       front-end curves (base_fe_*, det_fe_*, ctrl_*, penalty_fe_*)
  surrogate_delta_curves_ext_*of22.json    interference curves delta_{g->f}(s_g)
Only knots are produced here; nothing is refit. The three fragments were swept over three different
distortion intervals (VINE 34.8 to 47.3 dB, TrustMark 37.1 to 47.5, VideoSeal 41.6 to 50.6); after this
they share about 35 to 50 dB.
"""
import json, glob, os
import numpy as np
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
NEW = {"VideoSeal": [1.75, 2.0, 2.25, 2.5, 2.8, 3.2], "VINE": [0.14, 0.17], "TrustMark": [1.8, 2.0]}
out = {"base": {}, "d": {}, "frontend": {}, "delta": {}, "ranges": {}, "source": {"new_knots": NEW, "files": {}}}
def put(blk, key, x, y):
    c = out[blk].setdefault(key, {"xs": [], "ys": []}); c["xs"].append(float(x)); c["ys"].append(float(y))
# base + d from the in-process extension
for p in sorted(glob.glob(f"{SC}/perimage_ext/inprocess/*.json")):
    d = json.load(open(p)); F, s = d["frag"], float(d["strength"])
    for a, ys in d["ba"].items(): put("base", f"{F}|{a}", s, np.mean(ys))
    put("d", F, s, d["mse_mean"])
    out["source"]["files"].setdefault("inprocess", []).append(os.path.basename(p))
# base on the adversarial columns
for fam in ("ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker"):
    for p in sorted(glob.glob(f"{SC}/perimage_ext/{fam}/*.json")):
        d = json.load(open(p)); put("base", f"{d['frag']}|{fam}", float(d["strength"]), np.mean(d["ba"]))
        out["source"]["files"].setdefault(fam, []).append(os.path.basename(p))
# front-end curves at the new knots (each file: several keys, xs = the new knots of that fragment)
for p in sorted(glob.glob(f"{SC}/surrogate_fe_components_ext_*.json")):
    d = json.load(open(p))
    for k, v in (d.get("frontend") or {}).items():
        if not (isinstance(v, dict) and "xs" in v): continue
        for x, y in zip(v["xs"], v["ys"]):
            if isinstance(y, float) and y != y: continue                        # NaN knot: skip it, keep the rest
            put("frontend", k, x, y)
    out["source"]["files"].setdefault("frontend", []).append(os.path.basename(p))
# interference curves
for p in sorted(glob.glob(f"{SC}/surrogate_delta_curves_ext_*of22.json")):
    d = json.load(open(p))
    for k, v in d["delta"].items():
        for x, y in zip(v["xs"], v["ys"]): put("delta", k, x, y)
    out["source"]["files"].setdefault("delta", []).append(os.path.basename(p))
for blk in ("base", "d", "frontend", "delta"):
    for k, c in out[blk].items():
        order = np.argsort(c["xs"]); c["xs"] = [c["xs"][i] for i in order]; c["ys"] = [round(c["ys"][i], 5) for i in order]
for F, ks in NEW.items():
    have = sorted({x for k, c in out["base"].items() if k.startswith(F + "|") for x in c["xs"]})
    out["ranges"][F] = [min(have), max(have)] if have else None
json.dump(out, open(f"{SC}/surrogate_range_ext.json", "w"), indent=1)
print(f"wrote surrogate_range_ext.json: base {len(out['base'])} curves, d {len(out['d'])}, frontend {len(out['frontend'])}, delta {len(out['delta'])}")
print("measured new knots per fragment:", {F: sorted({x for k, c in out['base'].items() if k.startswith(F + '|') for x in c['xs']}) for F in NEW})
for F in NEW:
    ks = [k for k in out["base"] if k.startswith(F + "|")]
    print(f"  {F}: {len(ks)} columns have new knots" + (f" (missing: {sorted(set(a for a in ['jpeg25','blur','noise','bright','contrast','crop75','crop50','rot9','rs256','hflip','crop_jpeg','border20','vaeB','vaeC','regen','rinse2x','ctrlregen_s03','ctrlregen_s05','ctrlregen_s07','unmarker']) - set(k.split('|')[1] for k in ks))})" if len(ks) < 20 else ""))
