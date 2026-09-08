"""Unmeasured in-process cells of one class and domain after the measure arrays (certify_full.py measure).

A shard that was preempted or timed out leaves group files absent or attacks missing; the measure is
resumable per attack cell, so the chain re-submits the class/domain when this prints MISSING > 0.
Usage: python certify_full_check.py <class 0-4> [domain=indomain]
"""
import sys, os, json, glob
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
LIVE_OK = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
           "crop_jpeg", "border20", "vaeB", "vaeC", "regen", "rinse2x"}
K = ["C1", "C2", "C3", "C4", "C5"][int(sys.argv[1])]; DOM = sys.argv[2] if len(sys.argv) > 2 else "indomain"
OUT = f"{SC}/certify_full/{K}" if DOM == "indomain" else f"{SC}/certify_full_{DOM}/{K}"
recs = []
for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")): recs += json.load(open(f))["records"]
g = {}
for r in recs:
    s = r["solver"]
    if not s: continue
    key = (tuple(s["order"]), tuple(sorted(k for k, v in s["fe"].items() if v)), tuple(round(s["s"][f], 3) for f in s["order"]))
    g.setdefault(key, []).append(r)
keys = sorted(g, key=lambda k: (-len(g[k]), k))
missing = absent = cells = 0
for gi, k in enumerate(keys):
    want = sorted({a for m in g[k] for a in m["attacks"] if a in LIVE_OK}); cells += len(want)
    geo_cfg = any(st in k[1] for st in ("resync", "scale", "angle", "tile"))
    p = f"{OUT}/g{gi:04d}.json"
    if not os.path.exists(p): absent += 1; missing += len(want); continue
    rec = json.load(open(p))
    # a configuration with a geometric stage needs the two-view cell format (primary + cascade view, 2026-09-08)
    missing += sum(1 for a in want if rec["attacks"].get(a) is None or (geo_cfg and rec["attacks"][a].get("views") != 2))
print(f"{K} [{DOM}]: {len(keys)} groups ({absent} without a file), {cells} in-process cells wanted, MISSING {missing}")
