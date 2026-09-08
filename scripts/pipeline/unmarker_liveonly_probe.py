"""What C4 looks like when UnMarker is LIVE-ONLY rather than an offline table column.

W.live_required() declares unmarker ADVERSARIAL (per-image optimised, so the offline mean is not a
point estimate for the user's image) but nothing calls it: C4 was solved with unmarker as an ordinary
column. This measures the alternative the design asks for -- the solver discharges every other column
from the table and DEFERS unmarker to a live measurement on the chosen configuration -- by re-solving
each C4 request with unmarker dropped from the constraint set and reporting:
  deferred_sat   the request is feasible on everything the table can settle (live then decides unmarker)
  table_sat      what the deployed run reported, unmarker treated as a normal column
The gap is the set of requests whose SAT verdict currently RESTS on the unmarker curve.
Usage: python unmarker_liveonly_probe.py [shard=0] [n_shards=1]
"""
import sys, os, json, glob, time
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
for p in (f"{CF}/scripts/defense", SC): sys.path.insert(0, p)
SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
import solver_eval_continuous as SEC
from class_defs import classes, sample_class
C = classes(SEC.sg.attacks)[3]
SCEN = sample_class(C, 3, 2000, W.beta_from_fpr)
recs = []
for f in sorted(glob.glob(f"{SC}/class_eval_C4_shard*.json")): recs += json.load(open(f))["records"]
table_sat = {r["i"]: bool(r["solver"]) for r in recs}
idx = [i for i in range(2000) if i % NSHARD == SHARD]
print(f"C4 unmarker-deferred re-solve: {len(idx)} requests (shard {SHARD}/{NSHARD})", flush=True)
out = []; t0 = time.time()
for n, i in enumerate(idx):
    sc = dict(SCEN[i]); sc["attacks"] = [a for a in sc["attacks"] if a != "unmarker"]
    r = SEC.solve_free(sc)
    out.append({"i": i, "fpr": sc["fpr"], "cr": SCEN[i].get("_cr"), "min_ba": sc["min_ba"],
                "min_bits": sc["min_bits"], "min_psnr": sc["min_psnr"], "table_sat": table_sat.get(i),
                "deferred_sat": bool(r), "cfg": r})
    if (n + 1) % 25 == 0:
        d = sum(1 for x in out if x["deferred_sat"]); t = sum(1 for x in out if x["table_sat"])
        print(f"  {n+1}/{len(idx)}  deferred SAT {d}  table SAT {t}  [{time.time()-t0:.0f}s]", flush=True)
p = f"{SC}/unmarker_liveonly_{SHARD}of{NSHARD}.json"
json.dump({"rows": out}, open(p, "w"), indent=1)
print("wrote", p, "LIVEONLY_DONE", flush=True)
