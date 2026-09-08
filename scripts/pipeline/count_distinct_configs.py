"""How many DISTINCT configurations do 2,000 requests of a class collapse to?

Live certification costs per distinct configuration (embed once, attack once per column, decode), not
per request: two requests that receive the same configuration read the same measured cells at their own
thresholds. Strengths are continuous, but a certified optimum sits on the binding constraint, and there
are only a few binding columns and six budgets, so the strengths cluster.
Usage: python count_distinct_configs.py <class> <n_requests> [round=0.001]
"""
import os
import sys, json, collections, time
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
for p in (f"{CF}/scripts/defense", SC): sys.path.insert(0, p)
CLS = int(sys.argv[1]); N = int(sys.argv[2]); R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
import solver_eval_continuous as SEC
from class_defs import classes, sample_class
sg = SEC.sg; C = classes(sg.attacks)[CLS]; SCEN = sample_class(C, CLS, 2000, W.beta_from_fpr)
keys = collections.Counter(); att_union = collections.defaultdict(set); t0 = time.time(); sat = 0
for i in range(N):
    sc = SCEN[i]
    scen = dict(min_psnr=sc["min_psnr"], max_ms=sc["max_ms"], attacks=[a for a in sc["attacks"] if a in sg.attacks],
                min_ba=sc["min_ba"], allow_resync=True, allow_nested=True, min_bits=sc["min_bits"], resolution=512)
    built, m, _r, _c = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg,
                                            det_min=float(__import__("os").environ.get("DET_MIN", W.DEFAULT_DET_MIN)))
    if built is None: continue
    sat += 1; o, u, rs, ns, al, nf, ps, tm = built
    frags = [f for f in W.FR if str(m.eval(u[f])) == "True"]
    fe = tuple(sorted(k for k, v in W.frontend_decisions(o, m, rs, ns).items() if v))
    s = tuple(round(round(float(m.eval(o._svars[f]).as_fraction()) / R) * R, 3) for f in frags)
    pv = getattr(o, "_pvars", {})
    order = tuple(sorted(frags, key=lambda f: sum(1 for g in frags if g != f and any(
        (k == (g, f) and str(m.eval(v)) == "True") or (k == (f, g) and str(m.eval(v)) != "True") for k, v in pv.items()))))
    key = (order, fe, s); keys[key] += 1; att_union[key].update(scen["attacks"])
    if (i + 1) % 50 == 0: print(f"  {i+1}/{N} sat {sat} distinct {len(keys)} [{time.time()-t0:.0f}s]", flush=True)
cells = sum(len(v) for v in att_union.values())
print(f"{C['key']}: {N} requests, {sat} SAT, {len(keys)} distinct configurations at strength rounding {R}; "
      f"{cells} (config, attack) cells to measure; top: {keys.most_common(5)}", flush=True)
json.dump({"cls": C["key"], "n": N, "sat": sat, "distinct": len(keys), "cells": cells,
           "configs": [{"order": k[0], "fe": k[1], "s": k[2], "n_req": c, "attacks": sorted(att_union[k])} for k, c in keys.most_common()]},
          open(f"{SC}/distinct_configs_{C['key']}_n{N}.json", "w"), indent=1)
print("COUNT_DONE", flush=True)
