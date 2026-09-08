"""Ablation of the budget split: re-solve class requests with presence_threshold(min_ba, k) = min_ba for every k
(every zero-bit test at the single-fragment threshold), compare with the recorded solutions, and check whether each
ablated configuration still covers its request under the deployed (split) thresholds by pinning it inside the
corrected solver. Requests: SAT, non-stress, fpr in {1e-1, 1e-2} (where tau_k > tau_1), n per class, seeded.
Usage: python ablate_presence_split.py [n_per_class=100] [workers=24]
"""
import sys, os, json, glob, random, time
from multiprocessing import Pool
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
NPC = int(sys.argv[1]) if len(sys.argv) > 1 else 100; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 24
sys.argv = [sys.argv[0]]
BUDGETS = (1e-1, 1e-2); KEYS = ["C1", "C2", "C3", "C4"]
W = SEC = z3 = None
def _init():
    global W, SEC, z3
    for p in (f"{CF}/scripts/defense", SC):
        if p not in sys.path: sys.path.insert(0, p)
    import z3 as _z3; import watermark_smt_v2 as _W; import solver_eval_continuous as _SEC
    W, SEC, z3 = _W, _SEC, _z3
def sc_of(r):
    return dict(min_psnr=r["min_psnr"], max_ms=r["max_ms"], attacks=list(r["attacks"]), min_ba=r["min_ba"], fpr=r["fpr"],
                allow_resync=True, allow_nested=True, min_bits=r["min_bits"])
def pin_check(sc, cfg):
    """Is the configuration feasible for the request under the corrected thresholds? (pruning as in solve_free)"""
    o, u, rs, ns, al, nf, ps, tm = SEC._build(sc, order=True, prune=True)
    frags = cfg["frags"]; order = cfg["order"]
    for f in W.FR: o.add(u[f] == (f in frags))
    for f in frags: o.add(o._svars[f] >= z3.RealVal(cfg["s"][f] - 1e-6)); o.add(o._svars[f] <= z3.RealVal(cfg["s"][f] + 1e-6))
    for name, var in o._fevars.items():
        want = bool(cfg["fe"].get(name, False))
        if var is None:
            if want: return "stage_unavailable"
            continue
        o.add(var == z3.BoolVal(want))
    for (f, g), pv in getattr(o, "_pvars", {}).items():
        if f in frags and g in frags: o.add(pv == z3.BoolVal(order.index(f) < order.index(g)))
    return "sat" if o.check() == z3.sat else "unsat"
def same(a, b):
    return (a["frags"] == b["frags"] and a["order"] == b["order"] and all(bool(a["fe"].get(k)) == bool(b["fe"].get(k)) for k in set(a["fe"]) | set(b["fe"]))
            and all(abs(a["s"][f] - b["s"][f]) < 1e-3 for f in a["frags"]))
def work(r):
    sc = sc_of(r); orig = W.presence_threshold; t0 = time.time()
    W.presence_threshold = lambda min_ba, k, n_tx=100, t_corr=10: float(min_ba)
    try: abl = SEC.solve_free(sc)
    finally: W.presence_threshold = orig
    row = {"i": r["i"], "cls": r["cls"], "fpr": r["fpr"], "min_psnr": r["min_psnr"], "max_ms": r["max_ms"], "min_bits": r["min_bits"], "attacks": r["attacks"],
           "recorded": r["solver"], "ablated": abl, "secs": round(time.time() - t0, 1)}
    if abl is None: row["changed"] = True; row["pinned"] = "ablated_unsat"; return row
    row["changed"] = not same(abl, r["solver"]); row["dpsnr"] = abl["psnr_db"] - r["solver"]["psnr_db"]
    row["k_before"] = len(r["solver"]["frags"]); row["k_after"] = len(abl["frags"])
    row["pinned"] = "same" if not row["changed"] else pin_check(sc, abl)
    return row
if __name__ == "__main__":
    random.seed(7); jobs = []
    for K in KEYS:
        recs = [x for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")) for x in json.load(open(f))["records"]]
        pool = [x for x in recs if x["solver"] and not x.get("stress") and x["fpr"] in BUDGETS]
        jobs += random.sample(pool, min(NPC, len(pool)))
    print(f"{len(jobs)} requests ({NPC} per class, budgets {BUDGETS}), {NW} workers", flush=True)
    rows = []; t0 = time.time()
    with Pool(NW, initializer=_init) as P:
        for n, row in enumerate(P.imap_unordered(work, jobs), 1):
            rows.append(row)
            if n % 20 == 0: print(f"  {n}/{len(jobs)} [{time.time()-t0:.0f}s] changed so far {sum(r['changed'] for r in rows)}", flush=True)
    rows.sort(key=lambda r: (r["cls"], r["i"]))
    summ = {}
    for K in KEYS:
        rr = [r for r in rows if r["cls"] == K]; ch = [r for r in rr if r["changed"]]
        summ[K] = {"n": len(rr), "changed": len(ch), "k_up": sum(1 for r in ch if r.get("k_after", 0) > r.get("k_before", 0)),
                   "k_down": sum(1 for r in ch if r.get("k_after", 9) < r.get("k_before", 9)),
                   "dpsnr_changed_mean": (sum(r["dpsnr"] for r in ch if "dpsnr" in r) / max(1, sum(1 for r in ch if "dpsnr" in r))),
                   "dpsnr_changed_max": max([r["dpsnr"] for r in ch if "dpsnr" in r] or [0.0]),
                   "pinned_unsat": sum(1 for r in ch if r["pinned"] == "unsat"), "pinned_sat": sum(1 for r in ch if r["pinned"] == "sat"),
                   "other": sum(1 for r in ch if r["pinned"] not in ("sat", "unsat"))}
        print(f"{K}: n={summ[K]['n']} changed={summ[K]['changed']} (k up {summ[K]['k_up']}, down {summ[K]['k_down']}); among changed: "
              f"mean dPSNR {summ[K]['dpsnr_changed_mean']:+.2f} dB (max {summ[K]['dpsnr_changed_max']:+.2f}); ablated config infeasible at the split thresholds: "
              f"{summ[K]['pinned_unsat']}, still feasible: {summ[K]['pinned_sat']}, other: {summ[K]['other']}", flush=True)
    json.dump({"n_per_class": NPC, "budgets": BUDGETS, "summary": summ, "rows": rows}, open(f"{SC}/ablate_presence_split.json", "w"), indent=1)
    print("ABLATION_DONE", flush=True)
