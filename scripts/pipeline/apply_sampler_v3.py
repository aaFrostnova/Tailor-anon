"""Apply the v3 sampler to the solved shards without re-drawing anything: v2 and v3 share the random stream, so a
request differs only in the fields the new rules touch (C3/C5: min_psnr lowered by the payload-aware cap; C4:
max_ms raised by the ring reference for UnMarker). For every changed record the request fields are updated; the
solver is re-run where the change can alter the optimum (a previously UNSAT request whose floor dropped, or a
raised latency budget); the four fixed baselines are re-run for every changed record (a lower floor can admit a
pinned baseline). Records keep their previous solver output under "solver_before_v3". Originals are copied to
attic_class_eval_prev3_20260908/ first. Usage: python apply_sampler_v3.py <class keys> [workers=24]
"""
import sys, os, json, glob, time, shutil
from multiprocessing import Pool
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"); CF = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
KEYS = [a for a in sys.argv[1:] if a.startswith("C")]; NW = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 24
sys.argv = [sys.argv[0]]
ATTIC = f"{SC}/attic_class_eval_prev3_20260908"; os.makedirs(ATTIC, exist_ok=True)
SEC = None
def _init():
    global SEC
    for p in (f"{CF}/scripts/defense", SC):
        if p not in sys.path: sys.path.insert(0, p)
    import solver_eval_continuous as _SEC; SEC = _SEC
def sc_of(r): return dict(min_psnr=r["min_psnr"], max_ms=r["max_ms"], attacks=list(r["attacks"]), min_ba=r["min_ba"], fpr=r["fpr"], allow_resync=True, allow_nested=True, min_bits=r["min_bits"])
def work(job):
    path, idx, r, resolve = job; t0 = time.time(); sc = sc_of(r)
    sol = SEC.solve_free(sc) if resolve else r["solver"]
    fixed = {k: SEC.solve_fixed(sc, st, rv, nv) for k, (st, rv, nv) in SEC.FIXED.items()}
    return path, idx, sol, fixed, round(time.time() - t0, 1)
if __name__ == "__main__":
    v3 = json.load(open(f"{SC}/sampler_dump_v3.json")); jobs = []; shards = {}; nchg = 0
    for K in KEYS:
        for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")):
            shutil.copy2(f, ATTIC); d = json.load(open(f)); shards[f] = d
            for idx, r in enumerate(d["records"]):
                s = v3[K][r["i"]]
                assert s["attacks"] == r["attacks"] and s["fpr"] == r["fpr"] and s["min_bits"] == r["min_bits"], (K, r["i"])
                changed = {k: (r[k], s[k]) for k in ("min_psnr", "max_ms") if r[k] != s[k]}
                if not changed and bool(s["_stress"]) == bool(r["stress"]) and s["_stress_kind"] == r["stress_kind"]: continue
                nchg += 1; r["request_before_v3"] = {k: r[k] for k in ("min_psnr", "max_ms", "stress", "stress_kind")}
                r["min_psnr"], r["max_ms"], r["stress"], r["stress_kind"] = s["min_psnr"], s["max_ms"], bool(s["_stress"]), s["_stress_kind"]
                resolve = (r["solver"] is None) or ("max_ms" in changed and s["max_ms"] > changed["max_ms"][0])
                jobs.append((f, idx, r, resolve))
    print(f"{nchg} records change under v3 in {KEYS}; {sum(1 for j in jobs if j[3])} re-solved, {len(jobs)} baseline re-evaluations; {NW} workers", flush=True)
    t0 = time.time(); n = 0; newsat = 0
    with Pool(NW, initializer=_init) as P:
        for path, idx, sol, fixed, secs in P.imap_unordered(work, jobs):
            r = shards[path]["records"][idx]; r["solver_before_v3"] = r["solver"]; r["fixed_before_v3"] = r["fixed"]; r["solver"] = sol; r["fixed"] = fixed; r["sampler"] = "v3"
            n += 1; newsat += (sol is not None and r["solver_before_v3"] is None)
            if n % 25 == 0: print(f"  {n}/{len(jobs)} [{time.time()-t0:.0f}s] newly SAT {newsat}", flush=True)
    for f, d in shards.items():
        d["sampler_v3"] = "2026-09-08"; tmp = f + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, f)
    print(f"done: {n} records updated, newly SAT {newsat}; shards rewritten  APPLY_V3_DONE", flush=True)
