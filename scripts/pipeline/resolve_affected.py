"""Re-solve, in place, the class records whose recorded solution the resync-penalty fix invalidates:
resync on without TrustMark selected. Everything else in the shard is kept byte-for-byte (records, fixed
baselines, metadata); re-solved records get "resolved": "resync_penalty_fix" and keep their old solution
under "solver_before_fix". Usage: python resolve_affected.py <class keys, e.g. C3 C5> [workers=24]
"""
import sys, os, json, glob, time
from multiprocessing import Pool
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
KEYS = [a for a in sys.argv[1:] if a.startswith("C")]; NW = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 24
sys.argv = [sys.argv[0]]
SEC = None
def _init():
    global SEC
    for p in (f"{CF}/scripts/defense", SC):
        if p not in sys.path: sys.path.insert(0, p)
    import solver_eval_continuous as _SEC; SEC = _SEC
def affected(r):
    s = r["solver"]; return bool(s) and bool(s["fe"].get("resync")) and "TrustMark" not in s["frags"]
def work(job):
    path, idx, r = job; t0 = time.time()
    sc = dict(min_psnr=r["min_psnr"], max_ms=r["max_ms"], attacks=list(r["attacks"]), min_ba=r["min_ba"], fpr=r["fpr"],
              allow_resync=True, allow_nested=True, min_bits=r["min_bits"])
    return path, idx, SEC.solve_free(sc), round(time.time() - t0, 1)
if __name__ == "__main__":
    jobs = []; shards = {}
    for K in KEYS:
        for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")):
            d = json.load(open(f)); shards[f] = d
            jobs += [(f, i, r) for i, r in enumerate(d["records"]) if affected(r)]
    print(f"{len(jobs)} affected records in {KEYS} across {len(shards)} shards; {NW} workers", flush=True)
    t0 = time.time(); n = 0; changed = 0; unsat = 0
    with Pool(NW, initializer=_init) as P:
        for path, idx, sol, secs in P.imap_unordered(work, jobs):
            r = shards[path]["records"][idx]; r["solver_before_fix"] = r["solver"]; r["solver"] = sol; r["resolved"] = "resync_penalty_fix"
            n += 1; unsat += sol is None
            if sol is None or sol["frags"] != r["solver_before_fix"]["frags"] or any(bool(sol["fe"].get(k)) != bool(v) for k, v in r["solver_before_fix"]["fe"].items()): changed += 1
            if n % 25 == 0: print(f"  {n}/{len(jobs)} [{time.time()-t0:.0f}s]  config changed {changed}, now UNSAT {unsat}", flush=True)
    for f, d in shards.items():
        d["resync_penalty_fix"] = "2026-09-08"
        tmp = f + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, f)
    print(f"done: {n} re-solved, config changed {changed}, UNSAT {unsat}; shards rewritten  RESOLVE_DONE", flush=True)
