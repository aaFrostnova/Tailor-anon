"""Dry run of a sampler change: draw every class under the reference sampler (v2 matrix, SAMPLER_V=2) and the
current one, and report per class which request fields differ and what the change implies for the solved shards
(records to re-solve, records whose floor only moves). Usage: python sampler_diff.py"""
import os, sys, json, glob, subprocess, collections
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"); CF = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
PY = os.environ.get("WM_PY", "python")
DUMP = '''
import sys, json; sys.path.insert(0, "%s/scripts/defense"); sys.path.insert(0, "%s")
import watermark_smt_v2 as W, class_defs as CD, json
sg_attacks = json.load(open("%s/surrogate_canonical.json"))["attacks"]
CL = CD.classes(sg_attacks); out = {}
for ci, C in enumerate(CL):
    out[C["key"]] = [{k: v for k, v in s.items() if k in ("attacks","fpr","min_bits","min_psnr","max_ms","_stress","_stress_kind","_k")} for s in CD.sample_class(C, ci, 2000, W.beta_from_fpr)]
json.dump(out, open(sys.argv[1], "w"))
''' % (CF, SC, SC)
def dump(path, env):
    e = dict(os.environ); e.update(env); e["PYTHONPATH"] = f"{CF}:{CF}/scripts/defense:{SC}"
    r = subprocess.run([PY, "-c", DUMP, path], env=e, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); raise SystemExit(1)
    return json.load(open(path))
ref = dump(f"{SC}/sampler_dump_v2.json", {"SAMPLER_V": "2", "FEAS_MATRIX": f"{SC}/request_feasibility_matrix.v2.json"})
new = dump(f"{SC}/sampler_dump_v3.json", {"SAMPLER_V": "3"})
for K in ("C1", "C2", "C3", "C4", "C5"):
    recs = {r["i"]: r for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")) for r in json.load(open(f))["records"]}
    A, B = ref[K], new[K]; diff = collections.Counter(); resolve = 0; floor_only = 0; mism = 0
    for i, (a, b) in enumerate(zip(A, B)):
        r = recs.get(i)
        if r and (r["attacks"] != a["attacks"] or r["fpr"] != a["fpr"] or r["min_bits"] != a["min_bits"] or r["min_psnr"] != a["min_psnr"] or r["max_ms"] != a["max_ms"]): mism += 1
        ch = [k for k in a if a[k] != b[k]]
        for k in ch: diff[k] += 1
        if not ch: continue
        if set(ch) <= {"min_psnr", "_stress", "_stress_kind"} and b["min_psnr"] <= a["min_psnr"] and r and r["solver"]: floor_only += 1
        else: resolve += 1
    print(f"{K}: v2 sampler reproduces the shards: {'yes' if mism == 0 else f'NO ({mism} mismatches)'}; fields changed {dict(diff)}; "
          f"records needing a re-solve {resolve}, SAT records whose floor only moves down {floor_only}")
    kinds = collections.Counter((b.get("_stress_kind")) for b in B); print(f"     v3 stress kinds: {dict(kinds)}")
