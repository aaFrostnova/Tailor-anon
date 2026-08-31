"""Merge SLURM-array per-chunk eval_matrix JSONs -> one n-weighted-average {method}.json.
Usage: merge_chunks.py <dir_with_chunk_*_subdirs> <out_dir>"""
import json, glob, os, sys
from collections import defaultdict
indir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
bym = defaultdict(list)
for f in glob.glob(f"{indir}/chunk_*/*.json"):
    bym[os.path.basename(f)].append(json.load(open(f)))
for fname, js in bym.items():
    N = sum(j["n"] for j in js)
    def wavg(key):
        vs = [(j["n"], j.get(key)) for j in js if j.get(key) is not None]
        return sum(n * v for n, v in vs) / sum(n for n, _ in vs) if vs else None
    out = {"method": js[0]["method"], "n_bits": js[0]["n_bits"], "n": N, "tau": js[0].get("tau"),
           "psnr": wavg("psnr"), "ssim": wavg("ssim"), "n_chunks": len(js), "attacks": {}}
    atts = set().union(*[set(j["attacks"]) for j in js])
    for a in atts:
        nba = dba = ntp = dtp = 0.0
        for j in js:
            v = j["attacks"].get(a)
            if not v: continue
            if v.get("bit_acc") is not None: nba += j["n"] * v["bit_acc"]; dba += j["n"]
            if v.get("tpr") is not None: ntp += j["n"] * v["tpr"]; dtp += j["n"]
        out["attacks"][a] = {"bit_acc": (nba / dba if dba else None), "tpr": (ntp / dtp if dtp else None)}
    json.dump(out, open(f"{outdir}/{fname}", "w"), indent=2)
    print(f"{fname}: {len(js)} chunks -> N={N}")
