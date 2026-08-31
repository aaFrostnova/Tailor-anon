"""Merge sharded benchmark_composite_defense.py JSONs by weighted mean (exact for per-image
means over non-overlapping image ranges)."""
import glob, json, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "results/defense/shards_noPM/shard_*.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/defense/composite_noPM_n200.json"
shards = [json.load(open(f)) for f in sorted(glob.glob(pat))]
assert shards, f"no shards match {pat}"
N = sum(s["n_images"] for s in shards)
attacks = list(shards[0]["attacks"].keys())
merged = {"n_images": N, "tau": shards[0]["tau"], "n_shards": len(shards), "attacks": {}}
for a in attacks:
    merged["attacks"][a] = {}
    for k in shards[0]["attacks"][a].keys():
        vals = [(s["attacks"][a][k], s["n_images"]) for s in shards
                if a in s["attacks"] and s["attacks"][a].get(k) is not None]
        merged["attacks"][a][k] = float(sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else None
json.dump(merged, open(out_path, "w"), indent=2)

frags = [k for k in shards[0]["attacks"][attacks[0]].keys()
         if k not in ("fused_verify", "fused_zerobit", "composite_or", "fused_ba", "tm")]
print(f"merged {len(shards)} shards, N={N} -> {out_path}")
print(f"{'attack':<10}{'COMPOSITE':>10}{'fused_ver':>10}{'fused_0bit':>11}{'fused_ba':>9}{'TM':>6}" +
      "".join(f"{f[:5]:>7}" for f in frags))
for a in attacks:
    c = merged["attacks"][a]
    fv = "".join(f"{(c[f] if c[f] is not None else float('nan')):>7.2f}" for f in frags)
    print(f"{a:<10}{c['composite_or']:>10.2f}{c['fused_verify']:>10.2f}{c['fused_zerobit']:>11.2f}"
          f"{c['fused_ba']:>9.2f}{c['tm']:>6.2f}{fv}")
