"""Merge the full-pool shards: per class and family, per attack mean bit accuracy, standard error, detection
rate, pass against the representative's base threshold, per-source means; PSNR over the pool. Writes
fullpool.json and, with --tex, tab/fullpool.tex."""
import os
import json, glob, sys, collections
import numpy as np
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
out = {}
for C in ("C1", "C2", "C3", "C4", "C5"):
    for fam in ("inprocess", "diffusion"):
        sh = [json.load(open(f)) for f in sorted(glob.glob(f"{SC}/fullpool/{C}/{fam}_shard*.json"))]
        if not sh: continue
        n_sh = sh[0]["n_shards"]; attacks = sh[0]["attacks"]; thr = sh[0]["thr_base"]
        src = sum((s["src"] for s in sh), []); psnr = sum((s["psnr"] for s in sh), [])
        res = {"rep": sh[0]["rep"], "fpr": sh[0]["fpr"], "cfg": {k: sh[0]["cfg"][k] for k in ("order", "s", "fe")}, "thr_base": thr, "n_images": len(psnr),
               "complete": len(sh) == n_sh, "psnr": float(np.mean(psnr)), "sources": dict(collections.Counter(src)), "attacks": {}}
        for a in attacks:
            ba = np.array(sum((s["ba"][a] for s in sh), [])); det = np.array(sum((s["det"][a] for s in sh), []))
            by_src = {k: float(ba[[x == k for x in src]].mean()) for k in sorted(set(src))}
            res["attacks"][a] = {"in_scope": sh[0]["in_scope"][a], "ba": float(ba.mean()), "se": float(ba.std(ddof=1) / np.sqrt(len(ba))) if len(ba) > 1 else 0.0,
                                 "det": float(det.mean()), "pass": bool(ba.mean() >= thr - 1e-9), "by_source": by_src, "n": int(len(ba))}
        out.setdefault(C, {})[fam] = res
        print(f"{C} {fam}: request {res['rep']} {'+'.join(res['cfg']['order'])} n={res['n_images']}{'' if res['complete'] else ' (partial)'} psnr {res['psnr']:.2f} thr {thr:.3f}")
        for a, v in res["attacks"].items():
            print(f"    {a:10s} {'in ' if v['in_scope'] else 'out'} ba {v['ba']:.3f} +/- {v['se']:.3f} det {v['det']:.2f} {'pass' if v['pass'] else 'FAIL'}  " + " ".join(f"{k}={x:.2f}" for k, x in v["by_source"].items()))
json.dump(out, open(f"{SC}/fullpool.json", "w"), indent=1); print("wrote fullpool.json")
if "--tex" in sys.argv:
    path = sys.argv[sys.argv.index("--tex") + 1]
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{\textbf{Full-pool certification.} One certified request per class (the class's modal false-positive budget), embedded on all 10,000 pool images across the five sources; every attack the request names is run live and judged against the request's base threshold. \emph{Weakest} is the in-scope attack with the lowest mean bit accuracy, with its range across the five sources; the diffusion pair (regen, rinse2x) is measured on the first 1,000 images. Attacks the request did not name are also run; those that fall below the threshold are listed as out of scope.}",
         r"\label{tab:fullpool}", r"\footnotesize", r"\resizebox{\textwidth}{!}{\begin{tabular}{l l r r c l l}", r"\toprule",
         r"Class & design & PSNR & images & in scope & weakest (source range) & out of scope below threshold \\", r"\midrule"]
    for C, fams in out.items():
        res = fams.get("inprocess");
        if res is None: continue
        cells = {a: v for f in fams.values() for a, v in f["attacks"].items()}
        ins = {a: v for a, v in cells.items() if v["in_scope"]}; outs = [a for a, v in cells.items() if not v["in_scope"] and not v["pass"]]
        weak = min(ins.items(), key=lambda kv: kv[1]["ba"]) if ins else None
        tex = lambda a: a.replace("_", r"\_")
        wtxt = f"{tex(weak[0])} {weak[1]['ba']:.3f} ({min(weak[1]['by_source'].values()):.2f} to {max(weak[1]['by_source'].values()):.2f})" if weak else "n/a"
        L.append(f"{C} & {'$+$'.join(res['cfg']['order'])}{' $+$ ' + ', '.join(k for k, v in res['cfg']['fe'].items() if v) if any(res['cfg']['fe'].values()) else ''} & {res['psnr']:.1f} & {res['n_images']:,} & "
                 f"{sum(1 for v in ins.values() if v['pass'])}/{len(ins)} & {wtxt} & {', '.join(f'{tex(a)} {cells[a][chr(98)+chr(97)]:.3f}' for a in outs) if outs else 'none'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n"); print("wrote", path)
