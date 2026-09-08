"""Merge the per-class shards (class_scenarios.py) and, with --tex, write tab/classes.tex.

For each class: requests, solver satisfaction, delivered PSNR on satisfied requests, each fixed
baseline's satisfaction and its regret against the solver on the requests both satisfy, the number of
requests the solver reports infeasible (and how many of those every baseline also fails), and the
configuration mix. Usage: python merge_class_eval.py [--tex path]
"""
import os
import json, glob, sys, collections, math
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
KEYS = ["C1", "C2", "C3", "C4", "C5"]
out = {"classes": {}, "table_assembled_at": None, "margin": None}
for key in KEYS:
    shards = [json.load(open(f)) for f in sorted(glob.glob(f"{SC}/class_eval_{key}_shard*.json"))]
    if not shards:
        print(f"{key}: no shards yet"); continue
    ns = {d["n_shards"] for d in shards}
    assert len(ns) == 1, (key, ns)
    recs = [r for d in shards for r in d["records"]]
    idx = sorted(r["i"] for r in recs)
    complete = len(shards) == next(iter(ns)) and idx == list(range(shards[0]["N"]))
    out["table_assembled_at"] = shards[0]["table_assembled_at"]; out["margin"] = shards[0]["margin"]
    sat = [r for r in recs if r["solver"]]
    fixed_names = list(shards[0]["records"][0]["fixed"].keys()) if shards[0]["records"] else []
    base = {}
    for b in fixed_names:
        ok = [r for r in recs if r["fixed"].get(b)]
        both = [r for r in ok if r["solver"]]
        regret = [r["solver"]["psnr_db"] - r["fixed"][b]["psnr_db"] for r in both]
        base[b] = {"satisfaction": len(ok) / len(recs), "psnr_on_satisfied": (sum(r["fixed"][b]["psnr_db"] for r in ok) / len(ok)) if ok else None,
                   "regret_vs_solver_db": (sum(regret) / len(regret)) if regret else None, "n_both": len(both)}
    unsat = [r for r in recs if not r["solver"]]
    unsat_all = [r for r in unsat if not any(r["fixed"].get(b) for b in fixed_names)]
    mix = collections.Counter("+".join(r["solver"]["frags"]) + ("".join(f"+{k}" for k, v in r["solver"]["fe"].items() if v)) for r in sat)
    single = sum(v for k, v in mix.items() if len([p for p in k.split("+") if p in ("VINE", "TrustMark", "VideoSeal")]) == 1)
    by_fpr = {}
    for fpr in sorted({r["fpr"] for r in recs}, reverse=True):
        rr = [r for r in recs if r["fpr"] == fpr]; ss = [r for r in rr if r["solver"]]
        by_fpr[str(fpr)] = {"n": len(rr), "satisfaction": len(ss) / len(rr), "psnr": (sum(r["solver"]["psnr_db"] for r in ss) / len(ss)) if ss else None}
    by_cr = {}
    for cr in sorted({r.get("cr") for r in recs if r.get("cr")}):
        rr = [r for r in recs if r.get("cr") == cr]; ss = [r for r in rr if r["solver"]]
        by_cr[cr] = {"n": len(rr), "satisfaction": len(ss) / len(rr), "psnr": (sum(r["solver"]["psnr_db"] for r in ss) / len(ss)) if ss else None,
                     "baselines": {b: sum(1 for r in rr if r["fixed"].get(b)) / len(rr) for b in fixed_names}}
    # why each UNSAT request is infeasible, read off the feasibility matrix the sampler used: a column beyond
    # every fragment's reach at the request's budget (ceiling), a payload above a column's capacity (bits),
    # a fidelity floor above what the hardest column's strength allows (psnr), or none of these, in which
    # case the composition (one strength per fragment across all columns, interference, the k-fragment
    # threshold, latency) is what fails. Stress requests are counted apart: they are beyond the ceiling by design.
    try:
        FEAS = json.load(open(f"{SC}/request_feasibility_matrix.json")); FPRS = FEAS["budgets"]; N2 = set(FEAS["needs_second_fragment"])
        def _why(r):
            k = r.get("k") or (2 if any(a in N2 for a in r["attacks"]) else 1); bi = FPRS.index(r["fpr"]); why = []
            for a in r["attacks"]:
                nd = FEAS["need"].get(f"{a}|k{k}|{bi}")
                if nd is None: why.append(("ceiling", a))
                elif r["min_bits"] and FEAS["capacity_bits"][a] < r["min_bits"]: why.append(("bits", a))
            if not why:
                need_db = min(FEAS["need"][f"{a}|k{k}|{bi}"][0] for a in r["attacks"]) - (1.0 if k == 1 else 2.0)
                if r["min_psnr"] > need_db: why.append(("psnr", "-"))
            return why or [("composition", "-")]
        attribution = collections.Counter(); blocking = collections.Counter()
        for r in unsat:
            w = _why(r); kind = ("stress:" + str(r.get("stress_kind") or "budget")) if r.get("stress") else w[0][0]
            attribution[kind] += 1
            for c, a in w:
                if c == "ceiling": blocking[a] += 1
        unsat_attribution = {"by_cause": dict(attribution), "blocking_columns": dict(blocking.most_common(8))}
    except Exception as e:
        unsat_attribution = {"error": str(e)}
    # stress requests ask for a budget beyond the hardest column's ceiling on purpose (class_defs.STRESS):
    # infeasible by the fragments' limits, reported apart so the class satisfaction reads the method
    by_stress = {}
    for st in (False, True):
        rr = [r for r in recs if bool(r.get("stress")) == st]; ss = [r for r in rr if r["solver"]]
        if rr: by_stress["stress" if st else "within_ceiling"] = {"n": len(rr), "satisfaction": len(ss) / len(rr),
                                                                   "psnr": (sum(r["solver"]["psnr_db"] for r in ss) / len(ss)) if ss else None}
    by_bits = {}
    for b in sorted({r["min_bits"] for r in recs}):
        rr = [r for r in recs if r["min_bits"] == b]; ss = [r for r in rr if r["solver"]]
        by_bits[str(b)] = {"n": len(rr), "satisfaction": len(ss) / len(rr)}
    out["classes"][key] = {"name": shards[0]["name"], "core": shards[0]["core"], "optional": shards[0]["optional"], "N": len(recs), "complete": complete,
                           "satisfaction": len(sat) / len(recs), "psnr_on_satisfied": (sum(r["solver"]["psnr_db"] for r in sat) / len(sat)) if sat else None,
                           "unsat": len(unsat), "unsat_all_baselines_fail": len(unsat_all), "single_fragment_share": (single / len(sat)) if sat else None,
                           "config_mix": dict(mix.most_common(6)), "baselines": base, "by_fpr": by_fpr, "by_bits": by_bits, "by_cr": by_cr,
                           "by_stress": by_stress, "unsat_attribution": unsat_attribution}
    c = out["classes"][key]
    print(f"{key} {c['name']:32s} n={c['N']:4d}{'' if complete else ' (partial)'}  solver {c['satisfaction']:.3f} @ {c['psnr_on_satisfied'] or 0:.2f} dB  "
          f"UNSAT {c['unsat']} (all baselines fail {c['unsat_all_baselines_fail']})  single {c['single_fragment_share'] or 0:.0%}")
    for b, v in base.items():
        print(f"     {b:20s} sat {v['satisfaction']:.3f}  regret {v['regret_vs_solver_db'] if v['regret_vs_solver_db'] is not None else float('nan'):6.2f} dB")
    for cr, v in by_cr.items():
        print(f"     {cr:20s} n={v['n']:4d} sat {v['satisfaction']:.3f} @ {v['psnr'] or 0:.2f} dB  baselines " + " ".join(f"{b}={x:.2f}" for b, x in v["baselines"].items()))
    for st, v in by_stress.items():
        print(f"     {st:20s} n={v['n']:4d} sat {v['satisfaction']:.3f} @ {v['psnr'] or 0:.2f} dB")
    if unsat: print(f"     UNSAT attribution {unsat_attribution.get('by_cause')}  blocking columns {unsat_attribution.get('blocking_columns')}")
json.dump(out, open(f"{SC}/class_eval.json", "w"), indent=1)
print("wrote class_eval.json")
if "--tex" in sys.argv:
    path = sys.argv[sys.argv.index("--tex") + 1]
    LABEL = {"VINE-only@1.0": "VINE only", "TrustMark-only": "TrustMark only", "VINE+TM@.7": "VINE$+$TrustMark", "3frag@.7+resync": "always-full$+$resync"}
    C = out["classes"]; keys = [k for k in KEYS if k in C and C[k]["complete"]]   # a partial class never enters the paper table
    bl = list(next(iter(C.values()))["baselines"].keys())
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{\textbf{Five threat classes, 2,000 sampled requests each.} A class fixes the attacks a deployment must survive; each request samples the budgets it can afford (false-positive budget, fidelity floor, latency, payload) and which of the class's optional attacks are also in scope. \emph{Sat.} is the fraction of requests met; \emph{PSNR} the mean fidelity delivered on them; \emph{regret} the fidelity a fixed configuration gives up against the solver on the requests both satisfy. Every C4 request carries UnMarker and one CtrlRegen$+$ strength (0.3, 0.5 or 0.7, drawn at 0.4, 0.4, 0.2). Every solver answer is a certified optimum on the final table with the default safety allowance.}",
         r"\label{tab:classes}", r"\footnotesize", r"\resizebox{\textwidth}{!}{\begin{tabular}{l l r r r " + "r r " * len(bl) + "}", r"\toprule",
         "Class & threat & Sat. & PSNR & UNSAT & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{LABEL.get(b, b)}}}" for b in bl) + r" \\",
         " & & & (dB) & & " + " & ".join("Sat. & regret" for _ in bl) + r" \\", r"\midrule"]
    for k in keys:
        c = C[k]
        row = [k, c["name"].replace("+", "$+$").replace("&", r"\&"), f"{c['satisfaction']:.2f}", f"{c['psnr_on_satisfied']:.1f}" if c["psnr_on_satisfied"] else "n/a", str(c["unsat"])]
        for b in bl:
            v = c["baselines"][b]; row += [f"{v['satisfaction']:.2f}", (f"{v['regret_vs_solver_db']:.1f}" if v["regret_vs_solver_db"] is not None else "n/a")]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n"); print("wrote", path)
