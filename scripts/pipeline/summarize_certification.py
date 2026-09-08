"""One table from the per-class certification verdicts (certify_full_verdicts.py output files).

Rows: class x domain. Columns: SAT requests, distinct configurations, requests fully measured, live-feasible
rate under the mean and the rate criterion with the 95% image-bootstrap interval, the same for the
within-ceiling and the stress subsets, and the payload requests whose capacity line held live.
Also the solver-level numbers of the class (merge_class_eval.py's class_eval.json) so the two sit together:
satisfaction, delivered PSNR, UNSAT attribution. Writes certification_summary.json.
Usage: python summarize_certification.py            (reads certify_full_C*.json, class_eval.json)
"""
import json, os, glob
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
KEYS = ["C1", "C2", "C3", "C4", "C5"]; DOMS = ["indomain", "ood_content", "ood_generator"]
ce = json.load(open(f"{SC}/class_eval.json")).get("classes", {}) if os.path.exists(f"{SC}/class_eval.json") else {}
def fmt(ci): return f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "n/a"
out = {}
print(f"{'class':5s} {'domain':13s} {'SAT':>5s} {'cfg':>4s} {'done':>5s} {'mean-ok':>8s} {'95% CI':>15s} {'rate-ok':>8s} {'95% CI':>15s} {'ceiling mean/rate':>21s} {'stress mean/rate':>19s} {'cap':>7s}")
for K in KEYS:
    for dom in DOMS:
        p = f"{SC}/certify_full_{K}{'' if dom == 'indomain' else '_' + dom}.json"
        if not os.path.exists(p): print(f"{K:5s} {dom:13s} (no verdict file yet)"); continue
        d = json.load(open(p)); s = d["summary"]; b = s.get("bootstrap") or {}; rows = d["rows"]
        done = [r for r in rows if not r["pending"]]; pay = [r for r in done if r["min_bits"] > 0]
        cap = f"{sum(r['live_cap_ok'] for r in pay)}/{len(pay)}" if pay else "n/a"
        wc, st = b.get("within_ceiling"), b.get("stress")
        def sub(lab):
            rr = [r for r in done if (not r["stress"]) == (lab == "within_ceiling")]
            if not rr: return "n/a"
            return f"{sum(r['live_mean_ok'] for r in rr) / len(rr):.3f}/{sum(r['live_rate_ok'] for r in rr) / len(rr):.3f} (n={len(rr)})"
        print(f"{K:5s} {dom:13s} {s['n_sat']:5d} {s['n_groups']:4d} {s['n_done']:5d} "
              f"{(s['mean_ok'] if s['mean_ok'] is not None else float('nan')):8.3f} {fmt((b.get('all') or {}).get('mean_ok')):>15s} "
              f"{(s['rate_ok'] if s['rate_ok'] is not None else float('nan')):8.3f} {fmt((b.get('all') or {}).get('rate_ok')):>15s} "
              f"{sub('within_ceiling'):>21s} {sub('stress'):>19s} {cap:>7s}")
        out[f"{K}|{dom}"] = {"n_sat": s["n_sat"], "n_groups": s["n_groups"], "n_done": s["n_done"], "mean_ok": s["mean_ok"], "rate_ok": s["rate_ok"],
                             "bootstrap": b, "capacity_line_held": cap,
                             "pending_attacks": sorted({a for r in rows for a in r["pending"]})}
    if K in ce:
        c = ce[K]
        print(f"      solver: satisfaction {c.get('satisfaction', float('nan')):.3f} @ {c.get('psnr_on_satisfied') or float('nan'):.2f} dB, "
              f"UNSAT {c.get('unsat', '?')} attribution {(c.get('unsat_attribution') or {}).get('by_cause')} blocking {(c.get('unsat_attribution') or {}).get('blocking_columns')}")
json.dump(out, open(f"{SC}/certification_summary.json", "w"), indent=1)
print("wrote certification_summary.json")
