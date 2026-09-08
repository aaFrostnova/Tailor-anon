"""Per-request verdicts from the per-configuration measurements (certify_full.py), with image bootstrap.

Every SAT request of the class reads the cells of its own configuration at its own threshold:
  mean criterion   best-path mean bit accuracy >= max(threshold_base, capacity line of min_bits)
  rate criterion   fraction of images the decoder accepts at the request's threshold (keyed pass or best-path
                   >= threshold on the primary view, or a verified cascade view) >= det_min
A request is live-feasible under a criterion when every attack it named passes it; attacks whose
measurement has not landed (cross-environment chain still running) leave the request pending.

The class-level numbers rest on ONE held-out test set (100 images in domain, 100 per out-of-domain set),
shared by every configuration, so their uncertainty is that of a 100-image test set, not of 2,000
independent requests. The bootstrap resamples the IMAGES with replacement (B draws, one index draw per
image count so that every cell of that size sees the same images, i.e. the pairing across requests and
attacks is kept), recomputes every request's verdict, and reports the 2.5 / 97.5 percentiles of the
class-level live-feasible rate. Cells measured on fewer images (UnMarker, 30) are drawn separately.
Usage: python certify_full_verdicts.py <class 0-4> [det_min=0.9]      (CERT_DOMAIN, CERT_BOOT env)
"""
import sys, os, json, glob
import numpy as np
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
for p in (f"{CF}/scripts/defense", SC): sys.path.insert(0, p)
import watermark_smt_v2 as W


def key_of(s):
    return (tuple(s["order"]), tuple(sorted(k for k, v in s["fe"].items() if v)), tuple(round(s["s"][f], 3) for f in s["order"]))


def load(cls_key, domain="indomain"):
    recs = []
    for f in sorted(glob.glob(f"{SC}/class_eval_{cls_key}_shard*.json")): recs += json.load(open(f))["records"]
    gdir = f"{SC}/certify_full/{cls_key}" if domain == "indomain" else f"{SC}/certify_full_{domain}/{cls_key}"
    groups = {}
    for p in sorted(glob.glob(f"{gdir}/g*.json")):
        g = json.load(open(p)); groups[(tuple(g["order"]), tuple(g["fe"]), tuple(round(g["s"][f], 3) for f in g["order"]))] = g
    return recs, groups


def verdicts(recs, groups, det_min=0.9, B=1000, seed=0):
    """Rows (one per SAT request) and the class summary with bootstrap intervals over images."""
    # (gidx, attack) -> per image: primary best-path ba, keyed pass on the primary view, cascade verified,
    # cascade-view best-path ba (primary where no cascade), decoder verdict at 1%. Two-view cells (views = 2,
    # 2026-09-08) carry the cascade; older single-view cells (configurations without a geometric stage, where
    # the cascade never runs) are read as primary only.
    cells = {}
    for g in groups.values():
        for a, cell in g["attacks"].items():
            n = len(cell["det"]); order = g["order"]
            bp = np.array([max(cell["ba"][f][j] for f in order) for j in range(n)], float)
            if cell.get("views") == 2:
                idv = np.array(cell["idv"], bool); cok = np.array(cell["cas_ok"], bool)
                bc = np.array([max((cell["cas_ba"][f][j] if cell["cas_ba"][f][j] is not None else -1.0) for f in order) for j in range(n)], float)
                bc = np.where(cok, bc, bp)
            else:
                idv = np.array([any(cell["ver"][f][j] for f in order) for j in range(n)], bool); cok = np.zeros(n, bool); bc = bp
            cells[(g["gidx"], a)] = (bp, idv, cok, bc, np.asarray(cell["det"], float))
    rows = []
    for r in recs:
        s = r["solver"]
        if not s: continue
        g = groups.get(key_of(s)); k = len(s["order"]); tau = W.presence_threshold(r["min_ba"], k)
        tau_cap = W.bits_to_ba(r["min_bits"]) if r.get("min_bits", 0) > 0 else 0.0; line = max(tau, tau_cap)
        row = {"i": r["i"], "fpr": r["fpr"], "min_bits": r.get("min_bits", 0), "stress": bool(r.get("stress")), "k": k,
               "tau": tau, "tau_cap": tau_cap, "line": line, "attacks": {}, "pending": [], "gidx": g["gidx"] if g else None,
               "min_psnr": r.get("min_psnr"), "psnr_table": s.get("psnr_db"), "psnr_live": (g or {}).get("psnr_db")}
        # fidelity criterion: the embed the configuration delivers on the test images clears the request's floor
        row["live_psnr_ok"] = (row["psnr_live"] is not None) and (row["min_psnr"] is None or row["psnr_live"] >= row["min_psnr"] - 1e-9)
        for a in r["attacks"]:
            c = cells.get((g["gidx"], a)) if g else None
            if c is None: row["pending"].append(a); continue
            bp, idv, cok, bc, det = c
            # at this request's threshold: the decoder accepts on the primary view (keyed pass or presence at tau)
            # or on a verified cascade view; the mean is over the view it accepts
            prim = idv | (bp >= tau - 1e-12); acc = prim | cok; view = np.where(prim, bp, np.where(cok, bc, bp))
            row["attacks"][a] = {"mean": float(view.mean()), "rate": float(acc.mean()), "det": float(det.mean()), "n": len(bp),
                                 "cascade_used": float(np.mean(cok & ~prim))}
        m = row["attacks"].values()
        row["live_mean_ok"] = (not row["pending"]) and all(v["mean"] >= line - 1e-9 for v in m)
        row["live_rate_ok"] = (not row["pending"]) and all(v["rate"] >= det_min - 1e-9 for v in m)
        row["live_cap_ok"] = (not row["pending"]) and all(v["mean"] >= tau_cap - 1e-9 for v in m)
        rows.append(row)
    done = [r for r in rows if not r["pending"]]
    summary = {"n_sat": len(rows), "n_groups": len(groups), "n_done": len(done), "det_min": det_min,
               "mean_ok": (sum(r["live_mean_ok"] for r in done) / len(done)) if done else None,
               "rate_ok": (sum(r["live_rate_ok"] for r in done) / len(done)) if done else None}
    meas = [r for r in rows if r["psnr_live"] is not None]
    if meas:
        gaps = [r["psnr_live"] - r["psnr_table"] for r in meas if r["psnr_table"] is not None]
        summary["psnr"] = {"n": len(meas), "floor_ok": sum(r["live_psnr_ok"] for r in meas) / len(meas),
                           "live_minus_table_db": {"mean": float(np.mean(gaps)), "min": float(np.min(gaps)), "max": float(np.max(gaps))} if gaps else None}
    # ---- bootstrap over images ----
    if done and B > 0:
        rng = np.random.default_rng(seed)
        sizes = sorted({len(c[0]) for c in cells.values()})
        idx = {n: rng.integers(0, n, size=(B, n)) for n in sizes}
        cache = {}
        def stats(gidx, a, tau, line):
            key = (gidx, a, tau, line)
            if key not in cache:
                bp, idv, cok, bc, _ = cells[(gidx, a)]; I = idx[len(bp)]
                prim = idv | (bp >= tau - 1e-12); acc = prim | cok; view = np.where(prim, bp, np.where(cok, bc, bp))
                cache[key] = (view[I].mean(axis=1) >= line - 1e-9, acc[I].mean(axis=1) >= det_min - 1e-9)
            return cache[key]
        def ci(subset):
            if not subset: return None
            mean_ok = np.ones((B, len(subset)), bool); rate_ok = np.ones((B, len(subset)), bool)
            for qi, r in enumerate(subset):
                for a in r["attacks"]:
                    m_ok, r_ok = stats(r["gidx"], a, r["tau"], r["line"]); mean_ok[:, qi] &= m_ok; rate_ok[:, qi] &= r_ok
            return {"mean_ok": [round(float(x), 4) for x in np.percentile(mean_ok.mean(axis=1), [2.5, 97.5])],
                    "rate_ok": [round(float(x), 4) for x in np.percentile(rate_ok.mean(axis=1), [2.5, 97.5])], "n": len(subset)}
        summary["bootstrap"] = {"B": B, "image_counts": sizes, "all": ci(done),
                                "within_ceiling": ci([r for r in done if not r["stress"]]), "stress": ci([r for r in done if r["stress"]])}
    return rows, summary


if __name__ == "__main__":
    CLS = int(sys.argv[1]); DET_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    K = ["C1", "C2", "C3", "C4", "C5"][CLS]; DOMAIN = os.environ.get("CERT_DOMAIN", "indomain"); B = int(os.environ.get("CERT_BOOT", "1000"))
    recs, groups = load(K, DOMAIN); rows, summary = verdicts(recs, groups, DET_MIN, B)
    done = [r for r in rows if not r["pending"]]
    print(f"{K} [{DOMAIN}]: {summary['n_sat']} SAT requests, {summary['n_groups']} configurations measured, {summary['n_done']} requests fully measured, "
          f"{summary['n_sat'] - summary['n_done']} pending a cross-environment cell")
    if done:
        b = summary["bootstrap"]
        print(f"  live-feasible under the MEAN criterion : {sum(r['live_mean_ok'] for r in done)}/{len(done)} ({summary['mean_ok']:.3f})  95% CI over images {b['all']['mean_ok']}")
        print(f"  live-feasible under the RATE criterion  : {sum(r['live_rate_ok'] for r in done)}/{len(done)} ({summary['rate_ok']:.3f})  95% CI over images {b['all']['rate_ok']}  (det_min {DET_MIN})")
        for lab in ("within_ceiling", "stress"):
            if b[lab]: print(f"     {lab:15s} n={b[lab]['n']:4d}  mean-criterion CI {b[lab]['mean_ok']}  rate-criterion CI {b[lab]['rate_ok']}")
        if summary.get("psnr"):
            q = summary["psnr"]; g = q["live_minus_table_db"]
            print(f"  fidelity: measured PSNR clears the request's floor on {q['floor_ok']:.3f} of {q['n']} requests; live minus table {g['mean']:+.2f} dB (min {g['min']:+.2f}, max {g['max']:+.2f})" if g else "")
        pay = [r for r in done if r["min_bits"] > 0]
        if pay: print(f"  payload requests whose capacity line holds live: {sum(r['live_cap_ok'] for r in pay)}/{len(pay)}")
        worst = sorted(((r["i"], a, v["mean"] - r["tau"], v["rate"], v["det"]) for r in done for a, v in r["attacks"].items()), key=lambda c: c[3])[:8]
        print("  lowest per-image rates (request, attack, mean-tau, rate, det):")
        for i, a, gap, rate, det in worst: print(f"     q{i:04d} {a:14s} {gap:+.3f} {rate:.2f} {det:.2f}")
    json.dump({"cls": K, "domain": DOMAIN, "det_min": DET_MIN, "summary": summary, "rows": rows},
              open(f"{SC}/certify_full_{K}{'' if DOMAIN == 'indomain' else '_' + DOMAIN}.json", "w"))
    print("VERDICTS_DONE")
