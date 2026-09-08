"""Is the canonical table complete? Every cell the solver can read on any in-scope request must be measured.

What the solver reads (watermark_smt_v2.add_strength_order, build._fe_ms, surrogate_model.Surrogate):
  base(f, a)                      every fragment x every attack, spanning the fragment's strength range
  delta(g, f, a) at s_g           every ordered pair x every attack, spanning the OVERWRITING fragment's range
                                  (the UnMarker column keeps mid-strength constants: it is live-gated on every request)
  base_fe_<stage>_<f>|<a>         for every fragment a stage RE-EMBEDS (FE_AFFECTS: resync all, scale VINE,
                                  tile TrustMark) on every attack, spanning f's range; angle on the 7 geometric columns
  per-image cells                 plain: every fragment x attack; with a stage on: every re-embedded (stage, f) x attack,
                                  plus angle on the geometric columns (the stage's read changes there)
  latency_ms_<stage>|<a>          resync/scale/tile on every attack, angle on the geometric columns
  d(f), e(f, g)                   present; d spanning the range
Knot counts are checked on the RAW table (the thinned canonical keeps only the bends), spans on the canonical.
Usage: python audit_table_completeness.py [canonical.json] [canonical_raw.json]   -> prints the report, exits 1 on gaps
"""
import json, sys, os, math
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
# the offline table: WM_TABLE, else the scratch copy, else the one shipped in the repository (data/)
TABLE = os.environ.get("WM_TABLE") or (f"{SC}/surrogate_canonical.json" if os.path.exists(f"{SC}/surrogate_canonical.json") else f"{CF}/data/surrogate_canonical.json")
FR = ["VINE", "TrustMark", "VideoSeal"]
GEO7 = ["crop75", "crop50", "rot9", "rs256", "hflip", "crop_jpeg", "border20"]
AFFECTS = {"resync": FR, "scale": ["VINE"], "tile": ["TrustMark"]}
# The adversarial column is LIVE-ONLY (watermark_smt_v2.ADVERSARIAL): the table never settles it and a
# missing stage curve there does not restrict the solver, so completeness asks only for the plain cells
# and the search prior (VINE with the ring), not for every stage / per-image / latency cell.
LIVE_ONLY = {"unmarker"}
PRIOR = ["base_fe_scale_VINE|unmarker"]
# Cells whose knots start above the range but cannot mislead the solver, with the reason on record.
WHITELIST = {
    "base:VINE|unmarker": "plain VINE under UnMarker reads chance at every strength (0.497 at 0.2, 0.487 at 1.0); "
                          "the clamp below 0.2 returns chance as well",
    "perimage:VINE|unmarker": "the per-image cell of the same chance-level column (knots 0.2 to 1.0); no request "
                              "can clear a presence threshold on it at any strength",
}


def audit(canonical=TABLE, raw=f"{SC}/surrogate_canonical_raw.json"):
    T = json.load(open(canonical)); Rw = json.load(open(raw)) if os.path.exists(raw) else T
    A = T["attacks"]; R = T["ranges"]; issues = []; whitelisted = []
    def spans(c, lo, hi, tol=1e-9): return c["xs"][0] <= lo + tol and c["xs"][-1] >= hi - tol
    def nan(c): return any(isinstance(y, float) and y != y for y in c["ys"])
    def note(kind, key, msg):
        if f"{kind}:{key}" in WHITELIST: whitelisted.append((kind, key, WHITELIST[f"{kind}:{key}"]))
        else: issues.append(f"{kind} {key}: {msg}")
    # base
    for f in FR:
        for a in A:
            c = T["base"].get(f"{f}|{a}")
            if c is None: issues.append(f"base {f}|{a}: missing"); continue
            if nan(c): issues.append(f"base {f}|{a}: NaN knot")
            if not spans(c, *R[f]): note("base", f"{f}|{a}", f"spans {c['xs'][0]}..{c['xs'][-1]}, range {R[f]}")
    # delta, at the overwriting fragment's strength
    for g in FR:
        for f in FR:
            if g == f: continue
            for a in A:
                k = f"{g}|{f}|{a}"; c = T["delta"].get(k); cr = Rw["delta"].get(k, c)
                if c is None: issues.append(f"delta {k}: missing"); continue
                if nan(c): issues.append(f"delta {k}: NaN knot")
                if a == "unmarker": continue                      # mid-strength constant by design (live-gated column)
                if not spans(c, *R[g]): note("delta", k, f"spans {c['xs'][0]}..{c['xs'][-1]}, range of {g} {R[g]}")
                if len(cr["xs"]) < 5: note("delta", k, f"only {len(cr['xs'])} measured knots (a mid-strength constant, not a sweep)")
    # front-end replacement curves
    fe = T["frontend"]
    for k in PRIOR:
        if k not in fe: issues.append(f"frontend {k}: missing (the live-only column's search prior)")
    for stage, frs in AFFECTS.items():
        for f in frs:
            for a in A:
                if a in LIVE_ONLY: continue
                k = f"base_fe_{stage}_{f}|{a}"; c = fe.get(k)
                if c is None: issues.append(f"frontend {k}: missing"); continue
                if nan(c): issues.append(f"frontend {k}: NaN knot")
                if not spans(c, *R[f]): note("frontend", k, f"spans {c['xs'][0]}..{c['xs'][-1]}, range {R[f]}")
    for f in FR:
        for a in GEO7:
            if f"base_fe_angle_{f}|{a}" not in fe: issues.append(f"frontend base_fe_angle_{f}|{a}: missing")
    for stage in ("resync", "scale", "tile", "angle"):
        if f"penalty_fe_{stage}" not in fe and stage not in ("scale", "angle"): issues.append(f"frontend penalty_fe_{stage}: missing")
    if "nested_penalty" not in fe: issues.append("frontend nested_penalty: missing")
    # per-image cells
    P = T.get("perimage", {})
    for f in FR:
        for a in A:
            p = P.get(f"{f}|{a}")
            if p is None: issues.append(f"perimage {f}|{a}: missing"); continue
            if not spans(p, *R[f]): note("perimage", f"{f}|{a}", f"knots {p['xs'][0]}..{p['xs'][-1]}, range {R[f]}")
    if "scale:VINE|unmarker" not in P: issues.append("perimage scale:VINE|unmarker: missing (the live-only column's search prior)")
    for stage, frs in AFFECTS.items():
        for f in frs:
            for a in A:
                if a in LIVE_ONLY: continue
                k = f"{stage}:{f}|{a}"; p = P.get(k)
                if p is None: issues.append(f"perimage {k}: missing"); continue
                if not spans(p, *R[f]): note("perimage", k, f"knots {p['xs'][0]}..{p['xs'][-1]}, range {R[f]}")
    for f in FR:
        for a in GEO7:
            if f"angle:{f}|{a}" not in P: issues.append(f"perimage angle:{f}|{a}: missing")
    # latency
    L = T.get("latency", {})
    for stage in ("resync", "scale", "tile"):
        for a in A:
            if a in LIVE_ONLY: continue
            if f"latency_ms_{stage}|{a}" not in L: issues.append(f"latency latency_ms_{stage}|{a}: missing")
    for a in GEO7:
        if f"latency_ms_angle|{a}" not in L: issues.append(f"latency latency_ms_angle|{a}: missing")
    # distortion
    for f in FR:
        c = T["d"].get(f)
        if c is None or not spans(c, *R[f]): issues.append(f"d {f}: missing or not spanning {R[f]}")
    for pair in ("TrustMark|VINE", "VINE|VideoSeal", "TrustMark|VideoSeal"):
        if pair not in T["e"]: issues.append(f"e {pair}: missing")
    summary = {"attacks": len(A), "base": len(T["base"]), "delta": len(T["delta"]),
               "delta_sweeps": sum(1 for k, c in Rw["delta"].items() if len(c["xs"]) >= 5),
               "frontend": len(fe), "perimage": len(P), "latency": len(L), "issues": len(issues), "whitelisted": len(whitelisted)}
    return issues, whitelisted, summary


if __name__ == "__main__":
    args = sys.argv[1:]
    issues, wl, summary = audit(*args) if args else audit()
    print("summary:", json.dumps(summary))
    for kind, key, why in wl: print(f"  whitelisted {kind} {key}: {why}")
    for i in issues: print("  GAP", i)
    print("AUDIT_OK" if not issues else f"AUDIT_GAPS {len(issues)}")
    sys.exit(0 if not issues else 1)
