"""The five grounded threat classes (persona_scenarios.py, 2026-08-25) on the CONTINUOUS solver and the
final canonical table: certified optimum, continuous strengths, embed order, the four separately
selectable front-ends, and the presence threshold raised with the fragment count.

Everything the old script computed is recomputed under the new model: the base solve per class, the
single-axis adaptivity sweeps, the fidelity ceiling by binary search, the per-class constraint envelope,
the adversary-strength axis of C4 (now the four measured CtrlRegen+ columns), and the two-axis grids
where UNSAT emerges from a combination. C4's UnMarker column is included only if the table has it.

Usage: python persona_scenarios_v2.py <part>   part in base|adapt|boundary|env0..env4|adversary|combos|merge
Each part writes persona_scenarios_v2_<part>.json; `merge` joins them into persona_scenarios_v2.json.
"""
import sys, os, json, math, time, glob
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (f"{CF}/scripts/defense", SC):
    sys.path.insert(0, p)
import watermark_smt_v2 as W
from surrogate_model import Surrogate
PART = sys.argv[1] if len(sys.argv) > 1 else "base"
sg = Surrogate.from_dict(json.load(open(f"{SC}/surrogate_canonical.json")))
TABLE = json.load(open(f"{SC}/surrogate_canonical.json"))["source"].get("assembled_at")

SIG = ["jpeg25", "blur", "noise", "bright", "contrast"]
CR = ["ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"]   # s05_x2 is out of scope (no measured capacity)
UNM = ["unmarker"] if "unmarker" in sg.attacks else []
PERSONAS = [
    {"name": "C1 Signal / re-encoding", "attacks": SIG, "max_ms": 1000, "min_psnr": 36, "min_ba": 0.90, "min_bits": 0,
     "why": "platform re-encode + resize (social media, mobile share, e-commerce): only the signal family"},
    {"name": "C2 + Geometry (crop / rotation)", "attacks": SIG + ["crop75", "rot9"], "max_ms": 4000, "min_psnr": 34, "min_ba": 0.90, "min_bits": 20,
     "why": "cropping edits and print-then-scan add crop and rotation: the geometric front-ends become decisions"},
    {"name": "C3 + AI regeneration", "attacks": SIG + ["vaeB", "vaeC", "regen"], "max_ms": 4000, "min_psnr": 33, "min_ba": 0.90, "min_bits": 20,
     "why": "img2img platforms VAE-compress and regenerate: the latent fragment is forced up in strength"},
    {"name": "C4 + Adversarial removal (detection evasion)", "attacks": SIG + ["regen"] + CR + UNM, "max_ms": 4000, "min_psnr": 33, "min_ba": 0.63, "min_bits": 0,
     "why": "a knowledgeable adversary runs removal attacks (CtrlRegen+ at every measured strength" + (", UnMarker" if UNM else "") + "): presence is the realistic goal"},
    {"name": "C5 Broad + identity (provenance)", "attacks": SIG + ["crop75", "rot9", "vaeB", "vaeC", "regen"], "max_ms": 8000, "min_psnr": 36, "min_ba": 0.90, "min_bits": 37,
     "why": "C2PA-style provenance: a durable 37-bit identity across a broad transform set, offline ingest"},
]
for p in PERSONAS:
    missing = [a for a in p["attacks"] if a not in sg.attacks]
    assert not missing, f"{p['name']}: unmeasured attacks {missing}"

def db(D): return 10.0 * math.log10(255.0 ** 2 / max(D, 1e-12))
N_SOLVES = [0]
def solve(sc):
    scen = dict(min_psnr=float(sc["min_psnr"]), max_ms=float(sc["max_ms"]), attacks=list(sc["attacks"]),
                min_ba=float(sc["min_ba"]), allow_resync=True, allow_nested=True, min_bits=int(sc["min_bits"]), resolution=512)
    t0 = time.time(); N_SOLVES[0] += 1
    built, m, rounds, cert = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
    if built is None:
        return None
    assert cert, "optimum not certified"
    o, u, rs, ns, al, nf, ps, tm = built
    S = [f for f in W.FR if str(m.eval(u[f])) == "True"]
    pv = getattr(o, "_pvars", {})
    def before(x, y):
        for k, v in pv.items():
            if k == (x, y): return str(m.eval(v)) == "True"
            if k == (y, x): return str(m.eval(v)) != "True"
        return None
    order = sorted(S, key=lambda f: sum(1 for g in S if g != f and before(g, f) is True))
    s = {f: round(float(m.eval(o._svars[f]).as_fraction()), 3) for f in S}
    fe = [k for k, v in W.frontend_decisions(o, m, rs, ns).items() if v]
    return {"frags": order, "s": s, "fe": fe, "psnr": round(db(-float(m.eval(ps).as_fraction())), 2),
            "ms": round(float(m.eval(tm).as_fraction()), 1), "threshold": round(W.presence_threshold(scen["min_ba"], len(S)), 3),
            "rounds": rounds, "sec": round(time.time() - t0, 1)}
def cfgstr(r):
    if r is None: return "UNSAT"
    return "+".join(f"{f}@{r['s'][f]}" for f in r["frags"]) + (" +[" + ",".join(r["fe"]) + "]" if r["fe"] else "")
def base(min_psnr, max_ms, attacks, min_ba, min_bits):
    return dict(min_psnr=min_psnr, max_ms=max_ms, attacks=attacks, min_ba=min_ba, min_bits=min_bits)
def sc_of(p, **kw):
    sc = base(p["min_psnr"], p["max_ms"], p["attacks"], p["min_ba"], p["min_bits"]); sc.update(kw); return sc
def bmax(feas, lo, hi, tol):      # largest feasible value; feas is monotone-decreasing in the value
    if not feas(lo): return None
    if feas(hi): return hi
    while hi - lo > tol:
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if feas(mid) else (lo, mid)
    return round(lo, 2)
def bmin(feas, lo, hi, tol):      # smallest feasible value; feas is monotone-increasing in the value
    if not feas(hi): return None
    if feas(lo): return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2; lo, hi = (lo, mid) if feas(mid) else (mid, hi)
    return round(hi, 2)

out = {"part": PART, "table_assembled_at": TABLE, "unmarker_in_table": bool(UNM)}
t_start = time.time()
if PART == "base":
    out["personas"] = []
    for p in PERSONAS:
        r = solve(sc_of(p))
        out["personas"].append({**{k: p[k] for k in ("name", "why", "attacks", "max_ms", "min_psnr", "min_ba", "min_bits")},
                                "solver": cfgstr(r), "result": r, "sat": r is not None})
        print(f"  {p['name']:44s} {cfgstr(r):48s} {r['psnr'] if r else '---'} dB", flush=True)
elif PART == "adapt":
    def sweep(p, axis, values):
        traj = []
        for v in values:
            r = solve(sc_of(p, **{axis: v})); traj.append({"value": v, "config": cfgstr(r), "result": r})
            print(f"  {p['name'][:2]} {axis}={v}: {cfgstr(r)} ({r['psnr'] if r else '---'} dB)", flush=True)
        return traj
    out["adaptivity_sweeps"] = {
        "C1 Signal : max_ms 200->5000 (latency axis)": sweep(PERSONAS[0], "max_ms", [200, 500, 1000, 2000, 5000]),
        "C3 AI-regeneration : min_psnr 30->40 (fidelity axis)": sweep(PERSONAS[2], "min_psnr", [30, 32, 34, 36, 38, 40]),
        "C5 Provenance : min_ba 0.80->0.95 (robustness axis)": sweep(PERSONAS[4], "min_ba", [0.80, 0.85, 0.90, 0.95]),
    }
elif PART == "boundary":
    def psnr_ceiling(p, lo=28.0, hi=48.0, tol=0.25):
        feas = lambda mp: solve(sc_of(p, min_psnr=mp)) is not None
        if not feas(lo): return {"feasible_at_floor": False}
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if feas(mid): lo = mid
            else: hi = mid
        rl = solve(sc_of(p, min_psnr=lo)); rh = solve(sc_of(p, min_psnr=hi))
        return {"feasible_at_floor": True, "max_feasible_min_psnr": round(lo, 2), "config_just_below": cfgstr(rl), "just_above_is": cfgstr(rh)}
    out["psnr_boundary"] = {}
    for p in PERSONAS:
        out["psnr_boundary"][p["name"]] = psnr_ceiling(p); print(f"  {p['name']}: {out['psnr_boundary'][p['name']]}", flush=True)
elif PART.startswith("env"):
    p = PERSONAS[int(PART[3:])]
    ok = lambda **kw: solve(sc_of(p, **kw)) is not None
    out["constraint_envelope"] = {p["name"]: {
        "max_ms_floor": bmin(lambda v: ok(max_ms=v), 50.0, 8000.0, 25.0),
        "min_psnr_ceil": bmax(lambda v: ok(min_psnr=v), 28.0, 48.0, 0.25),
        "min_ba_ceil": bmax(lambda v: ok(min_ba=v), 0.50, 1.00, 0.01),
        "min_bits_ceil": max([bb for bb in (0, 10, 20, 37, 50) if ok(min_bits=bb)], default=None)}}
    print(f"  {p['name']}: {out['constraint_envelope'][p['name']]}", flush=True)
elif PART == "adversary":
    def adv_row(col):
        atts = SIG + ["regen", col] + UNM
        r = solve(base(33, 4000, atts, 0.63, 0))
        bits_ceil = max([bb for bb in (0, 10, 20, 37) if solve(base(33, 4000, atts, 0.63, bb)) is not None], default=None)
        ba_ceil = bmax(lambda v: solve(base(33, 4000, atts, v, 0)) is not None, 0.50, 1.00, 0.01)
        return {"config": cfgstr(r), "result": r, "min_bits_ceil": bits_ceil, "min_ba_ceil": ba_ceil}
    out["adversary_strength_C4"] = {}
    for col in CR:
        out["adversary_strength_C4"][col] = adv_row(col); print(f"  {col}: {out['adversary_strength_C4'][col]}", flush=True)
elif PART == "combos":
    def grid(attacks, max_ms, min_psnr, min_ba, rows_axis, cols_axis):
        (rn, rk, rv), (cn, ck, cv) = rows_axis, cols_axis
        g = {}
        for rval in rv:
            g[f"{rn}={rval}"] = {}
            for cval in cv:
                sc = base(min_psnr, max_ms, attacks, min_ba, 0)
                for key, val in ((rk, rval), (ck, cval)):
                    if key == "attacks_add": sc["attacks"] = attacks + (["rinse2x"] if val else [])
                    else: sc[key] = val
                r = solve(sc); g[f"{rn}={rval}"][f"{cn}={cval}"] = cfgstr(r)
                print(f"  {rn}={rval} {cn}={cval}: {cfgstr(r)}", flush=True)
        return g
    out["combination_unsat"] = {
        "identity x rinse (base: signal+regen, presence, 33dB, 4s)":
            grid(SIG + ["regen"], 4000, 33, 0.63, ("min_bits", "min_bits", [0, 20, 37]), ("rinse2x", "attacks_add", [False, True])),
        "fidelity x robustness (base: signal+regen, 4s, presence bits)":
            grid(SIG + ["regen"], 4000, 34, 0.80, ("min_psnr", "min_psnr", [34, 38, 41]), ("min_ba", "min_ba", [0.80, 0.90, 0.95])),
    }
elif PART == "merge":
    merged = {"table_assembled_at": TABLE, "unmarker_in_table": bool(UNM), "personas_def": PERSONAS}
    for fp in sorted(glob.glob(f"{SC}/persona_scenarios_v2_*.json")):
        d = json.load(open(fp)); assert d["table_assembled_at"] == TABLE, f"{fp} was computed on another table"
        for k, v in d.items():
            if k in ("part", "table_assembled_at", "unmarker_in_table", "n_solves", "sec"): continue
            if isinstance(v, dict) and isinstance(merged.get(k), dict): merged[k].update(v)
            else: merged[k] = v
    json.dump(merged, open(f"{SC}/persona_scenarios_v2.json", "w"), indent=2)
    print("merged:", sorted(k for k in merged if k not in ("personas_def",)), "->", f"{SC}/persona_scenarios_v2.json")
    sys.exit(0)
else:
    sys.exit(f"unknown part {PART}")
out["n_solves"] = N_SOLVES[0]; out["sec"] = round(time.time() - t_start, 1)
json.dump(out, open(f"{SC}/persona_scenarios_v2_{PART}.json", "w"), indent=2)
print(f"PERSONA_V2_{PART.upper()}_DONE  ({N_SOLVES[0]} certified solves, {out['sec']}s)", flush=True)
