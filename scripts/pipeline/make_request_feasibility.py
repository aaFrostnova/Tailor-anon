"""What the offline table says each column can reach, per false-positive budget: the request sampler's ceilings.

For every column and budget (threshold tau(k) for k = 1 or 2 selected fragments), the cheapest single
fragment, with any admissible front-end stage, whose MEAN bit accuracy clears tau + margin AND whose
per-image acceptance rate at tau reaches det_min (the two conditions the solver's coverage clause and
acceptance floor impose), with the strength that first does so and the fragment's solo PSNR there.
A column with no such fragment at a budget is beyond the table's reach at that budget for every
configuration: a request naming it there is infeasible by the fragments' limits, not by composition.
Also the capacity ceiling of every column (reliable bits at its best mean, W.ba_to_bits) and the columns
that VINE cannot carry even with the ring (they force a second fragment, k = 2).

Writes request_feasibility_matrix.json, read by class_defs.py. Regenerate after the table changes.
"""
import json, sys, os, math, subprocess
import numpy as np
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
# the offline table: WM_TABLE, else the scratch copy, else the one shipped in the repository (data/)
TABLE = os.environ.get("WM_TABLE") or (f"{SC}/surrogate_canonical.json" if os.path.exists(f"{SC}/surrogate_canonical.json") else f"{CF}/data/surrogate_canonical.json")
sys.path.insert(0, f"{CF}/scripts/defense")
import watermark_smt_v2 as W
sg = json.load(open(TABLE)); base, fe, pi, D = sg["base"], sg["frontend"], sg["perimage"], sg["d"]
FR = ["VINE", "TrustMark", "VideoSeal"]; A = sg["attacks"]
BUDGETS = [1e-1, 1e-2, 1e-4, 1e-6, 1e-9, 2.0 ** -37]
MARGIN = W.DEFAULT_MARGIN
# v4 (2026-09-08): the same margins the solver applies. The per-image floor is asked of the table at det_min plus
# its rate margin (0.96), the capacity line at bits_to_ba(bits) plus the mean margin, so that "reachable" here
# means "the solver will accept it". FEAS_V<4 reproduces the earlier, margin-free ceilings.
V = int(os.environ.get("FEAS_V", "4"))
DET_MIN = W.DEFAULT_DET_MIN + (W.DEFAULT_RATE_MARGIN if V >= 4 else 0.0)
CAP_M = MARGIN if V >= 4 else 0.0
def ev(c, x): return float(np.interp(x, c["xs"], c["ys"]))
def psnr(f, s): return 10 * math.log10(255.0 ** 2 / ev(D[f], s))
def rate_at(rec, s, tau):
    r = [float(np.mean((np.asarray(b) >= tau - 1e-12) | (np.asarray(v) if v is not None else False)))
         for b, v in zip(rec["ba"], rec["ver"] or [None] * len(rec["xs"]))]
    return float(np.interp(s, rec["xs"], r))
LAT = sg.get("latency", {})
# v3 (2026-09-08): the live-only column's reference route is the ring. UnMarker is not a measured column;
# its plain cells are an n=30 search prior, and at fpr 1e-1 the VideoSeal prior (0.59 at strength 1.25)
# outranked the ring by 0.6 dB of solo PSNR, so the sampler budgeted no scale-search latency and paired a
# second fragment with a one-fragment fidelity allowance (3 C4 requests UNSAT by construction). FEAS_V=2
# reproduces the previous matrix.
LIVE_ONLY = {"unmarker"}
def cells(f, a):
    """the plain cell and the stage cells of (f, a) that the solver can actually use for this column: a stage
    is selectable only where it is responsible for some column (its replacement curve gains >= 0.05 over the
    plain one there, watermark_smt_v2 fe_gain_min), so for the ceiling of a column only stages responsible on
    THAT column count; a stage enabled for another column may still apply here, but a ceiling that rests on
    that is one noisy knot away from being wrong (CtrlRegen+ at step 0.7 sat exactly on 0.90 that way).
    A stage cell without per-image values reads the plain ones. Each cell carries the latency it costs."""
    plain = base[f"{f}|{a}"]
    if V >= 3 and a in LIVE_ONLY:
        if f != "VINE": return []
        c = fe["base_fe_scale_VINE|unmarker"]
        return [("scale", c, pi.get(f"scale:{f}|{a}") or pi.get(f"{f}|{a}"), float(LAT.get(f"latency_ms_scale|{a}", 0.0)))]
    out = [("plain", plain, pi.get(f"{f}|{a}"), 0.0)]
    for st in ("resync", "scale", "angle", "tile"):
        c = fe.get(f"base_fe_{st}_{f}|{a}")
        if c is None: continue
        grid = sorted(set(c["xs"]) | set(plain["xs"]))
        if max(ev(c, x) - ev(plain, x) for x in grid) < 0.05: continue
        out.append((st, c, pi.get(f"{st}:{f}|{a}") or pi.get(f"{f}|{a}"), float(LAT.get(f"latency_ms_{st}|{a}", 0.0))))
    return out
need = {}
for a in A:
    for k in (1, 2):
        for bi, b in enumerate(BUDGETS):
            tau = W.presence_threshold(W.beta_from_fpr(b), k); cands = []
            for f in FR:
                lo, hi = sg["ranges"][f]; grid = np.linspace(lo, hi, 400)
                for st, c, rec, lat in cells(f, a):
                    ok = [s for s in grid if ev(c, s) >= tau + MARGIN and (rec is None or rate_at(rec, s, tau) >= DET_MIN)]
                    if ok: cands.append((round(psnr(f, min(ok)), 2), f, st, round(float(min(ok)), 3), round(lat, 1)))
            need[f"{a}|k{k}|{bi}"] = list(max(cands)) if cands else None      # [solo PSNR, fragment, stage, strength, stage latency ms]
# v3: the same reference at the PAYLOAD's line. The capacity clause holds the carrying fragment to
# bits_to_ba(min_bits) on every column whatever the budget, so a request's fidelity floor has to be
# drawn against max(tau + margin, capacity line); against the presence line alone, loose budgets paired
# 37/50-bit payloads with 38 dB floors that no configuration reaches (C5: 125 requests UNSAT by construction).
BITS = [10, 20, 37, 50]; need_cap = {}
if V >= 3:
    for a in A:
        for k in (1, 2):
            for bi, b in enumerate(BUDGETS):
                tau = W.presence_threshold(W.beta_from_fpr(b), k)
                for bits in BITS:
                    line = max(tau + MARGIN, W.bits_to_ba(bits) + CAP_M); cands = []
                    for f in FR:
                        lo, hi = sg["ranges"][f]; grid = np.linspace(lo, hi, 400)
                        for st, c, rec, lat in cells(f, a):
                            ok = [s for s in grid if ev(c, s) >= line and (rec is None or rate_at(rec, s, tau) >= DET_MIN)]
                            if ok: cands.append((round(psnr(f, min(ok)), 2), f, st, round(float(min(ok)), 3), round(lat, 1)))
                    need_cap[f"{a}|k{k}|{bi}|{bits}"] = list(max(cands)) if cands else None
# capacity ceiling per column: the payload whose line (plus the mean margin the solver applies) the best mean still reaches
capbits = {a: round(W.ba_to_bits(max(0.5, max(max(c["ys"]) for f in FR for _, c, _, _ in cells(f, a)) - CAP_M)), 1) for a in A}
# columns VINE cannot carry with or without the ring but another fragment can: a request naming one needs
# a second fragment for it, and the solver then holds every column to the two-fragment threshold. A column
# no fragment carries (CtrlRegen+ at step 0.7) is not one of them: VINE is still the best there, k stays 1.
def _best(f, a): return max(max(c["ys"]) for _, c, _, _ in cells(f, a))
needs2 = [a for a in A if _best("VINE", a) < 0.7 and max(_best("TrustMark", a), _best("VideoSeal", a)) >= 0.7]
try: git = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception: git = "unknown"
out = {"budgets": BUDGETS, "tau": {f"k{k}": [round(W.presence_threshold(W.beta_from_fpr(b), k), 3) for b in BUDGETS] for k in (1, 2, 3)},
       "margin": MARGIN, "det_min": DET_MIN, "need": need, "need_cap": need_cap, "version": V, "capacity_bits": capbits, "needs_second_fragment": needs2,
       "stage_latency_ms": {k: v for k, v in LAT.items()}, "fragment_ms": {f: W.cost_ms(f) for f in FR},
       "source": {"table": "surrogate_canonical.json", "table_measured_at": (sg.get("source") or {}).get("measured_at"), "git": git,
                  "note": "need[column|k<k>|<budget index>] = [solo PSNR dB, fragment, stage, strength, stage latency ms] of the cheapest fragment whose mean "
                          "clears tau+margin with per-image rate >= det_min at tau; null = beyond every fragment at that budget"}}
json.dump(out, open(f"{SC}/request_feasibility_matrix.json", "w"), indent=1)
print("tightest reachable budget per column (k=1 / k=2), capacity bits:")
for a in A:
    c1 = max([bi for bi in range(6) if need[f"{a}|k1|{bi}"]] or [-1]); c2 = max([bi for bi in range(6) if need[f"{a}|k2|{bi}"]] or [-1])
    print(f"  {a:14s} {BUDGETS[c1] if c1 >= 0 else 'none':>8} / {BUDGETS[c2] if c2 >= 0 else 'none':>8}   {capbits[a]:5.1f} bits" + ("   needs a second fragment" if a in needs2 else ""))
print("wrote request_feasibility_matrix.json")
