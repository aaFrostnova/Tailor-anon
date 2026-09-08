"""How many recorded solutions would the floor margins touch (route A of the 2026-09-08 margin decision)?

For every SAT record, at the configuration's strengths and the request's tau(k): a column is BINDING for the
per-image floor when no selected fragment reaches a table per-image rate >= RATE_NEW at tau (the stage cell
where the stage re-embeds the fragment, else the plain cell), and binding for the capacity line when
min_bits > 0 and no selected fragment's table mean reaches bits_to_ba(min_bits) + CAP_MARGIN (partner
interference ignored: it only makes the mean lower, so this count is a floor). A request is a re-solve
candidate if any column binds; it is a possible UNSAT if on some column no fragment can reach the new floor
at ANY strength of its range. Usage: python count_binding.py [RATE_NEW=0.96] [CAP_MARGIN=0.02]
"""
import os
import sys, json, glob, collections
import numpy as np
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
# the offline table: WM_TABLE, else the scratch copy, else the one shipped in the repository (data/)
TABLE = os.environ.get("WM_TABLE") or (f"{SC}/surrogate_canonical.json" if os.path.exists(f"{SC}/surrogate_canonical.json") else f"{CF}/data/surrogate_canonical.json")
RATE_NEW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.96; CAP_M = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
sg = json.load(open(TABLE)); base, fe, pi = sg["base"], sg["frontend"], sg["perimage"]
def ev(c, x): return float(np.interp(x, c["xs"], c["ys"]))
def rate_at(rec, s, tau):
    r = [float(np.mean((np.asarray(b) >= tau - 1e-12) | (np.asarray(v) if v is not None else False))) for b, v in zip(rec["ba"], rec["ver"] or [None] * len(rec["xs"]))]
    return float(np.interp(s, rec["xs"], r)), max(r)
FE_AFFECTS = {"resync": ["VINE", "TrustMark", "VideoSeal"], "scale": ["VINE"], "tile": ["TrustMark"]}   # which fragments a stage re-embeds
def cells(f, a, stage):
    """(mean curve, per-image cell) the solver reads for fragment f on column a under the configuration's stage."""
    if stage and f in FE_AFFECTS.get(stage, []):
        return fe.get(f"base_fe_{stage}_{f}|{a}") or base[f"{f}|{a}"], pi.get(f"{stage}:{f}|{a}")      # no stage cell -> no credit
    return base[f"{f}|{a}"], pi.get(f"{f}|{a}")
tot = {}
for K in ("C1", "C2", "C3", "C4", "C5"):
    recs = [r for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")) for r in json.load(open(f))["records"]]
    n = 0; bind_rate = 0; bind_cap = 0; bind_any = 0; maybe_unsat = 0; cols = collections.Counter()
    for r in recs:
        s = r["solver"]
        if not s: continue
        n += 1; k = len(s["frags"]); tau = W.presence_threshold(r["min_ba"], k); cap = W.bits_to_ba(r["min_bits"]) if r["min_bits"] > 0 else None
        stage = next((c for c, v in s["fe"].items() if v and c in ("resync", "scale", "tile")), None)
        br = bc = un = False
        for a in r["attacks"]:
            if a == "unmarker" or a.startswith("ctrlregen"): continue        # live-only / cross-environment columns judged live
            ok_r = False; ok_c = (cap is None); can_r = False; can_c = (cap is None)
            for f in s["frags"]:
                mc, rec = cells(f, a, stage)
                m = ev(mc, s["s"][f]); mean_ok = m >= tau + W.DEFAULT_MARGIN - 1e-9
                if rec is not None:
                    rt, rmax = rate_at(rec, s["s"][f], tau)
                    if mean_ok and rt >= RATE_NEW - 1e-9: ok_r = True
                    if rmax >= RATE_NEW - 1e-9 and max(mc["ys"]) >= tau + W.DEFAULT_MARGIN: can_r = True
                else:
                    if mean_ok: ok_r = True; can_r = True          # no per-image cell for this fragment/column: floor not applied there
                if cap is not None:
                    if m >= cap + CAP_M - 1e-9: ok_c = True
                    if max(mc["ys"]) >= cap + CAP_M - 1e-9: can_c = True
            if not ok_r: br = True; cols[a] += 1
            if not ok_c: bc = True; cols[a + "(cap)"] += 1
            if not can_r or not can_c: un = True
        bind_rate += br; bind_cap += bc; bind_any += (br or bc); maybe_unsat += ((br or bc) and un)
    tot[K] = (n, bind_rate, bind_cap, bind_any, maybe_unsat)
    print(f"{K}: SAT {n}; binding per-image floor {bind_rate}, binding capacity line {bind_cap}, re-solve candidates {bind_any} ({bind_any/n:.1%}); of which possibly UNSAT under the margins {maybe_unsat}; columns: {cols.most_common(6)}")
print("TOTAL re-solve candidates:", sum(v[3] for v in tot.values()), " possibly UNSAT:", sum(v[4] for v in tot.values()))
