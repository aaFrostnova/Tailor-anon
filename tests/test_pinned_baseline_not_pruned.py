"""A pinned baseline is judged on coverage, not on the solver's search heuristics.

The responsible-column rule fixes a front-end off when no requested column measurably benefits from it.
That is a device for the search. A fixed baseline that pins the SyncSeal mark ON is a question about
that operating point: applied to it, the rule contradicted the pin and every request without geometry
came back infeasible for a reason unrelated to coverage (the always-full baseline's satisfaction fell
from 0.75 to 0.35 between two revisions that changed nothing about its curves).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ("jpeg25",)


def _sg():
    sg = synthetic_surrogate(attacks=ATTACKS)
    for f in ("VINE", "TrustMark", "VideoSeal"):
        lo, hi = sg.range(f)
        sg._base[(f, "jpeg25")] = PWL([lo, hi], [0.99, 0.99])
    # resync changes every fragment's embed, so the measured-column rule wants its replacement curve for
    # each selected fragment on the requested column; give all three one that does not beat the plain
    # curve (a noise-level gain), which is exactly the case the pruning rule fixes the stage off for
    sg._frontend = {"penalty_fe_resync": PWL(list(sg.range("TrustMark")), [0.5, 0.5])}
    for f in ("VINE", "TrustMark", "VideoSeal"):
        lo, hi = sg.range(f)
        sg._frontend[f"base_fe_resync_{f}|jpeg25"] = PWL([lo, hi], [0.99, 0.99])
    return sg


def _pinned(fe_gain_min):
    o, u, rs, ns, *_ = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9, allow_resync=True,
                               allow_nested=True, min_bits=0, resolution=512, enable_order=True,
                               continuous_strength=True, surrogate=_sg(), margin=0.0, fe_gain_min=fe_gain_min)
    for f in W.FR: o.add(u[f])
    o.add(rs == True); o.add(ns == False)
    return o.check()


def test_pruning_would_reject_the_pinned_point_and_the_baseline_path_turns_it_off():
    assert _pinned(0.05) == z3.unsat          # the search rule, applied to a pin, contradicts it
    assert _pinned(-1.0) == z3.sat            # what solve_fixed now does: coverage decides, nothing else


def test_solve_fixed_builds_without_pruning():
    src = open(os.path.join("/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k", "solver_eval_continuous.py")).read()
    assert "_build(sc, prune=False)" in src.split("def solve_fixed")[1].split("def ")[0]
