"""Unit test for the z3-solved-model -> (frags, order, strengths) adapter (Phase 2 Task 2).

CPU-only, synthetic surrogate: solves a surrogate-mode scenario with watermark_smt_v2.build(...)
(enable_order=True, continuous_strength=True) and checks solved_config() reads the model into a
config LiveMeasurer.measure() can execute directly -- a selected-fragment subset, a total embed
order over exactly that subset, and per-fragment strengths inside each fragment's native range.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
sys.path.insert(0, "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
import z3
from surrogate_model import synthetic_surrogate
import watermark_smt_v2 as W
from live_fidelity import solved_config


def test_solved_config_reads_frags_order_strengths():
    sg = synthetic_surrogate(attacks=("jpeg25", "crop75"))
    o, u, rs, ns, al, nf, ps, tm = W.build(
        min_psnr=0.0, max_ms=1e9, attacks=["jpeg25", "crop75"], min_ba=0.60,
        allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
        enable_order=True, continuous_strength=True, surrogate=sg)
    o.maximize(ps)
    assert o.check() == z3.sat
    frags, order, strengths = solved_config(o, u, order_on=True)
    assert set(frags) <= {"VINE", "TrustMark", "VideoSeal"} and len(frags) >= 1
    assert set(order) == set(frags)                           # a total order over the selected set
    for f in frags:
        lo, hi = sg.range(f); assert lo - 1e-6 <= strengths[f] <= hi + 1e-6
