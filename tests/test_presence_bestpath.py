"""Presence is decided best-path too, with the false-positive budget split over the tests.

The coverage clause the solver reasons with is best-path: an attack is covered when SOME selected
fragment still decodes at the required bit accuracy. The deployed decoder applied that rule to the two
cryptographic tests but ran its presence (zero-bit) test on the FUSED codeword only, which a dead
fragment dilutes: on crop-then-JPEG the surviving VideoSeal read 0.63, the fused value 0.58, and the
decoder fired on 23.5% of images although the solver had declared the column covered. So the presence
test is run per fragment as well, and every zero-bit threshold is raised to keep the union of the tests
inside the same 1% budget; the solver's presence threshold rises with the fragment count the same way.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import numpy as np
import z3
from scipy.stats import binom
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W


def test_solver_presence_threshold_rises_with_fragment_count_and_leaves_identity_alone():
    b1 = W.beta_from_fpr(1e-2)
    assert abs(W.presence_threshold(b1, 1) - b1) < 1e-12
    b2, b3 = W.presence_threshold(b1, 2), W.presence_threshold(b1, 3)
    assert b1 < b2 <= b3, (b1, b2, b3)
    # never above the identity level, and an identity-level threshold is untouched
    bid = W.beta_from_fpr(2.0 ** -37)
    assert b3 <= bid and W.presence_threshold(bid, 3) == bid


def test_deployed_presence_thresholds_hold_the_union_bound():
    from eval_matrix import presence_tau
    n = 100
    for k in (1, 2, 3):
        n_tests = 1 if k == 1 else k + 1
        tau = presence_tau(k, n)
        # each zero-bit test at threshold tau has tail mass <= 0.01 / n_tests, so the union stays <= 1%
        tail = binom.sf(int(round(tau * n)) - 1, n, 0.5)
        assert tail * n_tests <= 0.01 + 1e-12, (k, tau, tail)
    assert presence_tau(1, n) < presence_tau(2, n) <= presence_tau(3, n)


def test_a_single_live_fragment_is_enough_for_presence():
    from eval_matrix import presence_detected, presence_tau
    tau2 = presence_tau(2, 100)
    # one fragment clearly alive, the other at chance: the fused value is diluted below tau but the
    # per-fragment test fires
    assert presence_detected({"a": 0.72, "b": 0.50}, fused_ba=0.58, n_frag=2) is True
    # nothing alive anywhere: no test fires
    assert presence_detected({"a": 0.55, "b": 0.52}, fused_ba=0.56, n_frag=2) is False
    # the fused test still counts when every fragment is weak but their fusion is not
    assert presence_detected({"a": 0.60, "b": 0.60}, fused_ba=tau2 + 0.01, n_frag=2) is True


def test_two_fragments_must_clear_the_raised_threshold():
    """A curve flat exactly at the one-fragment threshold clears it alone and fails with a partner."""
    sg = synthetic_surrogate(attacks=("jpeg25",))
    b1 = W.beta_from_fpr(1e-2)
    for f in ("TrustMark", "VINE"):
        lo, hi = sg.range(f)
        sg._base[(f, "jpeg25")] = PWL([lo, hi], [b1 + 0.002, b1 + 0.002])
    def solve(force):
        o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=["jpeg25"], min_ba=b1,
                                              allow_resync=False, allow_nested=False, min_bits=0,
                                              resolution=512, enable_order=True, continuous_strength=True,
                                              surrogate=sg)
        for f in W.FR: o.add(u[f] == (f in force))
        return o.check() == z3.sat
    assert solve(("TrustMark",)) is True
    assert solve(("TrustMark", "VINE")) is False
