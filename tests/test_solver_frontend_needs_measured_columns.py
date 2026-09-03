"""An embed-changing front-end may only be enabled when every requested column has its replacement curve.

The solver reads a stage's replacement curve wherever one was measured and the plain curve elsewhere.
For a stage that changes the embed (the tiled grid, the nested ring, the sync mark) the plain curve
describes an embed the deployment does not perform: tiled TrustMark under VAE compression read the
plain TrustMark curve, promised 0.94, and delivered 0.82 -- a false SAT certified live. So a request
naming a column without the stage's curve must not enable the stage together with the fragment it
changes; once the column is measured the stage is available again.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ("jpeg25", "rot9")


def _solve(sg, attacks, min_ba):
    o, u, rs, ns, al, nf, ps, tm = W.build(
        min_psnr=0.0, max_ms=1e9, attacks=list(attacks), min_ba=min_ba,
        allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
        enable_order=True, continuous_strength=True, surrogate=sg)
    o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"]))
    o.maximize(ps)
    if o.check() != z3.sat:
        return None
    m = o.model()
    return W.frontend_decisions(o, m, rs, ns)


def _table(with_jpeg_curve):
    sg = synthetic_surrogate(attacks=ATTACKS)
    lo, hi = sg.range("TrustMark")
    sg._base[("TrustMark", "rot9")] = PWL([lo, hi], [0.50, 0.50])     # plain TrustMark dies on rot9
    sg._base[("TrustMark", "jpeg25")] = PWL([lo, hi], [0.99, 0.99])   # and is fine on jpeg25
    fe = {"base_fe_tile_TrustMark|rot9": PWL([lo, hi], [0.99, 0.99]),  # the tiled grid rescues rot9
          "penalty_fe_tile": PWL([lo, hi], [0.0, 0.0])}
    if with_jpeg_curve:
        fe["base_fe_tile_TrustMark|jpeg25"] = PWL([lo, hi], [0.98, 0.98])
    sg._frontend = fe
    return sg


def test_stage_without_a_curve_on_a_requested_column_is_not_available():
    # rot9 alone: the tiled grid is measured there, so it may be used
    assert _solve(_table(False), ["rot9"], 0.90)["tile"] is True
    # rot9 + jpeg25: jpeg25 has no tiled curve, so the stage cannot be credited with the plain one;
    # TrustMark alone cannot clear rot9, so the request is honestly infeasible
    assert _solve(_table(False), ["rot9", "jpeg25"], 0.90) is None


def test_stage_becomes_available_once_the_column_is_measured():
    fe = _solve(_table(True), ["rot9", "jpeg25"], 0.90)
    assert fe is not None and fe["tile"] is True
