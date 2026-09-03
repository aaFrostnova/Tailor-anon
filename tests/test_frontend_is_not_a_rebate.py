"""A front-end is bought for coverage, never as a fidelity rebate, and an embed it changes carries no
unmeasured capacity.

The tiled grid's measured embed-side cost is slightly negative. Left signed, the solver attached the grid
to a TrustMark that covered nothing in a signal+VAE+regeneration request, for 0.04 dB; and because
capacity curves are measured on the plain embed, that tiled TrustMark was credited with 20 identity
bits it cannot carry (its accuracy under VAE sits near chance at that strength).
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
    lo, hi = sg.range("TrustMark")
    # the plain embed already covers the column; the tiled one measurably helps (so the stage is not
    # pruned as irresponsible) yet must still not be bought for its negative cost
    sg._base[("TrustMark", "jpeg25")] = PWL([lo, hi], [0.90, 0.90])
    sg._frontend = {"base_fe_tile_TrustMark|jpeg25": PWL([lo, hi], [0.99, 0.99]),
                    "penalty_fe_tile": PWL([lo, hi], [-1.0, -1.0])}          # a measured rebate
    sg._cap = {("TrustMark", "jpeg25"): PWL([lo, hi], [60.0, 60.0])}
    return sg


def _solve(sg, min_bits=0):
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                                          allow_resync=True, allow_nested=True, min_bits=min_bits,
                                          resolution=512, enable_order=True, continuous_strength=True, surrogate=sg)
    o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"]))
    o.maximize(ps)
    if o.check() != z3.sat:
        return None
    m = o.model()
    return {"fe": W.frontend_decisions(o, m, rs, ns), "D": -float(m.eval(ps).as_fraction())}


def test_a_negative_cost_does_not_buy_the_front_end():
    """With the rebate clamped, tile and no-tile tie on fidelity and the tie-break must not prefer it."""
    r = _solve(_sg())
    assert r is not None
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                                          allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
                                          enable_order=True, continuous_strength=True, surrogate=_sg())
    o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"])); o.add(o._fevars["tile"])
    o.maximize(ps); assert o.check() == z3.sat
    d_tile = -float(o.model().eval(ps).as_fraction())
    assert d_tile >= r["D"] - 1e-9, "the tiled grid was cheaper than the plain embed in the objective"


def test_tiled_trustmark_carries_no_capacity():
    sg = _sg()
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                                          allow_resync=True, allow_nested=True, min_bits=37, resolution=512,
                                          enable_order=True, continuous_strength=True, surrogate=sg)
    o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"]))
    o.add(o._fevars["tile"])
    assert o.check() == z3.unsat, "a tiled TrustMark must not satisfy an identity requirement on its plain capacity"
    o2, u2, *_ = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                         allow_resync=True, allow_nested=True, min_bits=37, resolution=512,
                         enable_order=True, continuous_strength=True, surrogate=sg)
    o2.add(u2["TrustMark"]); o2.add(z3.Not(u2["VINE"])); o2.add(z3.Not(u2["VideoSeal"])); o2.add(z3.Not(o2._fevars["tile"]))
    assert o2.check() == z3.sat, "the plain TrustMark still carries its measured capacity"
