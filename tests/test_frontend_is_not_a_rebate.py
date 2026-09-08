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
from surrogate_model import PWL, Surrogate, synthetic_surrogate
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
    # the tiled embed's per-image evidence (a re-embedded fragment without it gets no credit under the stage)
    sg._perimage[Surrogate.perimage_key("TrustMark", "jpeg25", "tile")] = {"xs": [lo, hi], "ba": [[0.99] * 10] * 2, "ver": None, "n": 10}
    return sg


def _solve(sg, min_bits=0):
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                                          allow_resync=True, allow_nested=True, min_bits=min_bits,
                                          resolution=512, enable_order=True, continuous_strength=True, margin=0.0, surrogate=sg)
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
                                          enable_order=True, continuous_strength=True, margin=0.0, surrogate=_sg())
    o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"])); o.add(o._fevars["tile"])
    o.maximize(ps); assert o.check() == z3.sat
    d_tile = -float(o.model().eval(ps).as_fraction())
    assert d_tile >= r["D"] - 1e-9, "the tiled grid was cheaper than the plain embed in the objective"


def test_tiled_trustmark_capacity_follows_the_tiled_curve():
    """Capacity is derived from the bit-accuracy expression, which under the tiled grid is the tiled
    replacement curve. Two columns: on jpeg25 the grid helps (0.90 to 0.99, so the stage stays
    responsible and is not pruned), on blur it hurts (0.90 to 0.80, 28 bits). A 37-bit payload is
    infeasible with the grid on and feasible with it off; a grid that reads 0.99 on both carries it."""
    atts = ("jpeg25", "blur")
    def table(tiled_blur):
        sg = synthetic_surrogate(attacks=atts); lo, hi = sg.range("TrustMark")
        for a in atts:
            sg._base[("TrustMark", a)] = PWL([lo, hi], [0.90, 0.90])
        sg._frontend = {"base_fe_tile_TrustMark|jpeg25": PWL([lo, hi], [0.99, 0.99]),
                        "base_fe_tile_TrustMark|blur": PWL([lo, hi], [tiled_blur, tiled_blur]),
                        "penalty_fe_tile": PWL([lo, hi], [0.0, 0.0])}
        for a, v in (("jpeg25", 0.99), ("blur", tiled_blur)):
            sg._perimage[Surrogate.perimage_key("TrustMark", a, "tile")] = {"xs": [lo, hi], "ba": [[v] * 10] * 2, "ver": None, "n": 10}
        return sg
    def build(sg, tile):
        o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(atts), min_ba=0.63,
                                              allow_resync=True, allow_nested=True, min_bits=37, resolution=512,
                                              enable_order=True, continuous_strength=True, margin=0.0, surrogate=sg)
        o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"]))
        o.add(o._fevars["tile"] if tile else z3.Not(o._fevars["tile"]))
        return o.check()
    assert build(table(0.80), tile=True) == z3.unsat, "a tiled TrustMark must not carry the plain embed's capacity"
    assert build(table(0.80), tile=False) == z3.sat, "the plain TrustMark carries its own"
    assert build(table(0.99), tile=True) == z3.sat, "a tiled embed that reads 0.99 carries the payload"
