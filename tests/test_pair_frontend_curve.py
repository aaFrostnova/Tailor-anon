"""A two-fragment front-end curve replaces single-curve-minus-delta when the partner is embedded after the host.

The single-fragment replacement curve was measured with the fragment alone and the interference term
delta_{g->f} with both fragments plain; a stage that changes the embed (the tiled grid, the ring, the sync
mark) can behave differently once a partner is written over it. The follow-up campaign measures the host's
curve with the partner embedded after it and the stage on; where that curve exists and the solver puts the
partner after the host, it is the level and the delta term for that pair is dropped.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ("crop75",)


def _sg(with_pair):
    sg = synthetic_surrogate(attacks=ATTACKS)
    lo, hi = sg.range("TrustMark")
    vlo, vhi = sg.range("VINE")
    sg._base[("TrustMark", "crop75")] = PWL([lo, hi], [0.60, 0.60])          # plain: below the threshold
    sg._base[("VINE", "crop75")] = PWL([vlo, vhi], [0.50, 0.50])             # VINE covers nothing here
    sg._delta[("VINE", "TrustMark", "crop75")] = PWL([vlo, vhi], [0.30, 0.30])   # plain interference
    sg._frontend = {"base_fe_tile_TrustMark|crop75": PWL([lo, hi], [0.95, 0.95]),   # tiled, alone
                    "penalty_fe_tile": PWL([lo, hi], [0.5, 0.5])}
    if with_pair:   # tiled TrustMark with VINE written over it: still readable
        sg._frontend["base_fe_tile_TrustMark|crop75|with_VINE"] = PWL([lo, hi], [0.93, 0.93])
    return sg


def _solve(sg, vine_after_trustmark):
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                                          allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
                                          enable_order=True, continuous_strength=True, surrogate=sg, clean_floor={})
    o.add(u["TrustMark"]); o.add(u["VINE"]); o.add(z3.Not(u["VideoSeal"])); o.add(o._fevars["tile"])
    pv = o._pvars
    o.add(pv[("TrustMark", "VINE")] if vine_after_trustmark else pv[("VINE", "TrustMark")])
    return o.check()


def test_pair_curve_is_the_level_when_partner_comes_after():
    # single curve 0.95 - delta 0.30 = 0.65 < 0.9 fails; the pair curve 0.93 clears
    assert _solve(_sg(with_pair=False), vine_after_trustmark=True) == z3.unsat
    assert _solve(_sg(with_pair=True), vine_after_trustmark=True) == z3.sat


def test_pair_curve_is_ignored_when_partner_comes_first():
    # VINE before TrustMark: nothing overwrites TrustMark, the single tiled curve 0.95 clears either way
    assert _solve(_sg(with_pair=True), vine_after_trustmark=False) == z3.sat
    assert _solve(_sg(with_pair=False), vine_after_trustmark=False) == z3.sat
