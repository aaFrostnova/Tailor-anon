"""A front-end may be switched on only where a requested column is one it measurably helps.

Every stage has replacement curves on the non-geometric columns too (they are needed so the solver
never reads the plain curve for an embed it changed), and those differ from the plain curves by
measurement noise. Left free, the optimizer bought that noise: resync was selected on a
signal+VAE+regeneration request for a 0.05 dB gain. A stage is now responsible for a column only when
its replacement curve beats the plain one by at least `fe_gain_min` somewhere in the range, and a stage
responsible for no requested column is fixed off.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ("jpeg25",)


def _sg(gain):
    sg = synthetic_surrogate(attacks=ATTACKS)
    lo, hi = sg.range("TrustMark")
    sg._base[("TrustMark", "jpeg25")] = PWL([lo, hi], [0.80, 0.80])
    sg._frontend = {"base_fe_tile_TrustMark|jpeg25": PWL([lo, hi], [0.80 + gain, 0.80 + gain]),
                    "penalty_fe_tile": PWL([lo, hi], [0.5, 0.5])}
    return sg


def _build(sg, force_tile):
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.63,
                                          allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
                                          enable_order=True, continuous_strength=True, surrogate=sg)
    o.add(u["TrustMark"]); o.add(z3.Not(u["VINE"])); o.add(z3.Not(u["VideoSeal"]))
    if force_tile: o.add(o._fevars["tile"])
    return o


def test_noise_level_gain_fixes_the_stage_off():
    o = _build(_sg(gain=0.01), force_tile=True)
    assert o.check() == z3.unsat, "tile was selectable on a column it does not measurably help"
    assert "tile" not in _build(_sg(gain=0.01), force_tile=False)._fe_responsible


def test_a_real_gain_keeps_the_stage_selectable():
    o = _build(_sg(gain=0.15), force_tile=True)
    assert o.check() == z3.sat
    assert o._fe_responsible["tile"] == ["jpeg25"]
