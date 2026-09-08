"""An embed-side front-end must cost something, or the solver will always switch it on.

A front-end can only raise coverage -- the cascade fires on a primary miss and its accept gate is
keyed, so it never turns a hit into a miss. If such a decision is free, every optimal model sets it
true and the returned configuration carries front-ends the request had no reason to buy. The nested
ring was charged; SyncSeal's mark and the tiled ring were not.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "defense"))
import z3
from surrogate_model import Surrogate, PWL
import watermark_smt_v2 as W

FR = ["VINE", "TrustMark"]
RNG = {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6)}


def _sur(penalty_key=None, penalty=0.0):
    flat = lambda lo, hi, v: PWL([lo, hi], [v, v])
    base = {(f, "rot9"): flat(*RNG[f], 0.60) for f in FR}
    delta = {(g, f, "rot9"): flat(*RNG[g], 0.0) for f in FR for g in FR if g != f}
    d = {f: flat(*RNG[f], 1.0) for f in FR}
    e = {tuple(sorted(FR)): PWL([0.6, 2.6], [0.0, 0.0])}
    fe = {}
    for f in FR:                                   # the stage rescues rot9 for either fragment
        fe[f"base_fe_tile_{f}|rot9"] = flat(*RNG[f], 0.95)
        fe[f"ctrl_fe_tile_{f}|rot9"] = flat(*RNG[f], 0.60)
    if penalty_key:
        fe[penalty_key] = flat(*RNG["TrustMark"], penalty)
    return Surrogate(FR, ["rot9"], RNG, base, delta, d, e, frontend=fe)


def _psnr_with(sur, tile_on):
    opt = z3.Optimize()
    u = {f: z3.Bool(f) for f in FR}
    for f in FR: opt.add(u[f])
    tile = z3.Bool("fe_tile")
    opt.add(tile if tile_on else z3.Not(tile))
    # a floor both sides clear, so the two models differ only in what the front-end costs
    _, _, psnr = W.add_strength_order(opt, u, ["rot9"], sur, 0.55, False,
                                      frontends={"tile": tile}, margin=0.0)
    opt.maximize(psnr)
    assert opt.check() == z3.sat
    return float(opt.model().eval(psnr).as_fraction())


def test_an_uncharged_front_end_is_refused():
    """A stage with a measured gain and no measured cost is free, and free coverage is bought in
    every optimal model. The request is refused rather than answered with a phantom front-end."""
    import pytest
    with pytest.raises(ValueError, match="no measured fidelity cost"):
        _psnr_with(_sur(), True)


def test_a_measured_penalty_is_deducted():
    sur = _sur("penalty_fe_tile", 0.40)
    on, off = _psnr_with(sur, True), _psnr_with(sur, False)
    assert abs((off - on) - 0.40) < 1e-9, f"expected the measured 0.40 MSE to be charged, got {off - on}"


def test_every_embed_side_front_end_has_a_host_fragment():
    """A penalty is a function of the strength of the fragment that carries it, so each embed-side
    stage has to name that fragment; a decode-side stage (the blind angle sweep) has none."""
    src = open(W.__file__).read()
    assert 'FE_HOST = {"resync": "TrustMark", "scale": "VINE", "tile": "TrustMark"}' in src
    assert '"angle"' not in src.split("FE_HOST = ")[1].split("}")[0], \
        "the blind angle sweep has no embed-side mark and must not be charged fidelity"


def _time_with(sur, tile_on):
    """The time budget with the stage forced on or off, fragments pinned so the only difference
    between the two models is what the stage costs."""
    b = W.build(0.0, 1e9, ["rot9"], 0.55, True, True, surrogate=sur, enable_order=True, margin=0.0)
    o, u, tms = b[0], b[1], b[7]
    # Pin EVERY fragment build() created, not just the ones the surrogate carries. A fragment left
    # free is chosen differently between the two solves and its own cost lands in the difference.
    for f, v in u.items(): o.add(v if f in FR else z3.Not(v))
    o.add(o._fevars["tile"] if tile_on else z3.Not(o._fevars["tile"]))
    assert o.check() == z3.sat
    return float(o.model().eval(tms, True).as_fraction())


def test_a_stage_that_never_fires_is_never_a_time_rebate():
    """Stage cost is (decode with) minus (decode without). Where the primary decode already
    succeeds the stage does not fire and that difference is noise around zero -- negative about
    half the time. Left signed, a negative cost pays the solver to switch on a stage that does
    nothing; -0.41 ms is what the rs256 column actually measured."""
    sur = _sur("penalty_fe_tile", 0.0)
    sur._latency["latency_ms_tile|rot9"] = -0.41
    assert _time_with(sur, True) - _time_with(sur, False) == 0.0, "a useless stage bought time"


def test_a_real_stage_cost_is_still_charged():
    """The clamp must not flatten genuine costs to zero."""
    sur = _sur("penalty_fe_tile", 0.0)
    sur._latency["latency_ms_tile|rot9"] = 250.0
    assert _time_with(sur, True) - _time_with(sur, False) == 250.0
