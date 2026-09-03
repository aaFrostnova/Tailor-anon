"""A fragment is never proposed below the strength at which unattacked images can be read.

The clean sweep (measure_clean_minimum.py) shows the table ranges start below the point where 99% of
clean images clear the level: TrustMark at 0.4 reaches the presence level on 98% of images and the
crypto-verify level on far fewer, yet the mean-curve clause was happy to select it there for identity
requests. The solver now takes a per-request floor: presence floor for presence requests, identity floor
when identity bits are required, interpolated in the threshold between the two.
"""
import os, sys, json, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ("jpeg25",)


def test_levels_from_sweep_rows_match_the_measured_floors():
    rows = {"TrustMark": [{"s": 0.4, "ba_mean": 0.8149, "ba_sd": 0.104, "frac_presence": 0.98},
                          {"s": 0.55, "ba_mean": 0.9365, "ba_sd": 0.061, "frac_presence": 1.0},
                          {"s": 0.7, "ba_mean": 0.987, "ba_sd": 0.019, "frac_presence": 1.0}],
            "VINE": [{"s": 0.14, "ba_mean": 0.868, "ba_sd": 0.093, "frac_presence": 0.99},
                     {"s": 0.2, "ba_mean": 0.915, "ba_sd": 0.076, "frac_presence": 1.0},
                     {"s": 0.3, "ba_mean": 0.963, "ba_sd": 0.047, "frac_presence": 1.0}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"rows": rows}, fh)
    lv = W.clean_floor_levels(fh.name)
    assert lv["TrustMark"] == (0.55, 0.7) and lv["VINE"] == (0.14, 0.3)
    # identity requests take the identity floor; presence requests interpolate in the threshold
    assert W.clean_floors(0.9, min_bits=37, levels=lv)["TrustMark"] == 0.7
    assert W.clean_floors(0.63, min_bits=0, levels=lv)["TrustMark"] == 0.55
    mid = W.clean_floors(0.73, min_bits=0, levels=lv)["TrustMark"]
    assert 0.55 < mid < 0.7
    assert W.clean_floors(0.95, min_bits=0, levels=lv)["TrustMark"] == 0.7   # clipped at the identity level


def test_missing_sweep_falls_back_to_the_baked_defaults():
    assert W.clean_floor_levels("/nonexistent/clean.json") == W.CLEAN_FLOOR_DEFAULT


def _sg():
    sg = synthetic_surrogate(attacks=ATTACKS)
    lo, hi = sg.range("TrustMark")
    sg._base[("TrustMark", "jpeg25")] = PWL([lo, hi], [0.99, 0.99])     # readable at any strength
    return sg


def _solve(sg, **kw):
    o, u, *_ = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=0.9,
                       allow_resync=False, allow_nested=False, min_bits=0, enable_order=True,
                       continuous_strength=True, surrogate=sg, **kw)
    o.add(u["TrustMark"])
    for f in u:
        if f != "TrustMark": o.add(z3.Not(u[f]))
    assert o.check() == z3.sat
    m = o.model()
    v = m.eval(o._svars["TrustMark"], model_completion=True)
    return float(v.numerator_as_long()) / float(v.denominator_as_long())


def test_solver_respects_a_pinned_floor_and_can_switch_it_off():
    sg = _sg()
    lo, hi = sg.range("TrustMark")
    floor = round(lo + 0.6 * (hi - lo), 4)
    assert _solve(sg, clean_floor={}) < floor - 1e-6            # off: the flat curve lets it sit at the bottom
    assert _solve(sg, clean_floor={"TrustMark": floor}) >= floor - 1e-9


def test_default_floors_are_the_measured_ones_and_bind_only_upwards():
    sg = _sg()
    lo, hi = sg.range("TrustMark")
    s = _solve(sg)                                                # clean_floor=None -> measured floors
    assert s >= max(lo, W.clean_floors(0.9, 0)["TrustMark"]) - 1e-9
    assert s <= hi + 1e-9
