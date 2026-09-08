"""Per-image values behind a curve, and the acceptance rate they give at a request's own threshold.

The curve blocks store E[ba]; the deployment accepts each image against a threshold set by the request,
so what a request needs is P(ba >= tau). The two agree only when the per-image distribution is narrow
and unimodal. Measured on UnMarker under the ring at s=0.3: mean 0.694 with half the images at 0.99 and
half at chance, so E[ba] clears a 0.63 threshold while half the images do not.
"""
import os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from surrogate_model import PWL, Surrogate, synthetic_surrogate
import watermark_smt_v2 as W

ATT = ("jpeg25",)


def _bimodal(n=30, p=0.5, hi=0.99, lo=0.50):
    k = int(round(p * n)); return [hi] * k + [lo] * (n - k)


def _sg_with_perimage(ba_rows, ver_rows=None, stage=None):
    sg = synthetic_surrogate(attacks=ATT)
    xs = [0.2, 0.6, 1.0]
    for f in W.FR:
        lo, hi = sg.range(f)
        sg._base[(f, "jpeg25")] = PWL([lo, hi], [0.99, 0.99])
    key = Surrogate.perimage_key("VINE", "jpeg25", stage)
    sg._perimage[key] = {"xs": xs, "ba": ba_rows, "ver": ver_rows, "n": len(ba_rows[0])}
    return sg


def test_bimodal_cell_mean_clears_but_rate_does_not():
    rows = [_bimodal(p=0.5)] * 3
    sg = _sg_with_perimage(rows)
    assert abs(np.mean(rows[0]) - 0.745) < 1e-9 and np.mean(rows[0]) >= 0.63   # the mean says covered
    r = sg.rate_curve("VINE", "jpeg25", 0.63)
    assert abs(r.eval(0.6) - 0.5) < 1e-9                                        # half the images are not


def test_rate_depends_on_the_threshold_not_on_a_stored_curve():
    rows = [[0.60, 0.70, 0.80, 0.95]] * 3
    sg = _sg_with_perimage(rows)
    assert sg.rate_curve("VINE", "jpeg25", 0.57).eval(0.6) == 1.0
    assert sg.rate_curve("VINE", "jpeg25", 0.65).eval(0.6) == 0.75
    assert sg.rate_curve("VINE", "jpeg25", 0.90).eval(0.6) == 0.25


def test_verification_flag_counts_as_accepted_below_the_threshold():
    rows = [[0.86, 0.86, 0.50, 0.50]] * 3
    ver = [[True, False, False, False]] * 3          # soft decoding verified one read at 0.86
    sg = _sg_with_perimage(rows, ver)
    assert sg.rate_curve("VINE", "jpeg25", 0.90).eval(0.6) == 0.25
    sg2 = _sg_with_perimage(rows, None)              # random-bit campaign: no flag, undercounts
    assert sg2.rate_curve("VINE", "jpeg25", 0.90).eval(0.6) == 0.0


def test_unmeasured_cell_returns_none_and_stage_key_is_separate():
    sg = _sg_with_perimage([[0.9] * 4] * 3, stage="scale")
    assert sg.rate_curve("VINE", "jpeg25", 0.63) is None
    assert sg.rate_curve("VINE", "jpeg25", 0.63, stage="scale").eval(0.6) == 1.0


def test_live_offset_shifts_the_per_image_values_with_the_curve():
    sg = _sg_with_perimage([[0.66, 0.66, 0.66, 0.66]] * 3)
    assert sg.rate_curve("VINE", "jpeg25", 0.65).eval(0.6) == 1.0
    sg2 = sg.with_live({("VINE", "jpeg25"): -0.05})
    assert sg2.rate_curve("VINE", "jpeg25", 0.65).eval(0.6) == 0.0
    assert sg.rate_curve("VINE", "jpeg25", 0.65).eval(0.6) == 1.0     # the original is untouched


def test_round_trip_through_to_dict():
    sg = _sg_with_perimage([[0.9, 0.5]] * 3, [[True, False]] * 3)
    sg2 = Surrogate.from_dict(sg.to_dict())
    assert sg2.rate_curve("VINE", "jpeg25", 0.95).eval(0.6) == 0.5


def test_solver_detection_floor_reads_the_per_image_rate():
    # mean 0.745 clears beta=0.63 on every fragment, but VINE's per-image rate is 0.5: with the floor on,
    # the solver may not use VINE for this column and must pick another fragment
    sg = _sg_with_perimage([_bimodal(p=0.5)] * 3)
    scen = dict(min_psnr=0.0, max_ms=1e9, attacks=list(ATT), min_ba=0.63, allow_resync=False,
                allow_nested=False, min_bits=0, resolution=512, margin=0.0)
    built, m, _r, _c = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg, det_min=0.9)
    assert built is not None
    o, u = built[0], built[1]
    chosen = [f for f in W.FR if str(m.eval(u[f])) == "True"]
    assert "VINE" not in chosen, chosen
