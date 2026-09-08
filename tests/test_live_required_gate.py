"""The offline table is not allowed to settle every column.

W.live_required_at names the two cases where the stored mean is not a point estimate for the image in
front of the deployment -- an adversarial attack that optimises per image, and any column whose table
value sits within k image-to-image standard deviations of the threshold a certified optimum sits on --
and solve_with_live must either measure those columns or say it did not. A request that comes back
"live-certified" while a required column was answered from the table is the failure this gate exists to
prevent: measured on the certified C4 requests, four of nine cleared the mean UnMarker threshold with
per-image detection between 0.30 and 0.80.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W
from live_calibration import solve_with_live

ATT = ("jpeg25", "unmarker")


def _sg(level):
    sg = synthetic_surrogate(attacks=ATT)
    for f in W.FR:
        lo, hi = sg.range(f)
        for a in ATT:
            sg._base[(f, a)] = PWL([lo, hi], [level, level])
    return sg


def _scen(beta, attacks=ATT):
    return dict(min_psnr=0.0, max_ms=1e9, attacks=list(attacks), min_ba=beta, allow_resync=False,
                allow_nested=False, min_bits=0, resolution=512, margin=0.0)


def _flat(cfg, a):
    return {f: (0.99, 0.005, 20) for f in cfg["order"]}


def test_adversarial_column_always_requires_live():
    assert W.live_required_at("unmarker", 0.99, 0.50)[0]        # far above the threshold, still required
    assert W.live_required_at("unmarker", 0.51, 0.50)[0]


def test_benign_column_far_from_threshold_does_not():
    need, why = W.live_required_at("jpeg25", 0.99, 0.63)
    assert not need and "sd from the threshold" in why


def test_any_column_near_the_threshold_requires_live():
    # a certified optimum sits ON the constraint, so this is the ordinary case, not a corner
    need, why = W.live_required_at("regen", 0.75, 0.74)
    assert need and "near-threshold" in why


def test_measured_sd_is_used_when_the_profile_has_the_column():
    # crop_jpeg is the widest column measured (sd 0.133); a 0.10 gap is inside 2 sd for it and outside
    # 2 sd for a narrow one, so the two must not answer the same way
    assert W.live_required_at("crop_jpeg", 0.73, 0.63)[0]
    assert not W.live_required_at("rs256", 0.73, 0.63)[0]


def test_no_executor_leaves_the_request_pending_not_certified():
    r = solve_with_live(_scen(0.90), _flat, _sg(0.99), live_ok={"jpeg25"})
    assert r["pending_live"] == ["unmarker"]
    assert r["verdict"].startswith("SAT (pending live: unmarker")
    assert r["provenance"]["unmarker"] == "live required (not executed)"


def test_an_executor_closes_it():
    r = solve_with_live(_scen(0.90), _flat, _sg(0.99), live_ok={"jpeg25"},
                        xenv_measure=_flat, xenv_ok=("unmarker",))
    assert r["pending_live"] == [] and r["verdict"] == "SAT (live-certified)"
    assert r["provenance"]["unmarker"] == "live"


def test_a_request_that_never_names_the_column_is_unaffected():
    r = solve_with_live(_scen(0.90, attacks=("jpeg25",)), _flat, _sg(0.99), live_ok={"jpeg25"})
    assert r["pending_live"] == [] and r["verdict"] == "SAT (live-certified)"


def test_a_column_the_gate_clears_is_still_pending_without_an_executor():
    """Every solve ends with a live check of every column it named: the gate's clearance is the expected
    risk of the column, not permission to settle it from the table."""
    sg = synthetic_surrogate(attacks=("jpeg25", "blur"))
    for f in W.FR:
        lo, hi = sg.range(f)
        for a in ("jpeg25", "blur"): sg._base[(f, a)] = PWL([lo, hi], [0.99, 0.99])
    r = solve_with_live(_scen(0.63, attacks=("jpeg25", "blur")), _flat, sg, live_ok={"jpeg25"})
    assert r["pending_live"] == ["blur"]
    assert r["provenance"]["blur"] == "live required (not executed)" and r["provenance"]["jpeg25"] == "live"
    assert r["live_reason"]["blur"].startswith("table value is more than")     # cleared by the gate, measured anyway
    assert r["verdict"].startswith("SAT (pending live: blur")
