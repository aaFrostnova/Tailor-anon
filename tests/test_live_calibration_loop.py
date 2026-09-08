"""The live-calibration loop: patches the curve it solved against, gates on the live standard error,
and stops when disagreement is inside noise.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W
from live_calibration import solve_with_live

ATT = ("jpeg25",)


def _sg(level):
    sg = synthetic_surrogate(attacks=ATT)
    for f in W.FR:
        lo, hi = sg.range(f)
        sg._base[(f, "jpeg25")] = PWL([lo, hi], [level[f], level[f]])
    return sg


def _scen(beta):
    # margin 0: these cases place the curves exactly on the threshold to exercise the gate, not the
    # request's safety allowance (W.DEFAULT_MARGIN), which would move the line they are placed on
    return dict(min_psnr=0.0, max_ms=1e9, attacks=list(ATT), min_ba=beta, allow_resync=False, allow_nested=False,
                min_bits=0, resolution=512, margin=0.0)


def test_no_contradiction_is_certified_in_one_round():
    sg = _sg({"VINE": 0.95, "TrustMark": 0.95, "VideoSeal": 0.95})
    r = solve_with_live(_scen(0.90), lambda cfg, a: {f: (0.96, 0.01, 20) for f in cfg["order"]}, sg, live_ok=set(ATT))
    assert r["verdict"] == "SAT (live-certified)" and r["rounds"] == 1 and r["patched"] == []
    assert r["provenance"]["jpeg25"] == "live"


def test_a_real_contradiction_patches_and_resolves_to_unsat():
    """Every fragment reads 0.95 in the table but 0.80 live at 0.90: the patch moves each curve down
    and the re-solve honestly finds nothing feasible."""
    sg = _sg({"VINE": 0.95, "TrustMark": 0.95, "VideoSeal": 0.95})
    r = solve_with_live(_scen(0.90), lambda cfg, a: {f: (0.80, 0.01, 20) for f in cfg["order"]}, sg, live_ok=set(ATT))
    assert r["verdict"] == "UNSAT" and r["rounds"] >= 2 and r["patched"], r
    assert all(p["offset"] < 0 for p in r["patched"])
    assert sg.base("VINE", "jpeg25").eval(0.5) == 0.95, "the shared table must not be written"


def test_disagreement_inside_noise_is_not_patched():
    """Table 0.905, live 0.895 with se 0.02: below 2 se, so nothing moves and the loop says so."""
    sg = _sg({"VINE": 0.905, "TrustMark": 0.905, "VideoSeal": 0.905})
    r = solve_with_live(_scen(0.90), lambda cfg, a: {f: (0.895, 0.02, 10) for f in cfg["order"]}, sg, live_ok=set(ATT))
    assert r["verdict"].startswith("SAT (live-inconclusive") and r["rounds"] == 1
    assert all(p["offset"] == 0.0 for p in r["patched"])
