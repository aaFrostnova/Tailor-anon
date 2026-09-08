"""The continuous path must honour the capacity derived from bit accuracy and the measured front-end curves.

These use a SYNTHETIC surrogate so they pin the mechanism rather than any particular measurement:
a payload the low end of the bit-accuracy curve cannot carry must force a stronger (or different) config,
and a front-end curve that is WORSE than its no-front-end control must let the solver decline that
front-end rather than being assumed to help.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, Surrogate, synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ("jpeg25", "rot9")


def _surrogate(cap=None, frontend=None):
    sg = synthetic_surrogate(attacks=ATTACKS)
    if cap:
        sg._cap = cap
    if frontend:
        sg._frontend = frontend
    return sg


def _solve(sg, attacks, min_ba=0.60, min_bits=0, allow_resync=True, force=(), forbid=()):
    o, u, rs, ns, al, nf, ps, tm = W.build(
        min_psnr=0.0, max_ms=1e9, attacks=list(attacks), min_ba=min_ba,
        allow_resync=allow_resync, allow_nested=False, min_bits=min_bits, resolution=512,
        enable_order=True, continuous_strength=True, surrogate=sg)
    for f in force:
        o.add(u[f])
    for f in forbid:
        o.add(z3.Not(u[f]))
    o.maximize(ps)
    if o.check() != z3.sat:
        return None
    m = o.model()
    frags = [f for f in W.FR if str(m.eval(u[f])) == "True"]
    return {"frags": frags,
            "s": {f: float(m.eval(o._svars[f]).as_fraction()) for f in frags},
            "resync": str(m.eval(rs)) == "True"}


def test_capacity_floor_forces_a_stronger_config():
    """A payload the low end of the bit-accuracy curve cannot carry must push the solve up the curve.

    Capacity is derived from bit accuracy (W.ba_to_bits): 0.60 carries 3 bits, 0.98 carries 86, so a
    70-bit payload (0.94 plus the request's 0.02 margin) is met only near the top of the range and a
    95-bit one (0.994) nowhere."""
    lo, hi = synthetic_surrogate().range("VINE")
    sg = _surrogate()
    sg._base[("VINE", "jpeg25")] = PWL([lo, hi], [0.60, 0.98])
    free = _solve(sg, ["jpeg25"], force=("VINE",), forbid=("TrustMark", "VideoSeal"))
    tight = _solve(sg, ["jpeg25"], min_bits=70, force=("VINE",), forbid=("TrustMark", "VideoSeal"))
    assert free is not None and tight is not None
    assert tight["s"]["VINE"] > free["s"]["VINE"], "capacity floor did not raise the strength"
    assert sg.base("VINE", "jpeg25").eval(tight["s"]["VINE"]) >= W.bits_to_ba(70) + W.DEFAULT_MARGIN - 1e-6
    # and a floor above the whole curve is infeasible, not silently ignored
    assert _solve(sg, ["jpeg25"], min_bits=95, force=("VINE",),
                  forbid=("TrustMark", "VideoSeal")) is None


def test_front_end_that_hurts_is_declined():
    """resync must be OFF when the effect it was measured to have is negative.

    A front-end contributes (its curve) minus (its same-run control), added to the solo curve, so
    switching it off returns exactly the solo curve. Here the solo curve already clears the
    threshold and the measured effect is negative, so enabling resync would break feasibility."""
    lo, hi = synthetic_surrogate().range("VideoSeal")
    frontend = {"base_resync_VideoSeal|rot9": PWL([lo, hi], [0.55, 0.60]),           # front-end on
                "raw_synced_noresync_VideoSeal|rot9": PWL([lo, hi], [0.90, 0.95])}   # its control
    sg = _surrogate(frontend=frontend)
    sg._base[("VideoSeal", "rot9")] = PWL([lo, hi], [0.95, 0.95])   # solo curve clears 0.85 alone
    r = _solve(sg, ["rot9"], min_ba=0.85, force=("VideoSeal",), forbid=("VINE", "TrustMark"))
    assert r is not None, "the solo curve clears 0.85, so this must be satisfiable"
    assert r["resync"] is False, "solver enabled a front-end its measured effect says costs accuracy"


def test_front_end_that_helps_is_enabled():
    """resync must be ON when only its curve clears the threshold."""
    lo, hi = synthetic_surrogate().range("TrustMark")
    frontend = {"base_resync_TrustMark|rot9": PWL([lo, hi], [0.99, 0.99]),           # front-end on
                "raw_synced_noresync_TrustMark|rot9": PWL([lo, hi], [0.50, 0.50])}   # its control
    sg = _surrogate(frontend=frontend)
    sg._base[("TrustMark", "rot9")] = PWL([lo, hi], [0.50, 0.50])   # solo curve alone fails 0.90
    r = _solve(sg, ["rot9"], min_ba=0.90, force=("TrustMark",), forbid=("VINE", "VideoSeal"))
    assert r is not None and r["resync"] is True, \
        "only the front-end's measured effect clears the threshold, so it must be enabled"
