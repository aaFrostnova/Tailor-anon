"""The two floors the solver reads from the table carry a margin, like the coverage clause.

The per-image floor is asked of the table at det_min + DEFAULT_RATE_MARGIN (a rate near 0.9 estimated from
100 images has a standard error of 0.03; the optimum sits on the floor, so without the margin half of the
live rates fall below it), and the capacity line at bits_to_ba(min_bits) + margin (the request's mean
margin; 334 C3 requests missed the bare line live by a median 0.004). 2026-09-08.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "defense"))
import z3
from surrogate_model import PWL, Surrogate, synthetic_surrogate
import watermark_smt_v2 as W

ATT = ("jpeg25",)


def _table():
    sg = synthetic_surrogate(attacks=ATT)
    lo, hi = sg.range("VINE")
    for f in W.FR:
        l, h = sg.range(f); sg._base[(f, "jpeg25")] = PWL([l, h], [0.50, 0.50])
        for g in W.FR:
            if g != f: sg._delta[(g, f, "jpeg25")] = PWL([sg.range(g)[0], sg.range(g)[1]], [0.0, 0.0])
    sg._base[("VINE", "jpeg25")] = PWL([lo, hi], [0.70, 1.00])          # mean rises linearly with strength
    # per-image rate rises 0 -> 1 across the range
    sg._perimage[Surrogate.perimage_key("VINE", "jpeg25")] = {"xs": [lo, hi], "ba": [[0.50] * 10, [0.99] * 10], "ver": None, "n": 10}
    return sg, lo, hi


def _vine_strength(sg, **kw):
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATT), min_ba=0.63, allow_resync=False,
                                          allow_nested=False, resolution=512, enable_order=True, continuous_strength=True, surrogate=sg, **kw)
    o.add(u["VINE"]); o.add(z3.Not(u["TrustMark"])); o.add(z3.Not(u["VideoSeal"])); o.maximize(ps)
    assert o.check() == z3.sat
    return float(o.model().eval(o._svars["VINE"]).as_fraction())


def test_per_image_floor_is_asked_with_its_margin():
    sg, lo, hi = _table()
    s = _vine_strength(sg, min_bits=0, margin=0.0, det_min=0.9)
    assert abs(s - (lo + (0.9 + W.DEFAULT_RATE_MARGIN) * (hi - lo))) < 1e-6, s


def test_the_margin_is_a_parameter():
    sg, lo, hi = _table()
    s = _vine_strength(sg, min_bits=0, margin=0.0, det_min=0.9, rate_margin=0.0)
    assert abs(s - (lo + 0.9 * (hi - lo))) < 1e-6, s


def test_capacity_line_carries_the_mean_margin():
    sg, lo, hi = _table()
    s0 = _vine_strength(sg, min_bits=37, margin=0.0, det_min=0.0)          # floor off: the capacity line alone binds
    s1 = _vine_strength(sg, min_bits=37, margin=0.02, det_min=0.0)
    line = W.bits_to_ba(37)
    assert abs(sg.base("VINE", "jpeg25").eval(s0) - line) < 1e-6 and abs(sg.base("VINE", "jpeg25").eval(s1) - (line + 0.02)) < 1e-6, (s0, s1)
