import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL

def test_pwl_eval_interpolates_and_clamps():
    p = PWL([0.0, 1.0, 2.0], [0.0, 10.0, 12.0])
    assert abs(p.eval(0.5) - 5.0) < 1e-9      # midpoint of first segment
    assert abs(p.eval(1.5) - 11.0) < 1e-9     # midpoint of second segment
    assert abs(p.eval(-3.0) - 0.0) < 1e-9     # clamp low
    assert abs(p.eval(9.0) - 12.0) < 1e-9     # clamp high


def _solve_y_at(p, xval):
    x = z3.Real("x")
    y, cons = p.add_to_z3(x, "t")
    s = z3.Solver(); s.add(cons); s.add(x == xval)
    assert s.check() == z3.sat
    m = s.model()
    return float(m.eval(y).as_fraction())

def test_pwl_add_to_z3_matches_eval():
    p = PWL([0.0, 1.0, 2.0], [0.0, 10.0, 12.0])
    for xv in [0.0, 0.5, 1.0, 1.5, 2.0]:
        assert abs(_solve_y_at(p, xv) - p.eval(xv)) < 1e-6

from surrogate_model import Surrogate, synthetic_surrogate

def test_synthetic_surrogate_shapes_and_monotone():
    sg = synthetic_surrogate()
    assert set(sg.fragments) == {"VINE", "TrustMark", "VideoSeal"}
    lo, hi = sg.range("VINE"); assert lo < hi
    a = sg.attacks[0]
    base = sg.base("VINE", a)
    assert base.eval(hi) >= base.eval(lo) - 1e-9          # robustness rises with strength
    assert sg.d("VINE").eval(hi) >= sg.d("VINE").eval(lo)  # distortion rises with strength
    assert sg.delta("TrustMark", "VINE", a).eval(hi) >= 0  # later fragment only hurts
    # round-trip
    sg2 = Surrogate.from_dict(sg.to_dict())
    assert abs(sg2.base("VINE", a).eval(hi) - base.eval(hi)) < 1e-9
