import sys
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
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
