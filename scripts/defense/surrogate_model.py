import numpy as np
import z3

class PWL:
    """Piecewise-linear curve through knots (xs, ys), clamped outside [xs[0], xs[-1]]."""
    def __init__(self, xs, ys):
        assert len(xs) == len(ys) >= 2
        assert all(xs[i] < xs[i+1] for i in range(len(xs)-1)), "xs must be strictly increasing"
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]

    def eval(self, x):
        return float(np.interp(x, self.xs, self.ys))  # np.interp clamps to end values

    def add_to_z3(self, x_var, name):
        """Return (y_var, constraints) encoding y_var == PWL(x_var) exactly.
        One boolean per segment; exactly one active; within the active segment y is the
        linear interpolant. x_var is assumed already bounded to [xs[0], xs[-1]] by the caller
        (or clamped here via the end segments)."""
        y = z3.Real(f"y_{name}")
        segs = [z3.Bool(f"seg_{name}_{i}") for i in range(len(self.xs) - 1)]
        cons = [z3.PbEq([(b, 1) for b in segs], 1)]  # exactly one segment active
        for i, b in enumerate(segs):
            x0, x1, y0, y1 = self.xs[i], self.xs[i+1], self.ys[i], self.ys[i+1]
            slope = (y1 - y0) / (x1 - x0)
            cons.append(z3.Implies(b, z3.And(x_var >= x0, x_var <= x1,
                                             y == y0 + slope * (x_var - x0))))
        return y, cons
