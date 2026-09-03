"""A live measurement refines the request that took it, and nothing else.

Measurements come from one user's images. Writing them back into the shared table would let that user's
data distribution bias every later request, and the bias would compound silently. The refinement is
therefore applied to a COPY of the surrogate for the duration of one solve.

The offset is applied to the whole curve rather than to the measured point alone. Patching one point
would leave the solver free to step to a neighbouring strength and read the original, optimistic value,
so a contradiction could be answered by moving rather than by accounting for it -- and the loop would
not converge.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from surrogate_model import synthetic_surrogate

ATT = ("jpeg25", "rot9")


def test_refinement_does_not_touch_the_shared_table():
    sg = synthetic_surrogate(attacks=ATT)
    before = sg.base("VINE", "rot9").eval(0.6)
    sg.with_live({("VINE", "rot9"): -0.2})
    assert sg.base("VINE", "rot9").eval(0.6) == before, "the shared surrogate was mutated"


def test_offset_moves_the_whole_curve_not_one_point():
    """Otherwise the next round steps aside and reads the un-patched value."""
    sg = synthetic_surrogate(attacks=ATT)
    off = sg.live_offset("VINE", "rot9", 0.6, sg.base("VINE", "rot9").eval(0.6) - 0.2)
    sg2 = sg.with_live({("VINE", "rot9"): off})
    for s in (0.4, 0.6, 0.8, 1.0):
        moved = sg.base("VINE", "rot9").eval(s) - sg2.base("VINE", "rot9").eval(s)
        assert moved > 0.05, f"curve barely moved at s={s}: {moved:.4f}"


def test_offset_reproduces_the_measurement():
    sg = synthetic_surrogate(attacks=ATT)
    target = 0.42
    off = sg.live_offset("VINE", "rot9", 0.6, target)
    assert abs(sg.with_live({("VINE", "rot9"): off}).base("VINE", "rot9").eval(0.6) - target) < 1e-6


def test_bit_accuracy_stays_a_probability():
    """A large negative offset must not push a curve below chance-of-nothing or above certainty."""
    sg = synthetic_surrogate(attacks=ATT)
    for off in (-5.0, +5.0):
        ys = sg.with_live({("VINE", "rot9"): off}).base("VINE", "rot9").ys
        assert all(0.0 <= y <= 1.0 for y in ys), ys


def test_untouched_cells_are_shared_not_copied_wrongly():
    """Only the measured cell changes; every other curve still answers exactly as before."""
    sg = synthetic_surrogate(attacks=ATT)
    sg2 = sg.with_live({("VINE", "rot9"): -0.3})
    for f in ("VINE", "TrustMark", "VideoSeal"):
        for a in ATT:
            if (f, a) == ("VINE", "rot9"): continue
            if (f, a) not in sg._base: continue
            assert sg.base(f, a).eval(0.6) == sg2.base(f, a).eval(0.6), (f, a)
