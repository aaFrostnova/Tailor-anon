"""A live contradiction measured under a front-end must move that front-end's replacement curve.

The CEGAR loop solves a configuration with a front-end on against the stage's replacement curve. Its
patch used to shift only the plain curve, which the solve was not reading, so the next round read the
same optimistic value, proposed the same configuration, and the loop spent its whole round budget
re-measuring one contradiction (three of eight requests on the final table).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from surrogate_model import PWL, synthetic_surrogate


def _sg():
    sg = synthetic_surrogate(attacks=("crop75",))
    lo, hi = sg.range("VideoSeal")
    sg._base[("VideoSeal", "crop75")] = PWL([lo, hi], [0.70, 0.76])
    sg._frontend = {sg.fe_key("resync", "VideoSeal", "crop75"): PWL([lo, hi], [0.90, 0.94])}
    return sg


def test_offset_on_a_front_end_curve_moves_only_that_curve():
    sg = _sg(); lo, hi = sg.range("VideoSeal")
    d = sg.live_offset("VideoSeal", "crop75", hi, 0.75, fe="resync")      # table 0.94, live 0.75
    assert abs(d - (0.75 - 0.94)) < 1e-9
    patched = sg.with_live({("fe", "resync", "VideoSeal", "crop75"): d})
    assert abs(patched.frontend(sg.fe_key("resync", "VideoSeal", "crop75")).eval(hi) - 0.75) < 1e-9
    assert patched.base("VideoSeal", "crop75").eval(hi) == 0.76, "the plain curve must not move"
    assert sg.frontend(sg.fe_key("resync", "VideoSeal", "crop75")).eval(hi) == 0.94, "the shared table must not move"


def test_offset_without_a_stage_still_moves_the_plain_curve():
    sg = _sg(); lo, hi = sg.range("VideoSeal")
    d = sg.live_offset("VideoSeal", "crop75", hi, 0.70)
    patched = sg.with_live({("VideoSeal", "crop75"): d})
    assert abs(patched.base("VideoSeal", "crop75").eval(hi) - 0.70) < 1e-9
    assert patched.frontend(sg.fe_key("resync", "VideoSeal", "crop75")).eval(hi) == 0.94
