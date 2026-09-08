"""The per-image acceptance floor under a front-end stage that does not change the fragment's embed.

`det_cond` reads the stage's per-image cell when one exists. Without one it fell back to the stage's
det_fe curve, a detection rate recorded at the decoder's default budget rather than at the request's
threshold, and a curve the live loop never patches. For a fragment whose embed the stage does not change
(FE_AFFECTS: tile re-embeds TrustMark only, scale re-embeds VINE only, angle nothing) the plain per-image
cell is the same embed read at the request's own threshold: on a column where the stage does not fire it
is the exact quantity, and where the stage's cascade fires it can only add acceptances. So the plain rate
is the floor there. A fragment the stage re-embeds with no per-image cell for that embed gets no credit
under the stage: the det_fe curves of the front-end campaign are records, never read.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, Surrogate, synthetic_surrogate
import watermark_smt_v2 as W

ATT = ("jpeg25",)
XS = [0.2, 0.6, 1.0]
PLAIN_FLOOR = 0.2 + (0.9 + W.DEFAULT_RATE_MARGIN) * 0.4     # the plain per-image rate rises 0 -> 1 between 0.2 and 0.6: the floor (0.9 plus its margin) is crossed there


def _table(stage, stage_perimage=None):
    """VINE is the fragment under test. TrustMark is present (the tiled grid is a TrustMark-side embed,
    so tile can only be on with TrustMark selected) but reads chance, so VINE alone must clear the column."""
    sg = synthetic_surrogate(attacks=ATT)
    for f in W.FR:
        l, h = sg.range(f); sg._base[(f, "jpeg25")] = PWL([l, h], [0.90 if f == "VINE" else 0.50, 0.90 if f == "VINE" else 0.50])
        for g in W.FR:
            if g != f: sg._delta[(g, f, "jpeg25")] = PWL([sg.range(g)[0], sg.range(g)[1]], [0.0, 0.0])
    lo, hi = sg.range("VINE"); tlo, thi = sg.range("TrustMark")
    # the stage measurably helps VINE on the column (so it is not pruned) and its det_fe curve is loose;
    # TrustMark's replacement curve exists (the admissibility rule asks for it) and stays at chance
    sg._frontend = {f"base_fe_{stage}_VINE|jpeg25": PWL([lo, hi], [0.99, 0.99]),
                    f"det_fe_{stage}_VINE|jpeg25": PWL([lo, hi], [1.0, 1.0]),
                    f"base_fe_{stage}_TrustMark|jpeg25": PWL([tlo, thi], [0.50, 0.50]),
                    f"penalty_fe_{stage}": PWL([tlo, thi], [0.0, 0.0])}
    # plain per-image cell: nothing clears the threshold at 0.2, everything does from 0.6 on
    sg._perimage[Surrogate.perimage_key("VINE", "jpeg25")] = {
        "xs": XS, "ba": [[0.50] * 10, [0.95] * 10, [0.95] * 10], "ver": None, "n": 10}
    if stage_perimage is not None:
        sg._perimage[Surrogate.perimage_key("VINE", "jpeg25", stage)] = {
            "xs": XS, "ba": stage_perimage, "ver": None, "n": 10}
    return sg


def _vine_strength(sg, stage):
    o, u, rs, ns, al, nf, ps, tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=list(ATT), min_ba=0.63,
                                          allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
                                          enable_order=True, continuous_strength=True, margin=0.0,
                                          surrogate=sg, det_min=0.9)
    o.add(u["VINE"]); o.add(z3.Not(u["VideoSeal"])); o.add(o._fevars[stage])
    o.add(u["TrustMark"] if stage == "tile" else z3.Not(u["TrustMark"]))
    o.maximize(ps)
    if o.check() != z3.sat: return None
    return float(o.model().eval(o._svars["VINE"]).as_fraction())


def test_unaffected_fragment_keeps_the_plain_floor_under_the_stage():
    """tile re-embeds TrustMark only: VINE under tile is the plain embed, so its floor is the plain one."""
    assert abs(_vine_strength(_table("tile"), "tile") - PLAIN_FLOOR) < 1e-6


def test_affected_fragment_without_a_stage_cell_gets_no_credit():
    """resync re-embeds every fragment: with no per-image cell for VINE under resync the stage cannot clear the
    column with VINE, whatever the loose det_fe curve says (it is not read), so forcing resync on is UNSAT."""
    assert _vine_strength(_table("resync"), "resync") is None


def test_a_stage_per_image_cell_wins_over_the_plain_one():
    sg = _table("tile", stage_perimage=[[0.95] * 10] * 3)      # under the stage every image clears at every knot
    assert _vine_strength(sg, "tile") < PLAIN_FLOOR - 0.1
