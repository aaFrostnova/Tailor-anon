"""The adversarial column is live-only: the table's cells there are a search prior, never the verdict.

Two consequences the solver has to honour. A front-end stage stays admissible on a request naming the
column even when no replacement curve was measured there (on a measured column a missing curve makes the
stage inadmissible with the fragment it re-embeds, because the plain curve would describe an embed nobody
measured; on the live-only column the plain curve is just the prior and live decides). And the request is
never certified from the table: the live gate requires the measurement whatever the table says.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import PWL, Surrogate, synthetic_surrogate
import watermark_smt_v2 as W

ATT = ("crop50", "unmarker")


def _table():
    """VideoSeal carries the adversarial column natively; TrustMark carries crop50 only with resync, and
    resync was measured on crop50 but not on the adversarial column."""
    sg = synthetic_surrogate(attacks=ATT)
    for f in W.FR:
        lo, hi = sg.range(f)
        sg._base[(f, "crop50")] = PWL([lo, hi], [0.55, 0.55])
        sg._base[(f, "unmarker")] = PWL([lo, hi], [0.95 if f == "VideoSeal" else 0.50] * 2)
    tlo, thi = sg.range("TrustMark"); vlo, vhi = sg.range("VideoSeal")
    # resync measured on crop50 for both fragments it re-embeds here (it helps TrustMark, not VideoSeal);
    # nothing measured for it on the adversarial column
    sg._frontend = {"base_fe_resync_TrustMark|crop50": PWL([tlo, thi], [0.95, 0.95]),
                    "base_fe_resync_VideoSeal|crop50": PWL([vlo, vhi], [0.55, 0.55]),
                    "penalty_fe_resync": PWL([tlo, thi], [0.0, 0.0])}
    # per-image evidence for the embeds resync changes, where the stage is meant to count (a re-embedded
    # fragment without a per-image cell gets no credit under the stage)
    sg._perimage[Surrogate.perimage_key("TrustMark", "crop50", "resync")] = {"xs": [tlo, thi], "ba": [[0.95] * 10] * 2, "ver": None, "n": 10}
    sg._perimage[Surrogate.perimage_key("VideoSeal", "unmarker", "resync")] = {"xs": [vlo, vhi], "ba": [[0.95] * 10] * 2, "ver": None, "n": 10}
    return sg


def _solve(sg):
    scen = dict(min_psnr=0.0, max_ms=1e9, attacks=list(ATT), min_ba=0.63, allow_resync=True, allow_nested=False,
                min_bits=0, resolution=512, margin=0.0)
    built, m, _r, _c = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
    if built is None: return None
    o, u = built[0], built[1]
    return {"frags": [f for f in W.FR if str(m.eval(u[f])) == "True"], "resync": str(m.eval(o._fevars["resync"])) == "True"}


def test_a_stage_stays_admissible_without_a_curve_on_the_live_only_column():
    r = _solve(_table())
    assert r is not None and r["resync"] and "TrustMark" in r["frags"] and "VideoSeal" in r["frags"], r


def test_the_same_gap_on_a_measured_column_makes_the_stage_inadmissible(monkeypatch):
    """Treat the column as a measured one: the missing resync curve on it now blocks TrustMark under resync,
    nothing else can cover crop50, and the request is UNSAT."""
    monkeypatch.setattr(W, "ADVERSARIAL", set())
    assert _solve(_table()) is None


def test_the_live_gate_never_settles_the_column_from_the_table():
    need, why = W.live_required_at("unmarker", 0.99, 0.63)
    assert need and "adversarial" in why
