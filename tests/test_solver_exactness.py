"""The solver must return the OPTIMUM, not merely a feasible configuration.

z3's `Optimize.maximize` is not complete on this encoding: on the measured surrogate it returns a
satisfying but sub-optimal model, non-deterministically, so two identical calls can disagree. That
breaks the property the whole method rests on -- that the returned configuration is the best one that
meets the request -- and it silently understates the solver against a grid baseline, since a grid that
happens to land on a better point then appears to beat an "exact" solver.

`solve_exact` therefore uses the optimizer only as a starting point and then CERTIFIES it: it asks a
fresh instance whether anything strictly better is satisfiable, and repeats until that query comes back
unsat. The final unsat is the optimality proof. These tests pin both properties -- the answer is
reproducible, and no better configuration exists.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3
from surrogate_model import synthetic_surrogate
import watermark_smt_v2 as W

ATTACKS = ["jpeg25", "rot9"]


def _scen(min_ba=0.60):
    return dict(min_psnr=0.0, max_ms=1e9, attacks=list(ATTACKS), min_ba=min_ba,
                allow_resync=True, allow_nested=False, min_bits=0, resolution=512)


def _kw(sg):
    return dict(enable_order=True, continuous_strength=True, surrogate=sg)


def test_solve_exact_is_reproducible():
    """Identical requests must give identical answers -- the optimum is a property of the request."""
    sg = synthetic_surrogate(attacks=ATTACKS)
    vals = [W.solve_exact(_scen(), **_kw(sg))[0] for _ in range(5)]
    assert len(set(round(v, 9) for v in vals)) == 1, f"non-reproducible optima: {vals}"


def test_solve_exact_admits_nothing_better():
    """The returned value must be certified: asserting a strictly better objective is unsat."""
    sg = synthetic_surrogate(attacks=ATTACKS)
    best, _, certified = W.solve_exact(_scen(), **_kw(sg))
    assert certified, "solver returned without an optimality certificate"
    o, u, rs, ns, al, nf, ps, tm = W.build(**_scen(), **_kw(sg))
    o.add(ps > best + 1e-9)
    assert o.check() == z3.unsat, f"a configuration better than {best} exists; not the optimum"


def test_solve_exact_never_worse_than_bare_maximize():
    """Certification can only improve on the optimizer's own answer, never degrade it."""
    sg = synthetic_surrogate(attacks=ATTACKS)
    o, u, rs, ns, al, nf, ps, tm = W.build(**_scen(), **_kw(sg))
    o.maximize(ps)
    assert o.check() == z3.sat
    bare = float(o.model().eval(ps).as_fraction())
    best, _, _ = W.solve_exact(_scen(), **_kw(sg))
    assert best >= bare - 1e-9, f"certified optimum {best} below bare maximize {bare}"


def test_solve_exact_reports_unsat():
    """An unsatisfiable request must come back as None rather than a fabricated optimum."""
    sg = synthetic_surrogate(attacks=ATTACKS)
    best, _, certified = W.solve_exact(_scen(min_ba=1.01), **_kw(sg))
    assert best is None and certified


def test_unmeasured_attack_is_refused_not_dropped():
    """A request naming an attack the surrogate never measured must fail loudly.

    Skipping it is the worst available behaviour: the constraint never enters the formula, the request
    comes back SAT, and a user who asked for protection against that attack is handed a configuration
    with no evidence behind it. This happened in practice -- a scenario set named a deep crop that had
    never been measured, so the set silently reduced to the shallower crop it also contained while
    reporting a satisfaction rate as if it had not.
    """
    import pytest
    sg = synthetic_surrogate(attacks=ATTACKS)
    scen = _scen(); scen["attacks"] = list(ATTACKS) + ["unmarker"]
    with pytest.raises(ValueError, match="no measured surrogate"):
        W.build(**scen, **_kw(sg))
