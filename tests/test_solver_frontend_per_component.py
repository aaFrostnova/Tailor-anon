"""The solver decides each geometric front-end separately, from measured curves.

Before the split there were two booleans standing for a cascade that ran all four of its stages
whenever either was set, and the effect of a front-end on a column that had never been measured
came from hardcoded `if attack in (...)` rules. These tests pin the replacement: one decision
variable per stage, every effect read from the table, and a SET of stages credited with the best
single stage rather than the sum -- the cascade accepts as soon as any stage verifies, so summing
would buy the same recovery twice.
"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "defense"))
import z3
from surrogate_model import Surrogate, PWL
from watermark_smt_v2 import add_strength_order

FR = ["VINE", "TrustMark"]
RNG = {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6)}


def _sur(frontend=None, latency=None, attacks=("rot9",)):
    flat = lambda lo, hi, v: PWL([lo, hi], [v, v])
    base = {(f, a): flat(*RNG[f], 0.60) for f in FR for a in attacks}
    delta = {(g, f, a): flat(*RNG[g], 0.0) for f in FR for g in FR if g != f for a in attacks}
    d = {f: flat(*RNG[f], 1.0) for f in FR}
    e = {tuple(sorted(FR)): PWL([0.6, 2.6], [0.0, 0.0])}
    return Surrogate(FR, list(attacks), RNG, base, delta, d, e,
                     frontend=frontend or {}, latency=latency or {})


def _solve(sur, fe_names, min_ba, attacks=("rot9",)):
    opt = z3.Optimize()
    u = {f: z3.Bool(f) for f in FR}
    fev = {n: z3.Bool(f"fe_{n}") for n in fe_names}
    for v in fev.values(): opt.add(v)              # every named front-end is switched on
    add_strength_order(opt, u, list(attacks), sur, min_ba, False, frontends=fev)
    return opt.check() == z3.sat


def _fe(name, f, a, on, off):
    """One stage's measured effect on one cell, plus the fidelity cost its embed-side mark carries.
    The solver refuses a stage that has the first and not the second -- free coverage would be
    switched on in every optimal model."""
    lo, hi = RNG[f]
    d = {f"base_fe_{name}_{f}|{a}": PWL([lo, hi], [on, on]),
         f"ctrl_fe_{name}_{f}|{a}": PWL([lo, hi], [off, off])}
    if name in ("resync", "tile"):
        th = RNG["TrustMark"]
        d[f"penalty_fe_{name}"] = PWL(list(th), [0.05, 0.05])
    return d


def test_a_measured_stage_supplies_its_own_effect():
    """base 0.60 + (0.95 - 0.60) = 0.95 clears a 0.90 floor; without the stage it does not."""
    fe = {}; fe.update(_fe("angle", "VINE", "rot9", 0.95, 0.60))
    fe.update(_fe("angle", "TrustMark", "rot9", 0.95, 0.60))
    assert _solve(_sur(fe), ["angle"], 0.90)
    assert not _solve(_sur({}), ["angle"], 0.90), "an unmeasured stage must not be credited"


def test_an_embed_changing_front_end_replaces_the_plain_curve():
    """The measured case this rule exists for. Plain TrustMark reads 1.000 on crop75; the tiled
    embed's own bare decode reads 0.497, because a mark written into 256px cells stops lining up
    with the decoder's canonical tiles once the picture is cropped and rescaled. Modelled as an
    effect added to the plain curve, the front-end contributes nothing on that column and the
    solver promises 1.000 for a configuration that delivers chance."""
    fe = {}
    for f in FR:
        lo, hi = RNG[f]
        fe[f"base_fe_tile_{f}|crop75"] = PWL([lo, hi], [0.497, 0.497])   # measured, tiled embed
        fe[f"ctrl_fe_tile_{f}|crop75"] = PWL([lo, hi], [0.497, 0.497])
        fe["penalty_fe_tile"] = PWL(list(RNG["TrustMark"]), [0.0, 0.0])
    sur = _sur(fe, attacks=("crop75",))
    for (f, a), c in list(sur._base.items()):
        sur._base[(f, a)] = PWL(list(RNG[f]), [1.0, 1.0])                # plain reads 1.000 here
    assert _solve(sur, [], 0.90, attacks=("crop75",)), "plain TrustMark clears 0.90 on crop75"
    assert not _solve(sur, ["tile"], 0.90, attacks=("crop75",)), \
        "switching on the tiled embed must move the fragment onto the tiled curve, not keep 1.000"


def test_two_front_ends_cannot_be_enabled_at_once():
    """Three of the four change the embed, so a pair is an embed nobody measured: the table holds
    a curve for the tiled grid and one for the sync mark, and none for both."""
    import watermark_smt_v2 as W
    src = open(W.__file__).read()
    assert "AtMost(*FEV.values(), 1)" in src


def test_a_stage_measured_on_one_column_is_not_extended_to_another():
    """The old rules gave resync a boost on every rotation-like column whether or not it had been
    measured there. A column with no curve gets no effect."""
    fe = _fe("resync", "VINE", "rot9", 0.95, 0.60)
    fe.update(_fe("resync", "TrustMark", "rot9", 0.95, 0.60))
    sur = _sur(fe, attacks=("rot9", "crop_jpeg"))
    assert not _solve(sur, ["resync"], 0.90, attacks=("rot9", "crop_jpeg")), \
        "crop_jpeg has no measured front-end curve, so nothing should clear it"


def test_latency_is_charged_per_stage_from_the_table():
    sur = _sur(latency={"latency_ms_angle|rot9": 412.5, "latency_ms_resync|rot9": 355.0})
    assert sur.latency("latency_ms_angle|rot9") == 412.5
    assert sur.latency("latency_ms_scale|rot9") is None, "an unmeasured stage cost must read as absent"


def test_build_exposes_one_variable_per_cascade_stage():
    import watermark_smt_v2 as W
    if not hasattr(W, "build"): pytest.skip("build() not available in this configuration")
    src = open(W.__file__).read()
    assert 'FEV={"resync":resync, "scale":nested, "angle":fe_angle, "tile":fe_tile}' in src, \
        "the cascade's stages must each have their own decision variable"
