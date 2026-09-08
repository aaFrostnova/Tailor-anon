"""The SyncSeal mark is paid whenever resync is on, not only when TrustMark is selected.

Its fidelity cost (penalty_fe_resync) is indexed by TrustMark's strength because that is how it was
measured, but the mark goes on the image whatever fragments are present. Gated on TrustMark being
selected, resync was free for every VINE+VideoSeal configuration: the solver reported 40.68 dB for
such a solution and the embed delivered 38.97 dB, the missing 2.68 MSE being the mark (2026-09-08).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "defense"))
import z3
from surrogate_model import Surrogate, PWL
import watermark_smt_v2 as W

FR = ["VINE", "VideoSeal"]
RNG = {"VINE": (0.2, 1.0), "VideoSeal": (0.5, 1.6)}
PEN = 2.6


def _sur(with_trustmark_host=False):
    flat = lambda lo, hi, v: PWL([lo, hi], [v, v])
    frs = FR + (["TrustMark"] if with_trustmark_host else [])
    rng = dict(RNG, TrustMark=(0.4, 2.0))
    base = {(f, "rot9"): flat(*rng[f], 0.60) for f in frs}
    delta = {(g, f, "rot9"): flat(*rng[g], 0.0) for f in frs for g in frs if g != f}
    d = {f: flat(*rng[f], 1.0) for f in frs}
    e = {tuple(sorted(p)): PWL([0.0, 5.0], [0.0, 0.0]) for p in [(a, b) for a in frs for b in frs if a < b]}
    fe = {"penalty_fe_resync": flat(0.4, 2.0, PEN)}
    for f in frs:                                   # resync rescues rot9 for every fragment it re-embeds
        fe[f"base_fe_resync_{f}|rot9"] = flat(*rng[f], 0.95)
        fe[f"ctrl_fe_resync_{f}|rot9"] = flat(*rng[f], 0.60)
    return Surrogate(frs, ["rot9"], rng, base, delta, d, e, frontend=fe), frs


def _psnr_with(sur, frs, resync_on, select):
    opt = z3.Optimize()
    u = {f: z3.Bool(f) for f in frs}
    for f in frs: opt.add(u[f] if f in select else z3.Not(u[f]))
    rs = z3.Bool("fe_resync")
    opt.add(rs if resync_on else z3.Not(rs))
    _, _, psnr = W.add_strength_order(opt, u, ["rot9"], sur, 0.55, False, frontends={"resync": rs}, margin=0.0)
    opt.maximize(psnr)
    assert opt.check() == z3.sat
    return float(opt.model().eval(psnr).as_fraction())


def test_the_mark_is_charged_without_trustmark_in_the_table():
    sur, frs = _sur(with_trustmark_host=False)
    on, off = _psnr_with(sur, frs, True, FR), _psnr_with(sur, frs, False, FR)
    assert abs((off - on) - PEN) < 1e-9, f"resync must cost the mark's {PEN} MSE without TrustMark, got {off - on}"


def test_the_mark_is_charged_without_trustmark_selected():
    sur, frs = _sur(with_trustmark_host=True)
    on, off = _psnr_with(sur, frs, True, FR), _psnr_with(sur, frs, False, FR)
    assert abs((off - on) - PEN) < 1e-9, f"resync must cost the mark's {PEN} MSE with TrustMark unselected, got {off - on}"


def test_with_trustmark_selected_the_charge_is_unchanged():
    sur, frs = _sur(with_trustmark_host=True)
    sel = FR + ["TrustMark"]
    on, off = _psnr_with(sur, frs, True, sel), _psnr_with(sur, frs, False, sel)
    assert abs((off - on) - PEN) < 1e-9, (off, on)
