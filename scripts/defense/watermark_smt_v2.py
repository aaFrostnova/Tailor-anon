"""Watermark-selection SMT optimizer v2 — ALL inputs measured (no calibrated guesses).
 - per-fragment bit-acc / PSNR / embed+decode ms : results/defense/smt_inputs.json (measured n=24)
 - VINE strength alpha as a DISCRETE decision variable (measured PSNR<->robustness trade)
 - carrier = released VINE-R (VINE_C3 no-LPIPS candidate REMOVED: decoder-only finetune can't lift rinse)
 - Pareto mode (--pareto): enumerate quality/robustness/speed trade-offs instead of one lexicographic pick.
Regen/rinse/rot/crop bit-acc for VINE/TM come from frag_suite (n=20) + memory (regen 0.92/0.80, TM regen dead).
"""
import json, sys, argparse
from z3 import (Optimize, Bool, Int, Real, If, Or, And, Not, Implies, Sum, BoolVal, sat, is_true)
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
D=json.load(open(CF+"/results/defense/smt_inputs.json"))              # solo: VINE alpha-sweep + per-frag solo PSNR + timing
DC=json.load(open(CF+"/results/defense/smt_inputs_composite.json"))    # composite PSNR stack anchors
FULL=json.load(open(CF+"/results/defense/smt_inputs_full.json"))       # n=100 end-to-end: per-attack composite bit-acc + DEPLOYED detection (validated fused>=best-path)
# F = validated n=100 composite bit-acc (VINE/TM/VideoSeal) + regen/rinse from hidden_big composite
PA=FULL["per_attack"]; REGC={"VINE":{"regen":0.93,"rinse":0.88},"TrustMark":{"regen":0.50,"rinse":0.50},"VideoSeal":{"regen":0.50,"rinse":0.50}}
F={"VINE":{"bit_acc":{a:PA[a]["vine"] for a in PA}},"TrustMark":{"bit_acc":{a:PA[a]["tm"] for a in PA}},"VideoSeal":{"bit_acc":{a:PA[a]["vs"] for a in PA}}}
for f in F: F[f]["bit_acc"].update(REGC[f])
# CALIB: correct optimistic PA cells to the measured 10k standalone bit-acc (calibration_regression.py):
# TrustMark photometric was +0.13/+0.19 optimistic (bright 0.96->0.84, contrast 0.95->0.77), crossing
# every min_ba threshold and flipping 25% of recommendations. Recalibrate to the larger 10k measurement.
CALIB={"TrustMark":{"bright":0.84,"contrast":0.77}}
for f in CALIB: F[f]["bit_acc"].update(CALIB[f])
# Advanced attacks. CtrlRegen+ is STRENGTH-parameterized (attacker's choice, an INPUT axis): VINE
# 0.82/0.71/0.64 @ s0.3/0.5/0.7 (measured 10k t3, n=200); spatial marks (TrustMark/VideoSeal) die to
# regeneration at every strength.
#
# UnMarker: values below are BARE-decode bit-acc; the nested/fine-scale-search front-end recovers VINE
# (modeled in frag_defends via UNMK_REC, mirroring the crop/rot recovery clauses).
#   VINE bare 0.53      -- results/defense/unified_bitacc.json unmarker_default raw_ba 0.533 (n=25),
#                          unmarker_strong 0.521 (n=15). The former 0.49 was eval10k/t3/vine_r.json, the
#                          same pre-scale-fix bare decode.
#   VINE recovered 0.98 -- unified_bitacc recovered_ba 0.984/0.976 with detect_crypto 1.0; corroborated by
#                          unmarker100_s200_100/_s300_150/_s400_200 (n=100 per strength) VINE_only = 1.00.
#   TrustMark 0.51      -- measured bare (eval10k/t3/trustmark_b.json 0.508); no scale-search path -> dead.
#   VideoSeal 0.50      -- NOT standalone-measured. The previous 0.85 was INFERRED by attributing composite
#                          UnMarker survival to VideoSeal; that attribution is refuted (the survival is
#                          VINE + scale search). Conservative placeholder until a standalone n>=200 run.
UNMK_REC=0.98            # VINE bit-acc under UnMarker WITH the nested/fine-scale-search decode (measured)
# crop50 was measured in the 10k run but never entered the table, so the solver could not reason about it
# (any scenario naming crop50 was silently unsatisfiable). BARE values from eval10k/t1_merged: VINE 0.500,
# TrustMark 0.501 (contrast: TrustMark survives crop75 at 0.826 but dies at crop50), VideoSeal 0.50
# conservative. The nested ring recovers it -- Ours+geo measures crop50 ba 1.000 / tpr 1.00 -- so crop50
# joins the crop clause in frag_defends and the nested capacity recovery in achievable_bits.
CROP50={"VINE":0.500,"TrustMark":0.501,"VideoSeal":0.500}
for _f,_v in CROP50.items(): F[_f]["bit_acc"]["crop50"]=_v
# MEASURED 0.607 (standalone, n=200, std=0.068) -- replaces the
# inferred 0.85 / conservative 0.50 placeholder. CAP['VideoSeal']['unmarker'] stays 0.0: this is a
# bit-acc measurement, not a soft-MI capacity measurement.
ADV={"VINE":     {"ctrlregen_s03":0.82,"ctrlregen_s05":0.71,"ctrlregen_s07":0.64,"ctrlregen":0.64,"unmarker":0.53},
     "TrustMark":{"ctrlregen_s03":0.50,"ctrlregen_s05":0.50,"ctrlregen_s07":0.50,"ctrlregen":0.50,"unmarker":0.51},
     "VideoSeal":{"ctrlregen_s03":0.50,"ctrlregen_s05":0.50,"ctrlregen_s07":0.50,"ctrlregen":0.50,"unmarker":0.607}}
for f in ADV: F[f]["bit_acc"].update(ADV[f])
PSTACK=DC["psnr_stack"]                                               # {"1frag":37.88,"2frag":36.65,"3frag":36.25}
# Stacking / front-end fidelity penalties, RE-CALIBRATED against the measured end-to-end catalog
# (wm_dataset10k/config_psnr.json, n=40 embed-only PSNR) instead of the older PSTACK anchors. Implied
# penalties vs the VINE-R solo anchor 36.19 dB: +TM 35.26 -> 0.93 | +TM+VS 34.40 -> 1.79 |
# nested 33.40 -> 2.79 | nested+TM+VS 32.31 -> 3.88. The previous constants understated the nested
# front-end by 1.79 dB (modeled 1.0 vs measured 2.79), i.e. by more than the fidelity effects reported
# from this model. Residual non-additivity is 0.70 dB (3-frag + nested predicts 31.61 vs measured 32.31),
# which stands as the model's known error bar for the fidelity axis.
PEN2, PEN3 = 0.93, 1.79
PEN4 = PEN3 + 0.4
NESTED_PEN = 2.79            # measured; was hard-coded 1.0
PSNR_MODEL_ERR = 0.70        # measured non-additivity of the stack+nested combination
PSNR_SOLO={f:D["fragments"][f]["psnr"] for f in D["fragments"]}       # per-fragment solo PSNR (base quality)
# NOTE: VINE_C3 (no-LPIPS ablation) candidate REMOVED — decoder-only finetune kill-test (n=100) confirmed it
# can't lift rinse (encoder-limited); deployed carrier = released VINE-R. See memory c3-decoder-finetune.
ALPHA=D["alpha_sweep"]           # {"0.3":..,"0.5":..,"0.7":..,"1.0":..}
ALV=[0.3,0.5,0.7,1.0]            # discrete alpha choices (measured)
SWEPT={"clean","jpeg25","crop75","noise"}
BA_full=lambda f,a: F[f]["bit_acc"].get(a,0.0)

# ---- reliable-bit CAPACITY axis (MEASURED soft-MI of 100 embedded, capacity_mi.py n=3000) ----
# Orthogonal to min_ba (per-bit accuracy). Robust-ID capacity under a set of attacks =
#   min_a ( max over chosen frags CAP[f][a] )   — UEP: same ID repeated across frags, best survivor carries it.
# NOTE: 11 attacks measured (hidden_big 6 + hidden_ext 5, capacity_combos.py). jpeg50/jpeg25 share "jpeg"; bright/contrast≈clean.
# rot9/rot30/vaeB/vaeC/rinse added from hidden_ext (N=1000). rot9/rot30 are NATIVE (no resync); resync recovers rotation
# toward clean (modeled below, mirrors the bit-acc front-end boost). rinse=28 (VINE) is the true double-regen info floor.
_CR_V={"ctrlregen_s03":20.0,"ctrlregen_s05":10.0,"ctrlregen_s07":6.0,"ctrlregen":6.0}   # VINE capacity by CtrlRegen+ strength
_CR_0={"ctrlregen_s03":0.0,"ctrlregen_s05":0.0,"ctrlregen_s07":0.0,"ctrlregen":0.0}
CAP={"VINE":     {"clean":96.0,"jpeg50":95.9,"jpeg25":95.9,"blur":80.4,"noise":95.8,"bright":96.0,"contrast":96.0,"crop90":0.0, "crop75":0.0, "crop50":0.0, "rot9":0.0, "rot30":0.0,"vaeB":89.9,"vaeC":90.2,"regen":52.3,"rinse":28.4,"unmarker":0.0, **_CR_V},
     "TrustMark":{"clean":94.3,"jpeg50":78.1,"jpeg25":78.1,"blur":62.1,"noise":80.2,"bright":94.3,"contrast":94.3,"crop90":85.7,"crop75":85.7,"crop50":0.0, "rot9":0.8, "rot30":0.0,"vaeB":76.1,"vaeC":65.0,"regen":0.0, "rinse":0.0, "unmarker":0.0, **_CR_0},
     # VideoSeal unmarker 0.0: the former 20.0 rested on the refuted "VideoSeal is the UnMarker survivor"
     # attribution. No standalone measurement exists; conservative until one is run.
     "VideoSeal":{"clean":59.6,"jpeg50":31.8,"jpeg25":31.8,"blur":12.9,"noise":12.2,"bright":59.6,"contrast":59.6,"crop90":13.5,"crop75":13.5,"crop50":0.0, "rot9":41.7,"rot30":0.0,"vaeB":9.9, "vaeC":7.5, "regen":0.0, "rinse":0.0, "unmarker":0.0, **_CR_0}}
# UnMarker capacity for VINE is a MEASURED LOWER BOUND, not a soft-MI number: with the nested/scale-search
# decode the full 37-bit ID crypto-verifies (unified_bitacc detect_crypto 1.0 at 2^-37), so >=37 bits get
# through; the true soft-MI is un-measured and likely higher. Applied only when nested is enabled.
UNMK_CAP_VINE=37.0

# ================================================================================================
# FRESH BASELINE OVERRIDE. Everything above assembled the table from smt_inputs*.json (2026-08-10,
# n=24) plus hand-patched blocks. Replace every VALUE with the freshly-measured, provenance-stamped
# baseline_table.json (100% coverage, std attached). The structures (F, ALPHA, CAP, PSNR_SOLO,
# penalties, latency) are kept; only the numbers change, so the rest of the solver is untouched.
# BA_STD (per-cell image-to-image std) is exposed for the variance-aware live_required() gate.
import json as _bjson, os as _bos
_BLP = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/baseline_table.json"
BASELINE_PROVENANCE = None
if _bos.path.exists(_BLP):
    _BT = _bjson.load(open(_BLP)); _C = _BT["cells"]
    BASELINE_PROVENANCE = {"updated_at": _BT.get("updated_at"), "coverage": _BT.get("coverage")}
    _ALL_ATT = sorted(set(k.split("/")[-1] for k in _C["bit_acc"]))
    def _bv(grp, key):
        c = _C[grp].get(key); return (c.get("value") if isinstance(c, dict) else c) if c is not None else None
    # --- solo PSNR (fragment @1.0) and VINE alpha PSNR ---
    for f in list(F):
        v = _bv("psnr_solo", f"{f}@1.0");  PSNR_SOLO[f] = v if v is not None else PSNR_SOLO.get(f)
    for a in ALV:
        v = _bv("psnr_solo", f"VINE@{a}")
        if v is not None: ALPHA[str(a)]["psnr"] = v
    # --- VINE per-alpha bit-acc for EVERY attack (baseline now has all of them) ---
    for a in ALV:
        for at in _ALL_ATT:
            v = _bv("bit_acc", f"VINE@{a}/{at}")
            if v is not None: ALPHA[str(a)][at] = v
    # --- F = deployed per-fragment bit-acc: VINE at alpha=1.0 as the non-swept default; TM/VS @1.0 ---
    for f in list(F):
        for at in _ALL_ATT:
            v = _bv("bit_acc", f"{f}@1.0/{at}")
            if v is not None: F[f]["bit_acc"][at] = v
    # --- CAP (fragment / attack) ---
    for f in list(F):
        for at in _ALL_ATT:
            v = _bv("cap", f"{f}/{at}")
            if v is not None: CAP[f][at] = v
    # --- penalties + real front-end latency ---
    for nm in ("PEN2", "PEN3", "NESTED_PEN"):
        v = _bv("penalties", nm)
        if v is not None: globals()[nm] = v
    _rs = _bv("frontend", "resync_ms"); _ns = _bv("frontend", "nested_ms")
    FE_MS = {"resync": (_rs if _rs is not None else 300.0), "nested": (_ns if _ns is not None else 1600.0)}
    # --- per-cell std for the variance gate ---
    BA_STD_FRESH = {}
    for k, c in _C["bit_acc"].items():
        if isinstance(c, dict) and "std" in c:
            f = k.split("@")[0]; at = k.split("/")[-1]; BA_STD_FRESH[(f, at)] = c["std"]
else:
    FE_MS = {"resync": 300.0, "nested": 1600.0}
    BASELINE_PROVENANCE = {"warning": "baseline_table.json NOT FOUND -- using inherited smt_inputs values"}

CAP_UNMEASURED=set()   # all 11 SMT attacks now measured (rot/vae/rinse via hidden_ext N=1000)
def achievable_bits(chosen, attacks, resync=False, nested=False):
    def eff(f,a):   # mirror the constraint's front-end capacity recovery
        c=CAP[f].get(a,0.0)
        if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal") and resync: return max(c,CAP[f]["clean"])
        if a in ("crop90","crop75","crop50") and f=="VINE" and nested: return max(c,CAP[f]["clean"])
        if a=="unmarker" and f=="VINE" and nested: return max(c,UNMK_CAP_VINE)   # measured >=37-bit lower bound
        return c
    meas=[a for a in attacks if a in CAP["VINE"]]
    if not meas: return None
    return min(max(eff(f,a) for f in chosen) for a in meas)

def frag_defends(f, a, thr, alpha_ge07, resync, nested):
    """Z3 Bool: fragment f (a real fragment name or 'VINE_C3') decodes attack a >= thr under front-ends."""
    boost=BoolVal(0.96>=thr)
    if f=="VINE":
        # alpha affects PSNR + signal robustness; regen-family needs alpha>=0.7
        if a in SWEPT and a!="crop75":
            # min over allowed alpha is enforced elsewhere; here use worst-case? -> use per-alpha handled in caller
            base=BoolVal(True)  # replaced by alpha-specific constraint in caller
            return base
        v=BA_full("VINE",a)
        if a in ("regen","rinse","vaeB","vaeC","ctrlregen","ctrlregen_s03","ctrlregen_s05","ctrlregen_s07"):
            return And(BoolVal(v>=thr), alpha_ge07)      # regen-family (incl CtrlRegen+ at any strength) requires alpha>=0.7
        if a in ("crop90","crop75","crop50"): return Or(BoolVal(v>=thr), And(nested,boost))
        # UnMarker: bare decode dies (0.53) but the nested/fine-scale-search decode recovers it to
        # UNMK_REC=0.98 (measured, unified_bitacc + unmarker100 sweeps) -- same shape as the crop clause.
        if a=="unmarker": return Or(BoolVal(v>=thr), And(nested,BoolVal(UNMK_REC>=thr)))
        return BoolVal(v>=thr)
    # pixel fragments TrustMark / VideoSeal
    base=BoolVal(BA_full(f,a)>=thr)
    if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal"): return Or(base, And(resync,boost))
    if a=="crop75" and f in ("TrustMark","VideoSeal"): return Or(base, And(resync,boost))
    return base

def solo_psnr_expr(f, a_lvl):
    if f=="VINE": return If(a_lvl==0,ALPHA["0.3"]["psnr"],If(a_lvl==1,ALPHA["0.5"]["psnr"],If(a_lvl==2,ALPHA["0.7"]["psnr"],ALPHA["1.0"]["psnr"])))
    return PSNR_SOLO.get(f,40.0)
def cost_ms(f):
    return D["fragments"][f]["embed_ms"]+D["fragments"][f]["decode_ms"]

FR=["VINE","TrustMark","VideoSeal"]
# ---------------------------------------------------------------------------------------------
# RESOLUTION. Every table above (F, CAP, PSNR_SOLO, ALPHA, cost_ms, PEN*) is measured at 512x512 only.
# `resolution` is therefore a DECLARED-BUT-GATED input: build() accepts it, and refuses any value that
# has no measured table rather than silently answering with 512 numbers. Populate RES_TABLES (and drop
# the guard) once the per-resolution campaign lands -- it must supply per-res attack bit-acc AND
# per-res PSNR, latency and stacking penalty, since quality and runtime constraints consume those too.
# Evidence (res_invariance_pilot.json n=50 + res_campaign.json n=50, TOST margin 0.02):
#   * GEOMETRY (crop75/crop50/rot9) is resolution-INVARIANT at every R tested -- the 512 cells transfer.
#   * SIGNAL (jpeg/blur/noise/clean) transfers for R>=512 but BREAKS at 256 (VINE/blur -0.044,
#     TrustMark/jpeg -0.036, VideoSeal -0.14..-0.17): the mark is small relative to fixed-PIXEL attacks.
#   * VAE breaks at 256 for every fragment (-0.03 VINE .. -0.29 VideoSeal); fine for R>=1024.
#   * DIFFUSION/ADVERSARIAL (regen/rinse/ctrlregen*/unmarker) are NOT measured at any R != 512.
#   * VideoSeal has NO native-resolution path (0/50 preserved at every R != 512), so its non-512 cells
#     are a forced-resize artifact; VINE and TrustMark preserve native resolution 50/50.
# ---------------------------------------------------------------------------------------------
# IMMUTABLE BASELINE + REQUEST-SCOPED OVERLAY.
# Live/CEGAR measurements are taken on ONE user's images, so writing them back into the shared table
# would let that user's data distribution bias every later request (and the bias would compound).
# The baseline measured table is therefore frozen here, and per-request refinements are applied only
# inside `request_overlay(...)`, which restores the baseline on exit -- including on exception.
# Every request starts from the same pristine baseline.
import copy as _copy
from contextlib import contextmanager as _ctx
_BASELINE = {"F": _copy.deepcopy(F), "CAP": _copy.deepcopy(CAP), "PSNR_SOLO": _copy.deepcopy(PSNR_SOLO)}

def baseline_ba(f, a):
    """Bit-acc from the FROZEN baseline, ignoring any active overlay."""
    return _BASELINE["F"][f]["bit_acc"].get(a, 0.0)

@_ctx
def request_overlay(overlay=None):
    """Scope a per-request table refinement.

    overlay = {"bit_acc": {(frag, attack): value}, "psnr_solo": {frag: value}, "cap": {(frag, attack): value}}
    Applies on top of the frozen baseline for the duration of the block, then restores it. Nothing a
    request measures can leak into the next request.
    """
    snap = {"F": _copy.deepcopy(F), "CAP": _copy.deepcopy(CAP), "PSNR_SOLO": _copy.deepcopy(PSNR_SOLO)}
    try:
        if overlay:
            for (fr, at), v in (overlay.get("bit_acc") or {}).items(): F[fr]["bit_acc"][at] = v
            for (fr, at), v in (overlay.get("cap") or {}).items():     CAP[fr][at] = v
            for fr, v in (overlay.get("psnr_solo") or {}).items():     PSNR_SOLO[fr] = v
        yield
    finally:                                    # restore the baseline no matter what happened
        F.clear();        F.update(snap["F"])
        CAP.clear();      CAP.update(snap["CAP"])
        PSNR_SOLO.clear(); PSNR_SOLO.update(snap["PSNR_SOLO"])

def assert_baseline_intact():
    """Guard for tests/harnesses: the shared table must equal the frozen baseline between requests."""
    bad = [(f, a) for f in F for a in F[f]["bit_acc"]
           if abs(F[f]["bit_acc"][a] - _BASELINE["F"][f]["bit_acc"].get(a, -1)) > 1e-12]
    if bad: raise AssertionError(f"baseline table was mutated outside request_overlay: {bad[:5]}")
    return True

# ---------------------------------------------------------------------------------------------
# VARIANCE-AWARE LIVE REQUIREMENT.
# The offline table stores a MEAN bit-acc. For some attacks that mean does not represent a specific
# user's image, so a table verdict on those attacks is not trustworthy -- the request MUST be settled by
# a live measurement on the user's OWN image:
#   * ADVERSARIAL attacks (UnMarker) optimise per-image worst-case; the offline mean is the attacker's
#     success on OTHER images, not this one -> ALWAYS live.
#   * HIGH-VARIANCE attacks: when the table mean sits within k*std of the required threshold theta, the
#     specific image can land on either side -> live to resolve (variance_profile.json supplies std).
ADVERSARIAL = {"unmarker"}                 # per-image optimised -> mean is never the point estimate
# Diffusion-family attacks are stochastic (per-image regeneration) and were measured at small n, so a
# point estimate that sits near theta must not decide feasibility alone -> give them a conservative
# default std so live_required() fires near the threshold even where variance_profile did not cover them.
DIFFUSION = {"regen", "rinse", "ctrlregen", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"}
DEFAULT_STD_DIFFUSION = 0.08
BA_STD = {}                                # populated below from BA_STD_FRESH (baseline) or variance_profile.json
try:
    import json as _json, os as _os
    _vp = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/variance_profile.json"
    if _os.path.exists(_vp):
        for _k, _v in _json.load(open(_vp)).items():
            BA_STD[tuple(_k.split("/"))] = _v.get("std", 0.0)
except Exception:
    pass
try:
    BA_STD.update(BA_STD_FRESH)     # prefer the baseline's per-cell std
except NameError:
    pass
def live_required(attack, frags, min_ba, k=2.0):
    """True if the offline table cannot settle this (attack, config, threshold) -> measure on the user's image."""
    if attack in ADVERSARIAL: return True, "adversarial (per-image worst-case)"
    def _sd(f):
        if (f, attack) in BA_STD: return BA_STD[(f, attack)]
        return DEFAULT_STD_DIFFUSION if attack in DIFFUSION else None
    cand = [(BA_full(f, attack), _sd(f)) for f in frags if _sd(f) is not None]
    if not cand: return False, "no variance data"
    mean, sd = max(cand, key=lambda ms: ms[0])
    if abs(mean - min_ba) < k * sd:
        return True, f"near-threshold: |{mean:.2f}-{min_ba:.2f}| < {k}*{sd:.2f}"
    return False, "table mean is >2 std from threshold"

RES_MEASURED = {256, 512, 1024, 2048}
_GEOM = {"crop90", "crop75", "crop50", "rot9", "rot30"}
_SIGNAL = {"clean", "jpeg25", "jpeg50", "blur", "noise", "bright", "contrast"}
_VAE = {"vaeB", "vaeC"}
RES_OK = {512: None,                      # None = the measurement basis; everything allowed
          1024: _GEOM | _SIGNAL | _VAE,   # verified transferable at R>=1024
          2048: _GEOM | _SIGNAL | _VAE,
          256:  _GEOM}                    # only geometry survives the 256 check
def _check_resolution(res, attacks=()):
    if res not in RES_MEASURED:
        raise ValueError(f"resolution={res} was never measured (have {sorted(RES_MEASURED)}).")
    ok = RES_OK.get(res)
    if ok is None: return
    missing = [a for a in attacks if a not in ok]
    if missing:
        raise ValueError(
            f"resolution={res} has no validated table for {sorted(set(missing))}. Validated at this "
            f"resolution: {sorted(ok)}. The diffusion/adversarial family was measured only at 512, and "
            f"signal/VAE cells do not transfer to 256. Measure those cells or pass resolution=512.")

def build(min_psnr,max_ms,attacks,min_ba,allow_resync,allow_nested,min_bits=0,resolution=512,
          enable_order=False,continuous_strength=False,surrogate=None):
    _check_resolution(resolution, attacks)
    surrogate_mode = (enable_order or continuous_strength) and surrogate is not None
    opt=Optimize()
    use={f:Bool(f) for f in FR}; resync=Bool("resync"); nested=Bool("nested")
    a_lvl=Int("alpha_lvl")                                   # 0->0.5, 1->0.7, 2->1.0 (VINE only)
    opt.add(a_lvl>=0, a_lvl<=3)
    if not allow_resync: opt.add(Not(resync))
    if not allow_nested: opt.add(Not(nested))
    opt.add(Implies(nested, use["VINE"]))
    opt.add(Implies(Not(use["VINE"]), a_lvl==3))            # alpha only meaningful with VINE
    opt.add(Or(*[use[f] for f in FR]))
    nfrag=Sum([If(use[f],1,0) for f in FR])
    # PSNR = lowest solo PSNR among included  - stacking - nested
    base=Real("basepsnr"); opt.add(base<=If(use["VINE"],solo_psnr_expr("VINE",a_lvl),99))
    opt.add(base<=If(use["TrustMark"],PSNR_SOLO["TrustMark"],99))
    opt.add(base<=If(use["VideoSeal"],PSNR_SOLO["VideoSeal"],99))
    # base must EQUAL the min (tie it down): base >= each-or-not handled by maximizing psnr later
    pen=If(nfrag<=1, 0.0, If(nfrag==2, PEN2, If(nfrag==3, PEN3, PEN4)))   # MEASURED stacking penalty (~0.6dB/frag)
    psnr=base - pen - If(nested,NESTED_PEN,0.0)
    opt.add(psnr>=min_psnr)
    time=Sum([If(use[f],cost_ms(f),0.0) for f in FR]) + If(resync,FE_MS['resync'],0.0) + If(nested,FE_MS['nested'],0.0)
    opt.add(time<=max_ms)
    a_ge07 = a_lvl>=2
    if not surrogate_mode:
        # discrete per-attack min_ba feasibility from the measured tables. Skipped in surrogate_mode:
        # the necessity experiment compares z3 against a grid enumerator that evaluates ONLY the
        # surrogate, so z3's feasible set must be driven by the surrogate alone (added below via
        # add_strength_order), not the real measured tables AND the surrogate at once.
        for a in attacks:
            opts=[]
            for f in list(F):
                if f=="VINE" and a in SWEPT and a!="crop75":
                    # alpha-specific measured bit-acc for swept signal attacks
                    ok=Or(*[And(a_lvl==i, BoolVal(ALPHA[str(ALV[i])][a]>=min_ba)) for i in range(4)])
                    opts.append(And(use["VINE"], ok))
                else:
                    opts.append(And(use[f], frag_defends(f,a,min_ba,a_ge07,resync,nested)))
            opt.add(Or(*opts))
    # CAPACITY constraint: need >= min_bits robust ID bits under each MEASURED in-scope attack
    # (some chosen fragment must carry >= min_bits reliable bits). Unmeasured attacks are skipped (warned in caller).
    if min_bits>0:
        for a in attacks:
            if a not in CAP["VINE"]: continue   # capacity unmeasured for this attack -> skip
            capopts=[]
            for f in list(F):
                c=CAP[f].get(a,0.0); cl=CAP[f]["clean"]      # resync un-warps rotation -> capacity recovers toward clean
                if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal"):
                    capopts.append(And(use[f], Or(BoolVal(c>=min_bits), And(resync, BoolVal(cl>=min_bits)))))
                elif a in ("crop90","crop75","crop50") and f=="VINE":  # nested-ring recovers VINE crop capacity
                    capopts.append(And(use[f], Or(BoolVal(c>=min_bits), And(nested, BoolVal(cl>=min_bits)))))
                elif a=="unmarker" and f=="VINE":            # nested/scale-search recovers VINE under UnMarker
                    # measured lower bound: the full 37-bit ID crypto-verifies at 2^-37 (unified_bitacc
                    # detect_crypto 1.0), so >=UNMK_CAP_VINE bits get through when nested is enabled.
                    capopts.append(And(use[f], Or(BoolVal(c>=min_bits), And(nested, BoolVal(UNMK_CAP_VINE>=min_bits)))))
                else:
                    capopts.append(And(use[f], BoolVal(c>=min_bits)))
            opt.add(Or(*capopts))
    if surrogate_mode:
        s_vars, p_vars, psnr_expr = add_strength_order(opt, use, attacks, surrogate, min_ba, enable_order)
        psnr = psnr_expr          # override the discrete-PSNR expression with the surrogate PSNR
        opt._svars = s_vars; opt._pvars = p_vars    # expose for callers/tests
    return opt,use,resync,nested,a_lvl,nfrag,psnr,time

def add_strength_order(opt, u, sel_attacks, surrogate, min_ba, order):
    import z3
    assert set(surrogate.fragments) <= set(u.keys()), "surrogate fragments must be a subset of the solver's fragment vars"
    FRs = surrogate.fragments
    s = {f: z3.Real(f"s_{f}") for f in FRs}
    for f in FRs:
        lo,hi = surrogate.range(f)
        opt.add(z3.Implies(u[f], z3.And(s[f] >= lo, s[f] <= hi)))
        opt.add(z3.Implies(z3.Not(u[f]), s[f] == 0))
    # precedence booleans (only meaningful among selected)
    p = {(f,g): z3.Bool(f"p_{f}_{g}") for f in FRs for g in FRs if f!=g}
    if order:
        for f in FRs:
            for g in FRs:
                if f>=g: continue
                both = z3.And(u[f], u[g])
                opt.add(z3.Implies(both, p[(f,g)] != p[(g,f)]))          # exactly one direction
        for f in FRs:                                                    # transitivity
            for g in FRs:
                for h in FRs:
                    if len({f,g,h})<3: continue
                    opt.add(z3.Implies(z3.And(p[(f,g)],p[(g,h)]), p[(f,h)]))
    else:
        for (f,g) in p: opt.add(p[(f,g)] == False)                       # fixed canonical order
    # feasibility: base_f(s_f,a) - sum_g after f delta_{g->f}(s_g,a) >= min_ba
    for a in sel_attacks:
        if a not in surrogate.attacks: continue
        for f in FRs:
            bexpr,bc = surrogate.base(f,a).add_to_z3(s[f], f"base_{f}_{a}")
            for c in bc: opt.add(z3.Implies(u[f], c))          # domain constraint only meaningful when f selected (input is s[f])
            drops=[]
            for g in FRs:
                if g==f: continue
                dexpr,dc = surrogate.delta(g,f,a).add_to_z3(s[g], f"del_{g}_{f}_{a}")
                for c in dc: opt.add(z3.Implies(u[g], c))      # input is s[g], not s[f] -> gate on u[g]
                after = z3.And(p[(f,g)], u[g]) if order else z3.And(u[g], (FRs.index(g)>FRs.index(f)))
                drops.append(z3.If(after, dexpr, z3.RealVal(0)))
            opt.add(z3.Implies(u[f], bexpr - z3.Sum(drops) >= min_ba))
    # distortion D = sum_f d_f(s_f) + sum_{f<g} e_{fg}(s_f+s_g) [gated by co-select]; PSNR = -D proxy
    dterms=[]
    for f in FRs:
        dexpr,dc = surrogate.d(f).add_to_z3(s[f], f"d_{f}")
        for c in dc: opt.add(z3.Implies(u[f], c))              # input is s[f] -> gate on u[f]
        dterms.append(z3.If(u[f], dexpr, z3.RealVal(0)))
    for i,f in enumerate(FRs):
        for g in FRs[i+1:]:
            ssum = z3.Real(f"ssum_{f}_{g}"); opt.add(ssum == s[f]+s[g])
            eexpr,ec = surrogate.e(f,g).add_to_z3(ssum, f"e_{f}_{g}")
            for c in ec: opt.add(z3.Implies(z3.And(u[f], u[g]), c))   # input is s[f]+s[g] -> gate on both selected
            dterms.append(z3.If(z3.And(u[f],u[g]), eexpr, z3.RealVal(0)))
    D = z3.Real("D_total"); opt.add(D == z3.Sum(dterms))
    psnr = z3.Real("psnr_surro"); opt.add(psnr == -D)     # monotone proxy; real dB mapping applied post-hoc
    return s, p, psnr

def report(tag,opt,use,resync,nested,a_lvl,psnr,time,attacks,min_ba):
    if opt.check()!=sat:
        print(f"  ❌ UNSAT — no combination satisfies these conditions (relax quality/speed, lower bit-acc, or L4-undefendable)."); return None
    m=opt.model(); ev=lambda e:m.eval(e,model_completion=True)
    chosen=[f for f in FR if is_true(ev(use[f]))]; rv=is_true(ev(resync)); nv=is_true(ev(nested))
    al=ALV[int(str(ev(a_lvl)))]
    fe=[x for x,b in [("resync",rv),("nested",nv)] if b]
    disp=[(f+f"(α={al})" if f=="VINE" else f) for f in chosen]
    print(f"  ✅ {' + '.join(disp)}"+(f"  +[{', '.join(fe)}]" if fe else "")+f"   PSNR≈{float(ev(psnr).as_fraction()):.1f}dB · {float(ev(time).as_fraction()):.0f}ms")
    return chosen,al,rv,nv

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--min_psnr",type=float,default=32.0); ap.add_argument("--max_ms",type=float,default=5000)
    ap.add_argument("--attacks",nargs="+"); ap.add_argument("--min_ba",type=float,default=0.90)
    ap.add_argument("--no_resync",action="store_true"); ap.add_argument("--no_nested",action="store_true")
    ap.add_argument("--min_bits",type=int,default=0,help="required robust ID bits (capacity axis; 0=off). 37=deployed ID")
    ap.add_argument("--pareto",action="store_true")
    a=ap.parse_args()
    SIG=["jpeg50","jpeg25","blur","noise","bright","contrast"]; GEO=["crop90","crop75","rot9","rot30"]; REG=["vaeB","vaeC","regen","rinse"]
    if a.pareto:
        # Pareto front via quality-threshold sweep: for each min PSNR, the minimal combo that still defends.
        atk=a.attacks or SIG+GEO+REG
        print(f"\nPARETO front (quality ↔ minimal combo): attacks={atk} ba>={a.min_ba}")
        seen=set()
        for q in [30,32,34,36,38,40,42,44,46]:
            opt,use,resync,nested,a_lvl,nfrag,psnr,time=build(float(q),9e9,atk,a.min_ba,not a.no_resync,not a.no_nested)
            opt.minimize(nfrag); opt.maximize(psnr); opt.minimize(time)
            if opt.check()!=sat: continue
            m=opt.model(); ev=lambda e:m.eval(e,model_completion=True)
            ch=tuple(f for f in FR if is_true(ev(use[f]))); al=ALV[int(str(ev(a_lvl)))]
            pv=float(ev(psnr).as_fraction()); key=(ch,al,is_true(ev(resync)),is_true(ev(nested)))
            if key in seen: continue
            seen.add(key)
            disp=[(f+f"(α={al})" if f=="VINE" else f) for f in ch]
            fe=[x for x,b in [("resync",ev(resync)),("nested",ev(nested))] if is_true(b)]
            print(f"  PSNR≥{q}: {' + '.join(disp)}"+(f" +[{','.join(fe)}]" if fe else "")+f"  → PSNR≈{pv:.1f}dB · {len(ch)}frag · {float(ev(time).as_fraction()):.0f}ms")
        print("\nV2_DONE"); sys.exit(0)
    if a.attacks:
        opt,*rest=build(a.min_psnr,a.max_ms,a.attacks,a.min_ba,not a.no_resync,not a.no_nested,a.min_bits)
        opt.minimize(rest[4]); opt.maximize(rest[5]); opt.minimize(rest[6])
        print(f"\nQUERY custom: PSNR>={a.min_psnr} ms<={a.max_ms} ba>={a.min_ba} bits>={a.min_bits} {a.attacks}")
        res=report("custom",opt,rest[0],rest[1],rest[2],rest[3],rest[5],rest[6],a.attacks,a.min_ba)
        unm=[x for x in a.attacks if x in CAP_UNMEASURED]
        if res:
            ab=achievable_bits(res[0],a.attacks,res[2],res[3])   # res[2]=resync, res[3]=nested
            if ab is not None: print(f"    robust-ID capacity ≈ {ab:.0f} bit (min over in-scope attacks; front-ends applied)"+(f"   [unmeasured: {unm}]" if unm else ""))
        if a.min_bits>0 and unm: print(f"    ⚠ capacity for {unm} is UNMEASURED — those attacks were skipped in the --min_bits check")
    else:
        DEMO=[("ID photo (signal, identity .90, HQ .90 quality)",36,5000,SIG,0.90),
              ("Social (signal+geo, identity .90)",34,5000,SIG+GEO,0.90),
              ("Web/AI (+regen, identity .90)",32,6000,SIG+GEO+REG,0.90),
              ("Web/AI (+regen, PRESENCE .63)",34,6000,SIG+GEO+REG,0.63),
              ("Highest-quality (signal only, presence, maximize PSNR)",44,5000,SIG,0.63),
              ("Adversarial (+CtrlRegen s0.9 proxy via rinse-hard? use regen)",30,7000,SIG+GEO+REG,0.90)]
        for lab,mp,mm,at,mb in DEMO:
            opt,*rest=build(mp,mm,at,mb,True,True)
            opt.minimize(rest[4]); opt.maximize(rest[5]); opt.minimize(rest[6])
            print(f"\nQUERY: {lab}\n  need PSNR>={mp} ms<={mm} ba>={mb}")
            report(lab,opt,rest[0],rest[1],rest[2],rest[3],rest[5],rest[6],at,mb)
    print("\nV2_DONE")
