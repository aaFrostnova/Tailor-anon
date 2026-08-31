"""Watermark selection as an SMT / optimization problem (Z3).

User gives their CONDITIONS as constraints:
  - image-quality floor   (min PSNR dB)
  - speed budget          (max embed+decode ms)
  - attack scenarios      (which attacks the image may face)
  - required bit-acc       (0.90 = crypto-identity, 0.63 = presence)
The solver returns the OPTIMAL fragment combination that satisfies them all
(minimal fragments, then best quality). If none exists -> UNSAT (e.g. L4 CtrlRegen+).

Decision variables (Bool): use each of {VINE, TrustMark, VideoSeal} + decode-side {resync, nested-ring}.
Robustness of a combo on attack a = "some included fragment (with its front-end boost) decodes a >= thr".
This is the deployed best-path/any-survives detector, encoded declaratively.
"""
from z3 import (Optimize, Bool, Real, Int, If, Or, And, Implies, Sum, BoolVal, sat, ToReal)

# ---------- MEASURED per-fragment bit-acc (frag_suite.log n=20 + composite evals) ----------
# VideoSeal signal/crop values partly estimated (its measured axis is rotation); flagged in README.
BA = {
 "VINE":      dict(clean=1.00,jpeg50=0.99,jpeg25=0.97,blur=0.97,noise=1.00,bright=1.00,contrast=1.00,
                   vaeB=0.87,vaeC=0.89,crop90=0.50,crop75=0.50,rot9=0.51,rot30=0.46,regen=0.92,rinse=0.80,
                   unmarker=0.99,ctrlregen_s05=0.82,ctrlregen_s07=0.56,ctrlregen_s09=0.29),
 "TrustMark": dict(clean=1.00,jpeg50=0.97,jpeg25=0.85,blur=1.00,noise=1.00,bright=1.00,contrast=1.00,
                   vaeB=0.58,vaeC=0.39,crop90=1.00,crop75=0.88,rot9=0.41,rot30=0.40,regen=0.39,rinse=0.39,
                   unmarker=0.55,ctrlregen_s05=0.50,ctrlregen_s07=0.50,ctrlregen_s09=0.50),
 "VideoSeal": dict(clean=1.00,jpeg50=0.95,jpeg25=0.85,blur=0.90,noise=0.90,bright=1.00,contrast=1.00,
                   vaeB=0.60,vaeC=0.50,crop90=0.70,crop75=0.50,rot9=0.83,rot30=0.55,regen=0.39,rinse=0.39,
                   unmarker=0.50,ctrlregen_s05=0.50,ctrlregen_s07=0.50,ctrlregen_s09=0.50),
}
FRAGS = ["VINE","TrustMark","VideoSeal"]
# embed+decode cost (ms, relative) and PSNR handled separately
T_EMBED = {"VINE":1800,"TrustMark":60,"VideoSeal":120}
T_DECODE= {"VINE":400,"TrustMark":40,"VideoSeal":80}
RESYNC_MS=300; NESTED_EMBED_MULT=3; NESTED_DECODE_MS=1600  # nested = 3-scale embed + scale-search decode

def frag_defends(f, a, thr, use_resync, use_nested):
    """Z3 Bool: does fragment f decode attack a at >= thr, given front-end flags?"""
    base = BoolVal(BA[f].get(a, 0.0) >= thr)
    boost = BoolVal(0.96 >= thr)   # resync/nested recover to ~0.96
    if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal"):
        return Or(base, And(use_resync, boost))          # resync re-aligns rotation for pixel frags
    if a == "crop75" and f in ("TrustMark","VideoSeal"):
        return Or(base, And(use_resync, boost))          # resync zoom-crop recovery
    if a in ("crop90","crop75") and f == "VINE":
        return Or(base, And(use_nested, boost))          # nested-ring scale-search (VINE crop -> crop50)
    return base

def py_defends(f, a, thr, resync, nested):
    """Plain-python mirror of frag_defends, for reporting the carrier of each attack."""
    if BA[f].get(a,0.0) >= thr: return True
    if 0.96 >= thr:
        if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal") and resync: return True
        if a=="crop75" and f in ("TrustMark","VideoSeal") and resync: return True
        if a in ("crop90","crop75") and f=="VINE" and nested: return True
    return False

def optimize(min_psnr, max_ms, attacks, min_ba, allow_resync=True, allow_nested=True, label=""):
    opt = Optimize()
    use = {f: Bool(f) for f in FRAGS}
    resync = Bool("resync"); nested = Bool("nested")
    # front-end availability
    if not allow_resync: opt.add(resync == False)
    if not allow_nested: opt.add(nested == False)
    opt.add(Implies(nested, use["VINE"]))                # nested-ring is a VINE placement
    opt.add(Or(*[use[f] for f in FRAGS]))                # at least one fragment
    nfrag = Sum([If(use[f],1,0) for f in FRAGS])
    # quality: base 38.5, -1.5 per fragment, -1.0 for nested embedding
    psnr = 38.5 - 1.5*ToReal(nfrag) - If(nested, 1.0, 0.0)
    opt.add(psnr >= min_psnr)
    # speed
    time = Sum([If(use[f], T_EMBED[f]+T_DECODE[f], 0) for f in FRAGS]) \
           + If(nested, T_EMBED["VINE"]*(NESTED_EMBED_MULT-1)+NESTED_DECODE_MS, 0) \
           + If(resync, RESYNC_MS, 0)
    opt.add(time <= max_ms)
    # robustness: every in-scope attack defended by some included fragment
    for a in attacks:
        opt.add(Or(*[And(use[f], frag_defends(f,a,min_ba,resync,nested)) for f in FRAGS]))
    # objective (lexicographic): fewest fragments -> highest PSNR -> fastest
    opt.minimize(nfrag); opt.maximize(psnr); opt.minimize(time)
    print(f"\n{'='*70}\nQUERY: {label}")
    print(f"  require: PSNR>={min_psnr}dB, time<={max_ms}ms, bit-acc>={min_ba} on {attacks}")
    if opt.check()==sat:
        from z3 import is_true
        m=opt.model()
        ev=lambda e: m.eval(e, model_completion=True)
        chosen=[f for f in FRAGS if is_true(ev(use[f]))]
        rv=is_true(ev(resync)); nv=is_true(ev(nested))
        fe=[x for x,b in [("resync",rv),("nested-ring",nv)] if b]
        pv=float(ev(psnr).as_fraction()); tv=int(str(ev(time)))
        print(f"  ✅ SOLUTION: {' + '.join(chosen)}" + (f"  +  [{', '.join(fe)}]" if fe else ""))
        print(f"     PSNR≈{pv:.1f}dB · time≈{tv}ms · {len(chosen)} fragment(s)")
        for a in attacks:  # who defends each
            car=[f for f in chosen if py_defends(f,a,min_ba,rv,nv)]
            print(f"       {a:14s} <- {', '.join(car) if car else '?'}")
    else:
        print("  ❌ UNSAT — no watermark combination satisfies these conditions.")
        print("     (relax quality/speed, lower required bit-acc, or accept these attacks are undefendable — cf. L4.)")

if __name__=="__main__":
    import sys, argparse
    if len(sys.argv)>1:   # custom single query
        ap=argparse.ArgumentParser(description="SMT watermark-combination optimizer")
        ap.add_argument("--min_psnr",type=float,default=32.0,help="image-quality floor (dB)")
        ap.add_argument("--max_ms",type=int,default=8000,help="speed budget (embed+decode ms)")
        ap.add_argument("--attacks",nargs="+",required=True,help="in-scope attacks (see BA keys)")
        ap.add_argument("--min_ba",type=float,default=0.90,help="required bit-acc (0.90=identity, 0.63=presence)")
        ap.add_argument("--no_resync",action="store_true"); ap.add_argument("--no_nested",action="store_true")
        a=ap.parse_args()
        optimize(a.min_psnr,a.max_ms,a.attacks,a.min_ba,not a.no_resync,not a.no_nested,label="custom")
        sys.exit(0)
    SIGNAL=["jpeg50","jpeg25","blur","noise","bright","contrast"]
    GEO=["crop90","crop75","rot9","rot30"]; REGEN=["vaeB","vaeC","regen","rinse"]
    # 1) ID photo: only signal, want identity (0.90), high quality, no speed pressure
    optimize(min_psnr=36, max_ms=6000, attacks=SIGNAL, min_ba=0.90,
             label="ID photo (signal only, identity 0.90, high quality)")
    # 2) Social media: signal+geometry, identity, resync allowed
    optimize(min_psnr=34, max_ms=6000, attacks=SIGNAL+GEO, min_ba=0.90,
             label="Social media (signal+geometry, identity 0.90)")
    # 3) Web / AI: +regeneration, identity
    optimize(min_psnr=32, max_ms=8000, attacks=SIGNAL+GEO+REGEN, min_ba=0.90,
             label="Web/AI (signal+geometry+regen, identity 0.90)")
    # 4) Web/AI but PRESENCE only (0.63) -> cheaper combo may suffice
    optimize(min_psnr=34, max_ms=8000, attacks=SIGNAL+GEO+REGEN, min_ba=0.63,
             label="Web/AI, PRESENCE-only 0.63 (cheaper OK?)")
    # 5) Fast + tiny: signal only, tight speed budget (VINE too slow) -> forces pixel frag
    optimize(min_psnr=37, max_ms=200, attacks=SIGNAL, min_ba=0.63,
             label="Fast&tiny (200ms, presence) — VINE too slow, must use pixel frag")
    # 6) Adversarial L4: include CtrlRegen+ s0.9 -> UNSAT
    optimize(min_psnr=30, max_ms=9000, attacks=SIGNAL+GEO+REGEN+["unmarker","ctrlregen_s09"], min_ba=0.63,
             label="Adversarial L4 (incl UnMarker + CtrlRegen+ s0.9)")
    # 7) Adversarial minus the killer: UnMarker + CtrlRegen+ s0.5 (defendable?)
    optimize(min_psnr=30, max_ms=9000, attacks=["unmarker","ctrlregen_s05"]+REGEN, min_ba=0.63,
             label="Strong-but-not-max (UnMarker + CtrlRegen+ s0.5, presence)")
    print("\nSMT_DONE")
