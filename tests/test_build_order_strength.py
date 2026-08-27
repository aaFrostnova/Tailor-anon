import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import z3, watermark_smt_v2 as W
from surrogate_model import synthetic_surrogate

BASE = dict(min_psnr=34.0, max_ms=4000.0, attacks=["jpeg25","crop75"], min_ba=0.72,
            allow_resync=True, allow_nested=True, min_bits=0, resolution=512)

def _solve_psnr(**kw):
    o,u,rs,ns,al,nf,ps,tm = W.build(**kw)
    o.maximize(ps)
    assert o.check()==z3.sat
    return float(o.model().eval(ps).as_fraction())

def test_default_path_unchanged():
    # with the new args absent/false, PSNR equals the current solver's value
    assert abs(_solve_psnr(**BASE) - _solve_psnr(**BASE, enable_order=False, continuous_strength=False)) < 1e-9

def test_strength_order_solve_valid_and_continuous():
    sg = synthetic_surrogate(attacks=("jpeg25","crop75"))
    o,u,rs,ns,al,nf,ps,tm = W.build(min_psnr=0.0, max_ms=1e9, attacks=["jpeg25","crop75"],
        min_ba=0.60, allow_resync=True, allow_nested=True, min_bits=0, resolution=512,
        enable_order=True, continuous_strength=True, surrogate=sg)
    o.maximize(ps)
    assert o.check()==z3.sat
    m=o.model()
    sel=[f for f in W.FR if str(m.eval(u[f]))=="True"]
    assert len(sel)>=1
    # every selected fragment has a strength inside its native range (continuous, not a grid point)
    for f in sel:
        lo,hi=sg.range(f); sv=float(m.eval(o._svars[f]).as_fraction()); assert lo-1e-9<=sv<=hi+1e-9
    # the precedence relation on the selected set is a valid strict total order
    for i in range(len(sel)):
        for j in range(i+1,len(sel)):
            a,b=sel[i],sel[j]
            pab=str(m.eval(o._pvars[(a,b)]))=="True"; pba=str(m.eval(o._pvars[(b,a)]))=="True"
            assert pab!=pba          # exactly one direction
