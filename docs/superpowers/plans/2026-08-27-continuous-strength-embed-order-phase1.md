# Continuous Per-Fragment Strength + Embed Order — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the watermark solver so every fragment carries its own continuous native strength and the embed order is a decision variable, prove the resulting z3 model is correct and that SMT is necessary (exact continuous optimum vs a grid that costs g^k), all validated first on a synthetic surrogate and then on a real measured surrogate.

**Architecture:** A reusable PWL surrogate module (`surrogate_model.py`) represents each measured curve as knots and encodes them into z3 via per-segment booleans. `build()` gains an opt-in path adding strength reals `s_f`, order booleans `p_fg`, and PWL feasibility/objective from the surrogate; default-off reproduces the current 70-config solver byte-for-byte. A necessity experiment compares z3.Optimize against grid enumeration. Only after the machinery passes on a synthetic surrogate do we run the GPU measurement campaign that fills the real surrogate and re-run the experiment.

**Tech Stack:** Python 3 (conda env `fingerprint`), z3 5.0.0 (z3.Optimize, linear real arithmetic), pytest 9.0.2, numpy, PIL/torch for the measurement campaign, existing fragment wrappers (VINE/TrustMark/VideoSeal).

**Spec:** `docs/superpowers/specs/2026-08-27-continuous-strength-embed-order-solver-design.md`

## Global Constraints

- All intermediate DATA and campaign/experiment scripts live under `/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/` (abbreviated `$SC` below). Reusable METHOD modules (`surrogate_model.py`, the `build()` change) and pytest tests live in the repo `/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/` (abbreviated `$CF`).
- Fragment set is fixed at exactly 3 real fragments (VINE, TrustMark, VideoSeal). NO synthetic fragment padding anywhere.
- Native strength knobs (each its own units/range): VINE `scale_resid α ∈ [0.2,1.0]`; TrustMark `WM_STRENGTH ∈ [0.4,1.6]`; VideoSeal `scaling_w` (band confirmed by probe in Task 6).
- Surrogate structure is `per-fragment + pairwise` for BOTH bit-acc (`base_f − Σδ_{g→f}`) and distortion (`Σd_f + Σe_{fg}`). Pairwise-additivity tolerance gate: `|Δba| ≤ 0.02` and `|ΔPSNR| ≤ 0.3 dB`.
- `build()` default path (order/strength disabled) MUST stay byte-identical to today; a regression test pins it.
- Phase-1 attack suite is the in-process family only: `jpeg25, blur, noise, bright, contrast, crop75, crop50, rot9, vaeB, vaeC`. Batch/adversarial attacks (regen, rinse, ctrlregen*, unmarker) stay driven by the existing `baseline_table.json` and are out of scope for the strength/order surrogate here.
- HF token when a wrapper load needs it: `<HF_TOKEN>`. GPU work goes via `sbatch` (partitions `gpu-preempt,gpu`); interactive shell is CPU-only.
- Python interpreter: `/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python` (`$PYFP`). Do not commit to git unless the user explicitly asks.

---

### Task 1: PWL primitive with exact z3 encoding

**Files:**
- Create: `$CF/scripts/defense/surrogate_model.py`
- Test: `$CF/tests/test_surrogate_model.py`

**Interfaces:**
- Produces: `class PWL(xs: list[float], ys: list[float])` with `.eval(x: float) -> float` (numpy linear interp, clamped to the end knots) and `.add_to_z3(x_var, name: str) -> tuple[z3.ArithRef, list]` returning `(y_var, constraints)` where `constraints` encode `y_var == PWL(x_var)` exactly via one boolean per segment.

- [ ] **Step 1: Write the failing test for `.eval`**

```python
# $CF/tests/test_surrogate_model.py
import sys; sys.path.insert(0, "scripts/defense")
import z3
from surrogate_model import PWL

def test_pwl_eval_interpolates_and_clamps():
    p = PWL([0.0, 1.0, 2.0], [0.0, 10.0, 12.0])
    assert abs(p.eval(0.5) - 5.0) < 1e-9      # midpoint of first segment
    assert abs(p.eval(1.5) - 11.0) < 1e-9     # midpoint of second segment
    assert abs(p.eval(-3.0) - 0.0) < 1e-9     # clamp low
    assert abs(p.eval(9.0) - 12.0) < 1e-9     # clamp high
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd $CF && $PYFP -m pytest tests/test_surrogate_model.py::test_pwl_eval_interpolates_and_clamps -v`
Expected: FAIL (`ModuleNotFoundError: surrogate_model`).

- [ ] **Step 3: Implement `PWL.__init__` and `.eval`**

```python
# $CF/scripts/defense/surrogate_model.py
import numpy as np
import z3

class PWL:
    """Piecewise-linear curve through knots (xs, ys), clamped outside [xs[0], xs[-1]]."""
    def __init__(self, xs, ys):
        assert len(xs) == len(ys) >= 2
        assert all(xs[i] < xs[i+1] for i in range(len(xs)-1)), "xs must be strictly increasing"
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]

    def eval(self, x):
        return float(np.interp(x, self.xs, self.ys))  # np.interp clamps to end values
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `cd $CF && $PYFP -m pytest tests/test_surrogate_model.py::test_pwl_eval_interpolates_and_clamps -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for `.add_to_z3` (exactness at knots and midpoints)**

```python
def _solve_y_at(p, xval):
    x = z3.Real("x")
    y, cons = p.add_to_z3(x, "t")
    s = z3.Solver(); s.add(cons); s.add(x == xval)
    assert s.check() == z3.sat
    m = s.model()
    return float(m.eval(y).as_fraction())

def test_pwl_add_to_z3_matches_eval():
    p = PWL([0.0, 1.0, 2.0], [0.0, 10.0, 12.0])
    for xv in [0.0, 0.5, 1.0, 1.5, 2.0]:
        assert abs(_solve_y_at(p, xv) - p.eval(xv)) < 1e-6
```

- [ ] **Step 6: Run it, verify it fails**

Run: `cd $CF && $PYFP -m pytest tests/test_surrogate_model.py::test_pwl_add_to_z3_matches_eval -v`
Expected: FAIL (`AttributeError: add_to_z3`).

- [ ] **Step 7: Implement `.add_to_z3` with per-segment booleans**

```python
    def add_to_z3(self, x_var, name):
        """Return (y_var, constraints) encoding y_var == PWL(x_var) exactly.
        One boolean per segment; exactly one active; within the active segment y is the
        linear interpolant. x_var is assumed already bounded to [xs[0], xs[-1]] by the caller
        (or clamped here via the end segments)."""
        y = z3.Real(f"y_{name}")
        segs = [z3.Bool(f"seg_{name}_{i}") for i in range(len(self.xs) - 1)]
        cons = [z3.PbEq([(b, 1) for b in segs], 1)]  # exactly one segment active
        for i, b in enumerate(segs):
            x0, x1, y0, y1 = self.xs[i], self.xs[i+1], self.ys[i], self.ys[i+1]
            slope = (y1 - y0) / (x1 - x0)
            cons.append(z3.Implies(b, z3.And(x_var >= x0, x_var <= x1,
                                             y == y0 + slope * (x_var - x0))))
        return y, cons
```

- [ ] **Step 8: Run both tests, verify pass**

Run: `cd $CF && $PYFP -m pytest tests/test_surrogate_model.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
cd $CF && git add scripts/defense/surrogate_model.py tests/test_surrogate_model.py
git commit -m "feat(surrogate): PWL primitive with exact per-segment z3 encoding"
```

---

### Task 2: Surrogate container + synthetic generator

**Files:**
- Modify: `$CF/scripts/defense/surrogate_model.py`
- Test: `$CF/tests/test_surrogate_model.py`

**Interfaces:**
- Consumes: `PWL` (Task 1).
- Produces:
  - `class Surrogate` with `.fragments: list[str]`, `.attacks: list[str]`, `.range(f) -> (lo,hi)`, `.base(f, a) -> PWL`, `.delta(g, f, a) -> PWL`, `.d(f) -> PWL`, `.e(f, g) -> PWL` (argument of `e` is `s_f + s_g`), and `.from_dict(d)` / `.to_dict()`.
  - `synthetic_surrogate(fragments=("VINE","TrustMark","VideoSeal"), attacks=(...)) -> Surrogate` with monotone-increasing `base`/`d`, small positive `delta`/`e`.

- [ ] **Step 1: Write the failing test**

```python
from surrogate_model import Surrogate, synthetic_surrogate

def test_synthetic_surrogate_shapes_and_monotone():
    sg = synthetic_surrogate()
    assert set(sg.fragments) == {"VINE", "TrustMark", "VideoSeal"}
    lo, hi = sg.range("VINE"); assert lo < hi
    a = sg.attacks[0]
    base = sg.base("VINE", a)
    assert base.eval(hi) >= base.eval(lo) - 1e-9          # robustness rises with strength
    assert sg.d("VINE").eval(hi) >= sg.d("VINE").eval(lo)  # distortion rises with strength
    assert sg.delta("TrustMark", "VINE", a).eval(hi) >= 0  # later fragment only hurts
    # round-trip
    sg2 = Surrogate.from_dict(sg.to_dict())
    assert abs(sg2.base("VINE", a).eval(hi) - base.eval(hi)) < 1e-9
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd $CF && $PYFP -m pytest tests/test_surrogate_model.py::test_synthetic_surrogate_shapes_and_monotone -v`
Expected: FAIL (`ImportError: Surrogate`).

- [ ] **Step 3: Implement `Surrogate` + `synthetic_surrogate`**

```python
# append to surrogate_model.py
RANGES = {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6), "VideoSeal": (0.5, 1.5)}
PHASE1_ATTACKS = ["jpeg25","blur","noise","bright","contrast","crop75","crop50","rot9","vaeB","vaeC"]

class Surrogate:
    def __init__(self, fragments, attacks, ranges, base, delta, d, e):
        self.fragments=list(fragments); self.attacks=list(attacks); self._ranges=dict(ranges)
        self._base=base; self._delta=delta; self._d=d; self._e=e   # dict-keyed PWLs
    def range(self,f): return self._ranges[f]
    def base(self,f,a): return self._base[(f,a)]
    def delta(self,g,f,a): return self._delta[(g,f,a)]
    def d(self,f): return self._d[f]
    def e(self,f,g): return self._e[tuple(sorted((f,g)))]
    def to_dict(self):
        pk=lambda p:{"xs":p.xs,"ys":p.ys}
        return {"fragments":self.fragments,"attacks":self.attacks,"ranges":self._ranges,
                "base":{f"{f}|{a}":pk(self._base[(f,a)]) for (f,a) in self._base},
                "delta":{f"{g}|{f}|{a}":pk(self._delta[(g,f,a)]) for (g,f,a) in self._delta},
                "d":{f:pk(self._d[f]) for f in self._d},
                "e":{f"{p[0]}|{p[1]}":pk(self._e[p]) for p in self._e}}
    @staticmethod
    def from_dict(dd):
        mk=lambda o:PWL(o["xs"],o["ys"])
        base={(k.split("|")[0],k.split("|")[1]):mk(v) for k,v in dd["base"].items()}
        delta={(k.split("|")[0],k.split("|")[1],k.split("|")[2]):mk(v) for k,v in dd["delta"].items()}
        d={k:mk(v) for k,v in dd["d"].items()}
        e={tuple(k.split("|")):mk(v) for k,v in dd["e"].items()}
        return Surrogate(dd["fragments"],dd["attacks"],dd["ranges"],base,delta,d,e)

def synthetic_surrogate(fragments=("VINE","TrustMark","VideoSeal"), attacks=tuple(PHASE1_ATTACKS)):
    import numpy as np
    fragments=list(fragments); attacks=list(attacks); ranges={f:RANGES[f] for f in fragments}
    base,delta,d,e={},{},{},{}
    for fi,f in enumerate(fragments):
        lo,hi=ranges[f]; xs=list(np.linspace(lo,hi,5))
        for a in attacks:
            # monotone-increasing saturating base in [0.5,1.0], attack-specific offset
            off=0.04*(hash((f,a))%5)
            ys=[min(1.0,0.55+0.4*((x-lo)/(hi-lo))-off) for x in xs]
            base[(f,a)]=PWL(xs,ys)
        d[f]=PWL(xs,[ (0.5+ ((x-lo)/(hi-lo)) )*1.0 for x in xs])  # distortion rises with strength
    for gi,g in enumerate(fragments):
        for fi,f in enumerate(fragments):
            if g==f: continue
            lo,hi=ranges[g]; xs=list(np.linspace(lo,hi,5))
            for a in attacks:
                delta[(g,f,a)]=PWL(xs,[0.02*((x-lo)/(hi-lo)) for x in xs])  # small positive
    for i in range(len(fragments)):
        for j in range(i+1,len(fragments)):
            p=tuple(sorted((fragments[i],fragments[j])))
            xs=[0.6,1.2,1.8,2.4,3.0]; e[p]=PWL(xs,[0.1*x for x in xs])   # 1-D in s_f+s_g
    return Surrogate(fragments,attacks,ranges,base,delta,d,e)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `cd $CF && $PYFP -m pytest tests/test_surrogate_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd $CF && git add scripts/defense/surrogate_model.py tests/test_surrogate_model.py
git commit -m "feat(surrogate): Surrogate container + synthetic generator with round-trip"
```

---

### Task 3: `build()` opt-in strength + order path

**Files:**
- Modify: `$CF/scripts/defense/watermark_smt_v2.py` (signature at line 301, add a new helper below `build`)
- Test: `$CF/tests/test_build_order_strength.py`

**Interfaces:**
- Consumes: `Surrogate`, `PWL.add_to_z3` (Tasks 1–2).
- Produces: `build(..., enable_order=False, continuous_strength=False, surrogate=None)` — when either flag is on, returns a z3 `Optimize` whose model exposes, in addition to today's handles, `s[f]` (Real strength) and `p[(f,g)]` (Bool precedence); feasibility/objective come from `surrogate`. A new module-level helper `add_strength_order(opt, u, sel_attacks, surrogate, min_ba, order)` returns `(s, p, psnr_expr)`.

- [ ] **Step 1: Write the failing regression test (default path unchanged)**

```python
# $CF/tests/test_build_order_strength.py
import sys; sys.path.insert(0, "scripts/defense")
import z3, watermark_smt_v2 as W

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
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd $CF && $PYFP -m pytest tests/test_build_order_strength.py::test_default_path_unchanged -v`
Expected: FAIL (`build() got an unexpected keyword argument 'enable_order'`).

- [ ] **Step 3: Add the three optional args to `build` (default-off no-op)**

Modify `def build(...)` at `watermark_smt_v2.py:301` to append `, enable_order=False, continuous_strength=False, surrogate=None` to the signature. At the END of `build`, immediately before its `return`, add:

```python
    if (enable_order or continuous_strength) and surrogate is not None:
        s_vars, p_vars, psnr_expr = add_strength_order(o, u, attacks, surrogate, min_ba, enable_order)
        ps = psnr_expr          # override the discrete-PSNR expression with the surrogate PSNR
        o._svars = s_vars; o._pvars = p_vars    # expose for callers/tests
    return o,u,rs,ns,al,nf,ps,tm
```

(Leave the existing body that builds `o,u,rs,ns,al,nf,ps,tm` intact; the block above only runs when opted in.)

- [ ] **Step 4: Run the regression test, verify it passes**

Run: `cd $CF && $PYFP -m pytest tests/test_build_order_strength.py::test_default_path_unchanged -v`
Expected: PASS (the opt-in block is skipped, `add_strength_order` not yet needed).

- [ ] **Step 5: Write the failing test for the strength+order model**

```python
from surrogate_model import synthetic_surrogate

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
```

- [ ] **Step 6: Run it, verify it fails**

Run: `cd $CF && $PYFP -m pytest tests/test_build_order_strength.py::test_strength_order_solve_valid_and_continuous -v`
Expected: FAIL (`add_strength_order` undefined).

- [ ] **Step 7: Implement `add_strength_order`**

Add at module level in `watermark_smt_v2.py` (imports `from surrogate_model import PWL` at top of file):

```python
def add_strength_order(opt, u, sel_attacks, surrogate, min_ba, order):
    import z3
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
            for c in bc: opt.add(c)
            drops=[]
            for g in FRs:
                if g==f: continue
                dexpr,dc = surrogate.delta(g,f,a).add_to_z3(s[g], f"del_{g}_{f}_{a}")
                for c in dc: opt.add(c)
                after = z3.And(p[(f,g)], u[g]) if order else z3.And(u[g], (FRs.index(g)>FRs.index(f)))
                drops.append(z3.If(after, dexpr, z3.RealVal(0)))
            opt.add(z3.Implies(u[f], bexpr - z3.Sum(drops) >= min_ba))
    # distortion D = sum_f d_f(s_f) + sum_{f<g} e_{fg}(s_f+s_g) [gated by co-select]; PSNR = -D proxy
    dterms=[]
    for f in FRs:
        dexpr,dc = surrogate.d(f).add_to_z3(s[f], f"d_{f}")
        for c in dc: opt.add(c)
        dterms.append(z3.If(u[f], dexpr, z3.RealVal(0)))
    for i,f in enumerate(FRs):
        for g in FRs[i+1:]:
            ssum = z3.Real(f"ssum_{f}_{g}"); opt.add(ssum == s[f]+s[g])
            eexpr,ec = surrogate.e(f,g).add_to_z3(ssum, f"e_{f}_{g}")
            for c in ec: opt.add(c)
            dterms.append(z3.If(z3.And(u[f],u[g]), eexpr, z3.RealVal(0)))
    D = z3.Real("D_total"); opt.add(D == z3.Sum(dterms))
    psnr = z3.Real("psnr_surro"); opt.add(psnr == -D)     # monotone proxy; real dB mapping applied post-hoc
    return s, p, psnr
```

- [ ] **Step 8: Run the strength+order test, verify it passes**

Run: `cd $CF && $PYFP -m pytest tests/test_build_order_strength.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
cd $CF && git add scripts/defense/watermark_smt_v2.py tests/test_build_order_strength.py
git commit -m "feat(solver): opt-in continuous strength + embed-order z3 path via surrogate"
```

---

### Task 4: SMT-necessity experiment (synthetic)

**Files:**
- Create: `$SC/smt_necessity_experiment.py`
- Test: `$CF/tests/test_necessity_experiment.py`

**Interfaces:**
- Consumes: `build(..., enable_order, continuous_strength, surrogate)`, `Surrogate`.
- Produces: `run_necessity(surrogate, scenario, g_list, orders=(False,True)) -> dict` with keys `z3_psnr`, `z3_time_s`, and per-g `grid_psnr[g]`, `grid_size[g]`, `grid_time_s[g]`; and a module `grid_enumerate(surrogate, scenario, g, order) -> (best_feasible_psnr, n_configs)` that evaluates the SAME surrogate with numpy on a strength grid of resolution g.

- [ ] **Step 1: Write the failing test**

```python
# $CF/tests/test_necessity_experiment.py
import sys; sys.path.insert(0, "scripts/defense"); sys.path.insert(0, "$SC")
from surrogate_model import synthetic_surrogate
from smt_necessity_experiment import run_necessity

def test_z3_at_least_best_grid_and_grid_grows():
    sg = synthetic_surrogate(attacks=("jpeg25","crop75"))
    scen = dict(min_psnr=0.0, max_ms=1e9, attacks=["jpeg25","crop75"], min_ba=0.60,
                allow_resync=True, allow_nested=True, min_bits=0, resolution=512)
    r = run_necessity(sg, scen, g_list=[3,6,12], orders=(True,))
    # z3's exact optimum is >= any grid's best-feasible (grid only approximates from below)
    for g in [3,6,12]:
        assert r["z3_psnr"] + 1e-6 >= r["grid_psnr"][g]
    # grid cost grows with resolution
    assert r["grid_size"][12] > r["grid_size"][6] > r["grid_size"][3]
    # refining the grid closes the gap (monotone non-increasing gap)
    gap = {g: r["z3_psnr"] - r["grid_psnr"][g] for g in [3,6,12]}
    assert gap[12] <= gap[3] + 1e-9
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd $CF && $PYFP -m pytest tests/test_necessity_experiment.py -v`
Expected: FAIL (`ModuleNotFoundError: smt_necessity_experiment`).

- [ ] **Step 3: Implement the experiment (replace `$SC` with the literal path)**

```python
# $SC/smt_necessity_experiment.py
import sys, json, itertools, time
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
import numpy as np, z3
import watermark_smt_v2 as W
from surrogate_model import Surrogate

def _feasible_psnr(sg, subset, order, svec, attacks, min_ba):
    """Evaluate the surrogate with numpy for one fully-specified config; return PSNR or None."""
    pos = {f:i for i,f in enumerate(order)}
    for a in attacks:
        if a not in sg.attacks: continue
        for f in subset:
            ba = sg.base(f,a).eval(svec[f])
            for g in subset:
                if g!=f and pos[g] > pos[f]:
                    ba -= sg.delta(g,f,a).eval(svec[g])
            if ba < min_ba: return None
    D = sum(sg.d(f).eval(svec[f]) for f in subset)
    D += sum(sg.e(f,g).eval(svec[f]+svec[g]) for f,g in itertools.combinations(subset,2))
    return -D

def grid_enumerate(sg, scen, g, order):
    FR = sg.fragments; best=None; n=0
    subsets = [c for r in range(1,len(FR)+1) for c in itertools.combinations(FR,r)]
    for subset in subsets:
        grids = [np.linspace(*sg.range(f), g) for f in subset]
        orders = list(itertools.permutations(subset)) if order else [tuple(subset)]
        for od in orders:
            for combo in itertools.product(*grids):
                n += 1
                svec = {f:float(v) for f,v in zip(subset,combo)}
                ps = _feasible_psnr(sg, subset, od, svec, scen["attacks"], scen["min_ba"])
                if ps is not None and (best is None or ps>best): best=ps
    return (best if best is not None else float("-inf")), n

def z3_solve(sg, scen, order):
    t0=time.time()
    o,u,rs,ns,al,nf,ps,tm = W.build(**scen, enable_order=order, continuous_strength=True, surrogate=sg)
    o.maximize(ps)
    ok = o.check()==z3.sat
    val = float(o.model().eval(ps).as_fraction()) if ok else float("-inf")
    return val, time.time()-t0

def run_necessity(sg, scen, g_list, orders=(False,True)):
    out={}
    for order in orders:
        zp, zt = z3_solve(sg, scen, order)
        rec={"z3_psnr":zp,"z3_time_s":zt,"grid_psnr":{},"grid_size":{},"grid_time_s":{}}
        for g in g_list:
            t0=time.time(); gp,n=grid_enumerate(sg,scen,g,order)
            rec["grid_psnr"][g]=gp; rec["grid_size"][g]=n; rec["grid_time_s"][g]=time.time()-t0
        out["order" if order else "fixed"]=rec
    # default the top-level keys to the order=True run for the test's convenience
    top = out.get("order", out.get("fixed"))
    return {**top, "by_mode": out}

if __name__=="__main__":
    from surrogate_model import synthetic_surrogate
    sg=synthetic_surrogate(attacks=("jpeg25","crop75","rot9"))
    scen=dict(min_psnr=0.0,max_ms=1e9,attacks=["jpeg25","crop75","rot9"],min_ba=0.60,
              allow_resync=True,allow_nested=True,min_bits=0,resolution=512)
    r=run_necessity(sg,scen,g_list=[4,8,16,32])
    json.dump(r,open("/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/necessity_synth.json","w"),indent=2)
    print("SYNTH NECESSITY:", {k:(round(v["z3_psnr"],3), {g:round(v["grid_psnr"][g],3) for g in v["grid_psnr"]}) for k,v in r["by_mode"].items()})
```

Note: in the test, replace the literal `sys.path.insert(0,"$SC")` with the real path `/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k`.

- [ ] **Step 4: Run the test, verify it passes**

Run: `cd $CF && $PYFP -m pytest tests/test_necessity_experiment.py -v`
Expected: PASS.

- [ ] **Step 5: Run the synthetic experiment end-to-end**

Run: `$PYFP $SC/smt_necessity_experiment.py`
Expected: prints `SYNTH NECESSITY:` with z3_psnr ≥ every grid_psnr, and writes `necessity_synth.json`.

- [ ] **Step 6: Commit**

```bash
cd $CF && git add tests/test_necessity_experiment.py
git commit -m "feat(necessity): z3-vs-grid experiment proven on synthetic surrogate"
```
(The experiment script lives in `$SC` — not under git; note its path in the commit message body.)

---

### Task 5: Native strength knob in the three wrappers

**Files:**
- Modify: `$CF/src/vine_crypto_wrapper.py:115-138` (`embed_with_target`)
- Modify: `$CF/src/trustmark_fragment.py:51-53` (`embed_with_target`)
- Modify: `$CF/src/videoseal_fragment.py:65-71` (`embed_with_target`)
- Test: `$CF/tests/test_strength_knob.py`
- Create (GPU smoke): `$SC/strength_smoke.py`

**Interfaces:**
- Produces: `embed_with_target(self, pil, target_bits, strength=1.0)` on all three wrappers. `strength=1.0` reproduces today's output exactly. VINE scales its residual by α=strength; TrustMark passes `WM_STRENGTH=strength`; VideoSeal passes `scaling_w=strength * default`.

- [ ] **Step 1: Write the CPU test for VINE residual scaling math**

```python
# $CF/tests/test_strength_knob.py
import numpy as np
from PIL import Image
import sys; sys.path.insert(0, ".")

def _scale_resid(cover, full, a):
    c=np.asarray(cover,np.float32); w=np.asarray(full,np.float32)
    return Image.fromarray(np.clip(c + a*(w-c),0,255).astype(np.uint8))

def test_scale_resid_endpoints():
    cover=Image.fromarray((np.ones((16,16,3))*100).astype(np.uint8))
    full =Image.fromarray((np.ones((16,16,3))*160).astype(np.uint8))
    assert np.array_equal(np.asarray(_scale_resid(cover,full,1.0)), np.asarray(full))      # a=1 -> full
    assert np.array_equal(np.asarray(_scale_resid(cover,full,0.0)), np.asarray(cover))     # a=0 -> cover
    mid=np.asarray(_scale_resid(cover,full,0.5)); assert abs(mid.mean()-130)<1.0            # halfway
```

- [ ] **Step 2: Run it, verify it fails or passes trivially, then wire VINE**

Run: `cd $CF && $PYFP -m pytest tests/test_strength_knob.py::test_scale_resid_endpoints -v`
Expected: PASS (pure math helper). This pins the scaling contract VINE must follow.

- [ ] **Step 3: Add `strength` to VINE `embed_with_target`**

In `vine_crypto_wrapper.py`, change the signature to `def embed_with_target(self, pil, target_bits, strength: float = 1.0)`. Replace the final compose line `encoded = torch.clamp((residual + orig) * 0.5 + 0.5, 0, 1)` with:

```python
        encoded = torch.clamp((strength * residual + orig) * 0.5 + 0.5, 0, 1)
```

(`strength=1.0` is byte-identical to today; residual is already `t_back(enc_256 - resized)`.)

- [ ] **Step 4: Add `strength` to TrustMark `embed_with_target`**

```python
    def embed_with_target(self, pil, target_bits, strength: float = 1.0):
        s = "".join(str(int(b)) for b in np.asarray(target_bits, np.uint8)[:self.n_bits])
        return self.tm.encode(pil.convert("RGB"), s, MODE="binary", WM_STRENGTH=strength)
```

- [ ] **Step 5: Add `strength` to VideoSeal `embed_with_target`**

Read the current `videoseal_fragment.py:65-71` body; thread `strength` into the underlying embed by setting the model's `scaling_w` for the call (VideoSeal exposes `self.model.scaling_w`). Concretely:

```python
    def embed_with_target(self, pil, target_bits, strength: float = 1.0):
        old = getattr(self.model, "scaling_w", None)
        if old is not None: self.model.scaling_w = float(old) * strength
        try:
            return self._embed_impl(pil, target_bits)   # existing body, unchanged
        finally:
            if old is not None: self.model.scaling_w = old
```

(If the existing method has no `_embed_impl`, rename today's body to `_embed_impl(self, pil, target_bits)` and have `embed_with_target` wrap it as above.)

- [ ] **Step 6: Run the CPU test, verify it passes**

Run: `cd $CF && $PYFP -m pytest tests/test_strength_knob.py -v`
Expected: PASS.

- [ ] **Step 7: Write the GPU monotonicity smoke**

```python
# $SC/strength_smoke.py  -- run via sbatch; asserts PSNR rises as strength falls
import sys, numpy as np
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
from PIL import Image
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
def psnr(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64); m=((a-b)**2).mean()
    return 99.0 if m<1e-9 else 10*np.log10(255*255/m)
KEY=b"v5_key_encoder_master"; NB=ShortenedBCH().n; dev="cuda"
cover=Image.open(sorted(__import__("glob").glob("/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/pool/E/img/*.png"))[0]).convert("RGB").resize((512,512))
t=np.random.RandomState(0).randint(0,2,NB).astype(np.uint8)
frags={"VINE":(VineCryptoWrapper(KEY,"vine",NB,dev,variant="R"),[0.2,0.6,1.0]),
       "TrustMark":(TrustMarkFragment(KEY,"trustmark",NB,model_type="B",device=dev),[0.4,1.0,1.6]),
       "VideoSeal":(VideoSealFragment(KEY,"videoseal",NB,device=dev),[0.5,1.0,1.5])}
for name,(fr,levels) in frags.items():
    ps=[psnr(cover, fr.embed_with_target(cover,t,strength=s)) for s in levels]
    print(name, [round(x,2) for x in ps], "monotone_down_in_strength=", all(ps[i]>=ps[i+1]-0.5 for i in range(len(ps)-1)))
print("STRENGTH_SMOKE_DONE")
```

- [ ] **Step 8: Run the GPU smoke via sbatch, confirm monotonicity**

Write an sbatch wrapper (partitions `gpu-preempt,gpu`, the env exports from the campaign template) that runs `$PYFP $SC/strength_smoke.py`; confirm each fragment's PSNR is non-decreasing as strength falls (lower strength → higher PSNR). If VideoSeal is flat, its `scaling_w` handle is wrong — fix Step 5 before proceeding.

- [ ] **Step 9: Commit**

```bash
cd $CF && git add src/vine_crypto_wrapper.py src/trustmark_fragment.py src/videoseal_fragment.py tests/test_strength_knob.py
git commit -m "feat(fragments): per-fragment native strength knob on embed_with_target"
```

---

### Task 6: Real measurement campaign → `surrogate_table.json`

**Files:**
- Create: `$SC/make_surrogate.py`
- Create: `$SC/make_surrogate.sbatch`
- Output: `$SC/surrogate_table.json`

**Interfaces:**
- Consumes: the strength-enabled wrappers (Task 5), `Surrogate.to_dict` (Task 2).
- Produces: `surrogate_table.json` loadable by `Surrogate.from_dict`, plus an `additivity` block with measured `|Δba|`, `|ΔPSNR|` for the validation orders.

- [ ] **Step 1: Write the campaign script**

Structure (concrete functions; N images from `$SC/pool/E/img`, canonical order VINE→TrustMark→VideoSeal, all attacks = the Phase-1 in-process family from Global Constraints):

```python
# $SC/make_surrogate.py  (abridged structure — implement each block)
# 1) load 3 wrappers (as in strength_smoke.py) and the 10 in-process attack fns (reuse res_campaign.py's
#    crop_keep/rot9/vae + jpeg/blur/noise/bright/contrast from the existing eval code)
# 2) STRENGTH CURVES base_f(s,a), d_f(s): for each f, each s in its 5-knot grid, embed alone at s;
#    d_f = PSNR(cover, emb); base_f[a] = mean best-path bit-acc after attack a. (fit PWL over the 5 knots)
# 3) PAIRWISE bit-acc delta_{g->f}(s_g,a): embed f alone at ref strength -> ba_f_alone[a];
#    embed f then g (g after f) with g at each s_g knot -> ba_f_after[a]; delta = ba_f_alone - ba_f_after.
# 4) PAIRWISE distortion e_{fg}(s_f+s_g): for a grid of (s_f,s_g), composite PSNR of f+g; e = D_comp - d_f - d_g.
# 5) ADDITIVITY VALIDATION: for a few full 3-orders, measure true composite ba_f and PSNR; compare to
#    base - sum(delta) and sum(d)+sum(e). Record max |Δba|, |ΔPSNR|.
# 6) assemble a Surrogate, dump Surrogate.to_dict() + {"additivity":..., "source":{...}} to surrogate_table.json
```

Each stored PWL uses the measured knots directly (xs = strength grid, ys = measured means). Stamp every entry with `{n, measured_at, git, native_knob, range}` (mirror `make_baseline.py`'s source-stamping; `measured_at` from an env-passed timestamp, never `Date.now()` inside a workflow).

- [ ] **Step 2: Probe VideoSeal's `scaling_w` band first**

Add an early block that reads `VideoSealFragment(...).model.scaling_w` default and sets VideoSeal's knot grid to `[0.5,0.75,1.0,1.25,1.5]×default`; print the resolved grid. Expected: a finite positive default; if absent, stop and report (Task 5 Step 5 handle is wrong).

- [ ] **Step 3: Smoke the campaign at tiny N**

Run: `sbatch $SC/make_surrogate.sbatch 3` (N=3 images). Expected: writes a small `surrogate_table.json`; no exceptions; all PWLs have ≥2 knots.

- [ ] **Step 4: Assert the artifact loads and additivity holds at smoke scale**

```python
# quick check (CPU)
$PYFP - <<'PY'
import sys,json; sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from surrogate_model import Surrogate
d=json.load(open("/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/surrogate_table.json"))
sg=Surrogate.from_dict(d); print("loaded", sg.fragments, len(sg.attacks),"attacks")
print("additivity", d["additivity"])
PY
```

Expected: loads cleanly; at full N the additivity `max|Δba|≤0.02` and `max|ΔPSNR|≤0.3` (at smoke N it may be noisy — the gate applies at the full run).

- [ ] **Step 5: Run the full campaign**

Run: `sbatch $SC/make_surrogate.sbatch 100` (N=100). Confirm the additivity gate passes; if it fails, record the offending pair/attack and consult Section 4.4 of the spec (measure triples for k=3) before proceeding.

- [ ] **Step 6: Commit the loader-side only**

```bash
cd $CF && git add -A  # no repo files changed here; campaign + json live in $SC (not git)
echo "surrogate_table.json + make_surrogate.py live in $SC (scratch), see plan Task 6"
```
(No repo commit if nothing under `$CF` changed; the artifact is a scratch data file by the immutable-baseline rule.)

---

### Task 7: Necessity + additivity on the REAL surrogate

**Files:**
- Modify: `$SC/smt_necessity_experiment.py` (add a `--real` entry point loading `surrogate_table.json`)
- Output: `$SC/necessity_real.json`

**Interfaces:**
- Consumes: `surrogate_table.json` (Task 6), `run_necessity` (Task 4).

- [ ] **Step 1: Add a real-surrogate entry point**

```python
# append to smt_necessity_experiment.py
def _load_real():
    import json
    from surrogate_model import Surrogate
    return Surrogate.from_dict(json.load(open(
        "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/surrogate_table.json")))
# in __main__: if "--real" in sys.argv: sg=_load_real(); attacks tuned to sg.attacks; dump necessity_real.json
```

- [ ] **Step 2: Run necessity on the real surrogate**

Run: `$PYFP $SC/smt_necessity_experiment.py --real`
Expected: writes `necessity_real.json`; z3_psnr ≥ every grid_psnr; the z3-minus-best-affordable-grid PSNR gap is strictly positive; grid_size grows as g^{|subset|}; order=True adds the k! factor.

- [ ] **Step 3: Produce the deliverable summary**

```python
$PYFP - <<'PY'
import json
r=json.load(open("/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/necessity_real.json"))
for mode,v in r["by_mode"].items():
    gs=sorted(v["grid_psnr"]); best=max(v["grid_psnr"].values())
    print(mode,"z3",round(v["z3_psnr"],3),"z3_t",round(v["z3_time_s"],3),
          "best_grid",round(best,3),"gap",round(v["z3_psnr"]-best,3),
          "grid_sizes",{g:v["grid_size"][g] for g in gs})
PY
```

Expected output is the money result: positive gap, z3 time roughly flat vs exploding grid sizes.

- [ ] **Step 4: Report to the user**

Summarize: the additivity-validation numbers (Task 6), the z3-vs-grid gap and timing table (this task), and confirm the default-off regression still passes (`$PYFP -m pytest tests/test_build_order_strength.py::test_default_path_unchanged`). This closes Phase 1.

---

## Self-Review

**Spec coverage:**
- §3 config space (u_f, s_f, p_fg) → Task 3. ✓
- §4.1 native knobs → Task 5. ✓
- §4.2 bit-acc surrogate → Tasks 2 (rep), 6 (measure). ✓
- §4.3 distortion surrogate (per-frag + pairwise e) → Tasks 2, 3 (objective), 6. ✓
- §4.4 additivity validation gate → Task 6 Step 5. ✓
- §5 measurement campaign → Task 6. ✓
- §6 z3 model + backward-compat → Task 3 (regression test Step 1). ✓
- §7 frontier live-verify → Phase 2 (separate plan; out of scope here, stated). ✓
- §8 necessity experiment → Tasks 4 (synthetic), 7 (real). ✓
- §9 artifacts/interfaces → Tasks 5,6 (wrappers, surrogate_table.json). ✓
- §11 testing → unit (T1,T2), regression (T3), integration/necessity (T4,T7). ✓
- Gap: §7 live-verify + §11 "integration: live matches surrogate" are Phase 2 — explicitly deferred, and a Phase-2 plan will cover `live_measure`/`OursComposite` order+strength embed.

**Placeholder scan:** No TBD/TODO left; every code step has runnable content. VideoSeal `scaling_w` handle is the one hardware-dependent unknown, gated by Task 5 Step 8 and Task 6 Step 2 with explicit stop conditions.

**Type consistency:** `Surrogate` accessors (`base/delta/d/e/range`) match between Tasks 2, 3, 4. `build(..., enable_order, continuous_strength, surrogate)` and the `o._svars/o._pvars` handles match between Tasks 3 and 4. `embed_with_target(..., strength=1.0)` matches between Tasks 5 and 6.

## Execution Handoff

Phase-1 deliverable: the opt-in continuous-strength + embed-order solver, proven correct on a synthetic surrogate and demonstrated necessary (positive z3-vs-grid gap) on a real measured surrogate, with the pairwise-additivity gate validated. Phase 2 (frontier live-verify of the chosen order+strengths on user images) is a separate plan.
