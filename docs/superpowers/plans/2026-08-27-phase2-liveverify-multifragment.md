# Phase 2 Implementation Plan — Live-Verify + Real Multi-Fragment

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make live-verification embed through each fragment's native strength knob so it actually validates z3's continuous config, then use a focused regen measurement to produce a real scenario where z3's optimum is multi-fragment with a definite embed order.

**Architecture:** `live_measure._embed` switches from post-hoc `scale_resid` to native `embed_with_target(strength=)`; a thin adapter feeds z3's `(frags, order, strengths)` into the measurer; a small regen mini-campaign adds regen columns to the surrogate as an immutable overlay; the `{regen, rot9}` scenario forces VINE+VideoSeal.

**Tech Stack:** Python (conda `fingerprint`; `ctrlregen` env for regen), z3, pytest, the Phase-1 surrogate/solver.

**Spec:** docs/superpowers/specs/2026-08-27-phase2-liveverify-multifragment-design.md

## Global Constraints
- Continue on branch feat/continuous-strength-order. Intermediate data + campaign/experiment scripts in `$SC` = /scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k; method code + tests in `$CF` = /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint. Note: `live_measure.py` currently lives in `$SC`.
- Native strength knobs (Phase 1): VINE residual-scale, TrustMark `WM_STRENGTH`, VideoSeal `blender.scaling_w`, all via `embed_with_target(pil, target, strength=<float>)`.
- The Phase-1 base `surrogate_table.json` is immutable; regen data is added as a separate `regen_overlay` block / file.
- Phase-1's 79 tests must keep passing.
- GPU via sbatch; regen via the `ctrlregen` env. `$PYFP`=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python. HF token <HF_TOKEN>.

---

### Task 1: Native-knob embed in the live path

**Files:** Modify `$SC/live_measure.py` (`_embed`, `measure`); Test `$CF/tests/test_live_embed_strength.py`.

**Interfaces:**
- Produces: `LiveMeasurer._embed(cover, frags, nested, strengths, order)` embedding each fragment via
  `FR[f].embed_with_target(img, t, strength=strengths.get(f, DEFAULT_S[f]))` (native knob); `measure(...,
  strengths=None, order=None)` unchanged signature but `strengths` is now the unified per-fragment map used
  for ALL fragments (no VINE-uses-alpha special-case).

- [ ] **Step 1: Write the failing unit test** (mock fragments so it runs CPU-only)

```python
# $CF/tests/test_live_embed_strength.py
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")

class _FakeFrag:
    def __init__(self, name): self.name=name; self.calls=[]
    def embed_with_target(self, pil, t, strength=1.0):
        self.calls.append(("emb", strength)); return pil   # identity; record the strength it was given

def test_embed_uses_native_strength_and_order(monkeypatch=None):
    import live_measure as LM
    lm = LM.LiveMeasurer.__new__(LM.LiveMeasurer)         # bypass __init__ (no GPU)
    lm.NB = 8
    lm.FR = {"VINE": _FakeFrag("VINE"), "VideoSeal": _FakeFrag("VideoSeal")}
    lm.scale_resid = lambda a,b,s: b
    lm.nested_vine_embed = lambda *a, **k: a[1]
    cover = Image.fromarray((np.ones((16,16,3))*100).astype(np.uint8))
    emb, tgts = lm._embed(cover, frags=["VINE","VideoSeal"], nested=False,
                          strengths={"VINE":0.4,"VideoSeal":1.3}, order=["VideoSeal","VINE"])
    # order respected: VideoSeal embedded before VINE
    assert lm.FR["VideoSeal"].calls[0][0]=="emb" and lm.FR["VINE"].calls[0][0]=="emb"
    # each fragment got its OWN native strength passed to embed_with_target
    assert abs(lm.FR["VINE"].calls[0][1]-0.4)<1e-9
    assert abs(lm.FR["VideoSeal"].calls[0][1]-1.3)<1e-9
```

- [ ] **Step 2: Run it, verify it fails**

Run: `$PYFP -m pytest /work/.../tests/test_live_embed_strength.py -v` → FAIL (current `_embed` calls `embed_with_target(img,t)` with no strength, then scale_resid).

- [ ] **Step 3: Rewrite `_embed`** in `$SC/live_measure.py`:

```python
    DEFAULT_S = {"VINE": 1.0, "TrustMark": 1.0, "VideoSeal": 1.0}
    def _embed(self, cover, frags, nested, strengths, order):
        """Embed the exact config through each fragment's NATIVE strength knob, in `order`."""
        img = cover; tgts = {}; strengths = strengths or {}
        for f in (order or list(frags)):
            if f not in frags: continue
            t = np.random.RandomState(abs(hash(f)) % 2**31).randint(0, 2, self.NB).astype(np.uint8)
            tgts[f] = t
            s = float(strengths.get(f, self.DEFAULT_S.get(f, 1.0)))
            if f == "VINE" and nested:
                img = self.nested_vine_embed(self.FR["VINE"], img, t, scales=(1.0,0.75,0.5), strength=s)
            else:
                img = self.FR[f].embed_with_target(img, t, strength=s)
            if img.size != cover.size: img = img.resize(cover.size)
        return img, tgts
```
Update `measure` to call `self._embed(cover, frags, nested, strengths, order)` (drop the `alpha` positional
into `strengths` if a caller still passes `alpha` — keep `measure(..., alpha=0.7, strengths=None, order=None)`
signature but fold `alpha` into VINE's strength when `strengths` lacks VINE: `strengths = {**({'VINE':alpha}
if 'VINE' not in (strengths or {}) else {}), **(strengths or {})}`). Ensure `nested_vine_embed` accepts a
`strength=` kwarg (thread it to the VINE embed; if it currently doesn't, add it).

- [ ] **Step 4: Run the unit test + the Phase-1 suite**

Run: `$PYFP -m pytest /work/.../tests/test_live_embed_strength.py /work/.../tests/ -v` → new test passes; 79 Phase-1 tests still pass.

- [ ] **Step 5: Commit** `git -C $CF add tests/test_live_embed_strength.py` (live_measure.py is in $SC, not git — note in commit body). Commit message: "feat(live): embed through native per-fragment strength knobs + unified strengths map".

---

### Task 2: z3→live adapter + surrogate-vs-reality fidelity check

**Files:** Create `$SC/live_fidelity.py`; Test `$CF/tests/test_solved_config_adapter.py`.

**Interfaces:**
- Consumes: `build(..., surrogate=...)` solved model (`o._svars`, `o._pvars`, `u`), `LiveMeasurer.measure`.
- Produces: `solved_config(opt, u, order_on) -> (frags, order, strengths)` — reads the model into a config.

- [ ] **Step 1: Write the failing adapter test** (uses synthetic_surrogate, CPU z3 only)

```python
# $CF/tests/test_solved_config_adapter.py
import sys, z3
sys.path.insert(0, "/work/.../scripts/defense"); sys.path.insert(0, "/scratch/.../wm_dataset10k")
from surrogate_model import synthetic_surrogate
import watermark_smt_v2 as W
from live_fidelity import solved_config

def test_solved_config_reads_frags_order_strengths():
    sg = synthetic_surrogate(attacks=("jpeg25","crop75"))
    o,u,rs,ns,al,nf,ps,tm = W.build(min_psnr=0.0,max_ms=1e9,attacks=["jpeg25","crop75"],min_ba=0.60,
        allow_resync=True,allow_nested=True,min_bits=0,resolution=512,
        enable_order=True,continuous_strength=True,surrogate=sg)
    o.maximize(ps); assert o.check()==z3.sat
    frags, order, strengths = solved_config(o, u, order_on=True)
    assert set(frags) <= {"VINE","TrustMark","VideoSeal"} and len(frags)>=1
    assert set(order)==set(frags)                            # a total order over the selected set
    for f in frags:
        lo,hi = sg.range(f); assert lo-1e-6 <= strengths[f] <= hi+1e-6
```

- [ ] **Step 2: Run it → fails** (`live_fidelity` missing).

- [ ] **Step 3: Implement `solved_config`** in `$SC/live_fidelity.py`:

```python
import z3
def solved_config(opt, u, order_on=True):
    m = opt.model()
    import watermark_smt_v2 as W
    frags = [f for f in W.FR if str(m.eval(u[f]))=="True"]
    sv = getattr(opt, "_svars", {}); pv = getattr(opt, "_pvars", {})
    strengths = {f: float(m.eval(sv[f]).as_fraction()) for f in frags if f in sv}
    if order_on and pv:
        # order = topological by precedence booleans p[(f,g)] = f before g
        before = {f: sum(1 for g in frags if g!=f and str(m.eval(pv[(f,g)]))=="True") for f in frags}
        order = sorted(frags, key=lambda f: -before[f])     # more "before" edges => earlier
    else:
        order = list(frags)
    return tuple(frags), order, strengths
```

- [ ] **Step 4: Run test → passes.**

- [ ] **Step 5: Add the fidelity harness** (bottom of `live_fidelity.py`, run under sbatch): load the REAL
`surrogate_table.json`, pick an in-process scenario (e.g. attacks=["jpeg25","crop75","rot9"], min_ba=0.72),
solve, `solved_config`, then `LiveMeasurer(n_images=24).measure(frags, a, strengths=strengths, order=order)`
for each attack; for each fragment/attack print `pred = surrogate.base(f,a).eval(strengths[f]) - Σδ` vs
`live per_fragment[f]`, and `pred_psnr = psnr_from_mse(Σd+Σe)` vs `live psnr`. Emit `live_fidelity.json` with
the max |Δba| and |ΔPSNR|.

- [ ] **Step 6: Run it via sbatch** (model env on `$SC/frontier.sbatch`); confirm it produces
`live_fidelity.json` and report the surrogate-vs-reality gaps. A large gap on TrustMark/VideoSeal would be a
finding (but native-knob embed should agree BETTER than the old scale_resid path).

- [ ] **Step 7: Commit** the repo test (`git -C $CF add tests/test_solved_config_adapter.py`); note
`live_fidelity.py` + `live_fidelity.json` live in $SC.

---

### Task 3: Focused regen mini-campaign → regen overlay

**Files:** Create `$SC/make_regen_overlay.py` + `.sbatch`; Output `$SC/surrogate_regen_overlay.json`.

**Interfaces:** Consumes the strength-enabled wrappers + `ctrlregen` regen application (reuse
`$SC/regen_offline_table.py` / `regen_table.sbatch` for how regen is applied cross-env). Produces
`surrogate_regen_overlay.json`: `base_VINE(s, "regen")` over VINE's 5-knot grid; `base_TrustMark("regen")`,
`base_VideoSeal("regen")` at default strength (expected ~chance); pairwise `δ_{VINE↔VideoSeal}` under regen
and rot9 at reference strengths. Source-stamped, N per cell.

- [ ] **Step 1** Read `$SC/regen_offline_table.py` and `regen_table.sbatch` to reuse the exact regen
application (ctrlregen env, batch flow, how a watermarked image is regenerated).
- [ ] **Step 2** Write `make_regen_overlay.py`: embed VINE alone at each strength → regen → decode
best-path bit-acc; embed TrustMark/VideoSeal alone at default → regen → decode (expect ~0.5); embed
VINE+VideoSeal (both orders) at reference strengths → {regen, rot9} → per-fragment bit-acc → δ. Fit PWLs;
dump `surrogate_regen_overlay.json` (base cells unchanged elsewhere).
- [ ] **Step 3** Smoke at N=3 via `sbatch` (ctrlregen env), confirm structure + that pixel marks read
~chance under regen and VINE survives.
- [ ] **Step 4** Full run at N=50 (regen is expensive — 50 is enough to anchor the scenario); report job id,
let the controller monitor.
- [ ] **Step 5** Validate: `base_VINE(s,"regen")` should be high (VINE survives), `base_{TM,VS}("regen")`
near chance, `base_VideoSeal(s,"rot9")` high (from Phase-1 base, cross-check), `base_VINE(s,"rot9")` low.
This is the evidence that {regen, rot9} forces stacking. No repo commit (scratch data).

---

### Task 4: The {regen, rot9} multi-fragment demonstration

**Files:** Modify `$SC/smt_necessity_experiment.py` (add `--scenario regen_rot9`); create a small
`$SC/multifrag_demo.py`; Output `$SC/multifrag_demo.json`.

**Interfaces:** Consumes the Phase-1 `surrogate_table.json` merged with `surrogate_regen_overlay.json`
(a loader that overlays regen columns), the solver, and the necessity experiment.

- [ ] **Step 1** Write a loader that returns a `Surrogate` whose attacks include `regen` (base/δ overlaid
from the regen file; everything else from the Phase-1 table). Assert the merged surrogate covers
{regen, rot9} for VINE and VideoSeal.
- [ ] **Step 2** `multifrag_demo.py`: solve `build(attacks=["regen","rot9"], min_ba=0.72, enable_order=True,
continuous_strength=True, surrogate=merged, ...)`. ASSERT z3 selects **both** VINE and VideoSeal (not a
single fragment) and returns a definite order + strengths. If a single fragment sneaks through, raise
min_ba until stacking is forced, and record the θ used.
- [ ] **Step 3** Show ORDER matters: evaluate the surrogate feasibility/PSNR of the chosen order vs the
reversed order; report the difference (feasible-only-in-one-order, or a PSNR delta).
- [ ] **Step 4** Run the necessity experiment on this scenario (`run_necessity` with the merged surrogate):
grid must now enumerate subsets × ORDERS × strength grid; report z3 vs grid gap + grid sizes (the k! factor
is now load-bearing, not just cost) + z3 time.
- [ ] **Step 5** Dump `multifrag_demo.json` (the chosen config, order-matters delta, necessity table). No
repo commit. This closes the Phase-1 honest boundary: a real multi-fragment + order optimum.

## Self-Review
- Spec (1) live-verify → Tasks 1,2. Spec (2)(A) regen multi-fragment → Tasks 3,4. ✓
- Placeholder scan: the only unknowns are runtime (regen throughput → N=50 fallback stated) and the exact θ
  that forces stacking (Task 4 Step 2 raises it until 2 fragments, records it). No TBDs.
- Type consistency: `solved_config(opt,u,order_on)->(frags,order,strengths)` used identically in Tasks 2,4;
  `_embed(...,strengths,order)` and `measure(...,strengths,order)` consistent Tasks 1,2.
