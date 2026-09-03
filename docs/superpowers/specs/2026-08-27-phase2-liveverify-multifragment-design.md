# Design Spec — Phase 2: Live-Verify Integration + Real Multi-Fragment Scenario

Date: 2026-08-27
Status: awaiting user review (do not implement until approved)
Branch: feat/continuous-strength-order (continues Phase 1)
Phase-1 spec: 2026-08-27-continuous-strength-embed-order-solver-design.md

## 1. Motivation

Phase 1 built a solver over continuous per-fragment strength + embed order, validated on a real measured
surrogate: z3 finds the exact continuous optimum that grid enumeration only approaches (4.04 dB gap at g=4,
grid 202,848 configs with order at g=32). Two gaps remain, which Phase 2 closes:

- **Live-verify consistency.** The surrogate and z3 use each fragment's NATIVE strength knob (VINE
  residual-scale, TrustMark `WM_STRENGTH`, VideoSeal `blender.scaling_w`). But `live_measure._embed`
  applies strength as a POST-HOC `scale_resid` on every fragment — which equals the native knob for VINE
  but NOT for TrustMark/VideoSeal. So a live check does not currently validate the exact config the
  surrogate predicted. Phase 2 makes the live path embed through the native knobs, so live-verify actually
  confirms z3's chosen (order, strengths).
- **Order/subset necessity on real data.** The Phase-1 real scenario's optimum was single-fragment because
  the in-process attack family has no regeneration to force VINE-stacking. Phase 2 adds a focused regen
  measurement so a real scenario forces a multi-fragment optimum where order matters.

## 2. Scope (two pieces, user-approved: (1) + (2)(A))

- **(1) Live-verify integration.** Make the live embed path use native strength knobs and a unified
  per-fragment strengths map; wire z3's continuous output into it; confirm z3 top-k configs on real images.
- **(2)(A) Real multi-fragment via a FOCUSED regen mini-campaign.** Measure only what is needed to force a
  real multi-fragment+order optimum — not a full regen × all-strengths × all-pairs campaign.

## 3. Piece (1): Live-verify integration

### 3.1 Native-knob embed in the live path
`live_measure._embed` currently does `img = scale_resid(img, FR[f].embed_with_target(img, t), s)`. Change to
embed through each fragment's native knob: `img = FR[f].embed_with_target(img, t, strength=s)` (Phase-1
Task-5 param), matching exactly how the surrogate measured `base_f(s,a)` and `d_f(s)`. VINE keeps nested-ring
support. The per-fragment strength comes from a unified `strengths: {fragment -> float}` map (drop the
VINE-uses-`alpha`, others-use-`strengths` special-case; the z3 model already exposes `o._svars[f]` for all
selected fragments).

### 3.2 Consuming the z3 continuous output
z3 returns `(subset, order π, strengths {s_f}, resync, nested)`. A thin adapter reads `o._svars`/`o._pvars`
from a solved `build(..., surrogate=...)` model into `(frags, order, strengths)` and calls
`LiveMeasurer.measure(frags, attack, strengths=strengths, order=order)`; the per-fragment best-path
bit-accuracy is checked against the FPR threshold θ, and real PSNR is recorded.

### 3.3 Frontier live-verify of the continuous config
For a z3-solved continuous config on a scenario's in-process attacks, live-verify it (and the top-k ranked
candidates) on the user's images: SAT if any verifies every attack at θ; otherwise measurement-backed. This
reuses the Phase-1 two-sided frontier structure; the only change is that candidates now carry
(order, strengths) and the measurer embeds through native knobs. Batch-only attacks (regen) still return
None from the in-process measurer and stay table-driven (Section 4).

### 3.4 Fidelity check (the payoff)
On held-out images, compare the live-measured per-fragment bit-accuracy and PSNR of z3's chosen config to
the surrogate's prediction. Report the gap. This quantifies the surrogate's real-world fidelity end to end
(the Phase-1 additivity gate checked internal consistency; this checks surrogate-vs-reality).

## 4. Piece (2)(A): Real multi-fragment scenario via a focused regen mini-campaign

### 4.1 The scenario that forces stacking
Attack set **{regen, rot9}**:
- `regen` (diffusion regeneration) destroys the pixel-space marks (TrustMark, VideoSeal) but VINE survives.
- `rot9` natively defeats VINE (geometry-weak) while VideoSeal is rotation-robust.
So no single fragment clears both — the solver must select **VINE + VideoSeal**, and the embed ORDER
between them affects mutual interference. This yields a real multi-fragment + order optimum.

### 4.2 What to measure (minimal)
Reuse the existing regen infrastructure (`regen_offline_table.py`, `ctrlregen` env, `regen_table.sbatch`).
Measure only:
- `base_VINE(s, regen)` over VINE's 5-knot strength grid (does VINE's regen survival depend on strength?).
- `base_{TrustMark,VideoSeal}(regen)` at their default strength — expected near chance (confirm they die).
- `base_VideoSeal(s, rot9)` and `base_VINE(s, rot9)` — already in the Phase-1 surrogate (rot9 is in-process);
  reuse, no re-measure.
- Pairwise `δ` between VINE and VideoSeal under {regen, rot9} at reference strengths (the interference that
  makes order matter) — a handful of measurements, not the full pair grid.
Merge these into a scenario-scoped surrogate overlay (regen columns added to the Phase-1
`surrogate_table.json` as an immutable extension; the base table is not mutated, per the immutability rule).

### 4.3 The demonstration
Solve `build(..., attacks=["regen","rot9"], surrogate=extended, ...)`: expect z3 to return VINE+VideoSeal
with a definite order and continuous strengths. Then:
- show the order MATTERS: the opposite order (or a fixed-order rule) is either infeasible or higher-distortion;
- run the necessity experiment on this scenario: z3 vs grid, now with a genuine 2-fragment + order optimum,
  so the grid must also enumerate orders (the k! factor is load-bearing, not just cost).
- live-verify the chosen config on real images for the in-process part (rot9); the regen part stays
  table-backed (batch-only), with a small cross-env live spot-check if practical.

## 5. Artifacts & interfaces

- `live_measure.py` — `_embed` uses native `embed_with_target(strength=)`; `measure`/`_embed` take a unified
  `strengths` map. Backward-compatible default (strengths omitted → prior per-fragment default).
- A small adapter (in the experiment layer, scratch) `solved_config(opt) -> (frags, order, strengths)`.
- `regen_offline_table.py` (reused, extended if needed) → regen columns; `surrogate_table.json` gains an
  immutable `regen_overlay` block (base cells unchanged).
- The necessity experiment gains a `--scenario regen_rot9` entry point.

## 6. Assumptions & risks

- **Live-vs-surrogate gap may be non-trivial** for TrustMark/VideoSeal now that the live path uses native
  knobs (it should IMPROVE agreement vs the old scale_resid mismatch). If a fragment's live bit-accuracy
  diverges from the surrogate beyond tolerance, that is a real finding: refine the surrogate knots (the
  Phase-1 CEGAR-style loop) — this is exactly what live-verify is for.
- **Regen is batch-only and cross-env** (ctrlregen). The mini-campaign is sized to a few hundred embeds, not
  the full matrix; if the ctrlregen throughput is too low, fall back to VINE-only regen at 3 strengths plus a
  chance-level assertion for the pixel marks (enough to force the scenario).
- **The scenario must genuinely force 2 fragments** — verify that neither VINE alone nor VideoSeal alone
  clears {regen, rot9} at the chosen θ before claiming a multi-fragment necessity; if VideoSeal's rot9 margin
  is large enough that a single fragment sneaks through, raise θ or add a third conflicting attack.

## 7. Testing

- **Unit** — the adapter maps a solved z3 model to (frags, order, strengths) correctly; `_embed` respects
  order and applies per-fragment native strength (embedding at strength=1.0 stays identical to prior for
  VINE, and now uses WM_STRENGTH/scaling_w for TM/VideoSeal).
- **Integration** — for a Phase-1 in-process scenario, live-verify of z3's config agrees with the surrogate
  within tolerance on held-out images (fidelity check).
- **Necessity (regen scenario)** — z3 returns a 2-fragment + order optimum; the grid enumerator (now over
  subsets × orders × strength grid) is worse and explodes; z3 stays flat.
- **Regression** — Phase-1 tests still pass (79) with the live_measure change (default path unchanged).

## 8. Phasing

- **2a**: live_measure native-knob embed + adapter + fidelity check on an in-process scenario. Deliverable:
  live-verify that actually validates z3's continuous config, + a surrogate-vs-reality fidelity number.
- **2b**: regen mini-campaign + the {regen, rot9} multi-fragment demonstration. Deliverable: a real scenario
  where z3 picks VINE+VideoSeal in a definite order, closing the Phase-1 honest boundary.
