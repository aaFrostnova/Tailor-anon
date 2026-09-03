# Design Spec — Continuous Per-Fragment Strength + Embed Order for the Watermark Solver

Date: 2026-08-27
Status: awaiting user review (do not implement until approved)

## 1. Motivation

The current solver maps user constraints to a fragment configuration over a space of ~70 discrete
configs (3 fragments × 4 front-end combos × VINE-only 4-level alpha). An independent adversarial review
verified that on this space the z3/SMT solver returns the identical optimum as brute-force enumeration on
111/111 cells — i.e. **SMT contributes nothing an enumeration loop does not**, because the space is tiny.
The measured 4.66 dB advantage over fixed rules comes from joint subset/front-end/alpha optimization, not
from the SMT machinery.

To make SMT genuinely necessary — and to deliver two axes the method has promised as outputs (per-fragment
strength and embed order) — we expand the configuration space to one that enumeration cannot traverse,
**at the fixed set of 3 real fragments (no synthetic padding):** every fragment gets its own continuous
native strength knob, and the embed order is a decision variable. Continuous strength alone makes the space
un-enumerable — a grid only approximates the optimum, and matching the exact optimum to within ε costs
`g^k` grid points (k = |subset|); embed order multiplies the discrete part by `k!`. The necessity rests on
continuous strength + order, not on growing the fragment count.

## 2. Goal & staging

Staged (user-approved):

- **Phase 1 — surrogate-backed solver + SMT-necessity demonstration.** Fit cheap piecewise-linear (PWL)
  surrogates from sparse real measurement, encode them into z3, let z3 solve the mixed
  real-strength / boolean-order / boolean-subset optimization exactly. Deliverable: the extended solver and
  the "z3 solves where enumeration times out, and finds a strictly better optimum" result.
- **Phase 2 — live validation.** Extend the existing frontier live-verification to embed in the chosen
  order at the chosen native strengths on the user's own image and certify the z3 optimum (top-k). The
  surrogate is only a ranker/predictor; live measurement is the ground truth, exactly as in the current
  two-sided frontier architecture.

## 3. Configuration space

Decision variables per request:

- `u_f ∈ {0,1}` — select fragment f (exists).
- `s_f ∈ ℝ`, gated by `u_f` (`¬u_f → s_f = 0`) — **native, continuous** strength of fragment f, each on
  its **own knob and range** (Section 4.1). NEW.
- `p_{fg} ∈ {0,1}` — precedence boolean: f is embedded before g. Defined for selected pairs. NEW.
- resync / nested front-end booleans (exist).

Why un-enumerable: `s_f` is continuous (LRA-exact only via a solver); the order booleans couple with the
strength reals in the feasibility constraints, so a discretizing enumeration costs `grid^{|subset|} · |subset|!`
per subset.

## 4. Surrogate model

Two empirical surrogates, both fit from sparse real measurement on the image pool and stored in an
immutable, source-stamped `surrogate_table.json` (same discipline as `baseline_table.json`).

### 4.1 Native strength knobs (each fragment its own)

Confirmed in-repo:

- **VINE** — `scale_resid(cover, watermarked, α)` = `cover + α·(watermarked − cover)` (post-hoc residual
  scaling; this is VINE's de-facto native knob in this codebase). Range to sweep: α ∈ [0.2, 1.0].
- **TrustMark** — library-native `TrustMark.encode(..., WM_STRENGTH=1.0)` (internally ×1.25, scales the
  residual before merge). Range to sweep: WM_STRENGTH ∈ [0.4, 1.6].
- **VideoSeal** — library-native `scaling_w` (message scaling). Range to be confirmed in the videoseal env;
  sweep across its supported band.

Each `s_f` is a Real with its own `[s_min^f, s_max^f]`. Ranges are not aligned across fragments — the
distortion surrogate (4.3) is what makes strengths comparable (a fragment "pays" PSNR on its own curve).

### 4.2 Bit-accuracy surrogate (feasibility)

For each fragment f, attack a:

    ba_f(config, a) ≈ base_f(s_f, a)  −  Σ_{g : embedded after f}  δ_{g→f}(s_g, a)

- `base_f(s, a)` — fragment f alone at strength s, best-path bit-accuracy under attack a. **Monotone PWL**
  in s (robustness increases with strength — validated for the swept families in prior work).
- `δ_{g→f}(s_g, a)` — the drop in f's bit-accuracy caused by embedding g **after** f, as a function of g's
  strength. Pairwise (2nd-order). Higher-order interactions assumed negligible and **validated** (4.4).

Feasibility per attack: `ba_f(config, a) ≥ θ(f_FPR)` for every selected f, where θ is the exact
binomial-tail threshold from the requested FPR (existing).

### 4.3 Distortion / PSNR surrogate (objective) — empirical composite

Per user decision, composite fidelity is **measured on real composites**, not assumed additive-in-power.
To stay z3-solvable (LRA cannot represent a general 2-D nonlinear surface), we fit it in the same
`per-fragment + pairwise-correction` form as the bit-acc model:

    D(subset, s-vector) ≈  Σ_f d_f(s_f)  +  Σ_{f<g} u_f u_g · e_{fg}(s_f, s_g)

- `d_f(s_f)` — distortion contributed by f alone at strength s_f (measured cover-vs-single-mark). Monotone PWL.
- `e_{fg}(·)` — the **measured** composite correction when f and g co-occur (real two-fragment composite
  distortion minus `d_f + d_g`). Kept z3-linear by fitting it as a 1-D PWL in `s_f + s_g` (a constant is the
  degenerate case); this captures composite super/sub-additivity without a bilinear term.

Reported PSNR = `monotone(D)`. Order's effect on PSNR is expected small (total residual energy is
near order-invariant); measured at one canonical order and validated (4.4). If order-on-PSNR proves
non-negligible, `e_{fg}` is measured per ordered pair (still z3-linear).

### 4.4 Validation of the pairwise assumption (the gate)

For a small set of full orders over the 3 fragments, measure the **true** composite `ba_f` and composite
distortion, and compare to the surrogate `base − Σδ` and `Σd + Σe`. Require the residual (3rd-order term)
below a tolerance (target: |Δba| ≤ 0.02, |ΔPSNR| ≤ 0.3 dB). If exceeded, either measure triples for the
(small) 3-fragment case or flag the offending pair as non-additive; the Phase-2 live-verify is the ultimate
safety net regardless.

## 5. Measurement campaign (Phase 1 data)

On the standard image pool (N per cell, best-path decode), all intermediate data under `/scratch`:

1. **Strength curves** `base_f(s,a)` and `d_f(s_f)`: 3 fragments × 5–8 native strength levels × attack
   suite. (VINE already has 4 levels; TrustMark/VideoSeal are new sweeps.)
2. **Pairwise bit-acc interference** `δ_{g→f}(s_g,a)`: 6 ordered pairs × a few s_g levels × attack suite.
3. **Pairwise distortion correction** `e_{fg}`: 3 unordered pairs × strength grid (composite PSNR).
4. **Additivity validation**: a handful of full 3-orders, measured composite ba + PSNR (Section 4.4).

Output: `surrogate_table.json` — PWL knots for every `base_f`, `δ_{g→f}`, `d_f`, `e_{fg}`, each cell
source-stamped {value, n, measured_at, git, native_knob, range}.

## 6. z3 model (`watermark_smt_v2.build()` extension)

New optional args: `enable_order=False`, `continuous_strength=False`, `surrogate=None`. Defaults off →
**bytewise-identical behavior to the current 70-config solver** (backward-compat, enforced by a regression
test).

When enabled:

- **Strength vars**: `s_f = Real`, `Implies(u_f, s_min^f ≤ s_f ≤ s_max^f)`, `Implies(Not u_f, s_f == 0)`.
- **Order vars**: `p_{fg} = Bool` for each fragment pair. For selected pairs: `p_{fg} + p_{gf} == 1`;
  transitivity `Implies(And(p_{fg}, p_{gh}), p_{fh})`; unselected fragments drop out of the order.
- **PWL encoding**: each `base_f(·,a)`, `δ_{g→f}(·,a)`, `d_f(·)`, `e_{fg}(·)` encoded by the standard
  convex-combination (λ) formulation → stays in linear real arithmetic.
- **Feasibility**: for each selected f and attack a:
  `base_f(s_f,a) − Σ_g ite(And(p_{fg}, u_g), δ_{g→f}(s_g,a), 0) ≥ θ(f_FPR)`.
- **Capacity / runtime**: existing `2^{C−f} ≥ req_id`; `Σ time + FE time ≤ max_ms` (order/strength do not
  materially change latency; FE latencies from the table).
- **Objective (unchanged lexicographic)**: `min Σ u_f` → `max PSNR = min D` (Section 4.3) → `min time`.

## 7. Frontier live-verify extension (Phase 2)

`OursComposite` embed path and `live_measure.measure()` extended to accept an **order** π and a
**per-fragment native strength** dict. Live-verify embeds in order π at `{s_f}`, applies the attack,
best-path decodes each fragment, checks `ba_f ≥ θ`, and measures real PSNR. z3's top-k candidates are
live-checked: SAT if any verifies; measurement-backed UNSAT over the frontier if all fail. Adversarial /
high-variance attacks remain force-live per the existing variance gate.

## 8. SMT-necessity experiment (the money result)

Fixed at the 3 real fragments (no synthetic padding). Compare **z3.Optimize** vs **brute-force grid
enumeration** along:

- (i) strength-grid density g per fragment → the continuous limit (the core driver);
- (ii) embed order off → on (× `k!` on the selected subset).

Per point record: z3 solve time + exact-optimum PSNR; grid-enumeration time + best-grid-feasible PSNR at
resolution g (grid size ≈ (# discrete structures: subsets × valid front-ends × orders) · `g^{|subset|}`).

Expected result (honest, and it does NOT rely on config count): z3 returns one exact continuous optimum in
~constant time; grid enumeration only approaches it from below, the gap closing as `O(1/g)` while its cost
grows as `g^{|subset|}`. Deliverables: (a) the "z3 exact optimum − best affordable grid" PSNR gap
(strictly positive — the continuous-optimization win), and (b) a solve-time / grid-size vs g curve showing
enumeration cost exploding as `g^k` while z3 stays flat. Turning order on shows the `k!` discrete multiplier
on top. The claim is precise: at 3 fragments SMT is necessary because the strength axis is continuous and
coupled to a combinatorial order — not because there are many fragments.

## 9. Artifacts & interfaces

- `surrogate_table.json` — new immutable baseline artifact; request-scoped overlay for live refinements
  (mirrors the baseline-table immutability rule; user data never mutates the shared surrogate).
- `watermark_smt_v2.build(...)` — new optional args, backward-compatible; regression test pins the
  disabled-path equivalence.
- Fragment wrappers — thread the native strength knob through embed:
  `VineCryptoWrapper` (expose α / scale_resid), `TrustMarkFragment.embed_with_target(..., strength=WM_STRENGTH)`,
  `VideoSealFragment.embed_with_target(..., strength=scaling_w)`.
- `live_measure.measure(...)` / `OursComposite` embed — accept order + per-fragment strength.
- New scripts (all writing intermediate data to `/scratch`): `make_surrogate.py` (campaign + fit),
  `smt_necessity_experiment.py` (Section 8).

## 10. Assumptions & risks

- **Pairwise additivity** (bit-acc and distortion): validated at the 3-fragment scale (4.4); live-verify
  backstops any surrogate error. If it fails badly, fall back to measured triples for k=3.
- **PWL fidelity**: z3's optimum is validated live; if live disagrees beyond tolerance, add a knot and
  re-solve (bounded CEGAR-style refinement) — the only place Approach B's measure-in-the-loop is used.
- **Native-knob ranges** differ per fragment and per library version; ranges are recorded in the surrogate
  origin record and must be re-measured if a fragment model changes.
- **Monotonicity** of `base_f`/`d_f` in s: assumed and checked at fit time; a non-monotone curve is kept as
  a general PWL (z3 handles it; only the "minimal feasible strength" intuition weakens).

## 11. Testing

- **Unit** — PWL encoding matches surrogate interpolation at knots and midpoints (z3 eval vs numpy);
  ordering constraints admit exactly the strict total orders on a selected subset; feasibility constraint
  matches a hand-computed composite ba on a 2-fragment toy.
- **Regression** — with order/continuous disabled, `build()` reproduces the current 70-config solver on the
  289-cell benchmark (0 config changes).
- **Integration** — on held-out images, z3's chosen (order, strengths) live-verified ba/PSNR match the
  surrogate prediction within tolerance (surrogate fidelity).
- **Necessity** — the Section 8 experiment reproduces the strictly-positive z3-minus-best-grid PSNR gap and
  the `g^k` enumeration-cost growth against z3's near-flat solve time (order on/off showing the `k!` factor).

## 12. Phasing & deliverables

- **Phase 1**: wrapper strength knobs → `make_surrogate.py` campaign + fit → `surrogate_table.json` →
  `build()` extension → `smt_necessity_experiment.py`. Deliverable: extended solver + necessity plot/table +
  additivity-validation numbers.
- **Phase 2**: `live_measure` / `OursComposite` order+strength embed → frontier live-verify of z3 top-k →
  surrogate-fidelity report. Deliverable: end-to-end deployed-solver validation on user images.
