# Code map: the deployed method and its evaluation pipeline

> The scratch pipeline scripts referred to below are checked in under `scripts/pipeline/` (paths read from `WM_REPO` / `WM_SCRATCH`, see its README); the measured table and the feasibility matrix are under `data/`.

Everything the current method needs lives in the 30 repository modules below plus the scratch pipeline
scripts. Older experiments, rejected fragments and superseded campaigns were moved (not deleted) to
`/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/attic/` with a `restore.sh`; the manifest is
`code_cleanup_manifest.json` there. Data lives in the scratch directory `$SC`.

## 1. The method (repository)

| module | role |
|---|---|
| `scripts/defense/watermark_smt_v2.py` | the z3 solver: fragments, continuous strengths, embed order, four front-end stages (at most one), replacement curves, capacity derived from bit accuracy (`ba_to_bits` / `bits_to_ba`, the BSC bound of the 100-bit codeword), clean-image floors, beta from the FPR budget, margin, latency, the per-image acceptance floor (`DEFAULT_DET_MIN`, one strength bound per cell), the live gate `live_required_at`, and the certified optimum `solve_exact_model` |
| `scripts/defense/surrogate_model.py` | the measured table the solver reads: PWL curves (base, delta, d, e, cap, frontend, latency) plus the per-image block and `rate_curve(f, a, tau, stage)`; `with_live` applies CEGAR offsets |
| `scripts/defense/live_calibration.py` | `solve_with_live`: solve, measure the returned configuration on the user's images, patch the curve the solve read (gated, shrunk, asymmetric offset), re-solve; must-live columns come back as `pending_live` unless an executor runs them |
| `scripts/defense/eval_matrix.py` | `OursComposite`: the deployed embed/decode (crypto codeword, per-fragment best-path verification, equal-weight soft fusion, presence test, geometric cascade on a primary miss) and the baseline wrappers of the reported matrix |
| `scripts/defense/composite_external_eval.py` | nested-ring VINE embed, scale/rotation helpers used by the cascade |
| `src/vine_crypto_wrapper.py`, `src/trustmark_fragment.py`, `src/videoseal_fragment.py` | the three fragments behind one keyed codeword |
| `src/shortened_bch.py`, `src/soft_bch.py`, `src/soft_fusion.py`, `src/payload.py` | BCH(100,37,t=10), Chase soft decoding + payload verification, MRC fusion, image-id payload |
| `src/syncseal_frontend.py`, `src/tiled_trustmark.py`, `src/angle_probe.py` | the resync, tile and angle stages |
| `src/attacks.py`, `src/image_pool.py`, `src/metrics.py`, `src/transforms.py` | the single attack definition (in-process family, diffusion regen/rinse, the cross-environment dispatch), the 10k five-source pool sampler, metrics |
| `scripts/attack/ctrlregen_batch.py`, `scripts/attack/unmarker_batch.py` | the two attacks that run in their own conda environments |
| `src/dft_kred_modules.py`, `src/quant_qim_modules.py`, `src/phasemark.py`, `src/maskwm_wrapper.py`, `src/learned_fragment_methods.py`, `src/latent_vae.py`, `src/sign_envelope.py`, `src/fusion_head3.py` | imported by the matrix harness for baselines only; not part of the deployed composite |

Tests: `tests/` (run with `PYTHONPATH=$CF:$CF/scripts/defense:$SC python -m pytest tests/`).

## 2. The pipeline (scratch `$SC`, in execution order)

**Table.** `make_surrogate_ext9.py` (base/delta/d/e/cap on the in-process family, 9 knots, N=100) →
`make_geo_overlay.py`, `make_signal_overlay.py`, `make_frontend_overlay.py`, `make_mildregen_overlay_9k.py`
(regen/rinse2x), `make_regen_overlay_ext.py` (CtrlRegen+ per step), `decode_unmarker_overlay.py` +
`embed_at_strength.py` (UnMarker), `unmk_ring_stage.py` / `unmk_ring_decode.py` / `merge_unmk_ring.py` /
`add_ring_det_curves.py` (the scale stage on the adversarial columns), `make_frontend_components.py`
(per-stage replacement curves, control with the cascade suppressed), `make_delta_curves.py` (interference
vs the overwriting strength), `measure_clean_minimum.py`, `variance_profile.py`, `make_ba_sd_profile.py` →
per-image values: `perimage_campaign.py` (plain cells, stage cells, retained attacked images) and
`merge_perimage.py` → range extension: `extend_knots_campaign.py`, `extend_xenv.py`, `merge_range_ext.py` →
**`make_canonical_surrogate.py`** (merges every overlay, splices the extension knots, attaches the per-image
block, then `fit_surrogate_curves.py` and `thin_surrogate_curves.py`) → `surrogate_canonical.json`.

**Classes.** `class_defs.py` (five threat classes and the request sampler), `class_scenarios.py` (solve 2,000
requests per class, four pinned baselines; `solver_eval_continuous.py` holds the solve helpers),
`merge_class_eval.py` (tables), `count_distinct_configs.py`.

**Certification.** `certify_full.py` measures every distinct configuration once on 100 cross-source images
(in-domain: pool offset 300; out-of-domain: `CERT_DOMAIN`/`CERT_IMAGES`), `certify_full_xenv.sbatch` runs
CtrlRegen+ / UnMarker on the kept embeds, `certify_full_verdicts.py` judges every request at its own
threshold (mean and per-image rate). `certify_classes.py` / `certify_classes_xenv.py` are the earlier 20-per-
class sample. `certify_fullpool.py` / `merge_fullpool.py`: one configuration per class on all 10,000 images.

**Live.** `solve_live_continuous.py` (the CEGAR loop on held-out images), `xenv_measure.py` (the
cross-environment executor), `live_xenv_demo.py`.

**Analyses behind paper claims** (`$SC/analysis/`): `fe_exhaustive_probe.py` (UNSAT is not a search
failure), `unmarker_liveonly_probe.py`, `bestpath_vs_fused.py`, `unmk_residual_*.py` / `unmk_proxy*.py`
(UnMarker decomposition), `smt_necessity_experiment.py` / `necessity_sweep.py`, `persona_scenarios_v2.py`,
`frontier_solver.py`, `validate_curve_error.py`, `certify_solver_eval.py`, `live_fidelity.py`.

## 3. Data files (`$SC`)

`surrogate_canonical.json` (what the solver reads), `surrogate_canonical_raw.json` (before fitting),
`surrogate_perimage.json`, `surrogate_range_ext.json`, `ba_sd_profile.json`, `clean_minimum_strength.json`,
`class_eval_<C>_shard*.json` → `class_eval.json`, `certify_full/<C>/g*.json` → `certify_full_<C>.json`,
`fullpool.json`, `pool/` (the 10k images, sources A to E).
