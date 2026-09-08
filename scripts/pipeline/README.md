# The evaluation pipeline

The solver and the deployed composite live in `scripts/defense/` and `src/` (see `CODE_MAP.md`). This
directory holds the campaign, evaluation and orchestration scripts around them, in the form they were last
run. Every script reads its locations from the environment and falls back to the paths of the reported run:

| variable | meaning | default |
|---|---|---|
| `WM_REPO` | this repository | the checkout the script lives in |
| `WM_SCRATCH` | working directory for data and outputs (`pool/`, shards, certification cells, logs) | `/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k` |
| `WM_PIPELINE` | directory the SLURM scripts source `_sbatch_header.inc` from | `$WM_SCRATCH` |
| `WM_TABLE`, `FEAS_MATRIX` | the offline table and the feasibility matrix | `$WM_SCRATCH/...` then `data/` |
| `WM_PY` | python of the `fingerprint` environment | `python` |
| `WM_PY_CTRLREGEN`, `WM_PY_UNMARKER` | pythons of the two attack environments (`scripts/attack/setup_*.sh`) | `python` |
| `HF_TOKEN`, `HF_HOME` | Hugging Face token and cache for the fragment models | unset / `$HF_HOME` |

SLURM scripts write their logs to `logs/` relative to the submission directory: submit them from
`$WM_SCRATCH` (`mkdir -p logs` first). `PYTHONPATH` must contain `$WM_REPO`, `$WM_REPO/scripts/defense`
and this directory (the SLURM scripts set it).

## 1. The offline table

Measured on the 10k five-source pool (`src/image_pool.py`; build it with `scripts/defense/build_wm_dataset.py`
if present in your checkout, otherwise any directory of 512 x 512 images per source), 100 images per cell.

1. Base curves, interference, distortion, capacity on the in-process attack family:
   `make_surrogate_ext9.sbatch` -> `make_surrogate_ext9.py` (9 knots per fragment).
2. Overlays that re-measure families with the shared operators: `geo_overlay.sbatch`, `signal_overlay.sbatch`,
   `frontend_overlay.sbatch` (`make_*_overlay.py`), the diffusion columns `make_mildregen9.sbatch`
   (`make_mildregen_overlay_9k.py`: regen / rinse2x), CtrlRegen+ per step `make_regen_overlay_ext.sbatch`
   and `make_ctrlregen*.sbatch` (`make_regen_overlay_ext.py`, runs in the `ctrlregen` environment), UnMarker
   `decode_unmarker_overlay.py` + `embed_at_strength.py` (the `unmarker` environment; see
   `scripts/attack/`).
3. Front-end stages: `fe_components.sbatch`, `fe_components_nongeo.sbatch` (`make_frontend_components.py`,
   one replacement curve per stage with the cascade suppressed as control), the ring on the adversarial
   columns `unmk_ring.sbatch` (`unmk_ring_stage.py`, `unmk_ring_decode.py`, `merge_unmk_ring.py`,
   `add_ring_det_curves.py`), and the cross-environment stage cells `fe_xenv.sbatch`
   (`fe_xenv_campaign.py`, `fe_xenv_merge.py`).
4. Interference over the overwriting fragment's strength: `delta_sweep.sbatch` (`make_delta_curves.py`,
   `DELTA_CELLS_FILE`), `delta_xenv.sbatch` (`make_delta_xenv.py`); consistency check `check_delta_sweeps.py`.
5. Per-image values: `perimage_campaign.sbatch`, `perimage_stage.sbatch`, `perimage_stage_regen.sbatch`
   (`perimage_campaign.py` plain / stage modes) -> `merge_perimage.py`.
6. Range extension: `extend_knots.sbatch` (`extend_knots_campaign.py`), `extend_xenv.sbatch` (`extend_xenv.py`)
   -> `merge_range_ext.py`. Clean-image floors and variance: `clean_minimum.sbatch`, `variance.sbatch`.
7. Assemble: `make_canonical_surrogate.py` (merges every overlay, splices the extension knots, attaches the
   per-image block; calls `fit_surrogate_curves.py` and `thin_surrogate_curves.py`) ->
   `surrogate_canonical.json`; then `audit_table_completeness.py` must print `AUDIT_OK`, and
   `make_ba_sd_profile.py` writes the live gate's standard deviations. `chain_complete_table.sh` runs the
   last completion campaign end to end; `smoke_complete.sbatch` is its 3-image smoke test.

UnMarker is a live-only column: the table keeps only a search prior for it, the audit and the solver's
admissibility rules exempt it, and every request that names it is measured live.

## 2. Requests and solving

- `make_request_feasibility.py` -> `request_feasibility_matrix.json`: per column, fragment count, budget and
  payload, the cheapest single cell that clears the solver's conditions (mean over threshold plus margin,
  per-image acceptance at the floor plus its margin, capacity line plus margin). Regenerate whenever the
  table or the solver's margins change (`FEAS_V` selects earlier rule sets).
- `class_defs.py`: the five threat classes and the request sampler (budget, payload, fidelity floor and
  latency drawn conditioned on the hardest column; a 6 percent stress minority beyond the ceiling;
  `SAMPLER_V` selects earlier rule sets).
- `class_scenarios.py <class 0-4> <shard> <n_shards> [N]`: solves every request of a shard to the certified
  optimum and poses it to the four pinned baselines (`solver_eval_continuous.py` holds the helpers).
  `class_scen40_gpu.sbatch` runs 40 shards per class as one array (`--array=0-199%40`, C5 first);
  `local_worker.sh <class> <workers>` solves shards of one class on the local node (finished shards are
  skipped by either side).
- `merge_class_eval.py [--tex path]` -> `class_eval.json` and the class table (satisfaction, delivered PSNR,
  baselines and regret, UNSAT attribution, stress split, configuration mix).

## 3. Certification

- `certify_full.py measure <class> <shard> <n_shards> [N]` embeds every distinct configuration of a class on
  the held-out images (in domain: pool offset 300; `CERT_DOMAIN=<name> CERT_IMAGES=<dir>` for an
  out-of-domain set built by `build_ood_sets.py`), applies every attack its requests named, and records two
  views per image: the primary read and, when no keyed test passes there, the crypto-verify-gated cascade
  view. `certify_full.sbatch <class> <shards> <N>` is the array; `certify_full_xenv.sbatch` runs CtrlRegen+
  and UnMarker on the kept embeds and decodes them (`certify_full.py decode_xenv`).
- `certify_full_check.py <class> [domain]` reports unmeasured cells (the chain re-submits a class with any).
- `certify_full_verdicts.py <class> [det_min]` judges every request at its own threshold (mean criterion,
  per-image rate criterion, fidelity floor, capacity line) with a bootstrap over images;
  `summarize_certification.py` collects the five classes and three domains into one table.
- `certify_fullpool.py` / `merge_fullpool.py`: one configuration per class on the whole pool.
- `chain_after_classes.sh` runs everything after the class arrays (merge, 30 measure arrays over five
  classes and three domains split between two partitions per `cert_preempt_shards.txt`, a retry round,
  the cross-environment chains, verdicts, summary); `orchestrate.sbatch` runs it as a SLURM job;
  `chain_resume.sh` / `orchestrate_resume.sbatch` resume after the measure arrays; `status_snapshot.sh`
  writes a one-screen status to `logs/STATUS.txt`.

## 4. Live loop

`solve_live_continuous.py` drives `scripts/defense/live_calibration.py` (solve, measure the returned
configuration on the user's images, patch the curve the solve read, re-solve); `xenv_measure.py` is the
cross-environment executor; `live_xenv_demo.py` is the end-to-end demonstration.

## 5. Analyses behind specific claims

`null_presence_union.py` (the joint null of the k+1 presence tests), `ablate_presence_split.py` and
`measure_ablation_live.py` (the budget split ablation, table and live), `count_binding.py` (how many
solutions sit on a floor), `sampler_diff.py` / `apply_sampler_v3.py` / `resolve_affected.py` (sampler and
solver revisions applied to solved shards without re-drawing), `bestpath_vs_fused.py`,
`unmarker_liveonly_probe.py`, `probe_vine_ctrlregen.py`, `count_distinct_configs.py`.

## Environments

`environment.yaml` / `requirements.txt` describe the `fingerprint` environment (torch, z3-solver, diffusers,
the VINE, TrustMark, VideoSeal and SyncSeal checkouts under `external/`, see `CODE_MAP.md`). CtrlRegen+ and
UnMarker run in their own environments: `scripts/attack/setup_ctrlregen.sh`, `scripts/attack/setup_unmarker.sh`.

## Known limitations of the reported runs

- The interference table (`delta`, 120 cells) is measured on the plain embeds only: it has no front-end
  dimension. A two-view re-measurement of the ring with a partner fragment (crop50, VINE 0.93 on the
  cascade view against 0.92 in the table) shows no error from this today, but a stage that changes the
  embed is not re-measured against every partner.
- UnMarker's table cells are a search prior measured on 30 images; the column is judged live only.
- Every curve is measured at 512 x 512; the solver refuses other resolutions until a table exists for them.
- The deployed decoder's presence test runs at a fixed 1 percent budget (`presence_detected`); the request's
  budget enters the solver and the verdicts (which re-threshold the recorded reads per request), not the
  decoder's own verdict, and the fused zero-bit test costs every two-fragment configuration one bit of
  threshold while adding almost no detections (the fused keyed verification is what helps, on 1 to 3 percent
  of images on the diffusion columns).
- The per-image acceptance floor and the capacity line carry margins (0.96 in the table for a 0.90
  deployed floor, +0.02 on the capacity line) from the margin run on; the archived no-margin run shows
  what happens without them (live rates 0.80 to 0.89 where the table said 0.90).
