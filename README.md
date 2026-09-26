# TAILOR: request-conditioned watermark composition

Code accompanying the submission. Given a deployment request
`u = (attacks, FPR budget, PSNR floor, latency ceiling)`, TAILOR selects and
certifies a composition of watermark fragments that satisfies all four.

## Layout

```
solver/        SMT model, request protocol, and the live-calibration loop
measurement/   unified detector, FPR accounting, and the measurement harness
fragments/     watermark fragments, attacks, image pool, and soft decoding
paper/         exporters that build the tables and figures from frozen results
inputs/        frozen offline measurements the solver reads
```

### solver/

| File | Role |
|---|---|
| `watermark_smt_v2.py` | the SMT encoding: coverage, FPR threshold, PSNR floor as an MSE ceiling, latency, and the optimization loop |
| `capacity_protocol.py` | the four-input request, per-image acceptance, and the candidate walk |
| `surrogate_model.py` | piecewise-linear response curves, the live offset, and request-scoped refinement |
| `live_topk_full.py` | live calibration: measure, compare, patch the request-local model, re-solve |
| `live_calibration.py` | the discrepancy measurement a failed candidate produces |
| `certify_full_frozen.py` | certification of a configuration against a frozen measurement set |
| `watermark_smt_topk.py` | enumeration of further candidates by blocking returned structures |
| `campaign.py`, `controller.py`, `prepare.py`, `prefetch.py`, `local_gpu.py` | campaign orchestration |
| `eval_matrix.py`, `composite_external_eval.py`, `solver_eval_continuous.py` | evaluation entry points |

### measurement/

`unified_detector.py` holds the verification rule every method is scored by:
the per-configuration threshold from the requested FPR budget, the union bound
over verification paths, and the aligned readouts. `final_measurement.py` runs
the frozen campaign; `rigor_protocol.py` and `final_holdout.py` enforce the
digest checks that keep the offline and live image slices disjoint. The three
`test_*.py` files check the detector, the selection adapter, and campaign
startup.

### fragments/

The watermark fragments (`vine_crypto_wrapper.py`, `trustmark_fragment.py`,
`videoseal_fragment.py`), the geometric stages (`syncseal_frontend.py`,
`tiled_trustmark.py`, `angle_probe.py`), the attack suite (`attacks.py`), the
image pool with its per-source offsets (`image_pool.py`), and soft decoding
(`soft_fusion.py`, `soft_bch.py`, `shortened_bch.py`).

Several modules here are alternatives that the evaluation measured and did not
deploy (`phasemark.py`, `quant_qim_modules.py`, `dft_kred_modules.py`,
`fusion_head3.py`, `maskwm_wrapper.py`); they are kept because the reported
ablations refer to them.

## Paths

Every path is a placeholder rooted at `/data/tailor`. Set the three environment
variables in `paths.py` to your own locations:

```bash
export TAILOR_PROJECT=/your/project
export TAILOR_WORKSPACE=/your/workspace
export TAILOR_ASSETS=/your/model/checkpoints
```

## Frozen inputs

`inputs/` carries the offline measurements the solver reads, so the selection
stage runs without re-running the measurement campaign:

| File | Contents |
|---|---|
| `surrogate_canonical.json` | the piecewise-linear response curves over fragment strength |
| `baseline_table.json` | per fragment, attack and strength: bit accuracy, distortion, latency |
| `clean_minimum_strength.json` | clean-image strength floors per fragment |
| `ba_sd_profile.json` | per-cell image-to-image standard deviation |
| `prior_width.json` | the prior width used to shrink a live offset |
| `request_feasibility_matrix.json` | precomputed feasibility per request class |
| `C1.json` ... `C5.json` | the evaluated request sets, one per scenario |

Not included, because of size: the per-image measurement cells (about 800 MB),
the image pool, and the fragment and attack model checkpoints. The cells are
reproduced by `measurement/final_measurement.py`; the checkpoints come from the
upstream watermark and diffusion projects listed in `requirements.txt`.

## Reproducing the reported numbers

1. Install `requirements.txt` and the upstream watermark packages.
2. Point the three roots at your copies.
3. Selection only, from the frozen inputs:
   `python solver/watermark_smt_topk.py --help`
4. Live calibration on your own images, which re-measures every candidate:
   `python solver/live_topk_full.py --help`
5. Tables and figures from frozen results: the `paper/export_*.py` and
   `paper/plot_*.py` scripts each read one committed result file and write one
   LaTeX table or figure.
