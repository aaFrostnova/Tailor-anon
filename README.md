# TAILOR: request-conditioned watermark composition

Code for the submission. Given a deployment request
`u = (attacks, FPR budget, PSNR floor, latency ceiling)`, TAILOR selects and
certifies a composition of watermark fragments that satisfies all four.

The repository carries the whole chain, from building the image pool to the
exporters that write the reported tables. The measurement artifacts themselves
are too large to distribute, so every stage that produces one is included and
documented below.

## Quickstart

The solver runs from a fresh clone. No GPU, no model checkpoints, and no
configuration: the offline measurements it reads are in `inputs/`.

```bash
pip install z3-solver numpy scipy
python solver/watermark_smt_v2.py --attacks jpeg25 --min_psnr 42 --max_ms 100
```

```
QUERY custom: PSNR>=42.0 ms<=100.0 ba>=0.9 bits>=0 ['jpeg25']
  VINE(alpha=0.3)   PSNR~45.6dB . 57ms
```

One fragment at low strength answers a compression-only request. Ask for more
and the composition grows:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 crop75 regen --min_psnr 38 --max_ms 500
```

```
QUERY custom: PSNR>=38.0 ms<=500.0 ba>=0.9 bits>=0 ['jpeg25', 'crop75', 'regen']
  VINE(alpha=0.7) + TrustMark   PSNR~38.3dB . 70ms
```

Ask for something the library cannot deliver and the answer is a statement about
the library, not a failed search:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 crop75 crop50 rot9 regen rinse --min_psnr 46 --max_ms 200
```

```
  UNSAT - no combination satisfies these conditions
```

## Layout

```
src/           watermark fragments, geometric stages, attacks, image pool, soft decoding
pipeline/      data generation: the measurement campaigns and the fits they feed
solver/        SMT model, request protocol, and the live-calibration loop
measurement/   unified detector, FPR accounting, and the frozen measurement harness
inputs/        the frozen offline measurements the solver reads (16 MB, included)
```

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/solver:$PWD/measurement:$PWD/pipeline
export TAILOR_PROJECT=/your/project        # source tree and fragment checkpoints
export TAILOR_WORKSPACE=/your/workspace    # campaigns, cells, artifacts
export TAILOR_ASSETS=/your/checkpoints     # diffusion, VAE and watermark weights
```

`TAILOR_PROJECT` is read by the solver; `CLEAN_MIN_JSON` and `BA_SD_PROFILE`
override two frozen inputs. Every other path in the measurement scripts is a
placeholder rooted at `/data/tailor` that you edit to your own location. The
watermark fragments and the attack models come from their upstream projects,
which `requirements.txt` lists.

**What runs without the measurement artifacts.** 32 of the 52 modules under
`solver/`, `measurement/` and `pipeline/` import with no data and no GPU, which
covers the solver, the request protocol, the response model and the detector.
The other 20 are campaign scripts: they read the artifacts a run produces, or
they need the watermark packages, so they raise rather than run until you point
them at your own campaign.

## The pipeline

Stages 1 to 3 build the performance database. They are the expensive part and
their outputs for the reported runs are in `inputs/`, so a reader who only wants
to reproduce selection can start at stage 4. The repository holds the method
itself; the exporters that turned the results into the reported tables are not
part of it.

### Stage 1. Image pool

```bash
python pipeline/build_wm_dataset.py --out $TAILOR_WORKSPACE/pool
```

Builds the ~10,000 image pool from five sources (splits A to E), converts every
image to RGB and resizes it to 512x512. `src/image_pool.py` draws from it by a
per-source offset, which is what keeps the offline and live slices disjoint:
offset 0 fits the response curves, offset 300 is the live set every reported
number is measured on.

### Stage 2. Offline measurement

Each script writes one family of measurements. All of them run the real
embed, attack and decode path on the GPU through `pipeline/live_measure.py`.

| Script | Produces |
|---|---|
| `pipeline/perimage_campaign.py` | per-image bit accuracy and keyed verification, per fragment, strength and attack |
| `pipeline/strength_sweep.py` | recovery across attack-strength sweeps (JPEG quality, blur sigma, noise sigma) |
| `pipeline/measure_order_strength.py` | the order and strength dimensions the solver optimizes over |
| `pipeline/make_delta_curves.py` | interference as a function of the overwriting fragment's strength |
| `pipeline/make_frontend_components.py` | each geometric stage measured separately |
| `pipeline/extend_knots_campaign.py` | range extension, so the three fragments share a distortion interval |
| `pipeline/measure_clean_minimum.py` | clean-image strength floors, `inputs/clean_minimum_strength.json` |
| `pipeline/measure_prior_width.py` | how far robustness moves when the images change, `inputs/prior_width.json` |
| `pipeline/make_ba_sd_profile.py` | per-attack image-to-image standard deviation, `inputs/ba_sd_profile.json` |
| `pipeline/embed_at_strength.py` | staging half of a cross-environment run, for attacks that need their own environment |

### Stage 3. Fit and freeze

```bash
python pipeline/fit_surrogate_curves.py       # knots -> fitted response curves
python pipeline/make_canonical_surrogate.py   # -> inputs/surrogate_canonical.json
python pipeline/make_baseline.py              # -> inputs/baseline_table.json
```

`make_canonical_surrogate.py` merges every measured piece into the one table the
solver and the live loop both read. Each curve is stored as the piecewise-linear
interpolant of its measured knots, which is what lets the solver treat strength
as a continuous variable while staying inside linear real arithmetic.

### Stage 4. Requests

```bash
python pipeline/class_scenarios.py             # five threat classes -> inputs/C1..C5.json
python pipeline/make_request_feasibility_v2.py # -> inputs/request_feasibility_matrix.json
```

### Stage 5. Selection and live calibration

`solver/live_topk_full.py` is the loop. It is staged so the GPU work is
separable from the bookkeeping:

```bash
python solver/live_topk_full.py enumerate <class> <n_requests> [k=3]   # solve: top-k candidates per request
python solver/live_topk_full.py measure   <class> [N=100]              # GPU: embed, attack, decode -> cells
python solver/live_topk_full.py walk      <class> [N=100]              # no GPU: verdicts from cached cells
python solver/live_topk_full.py patch     <class> <shard> <nshard>     # refine and re-solve where every candidate failed
python solver/live_topk_full.py verdict   <class>                      # pass@rank and delivered fidelity
```

**Generating the cells.** A cell is one `(configuration, attack)` pair measured
on the image slice: the per-image bit accuracy of each fragment, the fused
readout, the keyed verification flag, and the geometric views, recorded at every
threshold the six FPR budgets imply. `measure` writes one JSON per cell to
`$TAILOR_WORKSPACE/live_topk/cells/<cfg_id>_<attack>_<imgset>.json` and reuses
any cell already there, so the campaign is resumable and the same configuration
measured for one request is free for the next. The reported runs hold **30,333
cells, about 800 MB**, over the five request classes. `measure_rank` is the
sharded form for job arrays; run `walk` between ranks to decide which
configurations still need measuring.

`solver/controller.py` submits and resumes the whole campaign on a scheduler;
`solver/campaign.py` and `solver/prepare.py` handle request preparation and
checkpointing.

### Stage 6. Final measurement and audit

```bash
python measurement/final_measurement.py --help
```

Re-runs the accepted configurations under the frozen protocol and writes the
per-image records the paper reports. `measurement/unified_detector.py` holds the
verification rule every method is scored by: the threshold implied by the
requested FPR budget, the union bound over the configuration's verification
paths, and the aligned readouts. `measurement/rigor_protocol.py` and
`measurement/final_holdout.py` enforce the digest checks that keep the offline
and live slices disjoint.

## What is included, and what is not

`inputs/` ships the frozen outputs of stages 2 to 4, so stages 5 and 6 run
without repeating the measurement campaign:

| File | Contents |
|---|---|
| `surrogate_canonical.json` | the piecewise-linear response curves over fragment strength |
| `baseline_table.json` | per fragment, attack and strength: bit accuracy, distortion, latency |
| `clean_minimum_strength.json` | clean-image strength floors per fragment |
| `ba_sd_profile.json` | per-cell image-to-image standard deviation |
| `prior_width.json` | the prior width used to shrink a live offset |
| `request_feasibility_matrix.json` | precomputed feasibility per request class |
| `C1.json` ... `C5.json` | the evaluated request sets, one per scenario |
| `smt_inputs*.json` | the solo, composite and end-to-end measurements the SMT model reads at import |

Not included, because of size: the image pool, the 30,333 measurement cells
(about 800 MB), and the fragment and attack model checkpoints. Stages 1, 2 and 5
regenerate the first two; the checkpoints come from the upstream projects.

## A note on the fragment library

`src/` also holds alternatives the evaluation measured and did not deploy
(`phasemark.py`, `quant_qim_modules.py`, `dft_kred_modules.py`,
`fusion_head3.py`, `maskwm_wrapper.py`). They are kept because the reported
ablations refer to them.

## License

MIT, see `LICENSE`. The watermark fragments and the attack models are used
through their upstream packages and are not redistributed here; their own
licenses govern those packages.
