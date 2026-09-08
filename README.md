# Cryptographic composite watermark with an SMT solver

A keyed, error-corrected watermark carried by three fragments (VINE, TrustMark, VideoSeal), and a z3
solver that turns a deployment's request (attacks to survive, false-positive budget, fidelity floor, latency
budget, payload) into the configuration that meets it at the best fidelity, or proves that none does. The
solver reads a measured offline table of the fragments' behaviour, certifies every configuration it returns
on held-out images, and defers adversarial columns to live measurement.

## Layout

| path | contents |
|---|---|
| `scripts/defense/watermark_smt_v2.py` | the solver (fragments, continuous strengths, embed order, front-end stages, capacity from bit accuracy, per-image floor, budget-split thresholds, live gate) |
| `scripts/defense/surrogate_model.py` | the offline table the solver reads (piecewise-linear curves, per-image block, live offsets) |
| `scripts/defense/live_calibration.py` | the live loop: solve, measure, patch the table, re-solve |
| `scripts/defense/eval_matrix.py`, `scripts/defense/composite_external_eval.py` | the deployed composite: keyed codeword, per-fragment and fused verification, presence tests, geometric cascade; baseline wrappers |
| `src/` | fragments, BCH and soft decoding, fusion, front-end stages, attacks, the image pool |
| `scripts/attack/` | CtrlRegen+ and UnMarker in their own environments |
| `scripts/pipeline/` | the campaign, evaluation and orchestration scripts (table building, request sampling, class solves, certification, live loop, analyses); see its README |
| `data/` | the measured table, the feasibility matrix and the live gate's variance profile |
| `tests/` | the solver's behavioural tests (`PYTHONPATH=.:scripts/defense python -m pytest tests/`) |
| `docs/` | design notes and plans |
| `CODE_MAP.md`, `PIPELINE_REPORT_CN.md`, `WATERMARK_SMT_README.md` | module map, pipeline report, solver notes |

## Setup

```bash
conda env create -f environment.yaml        # the fingerprint environment (torch, z3-solver, diffusers, ...)
conda activate fingerprint
# fragment models and front-ends: VINE, TrustMark, VideoSeal, SyncSeal checkouts under external/ (see CODE_MAP.md)
export HF_TOKEN=...                          # for the gated model downloads
bash scripts/attack/setup_ctrlregen.sh       # optional: the two cross-environment attacks
bash scripts/attack/setup_unmarker.sh
```

## Using the solver

```python
import sys; sys.path[:0] = [".", "scripts/defense"]
import json, watermark_smt_v2 as W
from surrogate_model import Surrogate
sg = Surrogate.from_dict(json.load(open("data/surrogate_canonical.json")))
scen = dict(min_psnr=36.0, max_ms=4000.0, attacks=["jpeg25", "blur", "regen", "rot9"],
            min_ba=W.beta_from_fpr(1e-4), allow_resync=True, allow_nested=True, min_bits=37, resolution=512)
built, model, rounds, certified = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
# built is None  ->  UNSAT (no configuration meets the request)
```

`scripts/pipeline/solver_eval_continuous.py` (`solve_free`) wraps this call and returns the configuration
(fragments, embed order, strengths, front-end stages, delivered PSNR); `scripts/defense/live_calibration.py`
(`solve_with_live`) adds the live measurement loop.

## Reproducing the evaluation

`scripts/pipeline/README.md` walks through the four stages: building the offline table, sampling and
solving the five request classes, certifying every returned configuration on held-out images in domain
and out of domain, and the live loop. All scripts take their locations from `WM_REPO`, `WM_SCRATCH`,
`WM_PY` and related variables; the defaults are the paths of the reported runs.
