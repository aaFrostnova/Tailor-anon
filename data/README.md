# data/

The measured offline table and its derived files, as used for the reported runs.

| file | what it is |
|---|---|
| `surrogate_canonical.json` | the offline table the solver reads: per-fragment bit-accuracy curves per attack column (`base`), overwrite interference (`delta`), distortion (`d`, `e`), front-end replacement curves and costs (`frontend`), stage latencies (`latency`), and the per-image block (`perimage`); built by `scripts/pipeline/make_canonical_surrogate.py` |
| `request_feasibility_matrix.json` | what the table can reach per column, budget, fragment count and payload (`scripts/pipeline/make_request_feasibility.py`); the request sampler's ceilings |
| `ba_sd_profile.json` | per-attack image-to-image standard deviation of bit accuracy, read by the live gate (`live_required_at`) |
| `surrogate_range_ext.json` | the range-extension knots spliced into the table by the builder |

Every pipeline script looks for these files in `$WM_SCRATCH` first and falls back to this directory
(`WM_TABLE` / `FEAS_MATRIX` override the table and matrix paths explicitly).
