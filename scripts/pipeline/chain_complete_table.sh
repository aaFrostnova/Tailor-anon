#!/bin/bash
# Wait for the four table-completion arrays, then fold everything into the canonical table, audit it for
# completeness, run the tests, and (only then) release the held five-class re-solve.
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
PY=${WM_PY:-python}
export PYTHONPATH="$CF:$CF/scripts/defense:$SC" TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
until ! squeue -u mingzhel_umass_edu -h -o "%j" | grep -qE "^(delta_sw|delta_xv|fe_xenv|pi_regen)$"; do sleep 300; done
until [ -f $SC/local_worker.done ]; do sleep 120; done          # the local A100's slice (local_worker.sh)
echo "local worker: $(tail -1 $SC/logs/local_worker.out)"
echo "=== campaigns finished $(date) ==="
for n in delta_sw delta_xv fe_xenv pi_regen; do
  echo "$n: $(sacct -u mingzhel_umass_edu --starttime ${CHAIN_SINCE:-2026-09-07T18:00} -X -n -o JobName,State | awk -v n=$n '$1==n' | sort | uniq -c | tr '\n' ' ')"
done
echo "=== fold the front-end cells ==="; $PY $SC/fe_xenv_merge.py 2>&1 | tail -12
echo "=== range extension (ring knots 0.14/0.17) ==="; $PY $SC/merge_range_ext.py 2>&1 | tail -4
echo "=== rebuild (raw, for the per-image comparison) ==="; $PY $SC/make_canonical_surrogate.py 2>&1 | grep -E "merged surrogate_(delta_sweep|fe_xenv)|range extension|wrote|Error|Traceback|SKIPPING|below the campaign" | head -30
echo "=== merge per-image ==="; $PY $SC/merge_perimage.py 2>&1 | tail -16
echo "=== rebuild (final) ==="; $PY $SC/make_canonical_surrogate.py 2>&1 | grep -E "range extension|attached|wrote|attacks\(|OK$|Error|Traceback|SKIPPING|below the campaign" | head -12
echo "=== new interference sweeps vs the constants they replace ==="; $PY $SC/check_delta_sweeps.py 2>&1 | tail -12
echo "=== audit ==="; $PY $SC/audit_table_completeness.py 2>&1 | tail -40
AUD=$?
echo "=== request ceilings from the completed table (class_defs.py reads this) ==="
cp $SC/request_feasibility_matrix.json $SC/request_feasibility_matrix.before_completion.json
$PY $SC/make_request_feasibility.py 2>&1 | tail -22
$PY - <<'PYX'
import json
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
a = json.load(open(f"{SC}/request_feasibility_matrix.before_completion.json")); b = json.load(open(f"{SC}/request_feasibility_matrix.json"))
ch = [k for k in b["need"] if (a["need"].get(k) is None) != (b["need"][k] is None)]
print("reachability changes after completion:", ch or "none")
PYX
echo "=== tests ==="; PYTHONHASHSEED=0 $PY -m pytest $CF/tests/ -q -p no:cacheprovider 2>&1 | tail -3
if [ "$AUD" -eq 0 ] && PYTHONHASHSEED=0 $PY -m pytest $CF/tests/ -q -p no:cacheprovider 2>&1 | tail -1 | grep -q " passed" ; then
  echo "=== table complete and tests green: releasing the five-class re-solve ==="
  scontrol release 64055224
else
  echo "NOT RELEASED: audit rc=$AUD or tests failed"
fi
echo CHAIN_DONE $(date)
