#!/bin/bash
# Resume of chain_after_classes_v2.sh after the orchestrator (64094355) was cancelled mid-submission on
# 2026-09-08 16:34 UTC: the classes are merged and the C1..C4 certification arrays (64097305..64097329)
# are queued; C5's arrays are NOT submitted here (its sampler fix is pending the user's decision; the retry
# round below measures any class/domain with unmeasured cells, so C5 is certified either way).
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
PY=${WM_PY:-python}
export PYTHONPATH="$CF:$CF/scripts/defense:$SC"
SHARDS=16; NC=100; NU_IN=30; NU_OOD=10
LOCK=$SC/chain_resume.lock
if [ -f $LOCK ] && squeue -h -j "$(cat $LOCK)" 2>/dev/null | grep -q .; then echo "another resume ($(cat $LOCK)) is alive; exiting"; exit 0; fi
echo "${SLURM_JOB_ID:-local}" > $LOCK
wait_name() { until ! squeue -u mingzhel_umass_edu -h -o "%j" | grep -qE "^($1)$"; do sleep 300; done; }
submit_measure() {   # $1 class index, $2 domain, $3 array spec, then extra sbatch options
  local c=$1 dom=$2 arr=$3; shift 3
  if [ "$dom" = "indomain" ]; then sbatch --parsable --comment="cert C$((c+1)) $dom" "$@" --array=$arr $SC/certify_full.sbatch $c $SHARDS $NC
  else ( export CERT_DOMAIN=$dom CERT_IMAGES=$SC/$dom; sbatch --parsable --comment="cert C$((c+1)) $dom" "$@" --array=$arr $SC/certify_full.sbatch $c $SHARDS $NC ); fi
}
echo "=== resume $(date) (job ${SLURM_JOB_ID:-local}) ==="
sleep 60; wait_name "cert_full"
echo "=== measure arrays finished $(date); checking for unmeasured cells ==="
for dom in indomain ood_content ood_generator; do for c in 0 1 2 3 4; do
  line=$($PY $SC/certify_full_check.py $c $dom 2>/dev/null | tail -1); n=$(echo "$line" | grep -oE "MISSING [0-9]+" | awk '{print $2}')
  echo "  $line"
  [ "${n:-1}" -gt 0 ] && submit_measure $c $dom 0-$((SHARDS-1))
done; done
sleep 60; wait_name "cert_full"
echo "=== in-process certification finished $(date) ==="
for dom in indomain ood_content ood_generator; do for c in 0 1 2 3 4; do echo "  $($PY $SC/certify_full_check.py $c $dom 2>/dev/null | tail -1)"; done; done
bash $SC/status_snapshot.sh 2>/dev/null
for dom in indomain ood_content ood_generator; do
  DIR=$SC/certify_full; [ "$dom" != "indomain" ] && DIR=$SC/certify_full_$dom
  NU=$NU_IN; [ "$dom" != "indomain" ] && NU=$NU_OOD
  for c in 0 1 2 3 4; do
    K=$(echo C1 C2 C3 C4 C5 | cut -d' ' -f$((c+1)))
    n=$($PY -c "import json,glob; print(sum(1 for p in glob.glob('$DIR/$K/g*.json') if json.load(open(p)).get('xenv_needed')))")
    echo "  $dom $K: $n groups need the cross-environment chain (NC=$NC NU=$NU)"
    [ "$n" -gt 0 ] && ( export CERT_DOMAIN=$dom; sbatch --parsable --comment="xenv $K $dom" --array=0-$((n-1)) $SC/certify_full_xenv.sbatch $c $NC $NU )
  done
done
sleep 60; wait_name "cf_xenv"
echo "=== cross-environment chains finished $(date) ==="
for dom in indomain ood_content ood_generator; do
  for c in 0 1 2 3 4; do
    echo "--- class $c [$dom] ---"; CERT_DOMAIN=$dom $PY $SC/certify_full_verdicts.py $c 2>&1 | grep -vE "Warning|warn" | head -14
  done
done
[ -f $SC/summarize_certification.py ] && { echo "=== summary ==="; $PY $SC/summarize_certification.py 2>&1 | grep -vE "Warning|warn" | tail -60; }
bash $SC/status_snapshot.sh 2>/dev/null
echo CHAIN_CLASSES_DONE $(date)
