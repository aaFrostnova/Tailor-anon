#!/bin/bash
# After the five-class re-solve: merge (with UNSAT attribution and the stress split), certify every distinct
# configuration in domain and on the two out-of-domain sets, run the cross-environment chains, judge every
# request at its own line with bootstrap intervals over the images.
# v3 (2026-09-08 20:00 UTC): 40 shards per class (200 files). v2 (2026-09-08 15:00 UTC): 16 measure shards per class and domain instead of 8 (the 8-hour limit was a
# risk for C5); the shard range in cert_preempt_shards.txt (read at submission time, e.g. "8-15") goes to
# gpu-preempt on modern cards with --requeue (the measure is resumable per attack cell), the rest to
# superpod-a100 whose QoS allows 16 GPUs; after the arrays, any class/domain with unmeasured in-process
# cells is re-measured once on superpod before the chain moves on. Cross-environment chains stay on superpod.
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
PY=${WM_PY:-python}
export PYTHONPATH="$CF:$CF/scripts/defense:$SC"
SHARDS=16; NC=100; NU_IN=30; NU_OOD=10
PREEMPT_OPTS=(--partition=gpu-preempt --requeue "--constraint=a100|l40s|h100" --exclude=gpu058)
wait_name() { until ! squeue -u mingzhel_umass_edu -h -o "%j" | grep -qE "^($1)$"; do sleep 300; done; }
submit_measure() {   # $1 class index, $2 domain, $3 array spec, then extra sbatch options
  local c=$1 dom=$2 arr=$3; shift 3
  if [ "$dom" = "indomain" ]; then sbatch --parsable "$@" --array=$arr $SC/certify_full.sbatch $c $SHARDS $NC
  else ( export CERT_DOMAIN=$dom CERT_IMAGES=$SC/$dom; sbatch --parsable "$@" --array=$arr $SC/certify_full.sbatch $c $SHARDS $NC ); fi
}
wait_name "classes"
echo "=== classes finished $(date): shards $(ls $SC/class_eval_C*_shard*.json 2>/dev/null | wc -l)/200 ==="
[ "$(ls $SC/class_eval_C*_shard*.json 2>/dev/null | wc -l)" -ge 200 ] || { echo "NOT ALL SHARDS PRESENT, stopping"; exit 1; }
echo "=== merge ==="; $PY $SC/merge_class_eval.py 2>&1 | tail -60
bash $SC/status_snapshot.sh 2>/dev/null
PRE=$(cat $SC/cert_preempt_shards.txt 2>/dev/null | tr -d '[:space:]')
echo "=== certification: in domain + two out-of-domain sets ($SHARDS shards; gpu-preempt shards: ${PRE:-none}) ==="
for c in 0 1 2 3 4; do for dom in indomain ood_content ood_generator; do
  if [ -n "$PRE" ]; then
    lo=${PRE%-*}; hi=${PRE#*-}
    [ "$lo" -gt 0 ] && submit_measure $c $dom 0-$((lo-1))
    submit_measure $c $dom $PRE "${PREEMPT_OPTS[@]}"
    [ "$hi" -lt $((SHARDS-1)) ] && submit_measure $c $dom $((hi+1))-$((SHARDS-1))
  else submit_measure $c $dom 0-$((SHARDS-1)); fi
done; done
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
    [ "$n" -gt 0 ] && ( export CERT_DOMAIN=$dom; sbatch --parsable --array=0-$((n-1)) $SC/certify_full_xenv.sbatch $c $NC $NU )
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
