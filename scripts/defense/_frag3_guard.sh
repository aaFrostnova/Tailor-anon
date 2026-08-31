#!/bin/bash
# Patient GPU guard: wait for a STABLE >=15GB free window on the shared GPU, then run
# the 3-frag vs 2-frag eval. Retries on OOM. Logs to results/defense/_frag3_guard.log.
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/defense/_frag3_guard.log
RUN=results/defense/_frag3_run.log
chk(){ CUDA_VISIBLE_DEVICES=0 $PY -c "import torch;print(int(torch.cuda.mem_get_info()[0]/1e9))" 2>/dev/null; }
echo "guard start $(date +%H:%M:%S)" > $LOG
for i in $(seq 1 360); do
  F1=$(chk); echo "[guard $i] free=${F1:-0}GB @ $(date +%H:%M:%S)" >> $LOG
  if [ "${F1:-0}" -ge 15 ]; then
    sleep 5; F2=$(chk)
    if [ "${F2:-0}" -ge 15 ]; then
      echo "[guard] stable window ${F1}/${F2}GB -> run frag3" >> $LOG
      CUDA_VISIBLE_DEVICES=0 $PY scripts/defense/frag3_eval.py 40 > $RUN 2>&1
      if grep -qa FRAG3_DONE $RUN; then echo "[guard] SUCCESS $(date +%H:%M:%S)" >> $LOG; exit 0; fi
      echo "[guard] frag3 failed, retry" >> $LOG
    fi
  fi
  sleep 40
done
echo "[guard] TIMEOUT" >> $LOG
