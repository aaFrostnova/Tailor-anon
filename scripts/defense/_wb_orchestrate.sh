#!/bin/bash
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
# wait for eval chain (sto+instruct) AND local-mask download
while true; do
  done_edit=$(grep -qa WB_EDIT_DONE results/defense/_wb_edit.log 2>/dev/null && echo 1 || echo 0)
  err_edit=$(grep -qaE "Traceback|Error" results/defense/_wb_edit.log 2>/dev/null && echo 1 || echo 0)
  done_loc=$( { grep -qa LOCDL_DONE results/defense/_locdl.log 2>/dev/null || [ "$(find /work/pi_shiqingma_umass_edu/mingzheli/W-Bench/LOCAL_EDITING_5K/30-40/mask -name '*.png' 2>/dev/null|wc -l)" -ge 500 ]; } && echo 1 || echo 0)
  [ "$done_edit" = 1 ] && [ "$done_loc" = 1 ] && break
  [ "$err_edit" = 1 ] && { echo "EVAL_CHAIN_ERROR"; break; }
  sleep 30
done
if grep -qa WB_EDIT_DONE results/defense/_wb_edit.log 2>/dev/null; then
  echo "## running local task ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=1     /home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python scripts/defense/wbench_eval.py --task local --wb_dir /work/pi_shiqingma_umass_edu/mingzheli/W-Bench/LOCAL_EDITING_5K/30-40/image     --mask_dir /work/pi_shiqingma_umass_edu/mingzheli/W-Bench/LOCAL_EDITING_5K/30-40/mask --n 500 --strength 0.8 > results/defense/_wb_local.log 2>&1
fi
echo "===== ALL W-BENCH RESULTS ====="
for t in distortion sto_regen instruct local; do
  /home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python -c "import json;d=json.load(open('results/defense/wbench_$t.json'));print('$t:',{k:(round(v,3) if isinstance(v,float) else v) for k,v in d.items() if k not in ('rows',)}, d.get('rows',''))" 2>/dev/null || echo "$t: (no json)"
done
echo ALL_WB_DONE
