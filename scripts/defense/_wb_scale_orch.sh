#!/bin/bash
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=1
WB=/work/pi_shiqingma_umass_edu/mingzheli/W-Bench; PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
# wait for Wave1 (GPU free) + LOCAL bins downloaded
while true; do
  w1=$(grep -qa WAVE1_DONE /tmp/claude-3529/-home-mingzhel-umass-edu/028437e0-9eb9-4570-b229-0e3d330c7a0f/tasks/bhwa81kfe.output 2>/dev/null && echo 1 || echo 0)
  b1=$([ "$(find $WB/LOCAL_EDITING_5K/10-20/mask -name '*.png' 2>/dev/null|wc -l)" -ge 500 ] && echo 1 || echo 0)
  b2=$([ "$(find $WB/LOCAL_EDITING_5K/50-60/mask -name '*.png' 2>/dev/null|wc -l)" -ge 500 ] && echo 1 || echo 0)
  [ "$w1" = 1 ] && [ "$b1" = 1 ] && [ "$b2" = 1 ] && break
  sleep 30
done
echo "## LOCAL mask-size sweep ($(date +%H:%M))"
for bin in 10-20 50-60; do
  $PY scripts/defense/wbench_eval.py --task local --wb_dir $WB/LOCAL_EDITING_5K/$bin/image --mask_dir $WB/LOCAL_EDITING_5K/$bin/mask --n 500 --strength 0.8 --out results/defense/wbench_local_$bin.json && echo "LOCAL_$bin DONE"
done
echo "## SVD image-to-video (n=100)"
$PY scripts/defense/wbench_eval.py --task svd --wb_dir $WB/DISTORTION_1K/image --n 100 --out results/defense/wbench_svd.json && echo "SVD DONE"
echo ALL_SCALE_DONE
