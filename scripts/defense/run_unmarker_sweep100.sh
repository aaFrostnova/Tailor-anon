#!/bin/bash
# UnMarker pure-spectral STRENGTH SWEEP at n=100 (stage1/stage2 iters = strength axis).
# phased mode saves incrementally during stage2 -> monitor via png count. Runs alone.
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
GPU=${1:-0}; EMB=results/defense/ext_vtv100
for SP in "200 100" "300 150" "400 200"; do
  read S1 S2 <<< "$SP"
  ADIR=${EMB}_unmarker_s${S1}_${S2}
  echo "## UnMarker n=100 s1=$S1 s2=$S2 -> $ADIR  ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$GPU $UMPY scripts/attack/unmarker_batch.py --in_dir $EMB --out_dir $ADIR \
    --n 100 --no_preprocess --max_iter_stage1 $S1 --max_iter_stage2 $S2 > ${ADIR}.attacklog 2>&1
  echo "ATTACK_DONE s${S1}_${S2} pngs=$(ls $ADIR/*.png 2>/dev/null|wc -l)"
  CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/decode_breakdown.py --embed_dir $EMB --attacked_dir $ADIR \
    --attack_name unmarker100_s${S1}_${S2} --output results/defense/unmarker100_s${S1}_${S2}.json 2>&1 | grep -aE "\]|BREAKDOWN_DONE|Error"
done
echo UM_SWEEP100_DONE
