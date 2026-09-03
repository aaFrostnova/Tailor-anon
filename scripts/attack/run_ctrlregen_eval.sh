#!/bin/bash
# Full CtrlRegen+ eval of the no-PhaseMark composite: attack (ctrlregen env) at 3 strengths,
# then decode (fingerprint env). Embedded images must already exist in EMBED_DIR.
set -e
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
EMBED_DIR=${1:-results/defense/ext_noPM_coco}
GPU=${2:-5}
FRAGS="vine dft qim"

for STEP in 0.3 0.5 0.7; do
  TAG=$(echo $STEP | tr -d '.')
  ADIR=${EMBED_DIR}_ctrlregen_$TAG
  echo "=== CtrlRegen attack step=$STEP -> $ADIR ==="
  CUDA_VISIBLE_DEVICES=$GPU $CRPY scripts/attack/ctrlregen_batch.py \
    --in_dir $EMBED_DIR --out_dir $ADIR --step $STEP 2>&1 | grep -vE "warn|Warning|weights_only|state_dict|Loading pipeline|^\s*$|Migrating|cache for model" | tail -4
  echo "=== decode composite vs ctrlregen_$TAG ==="
  CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/composite_external_eval.py \
    --mode decode --fragments $FRAGS --embed_dir $EMBED_DIR --attacked_dir $ADIR \
    --attack_name ctrlregen_$STEP \
    --output results/defense/composite_noPM_ctrlregen_$TAG.json 2>&1 | grep -E "composite vs|COMPOSITE_or|DECODE_DONE"
done
echo "CTRLREGEN_EVAL_ALL_DONE"
