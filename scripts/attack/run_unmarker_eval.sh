#!/bin/bash
# UnMarker eval of the FINAL 4-fused composite: attack (unmarker env, full Vine.yaml config)
# then decode (fingerprint env). Reuses the embedded images from the CtrlRegen step.
set -e
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
EMB=${1:-results/defense/ext_4fused_coco}
N=${2:-10}
S1=${3:-2000}     # stage1 (high-freq) iters; paper Vine.yaml = 2000
S2=${4:-500}      # stage2 (low-freq) iters; paper Vine.yaml = 500
ADIR=${EMB}_unmarker
FRAGS="vine dft qim trustmark"

echo "=== UnMarker attack (n=$N, stage1=$S1 stage2=$S2) -> $ADIR ==="
CUDA_VISIBLE_DEVICES=0 $UMPY scripts/attack/unmarker_batch.py \
  --in_dir $EMB --out_dir $ADIR --n $N --max_iter_stage1 $S1 --max_iter_stage2 $S2 2>&1 \
  | grep -E "\[unmarker\]|\[[0-9]+/|UNMARKER_BATCH_DONE|Error|Traceback" | tail -20

echo "=== decode composite vs UnMarker ==="
CUDA_VISIBLE_DEVICES=0 $FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FRAGS \
  --embed_dir $EMB --attacked_dir $ADIR --attack_name unmarker \
  --output results/defense/composite_4fused_unmarker.json 2>&1 | grep -E "composite vs|COMPOSITE_or="
echo "UNMARKER_EVAL_DONE"
