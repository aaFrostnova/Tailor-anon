#!/bin/bash
# Cross-env UnMarker (pure-spectral, no-crop) for VINE+TM vs VINE+TM+VideoSeal.
# embed cached -> UnMarker attack (unmarker env) -> decode (fingerprint). Full paths.
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: do NOT export PYTHONPATH=$REPO — it shadows ai-watermark's own `src` package
# (ModuleNotFoundError: src.arguments). unmarker_batch sets its own path via chdir.
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
GPU=${1:-0}; N=12
VTM=results/defense/ext_vtm; VTV=results/defense/ext_vtv

for SPEC in "vine trustmark|$VTM|vtm" "vine trustmark videoseal|$VTV|vtv"; do
  IFS='|' read -r FRAGS EMB CFG <<< "$SPEC"
  ADIR=${EMB}_unmarker
  echo "## UnMarker (pure-spectral) cfg=$CFG  in=$EMB out=$ADIR"
  CUDA_VISIBLE_DEVICES=$GPU $UMPY scripts/attack/unmarker_batch.py --in_dir $EMB --out_dir $ADIR \
    --n $N --no_preprocess --max_iter_stage1 300 --max_iter_stage2 150 2>&1 \
    | grep -aE "\[unmarker\]|stage2 \[$N|UNMARKER_BATCH_DONE|OutOfMemory|Traceback" | tail -3
  nout=$(ls $ADIR/*.png 2>/dev/null | wc -l); echo "   attacked images: $nout"
  CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FRAGS \
    --embed_dir $EMB --attacked_dir $ADIR --attack_name unmarker_$CFG \
    --output results/defense/unmarker_${CFG}.json 2>&1 | grep -aE "COMPOSITE_or=|Error|Traceback" | tail -2
done
echo "UM3FUSED_DONE"
