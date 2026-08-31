#!/bin/bash
# Cross-env CtrlRegen+ comparison: VINE+TM (2-frag) vs VINE+TM+VideoSeal (3-frag).
# embed (fingerprint) -> CtrlRegen+ attack (ctrlregen env) -> decode (fingerprint).
# NOTE: full paths everywhere (a basename bug earlier ran attacks on 0 images); no set -e
# pipe-masking; embeds are skipped if already present.
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PYTHONPATH=$REPO
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
GPU=${1:-0}
N=24
VTM=results/defense/ext_vtm
VTV=results/defense/ext_vtv

emb () {  # $1=fragments  $2=dir
  if [ -f "$2/meta.json" ] && [ "$(ls $2/*.png 2>/dev/null | wc -l)" -ge "$N" ]; then echo "## $2 cached, skip embed"; return; fi
  echo "## embed $2 ($1)"
  CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/composite_external_eval.py --mode embed \
    --fragments $1 --n_images $N --embed_dir $2 2>&1 | grep -aE "embedded|EMBED_DONE|Error|Traceback" | tail -2
}
emb "vine trustmark"           $VTM
emb "vine trustmark videoseal" $VTV

for STEP in 0.3 0.5 0.7; do
  TAG=$(echo $STEP | tr -d '.')
  for SPEC in "vine trustmark|$VTM|vtm" "vine trustmark videoseal|$VTV|vtv"; do
    IFS='|' read -r FRAGS EMB CFG <<< "$SPEC"
    ADIR=${EMB}_ctrlregen_$TAG
    echo "## CtrlRegen step=$STEP cfg=$CFG  in=$EMB out=$ADIR"
    CUDA_VISIBLE_DEVICES=$GPU $CRPY scripts/attack/ctrlregen_batch.py --in_dir $EMB --out_dir $ADIR --step $STEP 2>&1 \
      | grep -aE "\[ctrlregen\]|CTRLREGEN_BATCH_DONE|Error|Traceback" | tail -2
    nout=$(ls $ADIR/*.png 2>/dev/null | wc -l)
    echo "   attacked images: $nout"
    CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FRAGS \
      --embed_dir $EMB --attacked_dir $ADIR --attack_name ctrlregen_${CFG}_$STEP \
      --output results/defense/ctrlregen_${CFG}_$TAG.json 2>&1 | grep -aE "composite vs|COMPOSITE_or=|Error|Traceback" | tail -2
  done
done
echo "CR3FUSED_DONE"
