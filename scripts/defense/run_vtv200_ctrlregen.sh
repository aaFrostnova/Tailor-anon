#!/bin/bash
# N=200 CtrlRegen+ for the 3-frag composite (VINE+TM+VideoSeal) + full fusion breakdown.
# embed 200 (fingerprint) -> CtrlRegen+ s0.3/0.5/0.7 one model load (ctrlregen) -> decode breakdown.
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PYTHONPATH=$REPO
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
GPU=${1:-0}; N=200
EMB=results/defense/ext_vtv200

# 1) embed 200 (skip if cached)
if [ -f "$EMB/meta.json" ] && [ "$(ls $EMB/*.png 2>/dev/null | wc -l)" -ge "$N" ]; then
  echo "## $EMB cached ($(ls $EMB/*.png|wc -l)), skip embed"
else
  echo "## embed $EMB (vine trustmark videoseal) N=$N"
  CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/composite_external_eval.py --mode embed \
    --fragments vine trustmark videoseal --n_images $N --embed_dir $EMB 2>&1 \
    | grep -aE "embedded|EMBED_DONE|Error|Traceback" | tail -3
fi

# 2) CtrlRegen+ all 3 strengths, one model load -> ${EMB}_ctrlregen_s03/_s05/_s07
echo "## CtrlRegen+ s0.3/0.5/0.7 on $N imgs (one model load)"
CUDA_VISIBLE_DEVICES=$GPU $CRPY scripts/attack/ctrlregen_batch.py \
  --in_dir $EMB --out_dir ${EMB}_ctrlregen --step_list 0.3,0.5,0.7 2>&1 \
  | grep -aE "\[ctrlregen\]|CTRLREGEN_BATCH_DONE|Error|Traceback" | tail -6

# 3) decode breakdown per strength
for TAG in 03 05 07; do
  ADIR=${EMB}_ctrlregen_s$TAG
  nout=$(ls $ADIR/*.png 2>/dev/null | wc -l); echo "## decode s$TAG  attacked=$nout"
  CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/decode_breakdown.py \
    --embed_dir $EMB --attacked_dir $ADIR --attack_name ctrlregen200_s$TAG \
    --output results/defense/ctrlregen200_s$TAG.json 2>&1 | grep -aE "\]|BREAKDOWN_DONE|Error|Traceback" | tail -2
done
echo "CR200_DONE"
