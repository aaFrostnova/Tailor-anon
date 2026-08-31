#!/bin/bash
# Full eval of the FINAL config (VINE+DFT+QIM+TrustMark-B, all fused) on GPU 0, sequential:
#   1) FPR  2) image quality  3) CtrlRegen+ (final config)  4) n=200 detection (full suite)
set -e
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export CUDA_VISIBLE_DEVICES=0
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
FRAGS="vine dft qim trustmark"

echo "########## STEP 1/4: FPR (n=200 clean / 50 regen / 50 wrong-id) ##########"
$FPPY scripts/defense/composite_fpr.py --fragments $FRAGS --n_clean 200 --n_regen 50 --n_wrong 50 \
  --output results/defense/composite_fpr_4fused.json 2>&1 | grep -E "\[FPR\]|FPR_DONE|Error|Traceback"

echo "########## STEP 2/4: image quality (n=20) ##########"
$FPPY scripts/defense/composite_quality.py --fragments $FRAGS --n_images 20 \
  --output results/defense/composite_quality_4fused.json 2>&1 | grep -E "stage|vine|dft|qim|trust|PSNR|QUALITY_DONE|Error|Traceback"

echo "########## STEP 3/4: CtrlRegen+ on final config ##########"
EMB=results/defense/ext_4fused_coco
$FPPY scripts/defense/composite_external_eval.py --mode embed --fragments $FRAGS --n_images 50 \
  --embed_dir $EMB 2>&1 | grep -E "embedded|EMBED_DONE|Error|Traceback"
for STEP in 0.3 0.5 0.7; do
  TAG=$(echo $STEP | tr -d '.')
  ADIR=${EMB}_ctrlregen_$TAG
  echo "--- CtrlRegen step=$STEP ---"
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $EMB --out_dir $ADIR --step $STEP 2>&1 | grep -E "CTRLREGEN_BATCH_DONE|Error|Traceback" | tail -2
  $FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FRAGS \
    --embed_dir $EMB --attacked_dir $ADIR --attack_name ctrlregen_$STEP \
    --output results/defense/composite_4fused_ctrlregen_$TAG.json 2>&1 | grep -E "composite vs|COMPOSITE_or="
done

echo "########## STEP 4/4: n=200 detection, full 18-attack suite ##########"
$FPPY scripts/defense/benchmark_composite_defense.py --fragments $FRAGS --tm_variant B --n_images 200 \
  --attacks clean jpeg blur noise bright contrast bm3d regen rinse2x rinse4x vae_b vae_c rs256 hflip crop75 crop50 rot9 crop_jpeg \
  --output results/defense/composite_4fused_n200.json 2>&1 | grep -E "^attack|^clean|^jpeg|^blur|^noise|^bright|^contrast|^bm3d|^regen|^rinse|^vae|^rs256|^hflip|^crop|^rot|\[done\]"

echo "FINAL_EVAL_ALL_DONE"
