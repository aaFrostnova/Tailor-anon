#!/bin/bash
set -e
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<200 && $3<10 {print $1; exit}')
export CUDA_VISIBLE_DEVICES=$GPU; echo "[gpu] $GPU"
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
IMG=/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image
COMMON="--fragments vine trustmark videoseal --syncseal --n_images 25 --start_idx 0 --image_dir $IMG --image_glob *.png"

echo "=== PHASE 1: embed 3-ring & 2-ring (fingerprint env) ==="
$FPPY scripts/defense/composite_external_eval.py --mode embed $COMMON --vine_scales 1.0,0.75,0.5 --embed_dir $SC/dr_embed_3ring 2>&1 | grep -aE "embedded|EMBED_DONE"
$FPPY scripts/defense/composite_external_eval.py --mode embed $COMMON --vine_scales 0.75,0.5     --embed_dir $SC/dr_embed_2ring 2>&1 | grep -aE "embedded|EMBED_DONE"

echo "=== PHASE 2: CtrlRegen attack both variants, s0.5 & s0.9 (ctrlregen env) ==="
for V in 3ring 2ring; do
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $SC/dr_embed_$V --out_dir $SC/dr_cr_$V --step_list 0.5,0.9 \
    2>&1 | grep -avE "warn|Warning|weights_only|state_dict|Loading pipeline|^\s*$|Migrating|cache for model" | tail -3
done

echo "=== PHASE 3: decode all 4 attacked sets (fingerprint env, full cascade) ==="
for V in 3ring 2ring; do
  for S in s05 s09; do
    ADIR=$SC/dr_cr_${V}_${S}
    [ -d "$ADIR" ] || ADIR=$SC/dr_cr_${V}_s0.${S:2:1}
    echo "--- $V $S  (dir=$(basename $ADIR)) ---"
    $FPPY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal \
      --geo_cascade --vine_scale_step 0.005 --resync_range 180 \
      --embed_dir $SC/dr_embed_$V --attacked_dir $ADIR --attack_name dr_${V}_${S} \
      --output results/defense/dr_regen_${V}_${S}.json 2>&1 | grep -aE "COMPOSITE_or|DECODE_DONE"
  done
done
echo DROP_RING_REGEN_ALL_DONE
