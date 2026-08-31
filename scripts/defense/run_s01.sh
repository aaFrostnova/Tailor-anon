#!/bin/bash
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache CUDA_VISIBLE_DEVICES=0
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
echo "=== attack p3embed_D at CtrlRegen s0.1 ==="
$CRPY scripts/attack/ctrlregen_batch.py --in_dir $SC/p3embed_D --out_dir $SC/p3cr_D --step_list 0.1 \
  2>&1 | grep -avE "warn|Warning|weights_only|state_dict|Loading pipeline|^\s*$|Migrating|cache for model" | tail -3
echo "=== decode s0.1 (bit-acc) ==="
$FPPY scripts/defense/bitacc_one.py --embed_meta $SC/p3embed_D/meta.json --attacked_dir $SC/p3cr_D_s01 --tag ctrlregen_s01 \
  2>&1 | grep -avE "FutureWarning|weights_only|warn\(" | grep -aE "ctrlregen_s01|BITACC_ONE_DONE|Error|Traceback"
echo S01_ALL_DONE
