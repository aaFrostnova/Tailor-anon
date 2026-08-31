#!/bin/bash
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache CUDA_VISIBLE_DEVICES=0
GR='grep -avE "FutureWarning|weights_only|warn\(|UserWarning"'
echo "=== [1/3] REGEN FPR-SAFE RE-DECODE (correct the report numbers) ==="
$FPPY scripts/defense/regen_fpr_safe_decode.py 2>&1 | grep -avE "FutureWarning|weights_only|warn\(|UserWarning" | grep -aE "gpu|crypto_only|DONE|Error|Traceback"
echo "=== [2/3] DECODE TIMING (deployed cascade, verify-only) ==="
$FPPY scripts/defense/decode_timing.py 2>&1 | grep -avE "FutureWarning|weights_only|warn\(|UserWarning" | grep -aE "gpu|det=|DONE|Error|Traceback"
echo "=== [3/3] MINIMAL EMBEDDING SWEEP (verify-only) ==="
$FPPY scripts/defense/minimal_embedding_sweep.py 2>&1 | grep -avE "FutureWarning|weights_only|warn\(|UserWarning" | grep -aE "gpu|layers=|DONE|Error|Traceback"
echo ALL_OPT_CHAIN2_DONE
