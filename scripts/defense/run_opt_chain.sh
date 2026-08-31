#!/bin/bash
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache CUDA_VISIBLE_DEVICES=0
# 1. wait for the drop-ring-regen job to release the GPU
while kill -0 2445600 2>/dev/null; do sleep 60; done
echo "=== drop-ring-regen finished — final regen comparison ==="
for V in 3ring 2ring; do for S in s05 s09; do
  f=results/defense/dr_regen_${V}_${S}.json
  [ -f "$f" ] && $FPPY -c "import json;d=json.load(open('$f'));print(f'  ${V} ${S}: detect={d[\"summary\"][\"composite_or\"]:.3f} (n={d[\"n\"]})')"
done; done
echo ""
echo "=== [1/2] DECODE TIMING PROFILE ==="
$FPPY scripts/defense/decode_timing.py 2>&1 | grep -avE "FutureWarning|weights_only|warn\(" | grep -aE "gpu|det=|DONE|Error|Traceback"
echo ""
echo "=== [2/2] MINIMAL EMBEDDING SWEEP ==="
$FPPY scripts/defense/minimal_embedding_sweep.py 2>&1 | grep -avE "FutureWarning|weights_only|warn\(" | grep -aE "gpu|layers=|DONE|Error|Traceback"
echo ALL_OPT_CHAIN_DONE
