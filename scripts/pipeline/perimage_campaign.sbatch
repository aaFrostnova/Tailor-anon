#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH --job-name=perimage
#SBATCH -t 3:00:00
#SBATCH -o logs/perimage_%A_%a.out
# array 0-26: inprocess (frag = id/9, knot = id%9); 27-29: ctrlregen s03/s05/s07; 30: unmarker; 31: ring
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
PY=${WM_PY:-python}
ID=${SLURM_ARRAY_TASK_ID}; FR=(VINE TrustMark VideoSeal)
if   [ $ID -le 26 ]; then ARGS="inprocess ${FR[$((ID/9))]} $((ID%9))"
elif [ $ID -eq 27 ]; then ARGS="ctrlregen s03"
elif [ $ID -eq 28 ]; then ARGS="ctrlregen s05"
elif [ $ID -eq 29 ]; then ARGS="ctrlregen s07"
elif [ $ID -eq 30 ]; then ARGS="unmarker"
else                      ARGS="ring"; fi
echo "node=$(hostname) task=$ID args=$ARGS $(date)"
$PY -u $SC/perimage_campaign.py $ARGS 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
