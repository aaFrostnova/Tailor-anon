#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH --job-name=extend
#SBATCH -t 4:00:00
#SBATCH -o logs/extend_%A_%a.out
# array 0-9: the ten extension knots (six VideoSeal, two VINE, two TrustMark)
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
CELLS=("VideoSeal 1.75" "VideoSeal 2.0" "VideoSeal 2.25" "VideoSeal 2.5" "VideoSeal 2.8" "VideoSeal 3.2" "VINE 0.14" "VINE 0.17" "TrustMark 1.8" "TrustMark 2.0")
ARGS=${CELLS[$SLURM_ARRAY_TASK_ID]}
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID cell=$ARGS $(date)"
${WM_PY:-python} -u $SC/extend_knots_campaign.py $ARGS 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
