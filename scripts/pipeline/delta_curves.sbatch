#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=delta_cv
#SBATCH -t 0-6:00
#SBATCH --exclude=gpu058
#SBATCH --array=0-21
#SBATCH -o logs/delta_curves_%A_%a.out
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
cd $CF
export PYTHONPATH="$CF:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TS=$(date -u +%FT%TZ) OUT_TAG=${OUT_TAG:-delta_curves}
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) cell=$SLURM_ARRAY_TASK_ID $(date)"
${WM_PY:-python} -u $SC/make_delta_curves.py "${1:-100}" "$SLURM_ARRAY_TASK_ID" 22 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
rc=${PIPESTATUS[0]}
[ -f "$SC/surrogate_${OUT_TAG}_${SLURM_ARRAY_TASK_ID}of22.json" ] || { echo "DELTA_ABORT: no output"; rc=1; }
echo "DONE $(date)"
exit ${rc:-0}
