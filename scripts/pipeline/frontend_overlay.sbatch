#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=fe_ovl
#SBATCH -t 0-12:00
#SBATCH -o logs/fe_ovl_%j.out
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
SC=$SCR/wm_dataset10k
cd $CF
export PYTHONPATH="$CF:$CF/scripts/defense:$SC" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TS=$(date -u +%FT%TZ)
N="${1:-100}"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) N=$N $(date)"
${WM_PY:-python} -u $SC/make_frontend_overlay.py "$N" 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|it/s"
rc=${PIPESTATUS[0]}   # the pipeline's python status, not grep's and not the trailing echo's
echo "DONE $(date)"
exit ${rc:-0}   # without this a crashed python still leaves the job COMPLETED
