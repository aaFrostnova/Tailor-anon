#!/bin/bash
# Mild regeneration (regen / rinse2x / rinse4x) at nine strength knots, one image shard per task.
# Usage: sbatch [--array=0-(NSHARD-1)] make_mildregen9.sbatch <N=100> <NSHARD=4>
#        then, on a login node: $FPPY make_mildregen_overlay_9k.py merge N NSHARD
#SBATCH --job-name=mildreg9
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 0-8:00
#SBATCH --array=0-3
#SBATCH -o logs/mildreg9_%A_%a.out
set -u -o pipefail
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$REPO/external/WatermarkAttacker:$SC" OMP_NUM_THREADS=8
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub" CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
N="${1:-100}"; NSHARD="${2:-4}"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) N=$N shard=$SLURM_ARRAY_TASK_ID/$NSHARD OUT_NAME=${OUT_NAME:-default} $(date)"
${WM_PY:-python} -u $SC/make_mildregen_overlay_9k.py measure "$N" "$NSHARD" "$SLURM_ARRAY_TASK_ID" 2>&1 \
  | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
rc=${PIPESTATUS[0]}   # the python status, not grep's and not the trailing echo's
echo "DONE $(date)"
exit ${rc:-0}
