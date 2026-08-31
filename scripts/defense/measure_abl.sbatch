#!/bin/bash
#SBATCH --job-name=vabl_measure
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --gres=gpu:a16:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=0:40:00
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/measure-%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/measure-%j.err
set -uo pipefail
REPO=/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface
export HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export TORCH_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/torch
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
RESDIR=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch
mkdir -p "$RESDIR"
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense/measure_abl_trajectory.py \
  --root /scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_abl_scratch \
  --out "$RESDIR/abl_position_trajectory.csv"
echo "=== measure exited rc=$? ==="
