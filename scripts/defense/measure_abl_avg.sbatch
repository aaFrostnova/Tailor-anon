#!/bin/bash
#SBATCH --job-name=vabl_avg --partition=gpu --account=pi_shiqingma_umass_edu --qos=normal
#SBATCH --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=0:40:00
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/measureavg-%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/measureavg-%j.err
set -uo pipefail
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
export PATH="$ENV/bin:$PATH" PYTHONPATH="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export TORCH_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/torch PYTHONUNBUFFERED=1
python /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense/measure_abl_avg.py \
  --root /scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_abl_scratch \
  --out /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/abl_position_avg.csv
echo "=== measureavg exited rc=$? ==="
