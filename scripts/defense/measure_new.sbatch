#!/bin/bash
#SBATCH --job-name=voffmeas2 --partition=gpu --account=pi_shiqingma_umass_edu --qos=normal
#SBATCH --gres=gpu:v100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=1:00:00
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/offmeas2-%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/offmeas2-%j.err
set -uo pipefail
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
export PATH="$ENV/bin:$PATH" PYTHONPATH="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export TORCH_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/torch PYTHONUNBUFFERED=1
python /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense/measure_abl_avg.py \
  --root /scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts \
  --out /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/abl_position_new.csv \
  --configs vine_official_C0 vine_official_C5 --minstep 45000
echo "=== offmeas2 exited rc=$? ==="
