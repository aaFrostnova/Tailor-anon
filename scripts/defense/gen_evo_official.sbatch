#!/bin/bash
#SBATCH --job-name=vevooff --partition=gpu --account=pi_shiqingma_umass_edu --qos=normal
#SBATCH --gres=gpu:v100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=0:40:00
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/evooff-%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/evooff-%j.err
set -uo pipefail
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
export PATH="$ENV/bin:$PATH" PYTHONPATH="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export TORCH_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/torch PYTHONUNBUFFERED=1
python /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense/gen_wm_evolution.py \
  --dir /scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_official_C0 \
  --steps 5000,10000,20000,30000,40000 \
  --out /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/fig_wm_evolution_official_C0.png \
  --tag "official-C0"
echo "=== evooff exited rc=$? ==="
