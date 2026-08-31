#!/bin/bash
#SBATCH --job-name=voffmeas3 --partition=gpu --account=pi_shiqingma_umass_edu --qos=normal
#SBATCH --gres=gpu:v100:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=0:40:00
#SBATCH --nodelist=gpu004
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/offmeas3-%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/offmeas3-%j.err
set -uo pipefail
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
export PATH="$ENV/bin:$PATH" PYTHONPATH="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export TORCH_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/torch PYTHONUNBUFFERED=1
S=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense/measure_abl_avg.py
R=/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts
O=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch
echo "### C0 60k+ ###"
python $S --root $R --out $O/abl_position_c0_60plus.csv --configs vine_official_C0 --minstep 60000
echo "### C2/C3 all ###"
python $S --root $R --out $O/abl_position_c2c3.csv --configs vine_official_C2 vine_official_C3 --minstep 0
echo "=== offmeas3 exited rc=$? ==="
