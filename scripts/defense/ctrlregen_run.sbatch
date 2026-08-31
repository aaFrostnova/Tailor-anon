#!/bin/bash
#SBATCH --job-name=vctrl --partition=gpu --account=pi_shiqingma_umass_edu --qos=normal
#SBATCH --gres=gpu:v100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=1:00:00
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/ctrl-%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/ctrl-%j.err
set -uo pipefail
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
VPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine/bin/python
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HUGGINGFACE_HUB_CACHE=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache/hub PYTHONUNBUFFERED=1
WORK=/scratch/workspace/mingzhel_umass_edu-ablator/regen_work
OUT=/scratch/workspace/mingzhel_umass_edu-ablator/regen_out; mkdir -p $OUT
for M in C0 C5 VINEB; do
  echo "=== ctrlregen attack $M (s0.3/0.5/0.7) ==="
  $CRPY $REPO/scripts/attack/ctrlregen_batch.py --in_dir $WORK/$M --out_dir $OUT/$M --step_list 0.3,0.5,0.7 2>&1 | grep -vE "warn|Warning|weights_only|Loading pipeline|^\s*$|Migrating|cache for" | tail -3
done
echo "=== 解码 ctrlregen 输出 ==="
export PYTHONPATH=/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo
$VPY $REPO/scripts/defense/decode_regen.py --tags 03,05,07
echo "=== ctrl exited rc=$? ==="
