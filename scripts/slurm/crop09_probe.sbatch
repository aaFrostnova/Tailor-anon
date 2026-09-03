#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=short
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=48G -t 0-1
#SBATCH --job-name=crop09
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
echo "=== crop0.9 WITH geo_cascade ==="
$PY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal --geo_cascade \
  --embed_dir "$SC/embed_ours" --attacked_dir "$SC/crop09_ours" --attack_name "crop09_geo" --output "$SC/out_crop09_geo.json" 2>&1 | grep -aE "COMPOSITE_or"
echo "=== crop0.9 WITHOUT geo_cascade (plain decode) ==="
$PY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal \
  --embed_dir "$SC/embed_ours" --attacked_dir "$SC/crop09_ours" --attack_name "crop09_plain" --output "$SC/out_crop09_plain.json" 2>&1 | grep -aE "COMPOSITE_or"
echo "CROP09_PROBE_DONE"
