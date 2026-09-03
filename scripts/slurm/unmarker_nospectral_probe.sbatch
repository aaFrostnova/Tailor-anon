#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-10
#SBATCH --job-name=um_nocrop
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
echo "=== UnMarker PURE-SPECTRAL (--no_preprocess, crop OFF) on embed_ours ==="
(cd external/ai-watermark && $UMPY $REPO/scripts/attack/unmarker_batch.py --in_dir "$SC/embed_ours" --out_dir "$SC/um_nocrop_ours" --n 25 --no_preprocess 2>&1 | tail -4)
echo "=== decode pure-spectral WITH geo_cascade ==="
$FPPY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal --geo_cascade \
  --embed_dir "$SC/embed_ours" --attacked_dir "$SC/um_nocrop_ours" --attack_name "um_nocrop" --output "$SC/out_um_nocrop_ours.json" 2>&1 | grep -aE "COMPOSITE_or"
echo "UM_NOCROP_PROBE_DONE"
