#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-8
#SBATCH --job-name=umnc
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
# PURE-SPECTRAL UnMarker (crop OFF) on the NESTED-VINE composite -- the missing cell
(cd external/ai-watermark && $UMPY $REPO/scripts/attack/unmarker_batch.py --in_dir $SC/p3embed_D \
    --out_dir $SC/p3um_nested_nocrop --n 15 --no_preprocess 2>&1 | tail -2)
$FPPY scripts/defense/phase3_adv.py --mode decode --config D --embed_dir $SC/p3embed_D \
    --attacked_dir $SC/p3um_nested_nocrop --out $SC/out_p3um_nested_nocrop.json 2>&1 | grep -aE "COMPOSITE_or"
echo UM_NESTED_NOCROP_DONE
