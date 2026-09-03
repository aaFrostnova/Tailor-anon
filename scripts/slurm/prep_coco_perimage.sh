#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-2
#SBATCH --job-name=fp_prep_perimage
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Parameterized per-image pixel fingerprint prep.
#   sbatch --export=ALL,N_IMAGES=200 scripts/slurm/prep_coco_perimage.sh

set -euo pipefail

N_IMAGES="${N_IMAGES:-200}"
EPS="${EPS:-0.0627}"
ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
OUT=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fp_eps16_${N_IMAGES}

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

PYTHON=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "Python: ${PYTHON}"
echo "Output: ${OUT}"
echo "N_IMAGES=${N_IMAGES}, epsilon=${EPS} (per-image key, not global)"

${PYTHON} scripts/prepare_coco_fingerprinted.py \
    --output_dir ${OUT} \
    --max_images ${N_IMAGES} \
    --epsilon ${EPS} \
    --num_fragments 8 \
    --resolution 512 \
    --skip_download

echo "Done: ${OUT}"
