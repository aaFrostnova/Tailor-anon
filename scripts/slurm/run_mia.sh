#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-3
#SBATCH --job-name=fp_mia
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

N_SAMPLES="${N_SAMPLES:-500}"
MODEL_SUBDIR="${MODEL_SUBDIR:-model_fp100pct}"
TAG="${TAG:-fp100}"
RESULT_DIR="${RESULT_DIR:-memorization_sd_per_image_10k}"
DATASET_DIR="${DATASET_DIR:-/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted}"

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
MODEL_NAME=/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5
MODEL_DIR=${ROOT_DIR}/results/${RESULT_DIR}/${MODEL_SUBDIR}
OUT=${ROOT_DIR}/results/mia_${TAG}.json

echo "MIA detection on ${MODEL_SUBDIR}, ${N_SAMPLES} samples"

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/mia_detection.py \
    --model_dir ${MODEL_DIR} \
    --dataset_dir ${DATASET_DIR} \
    --model_name ${MODEL_NAME} \
    --n_samples ${N_SAMPLES} \
    --output ${OUT}

echo "Done. Results: ${OUT}"
