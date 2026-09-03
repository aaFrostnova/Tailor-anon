#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-2
#SBATCH --job-name=fp_mia2
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

N_TRAIN="${N_TRAIN:-250}"
N_HELDOUT="${N_HELDOUT:-250}"
MODEL_SUBDIR="${MODEL_SUBDIR:-model_fp100pct}"
TAG="${TAG:-iter3}"
RESULT_DIR="${RESULT_DIR:-memorization_sd_sd_iter3_full_unet}"
DATASET_DIR="${DATASET_DIR:-/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_latent_fp_2k}"
MODE="${MODE:-latent}"
GLOBAL_FP="${GLOBAL_FP:-0}"

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
MODEL_NAME=/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5
MODEL_DIR=${ROOT_DIR}/results/${RESULT_DIR}/${MODEL_SUBDIR}
OUT=${ROOT_DIR}/results/mia_th_${TAG}.json

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

EXTRA=""
if [[ "${GLOBAL_FP}" == "1" ]]; then
    EXTRA="--global_fingerprint"
fi

python scripts/mia_train_vs_heldout.py \
    --model_dir ${MODEL_DIR} \
    --dataset_dir ${DATASET_DIR} \
    --model_name ${MODEL_NAME} \
    --n_train ${N_TRAIN} \
    --n_heldout ${N_HELDOUT} \
    --mode ${MODE} \
    ${EXTRA} \
    --output ${OUT}

echo "Done: ${OUT}"
