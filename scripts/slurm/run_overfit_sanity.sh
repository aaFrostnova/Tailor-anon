#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-4
#SBATCH --job-name=fp_overfit
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

N_IMAGES="${N_IMAGES:-50}"
MAX_STEPS="${MAX_STEPS:-15000}"              # HF: --max_train_steps=15000
LR="${LR:-1e-5}"                             # HF: --learning_rate=1e-05
LR_SCHEDULER="${LR_SCHEDULER:-constant}"     # HF: --lr_scheduler="constant"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"      # HF: --lr_warmup_steps=0
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"        # HF: --max_grad_norm=1
USE_EMA="${USE_EMA:-1}"                      # HF example: --use_ema
CENTER_CROP="${CENTER_CROP:-1}"              # HF example: --center_crop
RANDOM_FLIP="${RANDOM_FLIP:-1}"              # HF: --random_flip; WARNING destroys fingerprint
GRAD_CKPT="${GRAD_CKPT:-1}"                  # HF example: --gradient_checkpointing
SNR_GAMMA="${SNR_GAMMA:-}"
FULL_UNET="${FULL_UNET:-1}"                  # HF text2image finetune is full UNet
LORA_RANK="${LORA_RANK:-32}"                 # used only when FULL_UNET=0
TAG="${TAG:-sd_overfit50}"

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
DATASET_FULL=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_latent_fp_2k
DATASET_SMALL=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_latent_fp_${N_IMAGES}
OUTPUT_DIR=${ROOT_DIR}/results/memorization_${TAG}

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

# Build small dataset by copying first N_IMAGES of the full one
if [ ! -f "${DATASET_SMALL}/train.jsonl" ]; then
    mkdir -p ${DATASET_SMALL}/images ${DATASET_SMALL}/images_clean
    cp ${DATASET_FULL}/master.key ${DATASET_SMALL}/
    cp ${DATASET_FULL}/config.json ${DATASET_SMALL}/
    for i in $(seq 0 $((N_IMAGES-1))); do
        NAME=$(printf "%06d.png" $i)
        cp ${DATASET_FULL}/images/${NAME} ${DATASET_SMALL}/images/
        cp ${DATASET_FULL}/images_clean/${NAME} ${DATASET_SMALL}/images_clean/
    done
    head -n ${N_IMAGES} ${DATASET_FULL}/train.jsonl > ${DATASET_SMALL}/train.jsonl
    head -n ${N_IMAGES} ${DATASET_FULL}/metadata.jsonl > ${DATASET_SMALL}/metadata.jsonl
fi

PYTHON=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "Python: ${PYTHON}"
${PYTHON} -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

EXTRA_ARGS=""
[[ "${FULL_UNET}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --full_unet"
[[ "${USE_EMA}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --use_ema"
[[ "${CENTER_CROP}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --center_crop"
[[ "${RANDOM_FLIP}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --random_flip"
[[ "${GRAD_CKPT}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --gradient_checkpointing"
[[ -n "${SNR_GAMMA}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --snr_gamma ${SNR_GAMMA}"
echo "HF-aligned flags: ${EXTRA_ARGS}"

${PYTHON} scripts/run_memorization_sd.py \
    --model_name /project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5 \
    --dataset_dir ${DATASET_SMALL} \
    --output_dir ${OUTPUT_DIR} \
    --max_train_steps ${MAX_STEPS} \
    --n_samples 50 \
    --fp_ratios 1.0 \
    --batch_size 1 \
    --gradient_accumulation 4 \
    --lora_rank ${LORA_RANK} \
    --lr ${LR} \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_warmup_steps ${LR_WARMUP_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --mixed_precision fp16 \
    ${EXTRA_ARGS}

echo "Done: ${OUTPUT_DIR}"
