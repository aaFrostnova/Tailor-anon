#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH -t 0-4
#SBATCH --job-name=vine_fork
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Fork of VINE's exact training code with our multi-fragment crypto payload.
# Uses VINE's full pipeline: TransformNet, GAN, YUV loss, full finetune.

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
OUT=${ROOT_DIR}/results/vine_fork

export PYTHONUNBUFFERED=1
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub

mkdir -p ${ROOT_DIR}/slurm_logs ${OUT}
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/train_vine_fork.py \
    --dataset_folder ${COCO} \
    --val_folder ${ROOT_DIR}/images \
    --output_dir ${OUT} \
    --secret_size 127 \
    --train_batch_size 2 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --gradient_checkpointing \
    --allow_tf32 \
    --fixed_input \
    --no_im_loss_steps 1000 \
    --checkpointing_steps 2000 \
    --validation_steps 2000 \
    --viz_freq 500 \
    --report_to tensorboard \
    --tracker_project_name vine_crypto_frag \
    --imagenetc_step 5000 \
    --crop_resize_step 50000 \
    --ig_filter_step 50000

echo "Done: ${OUT}"
