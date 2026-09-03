#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=short
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=6 --mem=64G -t 0-2
#SBATCH --job-name=vine_geo_eval
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python scripts/eval_vine_geo.py --ckpt "$1" --n 24 --regen
echo SBATCH_DONE
