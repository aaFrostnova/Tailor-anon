#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=48G -t 0-10
#SBATCH --job-name=rerunfix
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python scripts/defense/rerun_fixed_search.py --n 50 --vine_scale_step 0.005 2>&1 | grep -aE "embed |crop staircase|off-center|UnMarker|CtrlRegen|DONE|Error|Traceback"
