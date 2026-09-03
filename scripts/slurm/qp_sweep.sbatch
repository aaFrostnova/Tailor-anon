#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=6 --mem=48G -t 0-4
#SBATCH --job-name=qp_$1
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/qpsw_%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/qpsw_%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
M=$1; case $M in ours) FR="vine trustmark videoseal"; EX="--geo_cascade";; *) FR="$M"; EX="";; esac
$PY scripts/defense/qp_sweep_method.py --fragments $FR $EX --n 100 --out results/defense/qp_curves/$M.json 2>&1 | grep -aE "==|s=|QP_SWEEP_DONE|Error|Traceback" | tail -40
echo "SBATCH_DONE"
