#!/bin/bash
#SBATCH --job-name=slv_cont
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -t 0-3:00
#SBATCH --array=0-19
#SBATCH -o logs/slv_cont_%A_%a.out
set -u
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$SC"
echo "node=$(hostname) shard=$SLURM_ARRAY_TASK_ID $(date)"
${WM_PY:-python} \
  $SC/solver_eval_continuous.py "$SLURM_ARRAY_TASK_ID" 20 2000
echo "SLV_CONT_DONE $(date)"
