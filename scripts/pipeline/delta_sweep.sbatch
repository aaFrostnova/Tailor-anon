#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=delta_sw
#SBATCH -t 3:00:00
#SBATCH -o logs/delta_sweep_%A_%a.out
# Interference over the overwriting fragment's strength, in-process columns: one (g, f, attack) cell per
# task from delta_sweep_cells.json (76 cells), N=100, host at its mid strength, g over its full range.
source ${WM_PIPELINE:-${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}}/_sbatch_header.inc
export DELTA_CELLS_FILE=$SC/delta_sweep_cells.json OUT_TAG=delta_sweep
export DELTA_GRID_JSON='{"VINE": [0.14, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "TrustMark": [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0], "VideoSeal": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.2]}'
NCELLS=$($FPPY -c "import json; print(len(json.load(open('$DELTA_CELLS_FILE'))))")
echo "node=$(hostname) cell=$SLURM_ARRAY_TASK_ID/$NCELLS $(date)"
$FPPY -u $SC/make_delta_curves.py 100 $SLURM_ARRAY_TASK_ID $NCELLS 2>&1 | grep --line-buffered -vE "$QUIET"
rc=${PIPESTATUS[0]}
grep -q "DELTA_CURVES_DONE" $SC/logs/delta_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || rc=1
echo "DONE $(date)"; exit ${rc:-0}
