#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH --job-name=pi_regen
#SBATCH -t 1:30:00
#SBATCH -o logs/perimage_stage_regen_%A_%a.out
# Per-image values of the stage cells on regen and rinse2x, for the fragments each stage re-embeds:
# (resync x 3 fragments, scale x VINE, tile x TrustMark) at every knot. Own output directory: the default
# one holds the 14-column cell of the same (stage, fragment, knot).
source ${WM_PIPELINE:-${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}}/_sbatch_header.inc
V="0.14 0.17 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"; T="0.4 0.55 0.7 0.85 1.0 1.15 1.3 1.45 1.6 1.8 2.0"; S="0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 1.75 2.0 2.25 2.5 2.8 3.2"
CELLS=()
for s in $V; do CELLS+=("resync VINE $s"); done; for s in $T; do CELLS+=("resync TrustMark $s"); done; for s in $S; do CELLS+=("resync VideoSeal $s"); done
for s in $V; do CELLS+=("scale VINE $s"); done
for s in $T; do CELLS+=("tile TrustMark $s"); done
read C F S <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
export STAGE_COLS=regen,rinse2x STAGE_OUT=$SC/perimage_ext/stage_${C}_regen
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID/${#CELLS[@]} cell=$C $F@$S -> $STAGE_OUT $(date)"
$FPPY -u $SC/perimage_campaign.py stage $C $F $S 2>&1 | grep --line-buffered -vE "$QUIET"
rc=${PIPESTATUS[0]}
grep -q "STAGE_DONE" $SC/logs/perimage_stage_regen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || rc=1
echo "DONE $(date)"; exit ${rc:-0}
