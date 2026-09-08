#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=delta_xv
#SBATCH -t 5:00:00
#SBATCH -o logs/delta_xenv_%A_%a.out
# Interference over the overwriting fragment's strength on the CtrlRegen+ columns: one ordered pair per
# task; embed (fingerprint env) -> CtrlRegen+ x3 (ctrlregen env) -> decode. N=50 like the base cells.
source ${WM_PIPELINE:-${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}}/_sbatch_header.inc
CELLS=("VINE TrustMark" "TrustMark VINE" "TrustMark VideoSeal" "VideoSeal TrustMark" "VINE VideoSeal" "VideoSeal VINE")
read G F <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
N=50; KD=$SC/delta_xenv/${G}_${F}; mkdir -p $KD
echo "node=$(hostname) pair=$G over $F N=$N $(date)"
[ -f $KD/meta.json ] || $FPPY -u $SC/make_delta_xenv.py embed $G $F $N $KD 2>&1 | grep -vE "$QUIET" | tail -1
[ -f $KD/meta.json ] || { echo "XENV_ABORT: embed failed"; exit 1; }
NEXP=$(ls $KD/embed/*.png | wc -l)
if [ "$(ls $KD/att_ctrlregen_s07/*.png 2>/dev/null | wc -l)" -lt "$NEXP" ]; then
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $KD/embed --out_dir $KD/att_ctrlregen --step_list 0.3,0.5,0.7 2>&1 | grep -vE "$QUIET" | tail -2
fi
echo "attacked: $(ls $KD/att_ctrlregen_s03/*.png 2>/dev/null | wc -l)/$NEXP per step $(date)"
$FPPY -u $SC/make_delta_xenv.py decode $G $F $KD 2>&1 | grep --line-buffered -vE "$QUIET"
rc=${PIPESTATUS[0]}
grep -q "DELTA_XENV_DONE" $SC/logs/delta_xenv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || rc=1
echo "DONE $(date)"; exit ${rc:-0}
