#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=fe_xenv
#SBATCH -t 6:00:00
#SBATCH -o logs/fe_xenv_%A_%a.out
# Front-end replacement cells on the cross-environment columns: one (stage, fragment, knot) per task,
# "stage frag strength NC NU": composite embed with the stage on -> CtrlRegen+ x3 (NC images) -> UnMarker
# (NU images; 0 = not at this knot) -> deployed per-image read + on/off decode + latency.
source ${WM_PIPELINE:-${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}}/_sbatch_header.inc
V="0.14 0.17 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"; T="0.4 0.55 0.7 0.85 1.0 1.15 1.3 1.45 1.6 1.8 2.0"; S="0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 1.75 2.0 2.25 2.5 2.8 3.2"
UV=" 0.14 0.3 0.5 0.7 1.0 "; UT=" 0.4 0.7 1.0 1.45 2.0 "; US=" 0.5 1.0 1.5 2.25 3.2 "
nu() { case "$2" in *" $1 "*) echo 30;; *) echo 0;; esac; }
CELLS=()
for s in $V; do CELLS+=("resync VINE $s 50 $(nu $s "$UV")"); done
for s in $T; do CELLS+=("resync TrustMark $s 50 $(nu $s "$UT")"); done
for s in $S; do CELLS+=("resync VideoSeal $s 50 $(nu $s "$US")"); done
for s in $T; do CELLS+=("tile TrustMark $s 50 $(nu $s "$UT")"); done
for s in 0.14 0.17; do CELLS+=("scale VINE $s 50 30"); done
read C F S NC NU <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
N=$(( NC > NU ? NC : NU )); KD=$SC/fe_xenv/${C}_${F}_$S; mkdir -p $KD
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID/${#CELLS[@]} cell=$C $F@$S NC=$NC NU=$NU $(date)"
[ "$(ls $KD/embed/*.png 2>/dev/null | wc -l)" -ge "$N" ] || $FPPY -u $SC/fe_xenv_campaign.py embed $C $F $S $N $KD 2>&1 | grep -vE "$QUIET" | tail -1
[ "$(ls $KD/embed/*.png 2>/dev/null | wc -l)" -ge "$N" ] || { echo "XENV_ABORT: embed failed"; exit 1; }
if [ "$(ls $KD/att_ctrlregen_s07/*.png 2>/dev/null | wc -l)" -lt "$NC" ]; then
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $KD/embed --out_dir $KD/att_ctrlregen --step_list 0.3,0.5,0.7 2>&1 | grep -vE "$QUIET" | tail -2
fi
if [ "$NU" -gt 0 ] && [ "$(ls $KD/att_unmarker/*.png 2>/dev/null | wc -l)" -lt "$NU" ]; then
  $UMPY scripts/attack/unmarker_batch.py --in_dir $KD/embed --out_dir $KD/att_unmarker --config attack_configs/Vine.yaml --n $NU --start_idx 0 --batch 1 2>&1 | grep -vE "$QUIET" | tail -2
fi
echo "attacked: ctrlregen $(ls $KD/att_ctrlregen_s03/*.png 2>/dev/null | wc -l)/$NC  unmarker $(ls $KD/att_unmarker/*.png 2>/dev/null | wc -l)/$NU  $(date)"
$FPPY -u $SC/fe_xenv_campaign.py decode $C $F $S $KD 2>&1 | grep --line-buffered -vE "$QUIET"
rc=${PIPESTATUS[0]}
grep -q "FEX_DECODE_DONE" $SC/logs/fe_xenv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || rc=1
echo "DONE $(date)"; exit ${rc:-0}
