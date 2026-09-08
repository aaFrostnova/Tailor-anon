#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=ext_xenv
#SBATCH -t 0-12:00
#SBATCH -o logs/extend_xenv_%A_%a.out
# One extension knot: composite embed -> CtrlRegen+ x3 (ctrlregen env, NC images) -> UnMarker (unmarker
# env, NU images; 0 = skip) -> per-image decode. Cells: "frag strength NC NU".
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
CELLS=("VideoSeal 1.75 50 0" "VideoSeal 2.0 50 30" "VideoSeal 2.25 50 0" "VideoSeal 2.5 50 30" "VideoSeal 2.8 50 0" "VideoSeal 3.2 50 30" "TrustMark 1.8 50 30" "TrustMark 2.0 50 30" "VINE 0.14 50 0" "VINE 0.17 50 0")
read F S NC NU <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
N=$(( NC > NU ? NC : NU )); KD=$SC/extend_xenv/${F}_$S; mkdir -p $KD
CRPY=${WM_PY_CTRLREGEN:-python}
UMPY=${WM_PY_UNMARKER:-python}
FPPY=${WM_PY:-python}
cd $CF
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
echo "node=$(hostname) cell=$F@$S NC=$NC NU=$NU $(date)"
[ "$(ls $KD/embed/*.png 2>/dev/null | wc -l)" -ge "$N" ] || $FPPY -u $SC/extend_xenv.py embed $F $S $N $KD 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating" | tail -1
[ "$(ls $KD/embed/*.png 2>/dev/null | wc -l)" -ge "$N" ] || { echo "XENV_ABORT: embed failed"; exit 1; }
if [ "$(ls $KD/att_ctrlregen_s07/*.png 2>/dev/null | wc -l)" -lt "$NC" ]; then
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $KD/embed --out_dir $KD/att_ctrlregen --step_list 0.3,0.5,0.7 2>&1 | grep -vE "warn|Warning|weights_only|Loading|Migrating|^\s*$" | tail -2
fi
if [ "$NU" -gt 0 ] && [ "$(ls $KD/att_unmarker/*.png 2>/dev/null | wc -l)" -lt "$NU" ]; then
  $UMPY scripts/attack/unmarker_batch.py --in_dir $KD/embed --out_dir $KD/att_unmarker --config attack_configs/Vine.yaml --n $NU --start_idx 0 --batch 1 2>&1 | grep -vE "warn|Warning|weights_only|^\s*$" | tail -2
fi
echo "attacked: ctrlregen $(ls $KD/att_ctrlregen_s03/*.png 2>/dev/null | wc -l)/$NC  unmarker $(ls $KD/att_unmarker/*.png 2>/dev/null | wc -l)/$NU  $(date)"
$FPPY -u $SC/extend_xenv.py decode $F $S $KD 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating"
rc=${PIPESTATUS[0]}
grep -q "XENV_DECODE_DONE" $SC/logs/extend_xenv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || rc=1
echo "DONE $(date)"
exit ${rc:-0}
