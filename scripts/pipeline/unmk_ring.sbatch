#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=unmk_ring
#SBATCH -t 0-12:00
#SBATCH --array=0-8
#SBATCH -o logs/unmk_ring_%A_%a.out
# One VINE strength knot: ring embed (fingerprint env) -> UnMarker (unmarker env) + CtrlRegen+ x3
# (ctrlregen env) -> cascade decode (fingerprint env). N images per knot as $1 (default 30).
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
N=${1:-30}
KNOTS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0); S=${KNOTS[$SLURM_ARRAY_TASK_ID]}
KD=$SC/unmarker9_ring/knot_$S; mkdir -p $KD
CRPY=${WM_PY_CTRLREGEN:-python}
UMPY=${WM_PY_UNMARKER:-python}
FPPY=${WM_PY:-python}
cd $CF
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export PYTHONPATH="$CF:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) knot=$S N=$N $(date)"
[ "$(ls $KD/embed/*.png 2>/dev/null | wc -l)" -ge "$N" ] || $FPPY -u $SC/unmk_ring_stage.py $S $N $KD/embed 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating" | tail -1
grep -q "STAGE_DONE" <(ls $KD/embed/*.png 2>/dev/null | wc -l | sed 's/.*/STAGE_DONE/') || { echo "RING_ABORT: embed failed"; exit 1; }
if [ "$(ls $KD/att_ctrlregen_s07/*.png 2>/dev/null | wc -l)" -lt "$N" ]; then
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $KD/embed --out_dir $KD/att_ctrlregen --step_list 0.3,0.5,0.7 2>&1 | grep -vE "warn|Warning|weights_only|Loading|Migrating|^\s*$" | tail -2
fi
if [ "$(ls $KD/att_unmarker/*.png 2>/dev/null | wc -l)" -lt "$N" ]; then
  $UMPY scripts/attack/unmarker_batch.py --in_dir $KD/embed --out_dir $KD/att_unmarker --config attack_configs/Vine.yaml --n $N --start_idx 0 --batch 1 2>&1 | grep -vE "warn|Warning|weights_only|^\s*$" | tail -2
fi
echo "attacked: unmarker $(ls $KD/att_unmarker/*.png 2>/dev/null | wc -l)/$N, ctrlregen s03/s05/s07 $(ls $KD/att_ctrlregen_s03/*.png 2>/dev/null | wc -l)/$(ls $KD/att_ctrlregen_s05/*.png 2>/dev/null | wc -l)/$(ls $KD/att_ctrlregen_s07/*.png 2>/dev/null | wc -l)  $(date)"
$FPPY -u $SC/unmk_ring_decode.py $S $KD 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating"
rc=${PIPESTATUS[0]}
grep -q "KNOT_DECODE_DONE" $SC/logs/unmk_ring_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || rc=1
echo "DONE $(date)"
exit ${rc:-0}
