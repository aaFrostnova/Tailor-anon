#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=vine_cr
#SBATCH -t 0-8:00
#SBATCH -o logs/probe_vine_ctrlregen_%j.out
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
N=${1:-50}; WD=$SC/vine_cr_probe
CRPY=${WM_PY_CTRLREGEN:-python}
FPPY=${WM_PY:-python}
cd $CF
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
echo "node=$(hostname) N=$N $(date)"
$FPPY -u $SC/probe_vine_ctrlregen.py embed $N $WD 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded"
for d in $WD/*/; do
  [ -d "$d/embed" ] || continue
  [ "$(ls $d/att_ctrlregen_s07/*.png 2>/dev/null | wc -l)" -ge "$N" ] && continue
  echo "attacking $(basename $d) $(date)"
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir "$d/embed" --out_dir "$d/att_ctrlregen" --step_list 0.3,0.5,0.7 2>&1 | grep -vE "warn|Warning|weights_only|Loading|Migrating|^\s*$" | tail -1
done
$FPPY -u $SC/probe_vine_ctrlregen.py decode $WD 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
