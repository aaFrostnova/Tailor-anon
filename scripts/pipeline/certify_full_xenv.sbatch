#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=cf_xenv
#SBATCH -t 0-12:00
#SBATCH -o logs/certify_full_xenv_%A_%a.out
# Cross-environment chain for the groups of one class that name CtrlRegen+ / UnMarker: $1 class index,
# $2 CtrlRegen+ images, $3 UnMarker images; the array index enumerates the groups with xenv_needed.
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
CRPY=${WM_PY_CTRLREGEN:-python}
UMPY=${WM_PY_UNMARKER:-python}
FPPY=${WM_PY:-python}
KEYS=(C1 C2 C3 C4 C5); K=${KEYS[$1]}; NC=${2:-100}; NU=${3:-30}
# CERT_DOMAIN (exported by the caller) selects the domain's group directory; certify_full.py decode_xenv reads it too
DOM=${CERT_DOMAIN:-indomain}; CDIR=$SC/certify_full; [ "$DOM" != "indomain" ] && CDIR=$SC/certify_full_$DOM
cd $CF
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
# the i-th group that needs the chain, and what it needs
read GI STEPS UNM <<< "$($FPPY - "$K" "$SLURM_ARRAY_TASK_ID" "$CDIR" <<'PY'
import json, glob, sys
K, idx, CDIR = sys.argv[1], int(sys.argv[2]), sys.argv[3]
need = []
for p in sorted(glob.glob(f"{CDIR}/{K}/g*.json")):
    g = json.load(open(p))
    if g.get("xenv_needed"): need.append(g)
g = need[idx]; steps = ",".join(sorted({"0." + a[-1] for a in g["xenv_needed"] if a.startswith("ctrlregen")})) or "-"
print(g["gidx"], steps, "1" if "unmarker" in g["xenv_needed"] else "0")
PY
)"
GD=$CDIR/$K/g$(printf %04d $GI)
echo "node=$(hostname) class=$K group=$GI steps=$STEPS unmarker=$UNM NC=$NC NU=$NU $(date)"
[ "$(ls $GD/embed/*.png 2>/dev/null | wc -l)" -ge 1 ] || { echo "XENV_ABORT: no embeds in $GD"; exit 1; }
if [ "$STEPS" != "-" ]; then
  mkdir -p $GD/embed_cr; ls $GD/embed/*.png | head -n $NC | xargs -I{} ln -sf {} $GD/embed_cr/
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $GD/embed_cr --out_dir $GD/att_ctrlregen --step_list $STEPS 2>&1 | grep -vE "warn|Warning|weights_only|Loading|Migrating|^\s*$" | tail -2
fi
if [ "$UNM" = "1" ] && [ "$(ls $GD/att_unmarker/*.png 2>/dev/null | wc -l)" -lt "$NU" ]; then
  $UMPY scripts/attack/unmarker_batch.py --in_dir $GD/embed --out_dir $GD/att_unmarker --config attack_configs/Vine.yaml --n $NU --start_idx 0 --batch 1 2>&1 | grep -vE "warn|Warning|weights_only|^\s*$" | tail -2
fi
$FPPY -u $SC/certify_full.py decode_xenv "$1" "$GI" 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
