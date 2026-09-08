#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=smoke_ct
#SBATCH -t 1:00:00
#SBATCH -o logs/smoke_complete_%j.out
# Two-image smoke of every new campaign path, writing under smoke_complete/ only (nothing the builder globs).
source ${WM_PIPELINE:-${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}}/_sbatch_header.inc
SM=$SC/smoke_complete; rm -rf $SM; mkdir -p $SM
echo "=== (1) fe_xenv: resync VINE 0.5, 2 images, 1 unmarker ==="
KD=$SM/fex; $FPPY -u $SC/fe_xenv_campaign.py embed resync VINE 0.5 2 $KD 2>&1 | grep -vE "$QUIET" | tail -1
$CRPY scripts/attack/ctrlregen_batch.py --in_dir $KD/embed --out_dir $KD/att_ctrlregen --step_list 0.3,0.5,0.7 2>&1 | grep -vE "$QUIET" | tail -1
$UMPY scripts/attack/unmarker_batch.py --in_dir $KD/embed --out_dir $KD/att_unmarker --config attack_configs/Vine.yaml --n 1 --start_idx 0 --batch 1 2>&1 | grep -vE "$QUIET" | tail -1
FEX_OUT=$SM/perimage/stage_resync_xenv $FPPY -u $SC/fe_xenv_campaign.py decode resync VINE 0.5 $KD 2>&1 | grep -vE "$QUIET"
echo "=== (2) delta xenv: VINE over TrustMark, 2 covers ==="
KD=$SM/dx; $FPPY -u $SC/make_delta_xenv.py embed VINE TrustMark 2 $KD 2>&1 | grep -vE "$QUIET" | tail -1
$CRPY scripts/attack/ctrlregen_batch.py --in_dir $KD/embed --out_dir $KD/att_ctrlregen --step_list 0.3,0.5,0.7 2>&1 | grep -vE "$QUIET" | tail -1
DELTA_OUT_DIR=$SM $FPPY -u $SC/make_delta_xenv.py decode VINE TrustMark $KD 2>&1 | grep -vE "$QUIET"
echo "=== (3) delta in-process: two cells, 2 covers ==="
echo '[["VINE","TrustMark","jpeg25"],["TrustMark","VideoSeal","regen"]]' > $SM/cells.json
export DELTA_CELLS_FILE=$SM/cells.json OUT_TAG=smoke_delta DELTA_GRID_JSON='{"VINE": [0.14, 0.6, 1.0], "TrustMark": [0.4, 1.0, 2.0]}'
$FPPY -u $SC/make_delta_curves.py 2 0 2 2>&1 | grep -vE "$QUIET" | tail -3
$FPPY -u $SC/make_delta_curves.py 2 1 2 2>&1 | grep -vE "$QUIET" | tail -3
mv $SC/surrogate_smoke_delta_*of2.json $SM/ 2>/dev/null; unset DELTA_CELLS_FILE OUT_TAG DELTA_GRID_JSON
echo "=== (4) stage per-image on regen/rinse2x: resync VINE 0.5, 2 images ==="
STAGE_COLS=regen,rinse2x STAGE_OUT=$SM/perimage/stage_resync_regen STAGE_N=2 $FPPY -u $SC/perimage_campaign.py stage resync VINE 0.5 2>&1 | grep -vE "$QUIET"
echo "SMOKE_DONE $(date)"
