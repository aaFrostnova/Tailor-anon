#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-8
#SBATCH --job-name=wbm_$1
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/wbm_%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/wbm_%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python; WB=/work/pi_shiqingma_umass_edu/mingzheli/W-Bench; N=200
M=$1; case $M in ours) FR="vine trustmark videoseal";; *) FR="$M";; esac
RES=results/defense/wbm/$M; mkdir -p $RES
echo "node=$(hostname) method=$M frags=$FR"
run(){ echo "##### $M / $1 #####"; $PY scripts/defense/wbench_eval_multi.py --fragments $FR --n $N "${@:2}" 2>&1 | grep -aE "task=|det|s=|frame_det|WBM_DONE|Error|Traceback" | tail -20; }
run distortion --task distortion --wb_dir $WB/DISTORTION_1K/image        --out $RES/distortion.json
run sto_sweep  --task sto_sweep  --wb_dir $WB/DISTORTION_1K/image        --out $RES/sto_sweep.json
run instruct   --task instruct   --wb_dir $WB/INSTRUCT_1K/image --caption_csv $WB/INSTRUCT_1K/captions_new.csv --strength 0.6 --out $RES/instruct.json
run local      --task local      --wb_dir $WB/LOCAL_EDITING_5K/30-40/image --mask_dir $WB/LOCAL_EDITING_5K/30-40/mask --out $RES/local.json
run svd        --task svd        --wb_dir $WB/DISTORTION_1K/image        --n 100 --out $RES/svd.json
echo "SBATCH_DONE $M"
