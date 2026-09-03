#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=80G -t 0-8
#SBATCH --job-name=svd_all
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python; WB=/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image
for M in ours vine trustmark videoseal maskwm; do
  case $M in ours) FR="vine trustmark videoseal";; *) FR="$M";; esac
  RES=results/defense/wbm1k/$M; mkdir -p $RES
  echo "##### svd $M #####"
  $PY scripts/defense/wbench_eval_multi.py --task svd --fragments $FR --wb_dir "$WB" --n 100 --out $RES/svd.json 2>&1 | grep -aE "frame_det|WBM_DONE|Error|Traceback|LocalEntry" | tail -6
done
echo "SVD_ALL_DONE"
