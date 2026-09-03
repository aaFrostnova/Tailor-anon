#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-8
#SBATCH --job-name=p3umsw
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
# weak = 500/125 ; strong = 4000/1000  (default 2000/500 already measured, det=0.16)
for S in "weak 500 125" "strong 4000 1000"; do
  set -- $S; NAME=$1; I1=$2; I2=$3
  (cd external/ai-watermark && $UMPY $REPO/scripts/attack/unmarker_batch.py --in_dir $SC/p3embed_Dum \
      --out_dir $SC/p3um_$NAME --n 15 --max_iter_stage1 $I1 --max_iter_stage2 $I2 2>&1 | tail -2)
  $FPPY scripts/defense/phase3_adv.py --mode decode --config D --embed_dir $SC/p3embed_Dum \
      --attacked_dir $SC/p3um_$NAME --out $SC/out_p3um_$NAME.json 2>&1 | grep -aE "COMPOSITE_or"
  echo "STRENGTH_$NAME iters=$I1/$I2 DONE"
done
echo P3UM_SWEEP_DONE
