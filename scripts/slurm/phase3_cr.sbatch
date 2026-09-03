#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-8
#SBATCH --job-name=p3cr
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
for C in D A; do
  $FPPY scripts/defense/phase3_adv.py --mode embed --config $C --out_dir $SC/p3embed_$C --n 25 2>&1 | grep -aE "EMBED_DONE"
  (cd external/CtrlRegen && $CRPY $REPO/scripts/attack/ctrlregen_batch.py --in_dir $SC/p3embed_$C --out_dir $SC/p3cr_$C --step_list "0.3,0.5,0.7,0.9" --steps 50 2>&1 | grep -aE "ctrlregen|Error")
  for TAG in 03 05 07 09; do
    $FPPY scripts/defense/phase3_adv.py --mode decode --config $C --embed_dir $SC/p3embed_$C --attacked_dir $SC/p3cr_${C}_s${TAG} --out $SC/out_p3cr_${C}_s${TAG}.json 2>&1 | grep -aE "COMPOSITE_or"
  done
done
echo P3CR_DONE
