#!/bin/bash
# Ablation: which component pushes VINE's watermark to the border? Short fine-tunes from VINE-B, each toggling
# ONE factor. 5 single-GPU jobs submitted in parallel to superpod-a100. Measure border/center ratio + PSNR after.
set -uo pipefail
REPO=/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
DATA=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco2017/train2017
ABLROOT=/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_abl   # big checkpoints -> scratch, not /work
GEN=$REPO/vine/src/_abl_sbatch
mkdir -p "$GEN"

# config name : extra train args
declare -a CFG=(
  "A_baseline:"
  "B_noPerc:--l2_loss_scale 0 --lpips_loss_scale 0 --G_loss_scale 0"
  "C_cropResize:--crop_resize_step 0"
  "D_noGAN:--G_loss_scale 0"
  "E_noLPIPS:--lpips_loss_scale 0"
)
port=29540
for entry in "${CFG[@]}"; do
  name="${entry%%:*}"; extra="${entry#*:}"; out="$ABLROOT/$name"; mkdir -p "$out"; port=$((port+1))
  sb="$GEN/$name.sbatch"
  cat > "$sb" <<EOF
#!/bin/bash
#SBATCH --job-name=vabl_$name
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=100G --time=6:00:00
#SBATCH --output=$out/slurm-%j.out
#SBATCH --error=$out/slurm-%j.err
set -uo pipefail
export PATH="$ENV/bin:\$PATH" PYTHONPATH="$REPO"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface
export HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export WANDB_MODE=offline WANDB_DIR="$out" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$REPO"
echo "host=\$(hostname)  config=$name  extra='$extra'"
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \\
  --main_process_port $port vine/src/train_crop_ft.py \\
    --dataset_folder "$DATA" --val_folder "$out/noval" --output_dir "$out" \\
    --resolution 256 --train_batch_size 8 --dataloader_num_workers 8 \\
    --max_train_steps 3000 --checkpointing_steps 1000 --checkpoints_total_limit 4 \\
    --no_im_loss_steps 0 --l2_loss_ramp 1 --secret_size 100 --seed 42 --report_to wandb \\
    $extra --key_change "abl_$name"
echo "=== $name exited rc=\$? ==="
EOF
  jid=$(sbatch --parsable "$sb")
  echo "submitted $name -> job $jid  (out=$out)"
done
echo "ALL_ABL_SUBMITTED"
