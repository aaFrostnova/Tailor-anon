#!/bin/bash
# FROM-SCRATCH ablation: what pushes VINE's watermark to the border?
# Faithful (not fine-tune): random-init VINE trained from step 0 so the payload EMERGES,
# discriminator CO-TRAINED from scratch (no GAN confound). 3 configs, each toggling ONE factor.
#   C0_full        : full VINE-B stage-2 config (secret+L2+LPIPS+GAN, blur/jpeg/noise augs) -> expect BORDER
#   C1_secretonly  : remove ALL perceptual loss (L2+LPIPS+GAN)                               -> expect UNIFORM (key test)
#   C5_crop        : full + crop-resize augmentation (attacks the border)                    -> expect payload OFF border
# 4x A100 DDP via train_ddp.py (shared-VAE double-wrap fixed). Data = OpenImage train_f (same dataset as paper).
# All checkpoints -> /scratch. Measure border/center ratio OFFLINE from each checkpoint afterwards.
set -uo pipefail
REPO=/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo
ENV=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/vine
DATA=/scratch/workspace/mingzhel_umass_edu-ablator/openimages/train_f            # OpenImage train shard (flat *.jpg)
ABLROOT=/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_abl_scratch # big checkpoints -> scratch
GEN=$REPO/vine/src/_abl_scratch_sbatch
mkdir -p "$GEN" "$ABLROOT"

# ---- guard: OpenImage must be extracted before we submit ----
# NOTE: use find, NOT `ls "$DATA"/*.jpg` -- the shell glob hits ARG_MAX ("argument list
# too long") past ~25k files and silently returns 0, which would falsely ABORT the submit.
n=$(find "$DATA" -maxdepth 1 -name '*.jpg' | wc -l)
if [ "$n" -lt 1000 ]; then
  echo "ABORT: only $n jpgs in $DATA (OpenImage not extracted yet). Re-run after EXTRACT_DONE."
  exit 1
fi
echo "OpenImage OK: $n jpgs in $DATA"

# config name : extra train args
declare -a CFG=(
  "C0_full:"
  "C1_secretonly:--l2_loss_scale 0 --lpips_loss_scale 0 --G_loss_scale 0"
  "C5_crop:--crop_resize_step 0"
)
port=29610
for entry in "${CFG[@]}"; do
  name="${entry%%:*}"; extra="${entry#*:}"; out="$ABLROOT/$name"; mkdir -p "$out"; port=$((port+1))
  sb="$GEN/$name.sbatch"
  cat > "$sb" <<EOF
#!/bin/bash
#SBATCH --job-name=vabl_s_$name
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --gres=gpu:a100:4 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=200G --time=16:00:00
#SBATCH --output=$out/slurm-%j.out
#SBATCH --error=$out/slurm-%j.err
set -uo pipefail
export PATH="$ENV/bin:\$PATH" PYTHONPATH="$REPO"
export HF_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface
export HF_HUB_CACHE=/project/pi_shiqingma_umass_edu/mingzheli/.cache/huggingface/hub
export TORCH_HOME=/project/pi_shiqingma_umass_edu/mingzheli/.cache/torch
export WANDB_MODE=offline WANDB_DIR="$out" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$REPO"
echo "host=\$(hostname)  config=$name  extra='$extra'"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
python -m accelerate.commands.launch \\
  --num_processes 4 --multi_gpu --num_machines 1 --main_process_port $port --mixed_precision no \\
  vine/src/train_ddp.py \\
    --dataset_folder "$DATA" --val_folder "$out/noval" --output_dir "$out" \\
    --resolution 256 --train_batch_size 8 --dataloader_num_workers 8 \\
    --max_train_steps 12000 --checkpointing_steps 1000 --checkpoints_total_limit 15 \\
    --secret_size 100 --seed 42 --report_to wandb \\
    $extra --key_change "abl_scratch_$name"
echo "=== $name exited rc=\$? ==="
EOF
  jid=$(sbatch --parsable "$sb")
  echo "submitted $name -> job $jid  (out=$out)"
done
echo "ALL_ABL_SCRATCH_SUBMITTED"
