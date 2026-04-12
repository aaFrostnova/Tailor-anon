#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion on fingerprinted COCO dataset.

Uses HuggingFace diffusers text-to-image fine-tuning pipeline.

Usage:
    # Quick test (few steps)
    python scripts/finetune_sd.py \
        --dataset_dir /project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted \
        --output_dir ./results/sd_finetuned \
        --max_train_steps 500 --batch_size 1

    # Full training
    python scripts/finetune_sd.py \
        --dataset_dir /project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted \
        --output_dir ./results/sd_finetuned \
        --max_train_steps 5000 --batch_size 4 --gradient_accumulation 4

Prerequisites:
    pip install diffusers accelerate transformers datasets
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class CaptionImageDataset(Dataset):
    """Load images + captions from the prepared fingerprinted dataset."""

    def __init__(self, dataset_dir, tokenizer, resolution=512, split="train"):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.resolution = resolution
        self.tokenizer = tokenizer

        # Load metadata
        jsonl_path = self.dataset_dir / f"{split}.jsonl"
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.image_dir / sample["file_name"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW
        image = image * 2.0 - 1.0  # [0,1] → [-1,1]

        # Tokenize caption
        tokens = self.tokenizer(
            sample["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SD on fingerprinted data")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to prepared fingerprinted dataset")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--model_name", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="Base model to fine-tune")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy imports (these are heavy)
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch.manual_seed(args.seed)

    # Load model components
    print(f"Loading model: {args.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Mixed precision
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    # Dataset
    dataset = CaptionImageDataset(args.dataset_dir, tokenizer, args.resolution)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.max_train_steps,
    )

    # Training loop
    unet.train()
    global_step = 0
    total_loss = 0

    print(f"\nTraining config:")
    print(f"  Dataset: {len(dataset)} images")
    print(f"  Batch size: {args.batch_size} × {args.gradient_accumulation} grad accum")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"  Steps: {args.max_train_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print()

    data_iter = iter(dataloader)
    optimizer.zero_grad()

    while global_step < args.max_train_steps:
        for accum_step in range(args.gradient_accumulation):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Encode text
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device, dtype=torch.long,
            )

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=(args.mixed_precision != "no")):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                loss = loss / args.gradient_accumulation

            loss.backward()
            total_loss += loss.item()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        # Logging
        if global_step % args.log_steps == 0:
            avg_loss = total_loss / args.log_steps
            print(f"Step {global_step}/{args.max_train_steps}: loss={avg_loss:.4f}")
            total_loss = 0

        # Save checkpoint
        if global_step % args.save_steps == 0:
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            unet.save_pretrained(str(ckpt_dir / "unet"))
            print(f"  Saved checkpoint to {ckpt_dir}")

    # Save final model
    print(f"\nSaving final model...")
    unet.save_pretrained(str(output_dir / "unet"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    # Save training info
    with open(output_dir / "training_info.json", "w") as f:
        json.dump({
            "base_model": args.model_name,
            "dataset_dir": args.dataset_dir,
            "max_train_steps": args.max_train_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "lr": args.lr,
            "resolution": args.resolution,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed,
        }, f, indent=2)

    print(f"Done! Model saved to {output_dir}")
    print(f"\nNext step: run detection")
    print(f"  python scripts/detect_fingerprint_in_model.py \\")
    print(f"    --model_dir {output_dir} \\")
    print(f"    --key {args.dataset_dir}/master.key")


if __name__ == "__main__":
    main()
