"""Train SD-pipeline-based watermark encoder (VINE-inspired, conservative route).

Architecture (borrowing VINE's image-to-image idea):
  1. ConditionAdaptor: fuse(image, S_pixel) → conditioned image
  2. VAE encode → latent z
  3. UNet 1-step denoise → modified latent
  4. VAE decode → watermarked image x_w

Detection: matched filter on (x_w - x_orig), same as before.

Training: ConditionAdaptor (from scratch) + UNet LoRA + VAE LoRA.
Loss: matched filter sign loss + MSE distortion + LPIPS perceptual loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.payload import BCHCodec, image_id_to_payload
from src.sign_envelope import derive_keyed_constants
from train_T_lat import (
    make_latent_region_masks, build_id_pool,
    load_sd_residual_pool, FlatImageDataset,
)

SD_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"


# ============================================================ modules

class ConditionAdaptor(nn.Module):
    """Fuse sign envelope S_pixel with input image.

    S_pixel (B, 1, 256, 256) with ±1 per region → dense features → concat with image → conv → fused image.
    """
    def __init__(self, n_bits=127):
        super().__init__()
        self.sign_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, image, S_pixel):
        sign_feat = self.sign_conv(S_pixel)
        fused = self.fuse(torch.cat([image, sign_feat], dim=1))
        return fused


def make_1step_sched(device):
    from diffusers import DDIMScheduler
    sched = DDIMScheduler.from_pretrained(SD_PATH, subfolder="scheduler")
    sched.set_timesteps(1, device=device)
    return sched


def load_sd_components(device):
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, AutoTokenizer

    vae = AutoencoderKL.from_pretrained(SD_PATH, subfolder="vae").to(device)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(SD_PATH, subfolder="unet").to(device)
    unet.requires_grad_(False)
    unet.train()

    # Add LoRA to UNet
    from peft import LoraConfig
    target_modules = []
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pat in ["to_k", "to_q", "to_v", "to_out.0", "conv1", "conv2", "conv_in", "conv_out"]:
            if pat in n:
                target_modules.append(n.replace(".weight", ""))
                break
    lora_config = LoraConfig(r=8, init_lora_weights="gaussian",
                             target_modules=target_modules, lora_alpha=8)
    unet.add_adapter(lora_config)

    n_lora = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"[setup] UNet LoRA params: {n_lora:,}")

    tokenizer = AutoTokenizer.from_pretrained(SD_PATH, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(SD_PATH, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    tokens = tokenizer("", max_length=tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt").input_ids
    text_emb = text_encoder(tokens.to(device))[0].detach()
    del text_encoder, tokenizer

    vae_scale = float(vae.config.scaling_factor)
    return vae, unet, text_emb, vae_scale


def sd_encode_watermark(image, S_pixel, adaptor, vae, unet, sched,
                        text_emb, vae_scale, timesteps):
    """Full SD pipeline watermark encoding."""
    B = image.shape[0]
    x_cond = adaptor(image, S_pixel)
    z = vae.encode(x_cond).latent_dist.mean * vae_scale
    emb = text_emb.repeat(B, 1, 1)
    model_pred = unet(z, timesteps[:B], encoder_hidden_states=emb).sample
    # Use scheduler step per sample (timestep is same for all)
    t = timesteps[0]
    z_out = sched.step(model_pred, t, z, return_dict=True).prev_sample
    x_w = vae.decode(z_out / vae_scale).sample.clamp(-1, 1)
    return x_w


# ============================================================ augmentation (simple for SD encoder)

def augment_simple(x_w, sd_pool=None, p_regen=0.30):
    B = x_w.shape[0]
    r = np.random.random()
    if sd_pool is not None and r < p_regen:
        idx = torch.randint(0, sd_pool.shape[0], (B,))
        return x_w + sd_pool[idx].to(x_w.device), False
    r -= p_regen
    if r < 0.15:
        return x_w, True
    r -= 0.15
    if r < 0.15:
        sigma = np.random.uniform(0.01, 0.05)
        return x_w + torch.randn_like(x_w) * sigma, True
    r -= 0.15
    if r < 0.20:
        from torchvision.transforms.functional import gaussian_blur
        k = int(np.random.choice([3, 5, 7]))
        s = float(np.random.uniform(0.5, 2.0))
        return gaussian_blur(x_w, kernel_size=k, sigma=s), True
    return x_w + torch.randn_like(x_w) * 0.01, True


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    codec = BCHCodec()
    master_key = args.master_key.encode("utf-8")
    id_pool_signs = build_id_pool(master_key, args.id_pool_size, codec)
    id_pool_t = torch.from_numpy(id_pool_signs).to(device)

    H, W = args.resolution, args.resolution
    grid = int(np.ceil(np.sqrt(codec.n)))
    cell_h, cell_w = H // grid, W // grid
    region_bounds = []
    for i in range(codec.n):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        region_bounds.append((y0, y1, x0, x1))

    # Load SD components
    vae, unet, text_emb, vae_scale = load_sd_components(device)
    sched = make_1step_sched(device)
    print(f"[setup] SD-v1.5 loaded, vae_scale={vae_scale:.4f}")

    # SD residual cache
    sd_pool = None
    if args.sd_residual_cache:
        sd_pool = load_sd_residual_pool(
            args.sd_residual_cache, strengths_to_load=args.attack_strengths,
        )
        if sd_pool is not None:
            sd_pool = sd_pool.to(device)
            print(f"[setup] SD residual pool: {sd_pool.shape}")

    # LPIPS loss
    lpips_net = None
    if args.lambda_lpips > 0:
        import lpips
        lpips_net = lpips.LPIPS(net='vgg').to(device)
        lpips_net.requires_grad_(False)
        print("[setup] LPIPS loaded")

    # Dataset
    ds = FlatImageDataset(args.image_dir, args.resolution, args.max_images)
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ConditionAdaptor
    adaptor = ConditionAdaptor(n_bits=codec.n).to(device)
    n_adaptor = sum(p.numel() for p in adaptor.parameters())
    print(f"[setup] ConditionAdaptor params: {n_adaptor:,}")

    # Trainable params: adaptor + UNet LoRA
    train_params = list(adaptor.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    print(f"[setup] total trainable params: {sum(p.numel() for p in train_params):,}")

    timesteps = sched.timesteps[0].repeat(args.batch_size).to(device)

    step = 0
    t0 = time.time()
    log_rows = []

    while step < args.steps:
        for x_batch in loader:
            if step >= args.steps:
                break
            x_batch = x_batch.to(device)
            B = x_batch.shape[0]
            if B != args.batch_size:
                continue

            id_indices = torch.randint(0, args.id_pool_size, (B,), device=device)
            sign_per_region = id_pool_t[id_indices]

            S_pixel = torch.zeros(B, 1, H, W, device=device)
            for i, (y0, y1, x0, x1) in enumerate(region_bounds):
                S_pixel[:, :, y0:y1, x0:x1] = sign_per_region[:, i].view(B, 1, 1, 1)

            # SD pipeline encoding
            x_w = sd_encode_watermark(
                x_batch, S_pixel, adaptor, vae, unet, sched,
                text_emb, vae_scale, timesteps,
            )

            distortion = F.mse_loss(x_w, x_batch)
            psnr = 10.0 * torch.log10(4.0 / (distortion + 1e-8))

            # Augment
            progress = step / max(args.steps, 1)
            x_attacked, is_diff = augment_simple(x_w, sd_pool=sd_pool, p_regen=args.p_regen)
            if not is_diff:
                x_attacked = x_attacked.detach()

            # Matched filter
            residual = x_attacked - x_batch
            rho_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_list.append(residual[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_attacked = torch.stack(rho_list, dim=1)

            residual_clean = x_w - x_batch
            rho_clean_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_clean_list.append(residual_clean[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_clean = torch.stack(rho_clean_list, dim=1)

            logit_scale = args.logit_scale
            sign_loss_att = F.mse_loss(torch.tanh(rho_attacked * logit_scale), sign_per_region)
            sign_loss_clean = F.mse_loss(torch.tanh(rho_clean * logit_scale), sign_per_region)

            # Losses
            if progress < args.warmup_frac:
                lambda_d = 0.0
                lambda_lp = 0.0
            else:
                ramp = (progress - args.warmup_frac) / (1.0 - args.warmup_frac)
                lambda_d = args.lambda_dist * ramp
                lambda_lp = args.lambda_lpips * ramp

            loss = args.att_weight * sign_loss_att + (1 - args.att_weight) * sign_loss_clean
            loss = loss + lambda_d * distortion
            if lpips_net is not None and lambda_lp > 0:
                lp = torch.mean(lpips_net(x_w, x_batch))
                loss = loss + lambda_lp * lp

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    mf_correct = ((rho_clean * sign_per_region) > 0).float().mean().item()
                    mf_att_correct = ((rho_attacked * sign_per_region) > 0).float().mean().item()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} "
                      f"sign_att={sign_loss_att.item():.4f} sign_clean={sign_loss_clean.item():.4f} "
                      f"psnr={psnr.item():.1f}dB mf_clean={mf_correct:.3f} mf_att={mf_att_correct:.3f} "
                      f"t={elapsed:.0f}s",
                      flush=True)
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "sign_att": float(sign_loss_att.item()),
                    "sign_clean": float(sign_loss_clean.item()),
                    "psnr": float(psnr.item()),
                    "mf_clean": float(mf_correct),
                    "mf_att": float(mf_att_correct),
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(adaptor, unet, args, log_rows)

            step += 1

    _save(adaptor, unet, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}")


def _save(adaptor, unet, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    save_dict = {
        "adaptor_state_dict": adaptor.state_dict(),
        "unet_lora_state_dict": {k: v for k, v in unet.named_parameters() if v.requires_grad},
        "config": vars(args),
        "n_bits": 127,
    }
    torch.save(save_dict, args.output_ckpt)
    log_path = args.output_ckpt.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--sd_residual_cache",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals")
    p.add_argument("--attack_strengths", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--id_pool_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_dist", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=0.5)
    p.add_argument("--logit_scale", type=float, default=100.0)
    p.add_argument("--warmup_frac", type=float, default=0.15)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--p_regen", type=float, default=0.30)
    p.add_argument("--att_weight", type=float, default=0.8)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
