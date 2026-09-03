"""Train SDXL-Turbo watermark encoder with skip connections + VAE finetune.

Full VINE-style architecture on SDXL-Turbo, with matched filter detection:
  1. ConditionAdaptor: fuse(image, S_pixel) → conditioned image
  2. VAE encode (with skip connection capture) → latent z
  3. UNet 1-step denoise → modified latent
  4. VAE decode (with skip connections from encoder) → watermarked image x_w

Improvements over v1:
  - SDXL-Turbo (1-step optimized) instead of SD-v1.5
  - VAE skip connections (encoder features → decoder)
  - VAE finetuning (full, not frozen)
  - LPIPS + MSE quality loss
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

SDXL_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo"


# ============================================================ VAE with skip connections (from VINE)

def vae_encoder_fwd_with_skip(self, sample):
    """Modified VAE encoder forward that saves intermediate features."""
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def vae_decoder_fwd_with_skip(self, sample, latent_embeds=None):
    """Modified VAE decoder forward that uses skip connections."""
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


# ============================================================ modules

class ConditionAdaptor(nn.Module):
    """Fuse sign envelope with input image."""
    def __init__(self):
        super().__init__()
        self.sign_dense = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, image, S_pixel):
        sign_feat = self.sign_dense(S_pixel)
        return self.fuse(torch.cat([image, sign_feat], dim=1))


def load_sdxl_turbo(device):
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer

    # VAE with skip connections
    vae = AutoencoderKL.from_pretrained(SDXL_PATH, subfolder="vae").to(device)
    vae.encoder.forward = vae_encoder_fwd_with_skip.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = vae_decoder_fwd_with_skip.__get__(vae.decoder, vae.decoder.__class__)

    # Add skip connection convs (VINE-style, initialized near-zero)
    block_out_ch = vae.config.block_out_channels  # [128, 256, 512, 512]
    vae.decoder.skip_conv_1 = nn.Conv2d(block_out_ch[3], 512, 1, bias=False).to(device)
    vae.decoder.skip_conv_2 = nn.Conv2d(block_out_ch[2], 512, 1, bias=False).to(device)
    vae.decoder.skip_conv_3 = nn.Conv2d(block_out_ch[1], 512, 1, bias=False).to(device)
    vae.decoder.skip_conv_4 = nn.Conv2d(block_out_ch[0], 256, 1, bias=False).to(device)
    for sc in [vae.decoder.skip_conv_1, vae.decoder.skip_conv_2, vae.decoder.skip_conv_3, vae.decoder.skip_conv_4]:
        nn.init.constant_(sc.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    vae.requires_grad_(True)
    vae.train()

    # UNet with LoRA
    unet = UNet2DConditionModel.from_pretrained(SDXL_PATH, subfolder="unet").to(device)
    unet.requires_grad_(False)
    unet.train()
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

    # Text embeddings (empty prompt)
    tokenizer = AutoTokenizer.from_pretrained(SDXL_PATH, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(SDXL_PATH, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    tokens = tokenizer("", max_length=tokenizer.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids
    text_emb_1 = text_encoder(tokens.to(device)).last_hidden_state.detach()

    tokenizer_2 = AutoTokenizer.from_pretrained(SDXL_PATH, subfolder="tokenizer_2", use_fast=False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(SDXL_PATH, subfolder="text_encoder_2").to(device)
    text_encoder_2.requires_grad_(False)
    tokens_2 = tokenizer_2("", max_length=tokenizer_2.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids
    out_2 = text_encoder_2(tokens_2.to(device))
    text_emb_2 = out_2.last_hidden_state.detach()
    pooled_emb = out_2.text_embeds.detach()

    text_emb = torch.cat([text_emb_1, text_emb_2], dim=-1)
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2

    # Scheduler (1-step)
    from diffusers import DDIMScheduler
    sched = DDIMScheduler.from_pretrained(SDXL_PATH, subfolder="scheduler")
    sched.set_timesteps(1, device=device)

    vae_scale = float(vae.config.scaling_factor)
    n_vae = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"[setup] VAE trainable params: {n_vae:,}")

    return vae, unet, text_emb, pooled_emb, sched, vae_scale


def sd_encode(image, S_pixel, adaptor, vae, unet, sched,
              text_emb, pooled_emb, vae_scale, timesteps):
    B = image.shape[0]
    x_cond = adaptor(image, S_pixel)

    # VAE encode (saves skip features in vae.encoder.current_down_blocks)
    z = vae.encode(x_cond).latent_dist.mean * vae_scale

    # UNet 1-step
    emb = text_emb.repeat(B, 1, 1)
    added_cond = {
        "text_embeds": pooled_emb.repeat(B, 1),
        "time_ids": torch.tensor([[256, 256, 0, 0, 256, 256]], device=image.device, dtype=image.dtype).repeat(B, 1),
    }
    model_pred = unet(z, timesteps[0], encoder_hidden_states=emb,
                      added_cond_kwargs=added_cond).sample

    # Scheduler step
    z_out = sched.step(model_pred, timesteps[0], z, return_dict=True).prev_sample

    # VAE decode with skip connections
    vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
    x_w = vae.decode(z_out / vae_scale).sample.clamp(-1, 1)
    return x_w


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

    vae, unet, text_emb, pooled_emb, sched, vae_scale = load_sdxl_turbo(device)
    timesteps = sched.timesteps.to(device)
    print(f"[setup] SDXL-Turbo loaded, vae_scale={vae_scale:.4f}, timesteps={timesteps.tolist()}")

    sd_pool = None
    if args.sd_residual_cache:
        sd_pool = load_sd_residual_pool(
            args.sd_residual_cache, strengths_to_load=args.attack_strengths,
        )
        if sd_pool is not None:
            sd_pool = sd_pool.to(device)
            print(f"[setup] SD residual pool: {sd_pool.shape}")

    lpips_net = None
    if args.lambda_lpips > 0:
        import lpips
        lpips_net = lpips.LPIPS(net='vgg').to(device)
        lpips_net.requires_grad_(False)
        print("[setup] LPIPS loaded")

    ds = FlatImageDataset(args.image_dir, args.resolution, args.max_images)
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    adaptor = ConditionAdaptor().to(device)
    print(f"[setup] ConditionAdaptor params: {sum(p.numel() for p in adaptor.parameters()):,}")

    # All trainable: adaptor + UNet LoRA + VAE (full)
    train_params = list(adaptor.parameters())
    train_params += [p for p in unet.parameters() if p.requires_grad]
    train_params += list(vae.parameters())
    print(f"[setup] total trainable params: {sum(p.numel() for p in train_params):,}")

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

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

            x_w = sd_encode(x_batch, S_pixel, adaptor, vae, unet, sched,
                           text_emb, pooled_emb, vae_scale, timesteps)

            distortion = F.mse_loss(x_w, x_batch)
            psnr = 10.0 * torch.log10(4.0 / (distortion + 1e-8))

            # Augment (simple: regen residual or identity)
            r = np.random.random()
            if sd_pool is not None and r < args.p_regen:
                idx = torch.randint(0, sd_pool.shape[0], (B,))
                x_attacked = (x_w + sd_pool[idx].to(device)).detach()
                is_diff = False
            else:
                x_attacked = x_w
                is_diff = True

            # Matched filter
            residual = x_attacked - x_batch
            rho_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_list.append(residual[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_att = torch.stack(rho_list, dim=1)

            residual_clean = x_w - x_batch
            rho_clean_list = []
            for y0, y1, x0, x1 in region_bounds:
                rho_clean_list.append(residual_clean[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3)))
            rho_clean = torch.stack(rho_clean_list, dim=1)

            logit_scale = args.logit_scale
            sign_loss_att = F.mse_loss(torch.tanh(rho_att * logit_scale), sign_per_region)
            sign_loss_clean = F.mse_loss(torch.tanh(rho_clean * logit_scale), sign_per_region)

            progress = step / max(args.steps, 1)
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
                loss = loss + lambda_lp * torch.mean(lpips_net(x_w, x_batch))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    mf_c = ((rho_clean * sign_per_region) > 0).float().mean().item()
                    mf_a = ((rho_att * sign_per_region) > 0).float().mean().item()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss={loss.item():.4f} "
                      f"sign_att={sign_loss_att.item():.4f} sign_clean={sign_loss_clean.item():.4f} "
                      f"psnr={psnr.item():.1f}dB mf_clean={mf_c:.3f} mf_att={mf_a:.3f} "
                      f"t={elapsed:.0f}s", flush=True)
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "sign_att": float(sign_loss_att.item()),
                    "sign_clean": float(sign_loss_clean.item()),
                    "psnr": float(psnr.item()),
                    "mf_clean": float(mf_c), "mf_att": float(mf_a),
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(adaptor, unet, vae, args, log_rows)

            step += 1

    _save(adaptor, unet, vae, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}")


def _save(adaptor, unet, vae, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    save_dict = {
        "adaptor_state_dict": adaptor.state_dict(),
        "vae_state_dict": vae.state_dict(),
        "config": vars(args),
        "n_bits": 127,
    }
    # Save UNet LoRA weights only
    lora_state = {}
    for n, p in unet.named_parameters():
        if p.requires_grad:
            lora_state[n] = p.data
    save_dict["unet_lora_state_dict"] = lora_state
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
    p.add_argument("--warmup_frac", type=float, default=0.10)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--p_regen", type=float, default=0.30)
    p.add_argument("--att_weight", type=float, default=0.8)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
