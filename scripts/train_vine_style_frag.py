"""VINE-style encoder + multi-fragment crypto payload + learned decoder.

Same neural architecture as train_vine_style.py. Only change:
  - FragmentedCodec (4 × BCH(31,16,t=3)) instead of single BCH(127,64,t=10)
  - Per-fragment crypto derivation: independent (σ_k, M_k) per fragment
  - Per-fragment bit accuracy logging during training

Neural pipeline is identical: ConditionAdaptor → VAE → UNet → VAE(skip) → ConvNeXt decoder.
The fragmentation is transparent to the neural network (still 127 bits).
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

from src.payload import image_id_to_payload
from src.fragment_payload import FragmentedCodec
from src.fragment_crypto import build_frag_id_pool
from train_T_lat import (
    make_latent_region_masks,
    load_sd_residual_pool, FlatImageDataset,
)

SD_TURBO_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/sd-turbo"


# ============================================================ VAE skip connections

def vae_encoder_fwd_skip(self, sample):
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


def vae_decoder_fwd_skip(self, sample, latent_embeds=None):
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
    """VINE-style: 127 bits → MLP → dense feature → fuse with image."""
    def __init__(self, n_bits=127):
        super().__init__()
        self.secret_dense1 = nn.Linear(n_bits, 64 * 64)
        self.secret_act1 = nn.ReLU()
        self.secret_dense2 = nn.Linear(64 * 64, 3 * 64 * 64)
        self.secret_act2 = nn.ReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 6, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, image, secret_bits):
        """image: (B,3,256,256), secret_bits: (B,127) floats in {0,1}."""
        s = 2 * (secret_bits - 0.5)
        s = self.secret_act1(self.secret_dense1(s))
        s = self.secret_act2(self.secret_dense2(s))
        s = s.reshape(-1, 3, 64, 64)
        s = F.interpolate(s, size=(image.shape[2], image.shape[3]), mode="nearest")
        fused = self.conv1(torch.cat([s, image], dim=1))
        return self.conv2(fused)


class WatermarkDecoder(nn.Module):
    """ConvNeXt-Base decoder: image → n_bits logits."""
    def __init__(self, n_bits=127):
        super().__init__()
        from torchvision import models
        self.backbone = models.convnext_base(weights=None)
        self.backbone.classifier.append(nn.Linear(1000, n_bits))
        self.backbone.classifier.append(nn.Sigmoid())

    def forward(self, x):
        x = (x + 1) / 2  # [-1,1] → [0,1]
        return self.backbone(x)


# ============================================================ SD pipeline

def load_sd_turbo(device):
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, AutoTokenizer
    from peft import LoraConfig

    vae = AutoencoderKL.from_pretrained(SD_TURBO_PATH, subfolder="vae").to(device)
    vae.encoder.forward = vae_encoder_fwd_skip.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = vae_decoder_fwd_skip.__get__(vae.decoder, vae.decoder.__class__)
    vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, 1, bias=False).to(device)
    vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, 1, bias=False).to(device)
    vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, 1, bias=False).to(device)
    vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, 1, bias=False).to(device)
    for sc in [vae.decoder.skip_conv_1, vae.decoder.skip_conv_2,
               vae.decoder.skip_conv_3, vae.decoder.skip_conv_4]:
        nn.init.constant_(sc.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    vae.requires_grad_(False)
    vae.train()
    vae_lora = LoraConfig(r=4, init_lora_weights="gaussian",
                          target_modules=["conv1", "conv2", "conv_in", "conv_shortcut",
                                          "conv_out", "to_k", "to_q", "to_v", "to_out.0"],
                          lora_alpha=4)
    vae.add_adapter(vae_lora, adapter_name="vae_skip")
    for sc in [vae.decoder.skip_conv_1, vae.decoder.skip_conv_2,
               vae.decoder.skip_conv_3, vae.decoder.skip_conv_4]:
        sc.requires_grad_(True)

    unet = UNet2DConditionModel.from_pretrained(SD_TURBO_PATH, subfolder="unet").to(device)
    unet.requires_grad_(False)
    unet.train()
    unet_targets = []
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pat in ["to_k", "to_q", "to_v", "to_out.0", "conv1", "conv2",
                    "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in",
                    "ff.net.2", "ff.net.0.proj"]:
            if pat in n:
                unet_targets.append(n.replace(".weight", ""))
                break
    unet_lora = LoraConfig(r=4, init_lora_weights="gaussian",
                           target_modules=unet_targets, lora_alpha=4)
    unet.add_adapter(unet_lora)

    tokenizer = AutoTokenizer.from_pretrained(SD_TURBO_PATH, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(SD_TURBO_PATH, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    tokens = tokenizer("", max_length=tokenizer.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids
    text_emb = text_encoder(tokens.to(device))[0].detach()
    del text_encoder, tokenizer

    sched = DDPMScheduler.from_pretrained(SD_TURBO_PATH, subfolder="scheduler")
    sched.set_timesteps(1, device=device)
    sched.alphas_cumprod = sched.alphas_cumprod.to(device)

    vae_scale = float(vae.config.scaling_factor)
    return vae, unet, text_emb, sched, vae_scale


def sd_encode(image, secret_bits, adaptor, vae, unet, sched, text_emb, vae_scale):
    B = image.shape[0]
    t = sched.config.num_train_timesteps - 1
    timesteps = torch.tensor([t] * B, device=image.device).long()

    x_cond = adaptor(image, secret_bits)
    z = vae.encode(x_cond).latent_dist.mean * vae_scale
    model_pred = unet(z, timesteps, encoder_hidden_states=text_emb.repeat(B, 1, 1)).sample
    z_out = torch.stack([
        sched.step(model_pred[i], timesteps[i], z[i], return_dict=True).prev_sample
        for i in range(B)
    ])
    vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
    x_w = vae.decode(z_out / vae_scale).sample.clamp(-1, 1)
    return x_w


# ============================================================ augmentation

def augment(x_w, sd_pool=None, p_regen=0.30):
    B = x_w.shape[0]
    r = np.random.random()
    if sd_pool is not None and r < p_regen:
        idx = torch.randint(0, sd_pool.shape[0], (B,))
        return (x_w + sd_pool[idx].to(x_w.device)).detach()
    if r < p_regen + 0.10:
        return x_w
    if r < p_regen + 0.20:
        sigma = np.random.uniform(0.01, 0.05)
        return x_w + torch.randn_like(x_w) * sigma
    if r < p_regen + 0.30:
        from torchvision.transforms.functional import gaussian_blur
        k = int(np.random.choice([3, 5, 7]))
        return gaussian_blur(x_w, kernel_size=k, sigma=float(np.random.uniform(0.5, 2.0)))
    if r < p_regen + 0.40:
        from io import BytesIO
        from PIL import Image
        import torchvision.transforms.functional as TF
        q = np.random.randint(30, 80)
        out = []
        for i in range(B):
            pil = TF.to_pil_image(((x_w[i].detach().cpu() + 1) / 2).clamp(0, 1))
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=q); buf.seek(0)
            out.append(TF.to_tensor(Image.open(buf).convert("RGB")) * 2 - 1)
        return torch.stack(out).to(x_w.device)
    return x_w + torch.randn_like(x_w) * 0.01


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    codec = FragmentedCodec(K=args.num_fragments, frag_t=args.frag_t)
    master_key = args.master_key.encode("utf-8")
    id_pool_signs = build_frag_id_pool(master_key, args.id_pool_size, codec)
    id_pool_t = torch.from_numpy(id_pool_signs).to(device)
    id_pool_bits = ((1 - id_pool_t) / 2).float()
    print(f"[setup] FragmentedCodec: K={codec.K}, frag_n={codec.frag_n}, "
          f"frag_t={codec.t}, total_n={codec.n}, data_bits={codec.data_bits}")

    vae, unet, text_emb, sched, vae_scale = load_sd_turbo(device)
    print(f"[setup] SD-Turbo loaded, vae_scale={vae_scale:.4f}")
    print(f"[setup] UNet LoRA: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    print(f"[setup] VAE LoRA+skip: {sum(p.numel() for p in vae.parameters() if p.requires_grad):,}")

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

    adaptor = ConditionAdaptor(n_bits=codec.n).to(device)
    decoder = WatermarkDecoder(n_bits=codec.n).to(device)
    print(f"[setup] ConditionAdaptor: {sum(p.numel() for p in adaptor.parameters()):,}")
    print(f"[setup] Decoder (ConvNeXt): {sum(p.numel() for p in decoder.parameters()):,}")

    train_params = list(adaptor.parameters()) + list(decoder.parameters())
    train_params += [p for p in unet.parameters() if p.requires_grad]
    train_params += [p for p in vae.parameters() if p.requires_grad]
    print(f"[setup] total trainable: {sum(p.numel() for p in train_params):,}")

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss()

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
            target_bits = id_pool_bits[id_indices]  # (B, 127) in {0, 1}

            # Encode
            x_w = sd_encode(x_batch, target_bits, adaptor, vae, unet, sched,
                           text_emb, vae_scale)

            distortion = F.mse_loss(x_w, x_batch)
            psnr = 10.0 * torch.log10(4.0 / (distortion + 1e-8))

            # Augment
            x_attacked = augment(x_w, sd_pool=sd_pool, p_regen=args.p_regen)

            # Decode
            pred_bits_att = decoder(x_attacked)
            pred_bits_clean = decoder(x_w)

            secret_loss_att = bce(pred_bits_att, target_bits)
            secret_loss_clean = bce(pred_bits_clean, target_bits)

            bit_acc_att = ((pred_bits_att > 0.5).float() == target_bits).float().mean().item()
            bit_acc_clean = ((pred_bits_clean > 0.5).float() == target_bits).float().mean().item()

            # Loss schedule (VINE-style: secret-only warmup)
            progress = step / max(args.steps, 1)
            no_im_loss = step < args.no_im_loss_steps

            if no_im_loss:
                loss = args.secret_loss_scale * secret_loss_att
            else:
                l2_scale = min(args.l2_loss_scale * step / max(args.l2_loss_ramp, 1), args.l2_loss_scale)
                lp_scale = min(args.lpips_loss_scale * step / max(args.lpips_loss_ramp, 1), args.lpips_loss_scale)
                loss = args.secret_loss_scale * (0.8 * secret_loss_att + 0.2 * secret_loss_clean)
                loss = loss + l2_scale * distortion
                if lpips_net is not None and lp_scale > 0:
                    loss = loss + lp_scale * torch.mean(lpips_net(x_w, x_batch))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                # Per-fragment bit accuracy
                frag_accs = []
                with torch.no_grad():
                    for k in range(codec.K):
                        s, e = codec.frag_range(k)
                        fk = ((pred_bits_clean[:, s:e] > 0.5).float() == target_bits[:, s:e]).float().mean().item()
                        frag_accs.append(fk)
                frag_str = " ".join([f"f{k}={a:.2f}" for k, a in enumerate(frag_accs)])
                print(f"[step {step:6d}] loss={loss.item():.4f} "
                      f"sec_att={secret_loss_att.item():.4f} "
                      f"psnr={psnr.item():.1f}dB "
                      f"acc_att={bit_acc_att:.3f} acc_cln={bit_acc_clean:.3f} "
                      f"[{frag_str}] "
                      f"t={elapsed:.0f}s", flush=True)
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "secret_loss_att": float(secret_loss_att.item()),
                    "secret_loss_clean": float(secret_loss_clean.item()),
                    "psnr": float(psnr.item()),
                    "bit_acc_att": float(bit_acc_att),
                    "bit_acc_clean": float(bit_acc_clean),
                    "frag_accs": frag_accs,
                    "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(adaptor, decoder, unet, vae, args, log_rows)

            step += 1

    _save(adaptor, decoder, unet, vae, args, log_rows)
    print(f"\n[done] saved -> {args.output_ckpt}")


def _save(adaptor, decoder, unet, vae, args, log_rows):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({
        "adaptor_state_dict": adaptor.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "unet_state_dict": unet.state_dict(),
        "vae_state_dict": vae.state_dict(),
        "config": vars(args),
        "n_bits": 127,
        "codec_type": "fragmented",
        "fragment_K": args.num_fragments,
        "frag_n": 31,
        "frag_t": args.frag_t,
    }, args.output_ckpt)
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
    # VINE-style loss scales
    p.add_argument("--secret_loss_scale", type=float, default=1.5)
    p.add_argument("--l2_loss_scale", type=float, default=2.0)
    p.add_argument("--l2_loss_ramp", type=int, default=10000)
    p.add_argument("--lpips_loss_scale", type=float, default=1.5)
    p.add_argument("--lpips_loss_ramp", type=int, default=10000)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--no_im_loss_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--p_regen", type=float, default=0.30)
    p.add_argument("--num_fragments", type=int, default=4)
    p.add_argument("--frag_t", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
