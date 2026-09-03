#!/usr/bin/env python3
"""Benchmark SD-Turbo pipeline encoder with matched filter detection.

Same attack suite as benchmark_v5.py but uses SD-Turbo pipeline for embedding.
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
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.payload import BCHCodec, image_id_to_payload
from src.sign_envelope import derive_keyed_constants
from train_T_lat import make_latent_region_masks, build_id_pool

DEVICE = "cuda"
SD_TURBO_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/sd-turbo"


# ============================================================ VAE skip (same as training)

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


class ConditionAdaptor(nn.Module):
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


# ============================================================ load model

def load_model(ckpt_path):
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, AutoTokenizer
    from peft import LoraConfig

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]

    # VAE with skip connections
    vae = AutoencoderKL.from_pretrained(SD_TURBO_PATH, subfolder="vae").to(DEVICE)
    vae.encoder.forward = vae_encoder_fwd_skip.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = vae_decoder_fwd_skip.__get__(vae.decoder, vae.decoder.__class__)
    vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, 1, bias=False).to(DEVICE)
    vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, 1, bias=False).to(DEVICE)
    vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, 1, bias=False).to(DEVICE)
    vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, 1, bias=False).to(DEVICE)
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
    vae.load_state_dict(ckpt["vae_state_dict"], strict=False)
    vae.eval()

    # UNet with LoRA
    unet = UNet2DConditionModel.from_pretrained(SD_TURBO_PATH, subfolder="unet").to(DEVICE)
    unet.requires_grad_(False)
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
    lora_config = LoraConfig(r=4, init_lora_weights="gaussian",
                             target_modules=unet_targets, lora_alpha=4)
    unet.add_adapter(lora_config)
    unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
    unet.eval()

    # Text embedding
    tokenizer = AutoTokenizer.from_pretrained(SD_TURBO_PATH, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(SD_TURBO_PATH, subfolder="text_encoder").to(DEVICE)
    text_encoder.requires_grad_(False)
    tokens = tokenizer("", max_length=tokenizer.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids
    text_emb = text_encoder(tokens.to(DEVICE))[0].detach()
    del text_encoder, tokenizer

    # Scheduler
    sched = DDPMScheduler.from_pretrained(SD_TURBO_PATH, subfolder="scheduler")
    sched.set_timesteps(1, device=DEVICE)
    sched.alphas_cumprod = sched.alphas_cumprod.to(DEVICE)

    # ConditionAdaptor
    adaptor = ConditionAdaptor().to(DEVICE)
    adaptor.load_state_dict(ckpt["adaptor_state_dict"])
    adaptor.eval()

    vae_scale = float(vae.config.scaling_factor)
    return adaptor, vae, unet, text_emb, sched, vae_scale


# ============================================================ embed + decode

def pil_to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1) * 2 - 1).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t):
    arr = ((t[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_pixel_region_bounds(n_bits, H=256, W=256):
    grid = int(np.ceil(np.sqrt(n_bits)))
    cell_h, cell_w = H // grid, W // grid
    bounds = []
    for i in range(n_bits):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        bounds.append((y0, y1, x0, x1))
    return bounds


def embed_sd_turbo(pil, adaptor, vae, unet, sched, text_emb, vae_scale,
                   master_key, image_id, codec, region_bounds):
    x = pil_to_tensor(pil)
    H, W = x.shape[2], x.shape[3]

    payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    codeword = codec.encode(payload)
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)

    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    bit_at_region = codeword[inv_perm]
    sign_per_region = M.astype(np.float32) * (1.0 - 2.0 * bit_at_region.astype(np.float32))

    S_pixel = torch.zeros(1, 1, H, W, device=DEVICE)
    for i, (y0, y1, x0, x1) in enumerate(region_bounds):
        S_pixel[0, 0, y0:y1, x0:x1] = float(sign_per_region[i])

    t = sched.config.num_train_timesteps - 1
    timestep = torch.tensor([t], device=DEVICE).long()

    with torch.no_grad():
        x_cond = adaptor(x, S_pixel)
        z = vae.encode(x_cond).latent_dist.mean * vae_scale
        emb = text_emb
        model_pred = unet(z, timestep, encoder_hidden_states=emb).sample
        z_out = sched.step(model_pred[0], timestep[0], z[0], return_dict=True).prev_sample.unsqueeze(0)
        vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
        x_w = vae.decode(z_out / vae_scale).sample.clamp(-1, 1)

    return tensor_to_pil(x_w)


def decode_matched_filter(pil_orig, pil_suspect, master_key, image_id, codec, region_bounds):
    x_orig = np.asarray(pil_orig, dtype=np.float32) / 255.0 * 2 - 1
    x_susp = np.asarray(pil_suspect, dtype=np.float32) / 255.0 * 2 - 1
    x_orig = x_orig.transpose(2, 0, 1)
    x_susp = x_susp.transpose(2, 0, 1)
    residual = (x_susp - x_orig).astype(np.float32)

    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    bits = np.zeros(codec.n, dtype=np.uint8)
    for j in range(codec.n):
        r = int(perm[j])
        y0, y1, x0, x1 = region_bounds[r]
        val = float(residual[:, y0:y1, x0:x1].mean())
        region_sign = 1 if val >= 0 else -1
        decoded_sign = region_sign * int(M[r])
        bits[j] = 0 if decoded_sign > 0 else 1

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)
    bit_acc = float(np.mean(bits == expected_cw))
    payload_dec, n_err = codec.decode(bits)
    bch_ok = payload_dec is not None and np.array_equal(payload_dec, expected_payload)

    return {"bit_accuracy": bit_acc, "bch_detected": bool(bch_ok), "n_errors": int(n_err)}


# ============================================================ attacks

def compute_psnr(pil_a, pil_b):
    a = np.asarray(pil_a, dtype=np.float64) / 255.0
    b = np.asarray(pil_b, dtype=np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 60.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(pil_a, pil_b):
    try:
        from skimage.metrics import structural_similarity
        a = np.asarray(pil_a, dtype=np.float64) / 255.0
        b = np.asarray(pil_b, dtype=np.float64) / 255.0
        return structural_similarity(a, b, channel_axis=2, data_range=1.0)
    except ImportError:
        return float("nan")


def tpr_threshold(n_bits=127, fpr=0.01):
    from scipy.stats import binom
    for k in range(n_bits, n_bits // 2, -1):
        if binom.sf(k - 1, n_bits, 0.5) > fpr:
            return (k + 1) / n_bits
    return 0.5


_SD_PIPE = None

def load_sd_regen():
    global _SD_PIPE
    if _SD_PIPE is None:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        SD_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_PATH, torch_dtype=torch.float16, safety_checker=None,
        ).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPE = pipe
    return _SD_PIPE


def apply_attack(name, pil):
    if name == "clean":
        return pil
    if name == "jpeg_50":
        from io import BytesIO
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=50); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "jpeg_30":
        from io import BytesIO
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=30); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "blur_1.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name == "blur_2.5":
        return pil.filter(ImageFilter.GaussianBlur(radius=2.5))
    if name == "noise_003":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = arr + np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.03
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "noise_005":
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = arr + np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * 0.05
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name == "crop_70":
        W, H = pil.size
        cw, ch = int(W * 0.7), int(H * 0.7)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    if name.startswith("regen_"):
        strength = int(name.split("_")[1]) / 100.0
        pipe = load_sd_regen()
        g = torch.Generator(DEVICE).manual_seed(42)
        out = pipe(prompt="", image=pil, strength=strength,
                   num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
        if out.size != pil.size:
            out = out.resize(pil.size, Image.BILINEAR)
        return out
    raise ValueError(f"Unknown attack: {name}")


# ============================================================ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--src_dir", required=True)
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--attack_set", default="all",
                   choices=["classical", "regen", "all"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_key = args.master_key.encode("utf-8")
    codec = BCHCodec()
    thr = tpr_threshold(codec.n, fpr=0.01)
    thr_01 = tpr_threshold(codec.n, fpr=0.001)
    print(f"[setup] BCH(n={codec.n}, k={codec.data_bits}, t={codec.t})")

    adaptor, vae, unet, text_emb, sched, vae_scale = load_model(args.ckpt)
    print(f"[setup] SD-Turbo encoder loaded from {args.ckpt}")

    region_bounds = build_pixel_region_bounds(codec.n, args.resolution, args.resolution)

    src_dir = Path(args.src_dir)
    image_files = sorted([f for f in src_dir.iterdir()
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.n_images]
    print(f"[setup] {len(image_files)} test images")

    ATTACKS_ALL = ["clean", "jpeg_50", "jpeg_30", "blur_1.5", "blur_2.5",
                   "noise_003", "noise_005", "crop_70",
                   "regen_010", "regen_020", "regen_030", "regen_040"]
    ATTACKS_CL = ["clean", "jpeg_50", "jpeg_30", "blur_1.5", "blur_2.5",
                  "noise_003", "noise_005", "crop_70"]
    ATTACKS_RE = ["regen_010", "regen_020", "regen_030", "regen_040"]
    attacks = {"all": ATTACKS_ALL, "classical": ATTACKS_CL, "regen": ATTACKS_RE}[args.attack_set]
    print(f"[setup] attacks: {attacks}")

    results_per_attack = {a: [] for a in attacks}
    psnr_list, ssim_list = [], []

    for idx, fp in enumerate(image_files):
        pil_clean = Image.open(fp).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS)
        image_id = f"bench_{idx:05d}"

        pil_wm = embed_sd_turbo(pil_clean, adaptor, vae, unet, sched, text_emb, vae_scale,
                                master_key, image_id, codec, region_bounds)

        psnr_val = compute_psnr(pil_clean, pil_wm)
        ssim_val = compute_ssim(pil_clean, pil_wm)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        for atk_name in attacks:
            pil_attacked = apply_attack(atk_name, pil_wm)
            res = decode_matched_filter(pil_clean, pil_attacked, master_key, image_id,
                                        codec, region_bounds)
            res["tpr_1pct"] = res["bit_accuracy"] >= thr
            res["tpr_01pct"] = res["bit_accuracy"] >= thr_01
            res["image"] = fp.name
            results_per_attack[atk_name].append(res)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(image_files)}] psnr={psnr_val:.1f}dB", flush=True)

    # Report
    summary = {
        "method": "sd_turbo_encoder_mf",
        "ckpt": args.ckpt,
        "n_images": len(image_files),
        "quality": {
            "psnr_mean": float(np.mean(psnr_list)),
            "psnr_std": float(np.std(psnr_list)),
            "ssim_mean": float(np.mean(ssim_list)),
            "ssim_std": float(np.std(ssim_list)),
        },
        "attacks": {},
    }

    print(f"\n{'='*70}")
    print(f"  PSNR:    {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB")
    print(f"  SSIM:    {np.mean(ssim_list):.4f}")
    print(f"{'='*70}")
    print(f"  {'Attack':<15s} {'bit_acc':>8s} {'BCH_det':>8s} {'TPR@1%':>8s} {'TPR@0.1%':>9s}")
    print(f"  {'-'*50}")

    for atk_name in attacks:
        runs = results_per_attack[atk_name]
        bit_acc = np.mean([r["bit_accuracy"] for r in runs])
        bch_det = np.mean([float(r["bch_detected"]) for r in runs])
        tpr1 = np.mean([float(r["tpr_1pct"]) for r in runs])
        tpr01 = np.mean([float(r["tpr_01pct"]) for r in runs])
        summary["attacks"][atk_name] = {
            "bit_accuracy_mean": float(bit_acc),
            "bch_detection_rate": float(bch_det),
            "tpr_at_1pct_fpr": float(tpr1),
            "tpr_at_01pct_fpr": float(tpr01),
        }
        print(f"  {atk_name:<15s} {bit_acc:>8.3f} {bch_det*100:>7.1f}% {tpr1*100:>7.1f}% {tpr01*100:>8.1f}%")

    print(f"{'='*70}\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "results_full.json", "w") as f:
        json.dump(results_per_attack, f, indent=2)
    print(f"[done] results -> {out_dir}")


if __name__ == "__main__":
    main()
