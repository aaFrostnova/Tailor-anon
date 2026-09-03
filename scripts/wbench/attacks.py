"""Attack suite for baseline watermark evaluation.

Two families:
  - Distortion suite: clean, jpeg, gaussian blur, gaussian noise, crop+resize, resize.
  - Regeneration (W-Bench "image regeneration"): SD img2img round-trip at a given
    strength and model (sd15/sd21/sdturbo/sdxlturbo).
  - Editing (W-Bench global/local), added in editing.py.

All attacks take and return a PIL.Image (RGB), size-preserving.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageFilter

SD_MODELS = {
    "sd15": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5",
    "sd21": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1",
    "sdturbo": "/project/pi_shiqingma_umass_edu/mingzheli/model/sd-turbo",
    "sdxlturbo": "/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo",
}

_SD_PIPES = {}


def _load_sd(model_key, device="cuda"):
    if model_key not in _SD_PIPES:
        from diffusers import (StableDiffusionImg2ImgPipeline,
                               StableDiffusionXLImg2ImgPipeline, DDIMScheduler)
        path = SD_MODELS[model_key]
        if model_key == "sdxlturbo":
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                path, torch_dtype=torch.float16).to(device)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                path, torch_dtype=torch.float16, safety_checker=None).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPES[model_key] = pipe
    return _SD_PIPES[model_key]


def distortion_attack(name, pil):
    """Size-preserving distortion attacks."""
    if name == "clean":
        return pil
    if name.startswith("jpeg_"):
        q = int(name.split("_")[1])
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=q); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name.startswith("blur_"):
        r = float(name.split("_")[1])
        return pil.filter(ImageFilter.GaussianBlur(radius=r))
    if name.startswith("noise_"):
        sigma = float(name.split("_")[1]) / 1000.0  # noise_50 -> 0.05
        arr = np.asarray(pil, np.float32) / 255.0
        arr += np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * sigma
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name.startswith("crop_"):
        ratio = int(name.split("_")[1]) / 100.0
        W, H = pil.size
        cw, ch = int(W * ratio), int(H * ratio)
        left, top = (W - cw) // 2, (H - ch) // 2
        return pil.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    if name.startswith("resize_"):
        f = float(name.split("_")[1]) / 100.0  # resize_110 -> 1.10
        W, H = pil.size
        Wb, Hb = int(W * f), int(H * f)
        big = pil.resize((Wb, Hb), Image.BILINEAR)
        if f >= 1.0:
            left, top = (Wb - W) // 2, (Hb - H) // 2
            return big.crop((left, top, left + W, top + H))
        return big.resize((W, H), Image.BILINEAR)
    raise ValueError(f"unknown distortion attack {name}")


def regen_attack(name, pil, device="cuda", seed=42):
    """W-Bench image regeneration via SD img2img. name = regen_<strength*100>_<model>."""
    parts = name.split("_")
    strength = int(parts[1]) / 100.0
    model_key = parts[2] if len(parts) > 2 else "sd15"
    pipe = _load_sd(model_key, device)
    W, H = pil.size
    ew, eh = (W // 8) * 8, (H // 8) * 8
    proc = pil.resize((ew, eh), Image.LANCZOS)
    g = torch.Generator(device).manual_seed(seed)
    kw = dict(prompt="", image=proc, strength=strength, generator=g)
    if model_key == "sdxlturbo":
        kw.update(num_inference_steps=max(2, int(50 * strength)) + 1, guidance_scale=0.0)
    else:
        kw.update(num_inference_steps=50, guidance_scale=1.0)
    out = pipe(**kw).images[0]
    if out.size != (W, H):
        out = out.resize((W, H), Image.BILINEAR)
    return out


def apply_attack(name, pil, device="cuda"):
    if name.startswith("regen_"):
        return regen_attack(name, pil, device)
    return distortion_attack(name, pil)


def psnr(pil_a, pil_b):
    a = np.asarray(pil_a.convert("RGB"), np.float64) / 255.0
    b = np.asarray(pil_b.convert("RGB"), np.float64) / 255.0
    if a.shape != b.shape:
        b = np.asarray(pil_b.convert("RGB").resize(pil_a.size), np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
