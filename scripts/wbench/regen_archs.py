"""Multi-architecture image-regeneration attacks.

Each class exposes .name, .arch, and .regen(pil)->pil (size-preserving), so a
watermarked image can be re-synthesised by a model of a DIFFERENT architecture.
The point is architecture diversity: a watermark may survive one generative
mechanism's reconstruction but not another's (continuous VAE vs discrete VQ vs
iterative diffusion vs consistency vs unCLIP cascade).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SD_MODELS = {
    "sd15": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5",
    "sd21": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1",
    "sdxlturbo": "/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo",
}


def _to_size(out, size):
    return out.resize(size, Image.BILINEAR) if out.size != size else out


# ---------------------------------------------------------- iterative latent diffusion (UNet)
class SDImg2Img:
    def __init__(self, model_key="sd15", strength=0.4, device="cuda"):
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        self.name = f"SD-img2img-{model_key}-s{strength}"
        self.arch = "latent-diffusion-UNet"
        self.strength = strength
        self.device = device
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODELS[model_key], torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def regen(self, pil):
        W, H = pil.size
        proc = pil.resize(((W // 8) * 8, (H // 8) * 8), Image.LANCZOS)
        g = torch.Generator(self.device).manual_seed(42)
        out = self.pipe(prompt="", image=proc, strength=self.strength,
                        num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
        return _to_size(out, (W, H))


# ---------------------------------------------------------- distilled SDXL (few-step UNet)
class SDXLTurboImg2Img:
    def __init__(self, strength=0.4, device="cuda"):
        from diffusers import StableDiffusionXLImg2ImgPipeline
        self.name = f"SDXL-Turbo-s{strength}"
        self.arch = "distilled-SDXL-UNet"
        self.strength = strength
        self.device = device
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SD_MODELS["sdxlturbo"], torch_dtype=torch.float16).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def regen(self, pil):
        W, H = pil.size
        proc = pil.resize((512, 512), Image.LANCZOS)
        g = torch.Generator(self.device).manual_seed(42)
        steps = max(2, int(round(4 / max(self.strength, 1e-3))))
        out = self.pipe(prompt="", image=proc, strength=self.strength,
                        num_inference_steps=steps, guidance_scale=0.0, generator=g).images[0]
        return _to_size(out, (W, H))


# ---------------------------------------------------------- continuous VAE round-trip (no diffusion)
class VAERoundTrip:
    def __init__(self, device="cuda", subfolder_model=None):
        from diffusers import AutoencoderKL
        self.name = "VAE-roundtrip"
        self.arch = "continuous-VAE-autoencoder"
        self.device = device
        path = subfolder_model or SD_MODELS["sd15"]
        self.vae = AutoencoderKL.from_pretrained(path, subfolder="vae",
                                                 torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def regen(self, pil):
        W, H = pil.size
        ew, eh = (W // 8) * 8, (H // 8) * 8
        x = torch.from_numpy(np.asarray(pil.resize((ew, eh), Image.LANCZOS), np.float32)
                             ).permute(2, 0, 1)[None].to(self.device) / 255.0 * 2 - 1
        x = x.half()
        lat = self.vae.encode(x).latent_dist.sample()
        rec = self.vae.decode(lat).sample
        rec = ((rec.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return _to_size(Image.fromarray(rec), (W, H))


# ---------------------------------------------------------- latent consistency model
class LCMImg2Img:
    def __init__(self, strength=0.4, device="cuda"):
        from diffusers import AutoPipelineForImage2Image
        self.name = f"LCM-s{strength}"
        self.arch = "latent-consistency"
        self.strength = strength
        self.device = device
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def regen(self, pil):
        W, H = pil.size
        proc = pil.resize((512, 512), Image.LANCZOS)
        g = torch.Generator(self.device).manual_seed(42)
        out = self.pipe(prompt="", image=proc, strength=self.strength,
                        num_inference_steps=4, guidance_scale=1.0, generator=g).images[0]
        return _to_size(out, (W, H))


# ---------------------------------------------------------- unCLIP cascade (Kandinsky 2.2)
class KandinskyImg2Img:
    def __init__(self, strength=0.4, device="cuda"):
        from diffusers import KandinskyV22Img2ImgPipeline, KandinskyV22PriorPipeline
        self.name = f"Kandinsky2.2-s{strength}"
        self.arch = "unCLIP-cascade"
        self.strength = strength
        self.device = device
        self.prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16).to(device)
        self.dec = KandinskyV22Img2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16).to(device)
        self.prior.set_progress_bar_config(disable=True)
        self.dec.set_progress_bar_config(disable=True)

    def regen(self, pil):
        W, H = pil.size
        proc = pil.resize((512, 512), Image.LANCZOS)
        g = torch.Generator(self.device).manual_seed(42)
        img_emb, neg_emb = self.prior("", generator=g).to_tuple()
        out = self.dec(image=proc, image_embeds=img_emb, negative_image_embeds=neg_emb,
                       strength=self.strength, num_inference_steps=50, height=512, width=512,
                       generator=g).images[0]
        return _to_size(out, (W, H))


# ---------------------------------------------------------- high-fidelity 16-channel VAE round-trip
class HiFiVAERoundTrip:
    """16-channel VAE round-trip (non-gated FLUX/SD3-class mirror). ~27 dB —
    the highest-fidelity regeneration we can run without gated FLUX/SD3 weights.
    Optional latent noise injects a mild 'regeneration' beyond pure reconstruction."""

    def __init__(self, device="cuda", repo="madebyollin/taef1", latent_noise=0.0):
        from diffusers import AutoencoderKL
        self.name = f"HiFiVAE16ch{'-n' + str(latent_noise) if latent_noise else ''}"
        self.arch = "16ch-VAE-autoencoder"
        self.device = device
        self.latent_noise = latent_noise
        # repo overridden by caller; default tries a 16ch VAE.
        self.vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def regen(self, pil):
        W, H = pil.size
        ew, eh = (W // 16) * 16, (H // 16) * 16
        x = torch.from_numpy(np.asarray(pil.resize((ew, eh), Image.LANCZOS), np.float32)
                             ).permute(2, 0, 1)[None].to(self.device).half() / 255.0 * 2 - 1
        lat = self.vae.encode(x).latent_dist.mode()
        if self.latent_noise > 0:
            lat = lat + torch.randn_like(lat) * self.latent_noise
        rec = self.vae.decode(lat).sample
        rec = ((rec.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return _to_size(Image.fromarray(rec), (W, H))


# ---------------------------------------------------------- discrete VQ autoencoder round-trip
class VQRoundTrip:
    def __init__(self, device="cuda", repo="CompVis/ldm-super-resolution-4x-openimages"):
        from diffusers import VQModel
        self.name = "VQGAN-roundtrip"
        self.arch = "discrete-VQ-autoencoder"
        self.device = device
        # VQModel weights from an LDM repo (vqvae subfolder); discrete codebook quantization.
        self.vq = VQModel.from_pretrained(repo, subfolder="vqvae",
                                          torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def regen(self, pil):
        W, H = pil.size
        ew, eh = (W // 4) * 4, (H // 4) * 4
        x = torch.from_numpy(np.asarray(pil.resize((ew, eh), Image.LANCZOS), np.float32)
                             ).permute(2, 0, 1)[None].to(self.device) / 255.0 * 2 - 1
        x = x.half()
        h = self.vq.encode(x).latents
        rec = self.vq.decode(h).sample
        rec = ((rec.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return _to_size(Image.fromarray(rec), (W, H))


FLUX_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/FLUX.1-dev"


# ---------------------------------------------------------- FLUX 16-channel VAE round-trip
class FluxVAERoundTrip:
    """FLUX.1 16-channel VAE encode->decode round-trip: the highest-fidelity
    reconstruction bottleneck available (true gated 16ch VAE, ~27-31 dB), no diffusion."""

    def __init__(self, device="cuda", repo=FLUX_PATH, latent_noise=0.0):
        from diffusers import AutoencoderKL
        self.name = f"FLUX-VAE16ch{'-n' + str(latent_noise) if latent_noise else ''}"
        self.arch = "16ch-VAE-FLUX"
        self.device = device
        self.latent_noise = latent_noise
        self.vae = AutoencoderKL.from_pretrained(
            repo, subfolder="vae", torch_dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def regen(self, pil):
        W, H = pil.size
        ew, eh = (W // 16) * 16, (H // 16) * 16
        x = torch.from_numpy(np.asarray(pil.resize((ew, eh), Image.LANCZOS), np.float32)
                             ).permute(2, 0, 1)[None].to(self.device).bfloat16() / 255.0 * 2 - 1
        lat = self.vae.encode(x).latent_dist.mode()
        if self.latent_noise > 0:
            lat = lat + torch.randn_like(lat) * self.latent_noise
        rec = self.vae.decode(lat).sample
        rec = ((rec.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return _to_size(Image.fromarray(rec), (W, H))


# ---------------------------------------------------------- FLUX flow-matching img2img regeneration
class FluxImg2Img:
    """FLUX.1-dev img2img regeneration (12B flow-matching transformer). At low strength
    this is a high-fidelity generative re-synthesis -- the strongest imperceptible eraser."""

    def __init__(self, strength=0.3, steps=28, guidance=3.5, device="cuda", repo=FLUX_PATH):
        from diffusers import FluxImg2ImgPipeline
        self.name = f"FLUX-img2img-s{strength}"
        self.arch = "flow-matching-FLUX"
        self.strength = strength; self.steps = steps; self.guidance = guidance
        self.device = device
        self.pipe = FluxImg2ImgPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def regen(self, pil, strength=None):
        s = self.strength if strength is None else strength
        W, H = pil.size
        proc = pil.resize(((W // 16) * 16, (H // 16) * 16), Image.LANCZOS)
        g = torch.Generator("cpu").manual_seed(42)
        # FLUX needs enough steps that strength*steps >= ~1 even at low strength
        steps = max(self.steps, int(np.ceil(1.0 / max(s, 1e-3))) + 1)
        out = self.pipe(prompt="", image=proc, strength=s,
                        num_inference_steps=steps, guidance_scale=self.guidance,
                        generator=g).images[0]
        return _to_size(out, (W, H))


def build_regens(specs, device="cuda"):
    """specs: list of (key, kwargs). Returns {name: obj}, skipping failures."""
    factory = {
        "flux_vae": lambda **k: FluxVAERoundTrip(**k),
        "flux_i2i": lambda **k: FluxImg2Img(**k),
        "sd15": lambda **k: SDImg2Img("sd15", **k),
        "sd21": lambda **k: SDImg2Img("sd21", **k),
        "sdxlturbo": lambda **k: SDXLTurboImg2Img(**k),
        "vae": lambda **k: VAERoundTrip(**k),
        "hifivae": lambda **k: HiFiVAERoundTrip(repo="ostris/vae-kl-f8-d16", **k),
        "lcm": lambda **k: LCMImg2Img(**k),
        "kandinsky": lambda **k: KandinskyImg2Img(**k),
        "vq": lambda **k: VQRoundTrip(**k),
    }
    reg = {}
    for key, kw in specs:
        try:
            obj = factory[key](**kw)
            reg[obj.name] = obj
            print(f"[regen] loaded {obj.name} (arch={obj.arch})", flush=True)
        except Exception as e:
            print(f"[regen] SKIP {key} {kw}: {type(e).__name__}: {e}", flush=True)
    return reg
