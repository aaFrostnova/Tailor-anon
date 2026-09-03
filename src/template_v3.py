"""v3 hybrid template: trainable carrier + crypto-keyed envelope.

Provides two ways to obtain the template T (the shared "look-for" pattern
that the keyed sign envelope flips per image_id):

  - construct_template_T_via_vae  (Plan A: no training, VAE-projected from
        an AES-CTR Gaussian; image-id-independent, master_key-derived).
  - LearnableTemplate              (Plan B: a torch nn.Module storing a
        learnable 3×H×W parameter; initialized from Plan A's output, then
        gradient-descent-trained against augmentation-augmented decoding loss).

Both produce a (3, H, W) numpy/tensor in [-EPS, +EPS] that we later
elementwise-multiply with the keyed sign envelope and a JND mask before
adding to the cover image.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import hashlib
import hmac

import numpy as np
import torch
from torch import nn

from .fragment import generate_fragment


def _derive_template_subkey(master_key: bytes, salt: bytes) -> bytes:
    """HKDF-extract → 32-byte AES key for the template seed."""
    return hmac.new(master_key, salt, hashlib.sha256).digest()


SD_VAE_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"


# ---------------------------------------------------------- Plan A: VAE-projected T

_CACHED_VAE = None


def _load_vae(model_path: str = SD_VAE_PATH, device: str = "cuda"):
    """Load SD-v1-5 VAE once, cache on module."""
    global _CACHED_VAE
    if _CACHED_VAE is None:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        vae = vae.to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        _CACHED_VAE = vae
    return _CACHED_VAE


def construct_template_T_via_vae(
    master_key: bytes,
    shape: Tuple[int, int, int] = (3, 256, 256),
    eps: float = 16.0 / 255.0,
    device: str = "cuda",
    vae_path: str = SD_VAE_PATH,
) -> np.ndarray:
    """Deterministic, manifold-aligned, key-derived template (Plan A).

    Pipeline:
      1. master_key + salt → AES-CTR Gaussian g of `shape`.
      2. Clamp g into [-1, 1] (VAE input range).
      3. Pass through SD-v1-5 VAE encode→decode (one round-trip).
      4. Center + scale to fill the [-eps, +eps] budget.

    The VAE round-trip projects arbitrary tensors onto its learned natural-
    image manifold approximation, so the resulting T lives close to the SD
    flow's tangent space and survives one img2img regen pass much better than
    raw AES-CTR Gaussian.
    """
    # Single deterministic subkey via HMAC-SHA256 (32-byte AES key)
    subkey = _derive_template_subkey(master_key, b"v3/template_seed")
    seed = generate_fragment(subkey, shape, epsilon=1.0, fragment_index=0)
    # seed is unit-variance Gaussian; clamp to VAE [-1, 1] input range
    g = np.clip(seed, -1.0, 1.0).astype(np.float32)

    vae = _load_vae(vae_path, device=device)
    with torch.no_grad():
        g_t = torch.from_numpy(g).unsqueeze(0).to(device)
        latent = vae.encode(g_t).latent_dist.mean
        out = vae.decode(latent).sample

    T = out[0].cpu().numpy().astype(np.float32)
    T = T - T.mean()
    max_abs = max(float(np.abs(T).max()), 1e-8)
    T = T / max_abs * float(eps)
    return T.astype(np.float32)


def construct_template_T_lowpass(
    master_key: bytes,
    shape: Tuple[int, int, int] = (3, 256, 256),
    eps: float = 16.0 / 255.0,
    cutoff_frac: float = 0.125,
) -> np.ndarray:
    """Fallback: low-pass-filtered AES-CTR Gaussian (no VAE call).

    Use when SD VAE is unavailable. Low-pass to roughly the spatial bandwidth
    that SD's VAE preserves (~1/8 of Nyquist matches the VAE 8× downsample).
    """
    subkey = _derive_template_subkey(master_key, b"v3/lowpass_seed")
    g = generate_fragment(subkey, shape, epsilon=1.0, fragment_index=0)
    C, H, W = shape
    # 2D low-pass via FFT mask
    fy = np.fft.fftfreq(H).reshape(-1, 1)
    fx = np.fft.fftfreq(W).reshape(1, -1)
    r = np.sqrt(fy ** 2 + fx ** 2)
    mask = (r <= cutoff_frac).astype(np.float32)
    out = np.zeros_like(g)
    for c in range(C):
        Fg = np.fft.fft2(g[c]) * mask
        out[c] = np.real(np.fft.ifft2(Fg)).astype(np.float32)
    out = out - out.mean()
    max_abs = max(float(np.abs(out).max()), 1e-8)
    return (out / max_abs * eps).astype(np.float32)


# ---------------------------------------------------------- Plan B: LearnableTemplate

class LearnableTemplate(nn.Module):
    """A trainable (3, H, W) template parameter, bounded by tanh × EPS.

    Initialized from Plan A's VAE-projected T (a strong starting point that
    is already manifold-aligned). Training then refines T to maximize the
    statistical decoder's bit accuracy under augmentation.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int] = (3, 256, 256),
        eps: float = 16.0 / 255.0,
        init_value: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.eps = float(eps)
        if init_value is None:
            # arctanh inverse: parameter starts near zero
            param_init = torch.zeros(shape)
        else:
            arr = np.clip(init_value / eps, -0.999, 0.999).astype(np.float32)
            param_init = torch.from_numpy(np.arctanh(arr))
        self.t_raw = nn.Parameter(param_init)

    def forward(self) -> torch.Tensor:
        """Return template T ∈ [-eps, +eps] of shape (3, H, W)."""
        return torch.tanh(self.t_raw) * self.eps

    def detach_numpy(self) -> np.ndarray:
        """Return current template as numpy (3, H, W)."""
        with torch.no_grad():
            return self().detach().cpu().numpy().astype(np.float32)

    @classmethod
    def from_vae(
        cls,
        master_key: bytes,
        shape: Tuple[int, int, int] = (3, 256, 256),
        eps: float = 16.0 / 255.0,
        device: str = "cuda",
    ) -> "LearnableTemplate":
        T0 = construct_template_T_via_vae(master_key, shape, eps, device=device)
        return cls(shape=shape, eps=eps, init_value=T0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "t_raw": self.t_raw.detach().cpu(),
            "eps": self.eps,
            "shape": tuple(self.t_raw.shape),
        }, path)

    @classmethod
    def load(cls, path: str) -> "LearnableTemplate":
        st = torch.load(path, map_location="cpu")
        m = cls(shape=tuple(st["shape"]), eps=float(st["eps"]))
        m.t_raw.data.copy_(st["t_raw"])
        return m
