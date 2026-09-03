"""Wrappers turning the trained DFT-Kred and Quant-QIM modules into
shared-codeword, crypto-keyed watermark METHODS for the fused detector.

Each wrapper:
  - loads a trained checkpoint (encoder + decoder + canonical carriers);
  - embeds an explicit 100-bit transmitted codeword `tx`, scrambled per-image
    via (perm, M) = derive_method_keyed_constants(master_key, image_id, name);
  - exposes raw per-bit logits at decode (LLR-scaled: logit > 0 -> bit 1);
  - exposes carrier metadata (frequency bins / block indices, radius, bit_flip,
    delta) used by the attack-characterization and tamper-localization stages.

All operate at the model's native resolution (256x256), green channel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.dft_kred_modules import DFTKredEncoder, DFTKredDecoder, FFTAwareDecoder
from src.quant_qim_modules import QuantQIMEncoder, QuantQIMDecoder
from src.vine_crypto_wrapper import derive_method_keyed_constants, apply_crypto


def _to_bytes(x):
    return x.encode("utf-8") if isinstance(x, str) else x


class _LearnedMethodBase:
    n_bits = 100

    def __init__(self, master_key: bytes, method_name: str, device: str = "cuda"):
        self.master_key = master_key
        self.method_name = method_name
        self.device = device
        self.resolution = 256
        self.enc = None
        self.dec = None

    def get_perm_M(self, image_id: str):
        return derive_method_keyed_constants(
            self.master_key, image_id, self.method_name, self.n_bits,
        )

    def _img_to_tensor(self, pil: Image.Image) -> torch.Tensor:
        t = transforms.Compose([
            transforms.Resize(self.resolution,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
        ])
        return t(pil.convert("RGB")).unsqueeze(0).to(self.device)

    def embed(self, pil: Image.Image, image_id: str, tx: np.ndarray) -> Image.Image:
        """Embed the shared 100-bit codeword `tx`, scrambled by this method's key.

        The model runs at its native 256x256; for larger inputs the residual is
        computed at 256 and upscaled back to the original size and added to the
        original (same 256-roundtrip strategy VINE uses), so the pipeline can run
        at 512 where SD regeneration is in-distribution.
        """
        import torch.nn.functional as F
        perm, M = self.get_perm_M(image_id)
        secret = apply_crypto(np.asarray(tx, dtype=np.uint8), perm, M)  # (100,) float {0,1}
        secret_t = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).to(self.device)
        pil = pil.convert("RGB")
        W, H = pil.size
        img256 = self._img_to_tensor(pil)               # (1,3,256,256)
        with torch.no_grad():
            wm256 = self.enc(img256, secret_t)
        if (W, H) == (self.resolution, self.resolution):
            return transforms.ToPILImage()(wm256[0].clamp(0, 1).cpu())
        resid256 = wm256 - img256
        resid_up = F.interpolate(resid256, size=(H, W), mode="bilinear", align_corners=False)
        orig = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)
        out = (orig + resid_up).clamp(0, 1)
        return transforms.ToPILImage()(out[0].cpu())

    def raw_logits(self, pil: Image.Image) -> np.ndarray:
        """Per-bit secret logits (target domain), LLR-scaled (logit>0 -> bit 1)."""
        img = self._img_to_tensor(pil)
        with torch.no_grad():
            logits = self.dec(img)[0].cpu().numpy()
        return logits


class DFTKredMethod(_LearnedMethodBase):
    """DFT-magnitude QIM carrier method (frequency domain, location-aware)."""

    def __init__(self, ckpt_path: str, master_key: bytes,
                 method_name: str = "dft_kred", device: str = "cuda"):
        super().__init__(master_key, method_name, device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.n_bits = int(ck["n_bits"])
        self.resolution = int(ck["resolution"])
        self.K = int(ck["K"]); self.M = int(ck["M"])
        self.r_lo = float(ck["r_lo"]); self.r_hi = float(ck["r_hi"])
        ckey = _to_bytes(ck["canonical_key"]); cid = ck["canonical_id"]
        self.enc = DFTKredEncoder(
            n_bits=self.n_bits, K=self.K, M=self.M, resolution=self.resolution,
            r_lo=self.r_lo, r_hi=self.r_hi, canonical_key=ckey, canonical_id=cid,
        ).to(device).eval()
        decoder_type = (ck.get("config") or {}).get("decoder_type", "fft_aware")
        if decoder_type == "fft_aware":
            self.dec = FFTAwareDecoder(
                n_bits=self.n_bits, K=self.K, M=self.M, resolution=self.resolution,
                r_lo=self.r_lo, r_hi=self.r_hi, canonical_key=ckey, canonical_id=cid,
            ).to(device).eval()
        else:
            self.dec = DFTKredDecoder(n_bits=self.n_bits).to(device).eval()
        self.enc.load_state_dict(ck["encoder_state_dict"])
        self.dec.load_state_dict(ck["decoder_state_dict"])
        # carrier metadata for characterization/localization: (n_bits, K*M, 2)
        self.carriers = self.enc.carriers.cpu().numpy()
        self.bit_flip = self.enc.bit_flip.cpu().numpy()
        self.carrier_kind = "dft_freq"

    @torch.no_grad()
    def carrier_magnitudes(self, pil: Image.Image) -> np.ndarray:
        """FFT-magnitude at each carrier, shape (n_bits, K*M) — for B/C analysis."""
        img = self._img_to_tensor(pil)
        green = img[:, 1]
        Fc = torch.fft.fft2(green)
        fy = self.dec.carriers[..., 0].reshape(-1)
        fx = self.dec.carriers[..., 1].reshape(-1)
        mag = torch.abs(Fc[:, fy, fx])[0].cpu().numpy()
        return mag.reshape(self.n_bits, self.K * self.M)

    def decode_delta(self) -> float:
        import torch.nn.functional as F
        return float(F.softplus(self.dec.decode_delta) + 1.0)

    def carrier_radii(self) -> np.ndarray:
        """Radial frequency of each carrier, shape (n_bits, K*M)."""
        cy = cx = self.resolution / 2.0
        fy = self.carriers[..., 0].astype(np.float64)
        fx = self.carriers[..., 1].astype(np.float64)
        # bins are 0..res-1; wrap to signed frequency for radius
        sy = np.where(fy > self.resolution / 2, fy - self.resolution, fy)
        sx = np.where(fx > self.resolution / 2, fx - self.resolution, fx)
        return np.sqrt(sy ** 2 + sx ** 2)


class QuantQIMMethod(_LearnedMethodBase):
    """Block-mean QIM carrier method (spatial domain, location-aware)."""

    def __init__(self, ckpt_path: str, master_key: bytes,
                 method_name: str = "quant_qim", device: str = "cuda"):
        super().__init__(master_key, method_name, device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.n_bits = int(ck["n_bits"])
        self.resolution = int(ck["resolution"])
        self.n_pos = int(ck["n_pos"]); self.block_size = int(ck["block_size"])
        ckey = _to_bytes(ck["canonical_key"]); cid = ck["canonical_id"]
        init_delta = float((ck.get("config") or {}).get("init_delta", 0.06))
        self.enc = QuantQIMEncoder(
            n_bits=self.n_bits, n_pos=self.n_pos, block_size=self.block_size,
            resolution=self.resolution, canonical_key=ckey, canonical_id=cid,
            init_delta=init_delta,
        ).to(device).eval()
        self.dec = QuantQIMDecoder(
            n_bits=self.n_bits, n_pos=self.n_pos, block_size=self.block_size,
            resolution=self.resolution, canonical_key=ckey, canonical_id=cid,
            init_delta=init_delta,
        ).to(device).eval()
        self.enc.load_state_dict(ck["encoder_state_dict"])
        self.dec.load_state_dict(ck["decoder_state_dict"])
        self.n_by = self.enc.n_by; self.n_bx = self.enc.n_bx
        self.carriers = self.enc.carriers.cpu().numpy()       # (n_bits, n_pos) flat block idx
        self.bit_flip = self.enc.bit_flip.cpu().numpy()
        self.carrier_kind = "qim_block"

    @torch.no_grad()
    def carrier_means(self, pil: Image.Image) -> np.ndarray:
        """Block means at each carrier, shape (n_bits, n_pos) — for B/C analysis."""
        import torch.nn.functional as F
        img = self._img_to_tensor(pil)
        green = img[:, 1]
        mu = F.avg_pool2d(green.unsqueeze(1), self.block_size).squeeze(1).reshape(1, -1)
        idx = self.dec.carriers.reshape(-1)
        sel = mu[:, idx][0].cpu().numpy()
        return sel.reshape(self.n_bits, self.n_pos)

    def decode_delta(self) -> float:
        import torch.nn.functional as F
        return float(F.softplus(self.dec.decode_delta) + 1e-4)

    def carrier_block_coords(self) -> np.ndarray:
        """(by, bx) spatial block coords of each carrier, shape (n_bits, n_pos, 2)."""
        by = self.carriers // self.n_bx
        bx = self.carriers % self.n_bx
        return np.stack([by, bx], axis=-1)
