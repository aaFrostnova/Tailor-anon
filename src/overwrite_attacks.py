"""Universal per-class watermark overwrite attacks (Phase B).

An overwrite attack removes (or spoofs) a victim watermark by COLLIDING with its
embedding subspace -- re-embedding a competing, key-blind signal in the same
representation. The attacker knows only the TRANSFORM (public), never the
victim's key or carrier subset.

Analytic saturators (B.1): the analytic subspaces are public, so we reconstruct
the FULL candidate set (a superset of any victim's keyed carriers) and randomize
the entire subspace:
  - FFTAnnulusSaturator  -> green-channel FFT-magnitude QIM annulus (kills DFT-Kred / ScaleAware)
  - BlockMeanSaturator   -> green-channel block-mean QIM grid       (kills Quant-QIM)
  - DWTDCTSaturator      -> re-embed random dwtDct + dwtDctSvd marks (kills the DWT-DCT pair)
  - AnalyticUniversal    -> chains all three under one PSNR budget

Each exposes overwrite(pil, target_psnr) -> PIL and binary-searches a residual
scale to land on a stated PSNR budget (so removal can be compared to regeneration
at EQUAL fidelity). No victim key is ever referenced.

Mirrors the carrier-write pattern in scripts/prototype_dft_kredundant.py:qim_embed_k.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


# ----------------------------------------------------------------- utilities
def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 99.0


def _budget_blend(orig_u8: np.ndarray, sat_u8: np.ndarray, target_psnr: float | None) -> np.ndarray:
    """Blend attacked = clip(orig + a*(sat-orig)); binary-search a in [0,1] for the
    STRONGEST attack whose PSNR >= target_psnr. None -> full saturation (a=1)."""
    if target_psnr is None:
        return sat_u8
    orig = orig_u8.astype(np.float64)
    resid = sat_u8.astype(np.float64) - orig
    if _psnr(orig_u8, sat_u8) >= target_psnr:
        return sat_u8  # full saturation already within budget
    lo, hi = 0.0, 1.0
    out = orig_u8
    for _ in range(18):
        a = 0.5 * (lo + hi)
        cand = np.clip(orig + a * resid, 0, 255).astype(np.uint8)
        if _psnr(orig_u8, cand) >= target_psnr:
            out = cand; lo = a      # within budget -> push stronger
        else:
            hi = a                  # too strong -> back off
    return out


# ----------------------------------------------------------- FFT annulus QIM
class FFTAnnulusSaturator:
    """Randomize every green-channel FFT-magnitude bin in a wide annulus to a QIM
    lattice phase. The victim's keyed carriers are a subset of this annulus, so
    its QIM readout is driven to chance. Key-blind."""

    def __init__(self, resolution: int = 256, r_lo: float = 16.0, r_hi: float = 80.0,
                 delta: float = 50.0, channel: int = 1, seed: int = 0):
        self.res = resolution
        self.r_lo, self.r_hi = r_lo, r_hi
        self.delta = delta
        self.channel = channel
        self.seed = seed
        self._bins = self._annulus_bins(resolution, r_lo, r_hi)

    @staticmethod
    def _annulus_bins(N: int, r_lo: float, r_hi: float):
        bins = []
        for fy in range(1, N // 2):
            for fx_signed in range(-(N // 2 - 1), N // 2):
                r = (fy ** 2 + fx_signed ** 2) ** 0.5
                if r_lo <= r <= r_hi:
                    bins.append((fy, fx_signed % N))
        return bins

    def _saturate_small(self, small_u8: np.ndarray, rng) -> np.ndarray:
        img = small_u8.astype(np.float64) / 255.0
        N = self.res
        ch = img[:, :, self.channel].copy()
        F = np.fft.fft2(ch)
        d = self.delta
        for (fy, fx) in self._bins:
            rbit = int(rng.integers(0, 2))
            mag = abs(F[fy, fx]); phase = np.angle(F[fy, fx])
            tgt = np.round(mag / d) * d + (d / 2.0) * rbit
            val = tgt * np.exp(1j * phase)
            F[fy, fx] = val
            F[(-fy) % N, (-fx) % N] = np.conj(val)
        ch_w = np.clip(np.real(np.fft.ifft2(F)), 0.0, 1.0)
        out = img.copy(); out[:, :, self.channel] = ch_w
        return (out * 255.0 + 0.5).astype(np.uint8)

    def overwrite(self, pil: Image.Image, target_psnr: float | None = 38.0) -> Image.Image:
        pil = pil.convert("RGB")
        W, H = pil.size
        orig_u8 = np.asarray(pil)
        rng = np.random.default_rng(self.seed)
        small = np.asarray(pil.resize((self.res, self.res), Image.BICUBIC))
        sat_small = self._saturate_small(small, rng)
        # residual at small res, upscaled to native and added (symmetric to the victim embed)
        resid_small = sat_small.astype(np.float64) - small.astype(np.float64)
        resid_img = Image.fromarray(np.clip(resid_small + 128, 0, 255).astype(np.uint8))
        resid_up = np.asarray(resid_img.resize((W, H), Image.BILINEAR)).astype(np.float64) - 128.0
        sat_u8 = np.clip(orig_u8.astype(np.float64) + resid_up, 0, 255).astype(np.uint8)
        return Image.fromarray(_budget_blend(orig_u8, sat_u8, target_psnr))


# --------------------------------------------------------- block-mean QIM
class BlockMeanSaturator:
    """Shift every green-channel block mean to a random QIM lattice phase. The
    victim's keyed blocks are a subset, so its block-QIM readout -> chance."""

    def __init__(self, resolution: int = 256, block_size: int = 8,
                 delta: float = 0.12, channel: int = 1, seed: int = 0):
        self.res = resolution
        self.bs = block_size
        self.delta = delta
        self.channel = channel
        self.seed = seed

    def _saturate_small(self, small_u8: np.ndarray, rng) -> np.ndarray:
        img = small_u8.astype(np.float64) / 255.0
        N = self.res; bs = self.bs; nb = N // bs
        ch = img[:, :, self.channel]
        blocks = ch[: nb * bs, : nb * bs].reshape(nb, bs, nb, bs)
        mu = blocks.mean(axis=(1, 3))                       # (nb,nb) block means
        rbit = rng.integers(0, 2, size=mu.shape)
        d = self.delta
        tgt = np.round(mu / d) * d + (d / 2.0) * rbit
        shift = (tgt - mu)[:, None, :, None]                # broadcast per-block constant
        ch2 = ch.copy()
        ch2[: nb * bs, : nb * bs] = np.clip(blocks + shift, 0.0, 1.0).reshape(nb * bs, nb * bs)
        out = img.copy(); out[:, :, self.channel] = ch2
        return (out * 255.0 + 0.5).astype(np.uint8)

    def overwrite(self, pil: Image.Image, target_psnr: float | None = 38.0) -> Image.Image:
        pil = pil.convert("RGB")
        W, H = pil.size
        orig_u8 = np.asarray(pil)
        rng = np.random.default_rng(self.seed + 1)
        small = np.asarray(pil.resize((self.res, self.res), Image.BICUBIC))
        sat_small = self._saturate_small(small, rng)
        resid_small = sat_small.astype(np.float64) - small.astype(np.float64)
        resid_img = Image.fromarray(np.clip(resid_small + 128, 0, 255).astype(np.uint8))
        resid_up = np.asarray(resid_img.resize((W, H), Image.BILINEAR)).astype(np.float64) - 128.0
        sat_u8 = np.clip(orig_u8.astype(np.float64) + resid_up, 0, 255).astype(np.uint8)
        return Image.fromarray(_budget_blend(orig_u8, sat_u8, target_psnr))


# ------------------------------------------------------------- DWT-DCT (lib)
class DWTDCTSaturator:
    """Re-embed random dwtDct + dwtDctSvd marks (same-algorithm = in-subspace
    saturation of the wavelet-DCT coefficients). Uses the invisible-watermark lib."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        import cv2
        from imwatermark import WatermarkEncoder
        self._cv2 = cv2
        self._Enc = WatermarkEncoder

    def _embed(self, pil, method, bits):
        bgr = self._cv2.cvtColor(np.asarray(pil.convert("RGB")), self._cv2.COLOR_RGB2BGR)
        enc = self._Enc()
        enc.set_watermark("bits", list(int(b) for b in bits))
        wm = enc.encode(bgr, method)
        return Image.fromarray(self._cv2.cvtColor(wm, self._cv2.COLOR_BGR2RGB))

    def overwrite(self, pil: Image.Image, target_psnr: float | None = 38.0) -> Image.Image:
        pil = pil.convert("RGB")
        orig_u8 = np.asarray(pil)
        rng = np.random.default_rng(self.seed + 2)
        out = self._embed(pil, "dwtDct", rng.integers(0, 2, 32))
        out = self._embed(out, "dwtDctSvd", rng.integers(0, 2, 32))
        sat_u8 = np.asarray(out.resize(pil.size))
        return Image.fromarray(_budget_blend(orig_u8, sat_u8, target_psnr))


# ---------------------------------------------------------- universal chain
class AnalyticUniversal:
    """Chain FFT + block + DWT saturators under one shared PSNR budget."""

    def __init__(self, resolution: int = 256, fft_delta: float = 120.0,
                 block_delta: float = 0.06, seed: int = 0, include_dwt: bool = True):
        self.fft = FFTAnnulusSaturator(resolution=resolution, delta=fft_delta, seed=seed)
        self.block = BlockMeanSaturator(resolution=resolution, delta=block_delta, seed=seed)
        self.dwt = DWTDCTSaturator(seed=seed) if include_dwt else None

    def overwrite(self, pil: Image.Image, target_psnr: float | None = 38.0) -> Image.Image:
        orig_u8 = np.asarray(pil.convert("RGB"))
        cur = self.fft.overwrite(pil, target_psnr=None)
        cur = self.block.overwrite(cur, target_psnr=None)
        if self.dwt is not None:
            cur = self.dwt.overwrite(cur, target_psnr=None)
        sat_u8 = np.asarray(cur.resize(pil.size))
        return Image.fromarray(_budget_blend(orig_u8, sat_u8, target_psnr))
