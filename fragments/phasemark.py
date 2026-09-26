"""PhaseMark (APM variant) — reimplementation.

PhaseMark: A Post-hoc, Optimization-Free Watermarking of AI-generated Images in the
Latent Frequency Domain (Lee & Cho, SNU; ICASSP 2026; arXiv:2601.13128). No official code;
reimplemented from the paper's algorithm.

Post-hoc (works on any existing image), optimization-free (one VAE encode + FFT + phase
edit + IFFT + decode to embed; one VAE encode + FFT to detect), multi-bit (128 bits =
32 blocks x 4 latent channels). Robust to diffusion regeneration because the mark lives in
the VAE latent mid-band phase, which the regeneration manifold preserves.

APM (Absolute Phase Modulation, the robust variant): set the phase of selected mid-band
latent FFT coefficients to +pi/2 (bit 1) or -pi/2 (bit 0), keep magnitude; enforce
Hermitian symmetry so the IFFT is real. Detect: read the sign of the summed phase.

Detection is MODEL-BOUND: detect must use the SAME VAE as embed.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.latent_vae import LatentVAE
from src.payload import BCHCodec, image_id_to_payload
from src.vine_crypto_wrapper import (
    apply_crypto, derive_method_keyed_constants, undo_crypto,
)

LAT = 64          # SD latent spatial size for 512px image
CROP = 44         # central crop of the latent
OFF = (LAT - CROP) // 2   # = 10
R_LO, R_HI = 9.0, 19.0    # mid-band annulus on the 44x44 crop
N_BLOCKS = 32             # blocks per channel
N_CH = 4
PHI = np.pi / 2


def _signed_freq(n, N):
    return n if n <= N // 2 else n - N


def _build_blocks(r_lo: float = R_LO, r_hi: float = R_HI, seed: int = 0):
    """Deterministic list of N_BLOCKS non-overlapping 2x2 coeff-blocks per channel in the
    half-plane (fv>0) mid-band annulus, off the principal axes. Returns list of blocks,
    each a list of 4 (i,j) natural-FFT indices into the CROPxCROP spectrum."""
    f = np.array([_signed_freq(n, CROP) for n in range(CROP)])
    cand = []
    for i in range(CROP - 1):
        for j in range(CROP - 1):
            coeffs = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
            fus = [f[a] for a, _ in coeffs]
            fvs = [f[b] for _, b in coeffs]
            if any(fu == 0 for fu in fus) or any(fv == 0 for fv in fvs):
                continue                       # off principal axes
            if not all(fv > 0 for fv in fvs):  # half-plane -> avoids conjugate-pair collisions
                continue
            rs = [np.hypot(fu, fv) for fu, fv in zip(fus, fvs)]
            if not all(r_lo <= r <= r_hi for r in rs):
                continue
            cand.append((np.mean(rs), coeffs))
    cand.sort(key=lambda t: t[0])              # inner-to-outer, deterministic
    blocks, used = [], set()
    for _, coeffs in cand:
        if any(c in used for c in coeffs):
            continue
        blocks.append(coeffs)
        used.update(coeffs)
        if len(blocks) == N_BLOCKS:
            break
    if len(blocks) < N_BLOCKS:
        raise RuntimeError(f"only {len(blocks)} blocks found in annulus [{r_lo},{r_hi}]; widen it")
    return blocks


_BLOCKS = _build_blocks()


def _mirror(i, j):
    return (-i) % CROP, (-j) % CROP


class PhaseMark:
    """Raw multi-bit PhaseMark over an SD VAE. n_bits up to N_BLOCKS*len(active_ch) (128).

    Variants (paper trades detection<->quality across 4 variants):
      gamma=1.0           -> APM (Absolute Phase Modulation): hard set phase to +/-pi/2
                             (most robust, lowest PSNR -- our original).
      gamma in (0,1)      -> SPS-style Soft Phase Steering: rotate each coeff only a
                             fraction gamma toward +/-pi/2 (keeps magnitude). Smaller
                             pixel-delta per unit of phase margin -> better PSNR frontier.
      perceptual=True     -> mask the pixel-domain residual by local image activity
                             (hide the mark in texture/edges); raises PSNR/SSIM cheaply.
      band=(lo,hi)        -> mid-band annulus (paper uses [10,18]; default [9,19]).
      active_ch           -> latent channels to modulate (subset -> fewer coeffs -> +PSNR).
    Detection (_soft) is unchanged: signed phase-sum over each block; sign reads under
    both hard and soft embedding because the magnitude is preserved.
    """

    def __init__(self, vae_key: str = "sd21", device: str = "cuda",
                 gamma: float = 1.0, perceptual: bool = False, perc_floor: float = 0.5,
                 band: tuple = (R_LO, R_HI), active_ch: tuple | None = None):
        self.vae = LatentVAE(vae_key, device)
        self.device = device
        self.gamma = float(gamma)
        self.perceptual = bool(perceptual)
        self.perc_floor = float(perc_floor)
        self.blocks = _build_blocks(band[0], band[1]) if tuple(band) != (R_LO, R_HI) else _BLOCKS
        self.active_ch = tuple(range(N_CH)) if active_ch is None else tuple(active_ch)
        self.capacity = N_BLOCKS * len(self.active_ch)

    def embed(self, pil: Image.Image, bits: np.ndarray, residual: bool = True,
              strength: float = 1.0) -> Image.Image:
        """residual=True (default): add only the watermark perturbation
        (decode(edited)-decode(orig)) to the CLEAN original, cancelling the VAE
        round-trip loss -> high PSNR. residual=False returns the raw VAE decode.

        strength<1.0: blend the latent-crop perturbation toward the original
        (soft modulation in latent space) -- trades regeneration robustness for PSNR.
        (gamma/perceptual/band/active_ch are set on the instance.)"""
        bits = np.asarray(bits, np.uint8)
        assert len(bits) <= self.capacity, f"{len(bits)} bits > capacity {self.capacity}"
        b = np.zeros(self.capacity, np.uint8); b[:len(bits)] = bits
        orig_lat = self.vae.encode(pil)                         # [1,4,64,64]
        lat = orig_lat.clone()
        orig_crop = lat[:, :, OFF:OFF + CROP, OFF:OFF + CROP].clone()
        crop = orig_crop.clone()
        for pos, ch in enumerate(self.active_ch):
            F = torch.fft.fft2(crop[0, ch].float())
            for bi, block in enumerate(self.blocks):
                tgt = PHI if int(b[pos * N_BLOCKS + bi]) == 1 else -PHI
                for (i, j) in block:
                    mag = torch.abs(F[i, j])
                    if self.gamma >= 1.0:
                        new_ang = float(tgt)
                    else:                                       # soft phase steering
                        cur = float(torch.angle(F[i, j]))
                        d = (tgt - cur + np.pi) % (2 * np.pi) - np.pi
                        new_ang = cur + self.gamma * d
                    val = mag * torch.exp(torch.tensor(1j * new_ang, device=F.device))
                    F[i, j] = val
                    mi, mj = _mirror(i, j)
                    F[mi, mj] = torch.conj(val)                 # Hermitian symmetry -> real IFFT
            crop[0, ch] = torch.real(torch.fft.ifft2(F)).to(crop.dtype)
        if strength != 1.0:                                     # soft modulation (latent space)
            crop = orig_crop + strength * (crop - orig_crop)
        lat[:, :, OFF:OFF + CROP, OFF:OFF + CROP] = crop
        wm_recon = self.vae.decode(lat, size=pil.size)
        if not residual:
            return wm_recon
        orig_recon = self.vae.decode(orig_lat, size=pil.size)
        delta = np.asarray(wm_recon, np.float64) - np.asarray(orig_recon, np.float64)
        if self.perceptual:                                     # hide residual in texture/edges
            g = np.asarray(pil.convert("L").resize(wm_recon.size), np.float32)
            act = np.abs(np.gradient(g, axis=0)) + np.abs(np.gradient(g, axis=1))
            hi = np.percentile(act, 90) + 1e-6                   # full strength on textured pixels;
            m = np.clip(act / hi, self.perc_floor, 1.0)[..., None]  # only flats are attenuated
            delta = delta * m
        out = np.clip(np.asarray(pil.convert("RGB").resize(wm_recon.size), np.float64) + delta, 0, 255)
        return Image.fromarray(out.astype(np.uint8))

    def _soft(self, pil: Image.Image) -> np.ndarray:
        """Per-bit signed score (sum of phases over the block); sign = bit, |.| = confidence."""
        lat = self.vae.encode(pil)
        crop = lat[:, :, OFF:OFF + CROP, OFF:OFF + CROP]
        out = np.zeros(self.capacity, np.float64)
        for pos, ch in enumerate(self.active_ch):
            F = torch.fft.fft2(crop[0, ch].float())
            for bi, block in enumerate(self.blocks):
                s = sum(float(torch.angle(F[i, j])) for (i, j) in block)
                out[pos * N_BLOCKS + bi] = s
        return out

    def detect(self, pil: Image.Image, n_bits: int | None = None) -> np.ndarray:
        soft = self._soft(pil)
        bits = (soft > 0).astype(np.uint8)
        return bits[:n_bits] if n_bits else bits

    def soft_scores(self, pil: Image.Image, n_bits: int | None = None) -> np.ndarray:
        s = self._soft(pil)
        return s[:n_bits] if n_bits else s


class PhaseMarkWrapper:
    """Crypto wrapper: PhaseMark carrying the shared shortened-BCH codeword, keyed
    per-image by (perm, M). Mirrors VineCryptoWrapper / MaskWMWrapper so it drops into
    the fused detector and the overwrite matrix. Role: the post-hoc REGENERATION tier
    (latent-frequency phase) -- a different subspace from VINE/TrustMark/DFT/QIM."""

    def __init__(self, master_key: bytes = b"v5_key_encoder_master",
                 method_name: str = "phasemark", n_bits: int = 100,
                 vae_key: str = "sd21", device: str = "cuda",
                 detection_threshold: float = 0.75, strength: float = 1.0,
                 gamma: float = 1.0, perceptual: bool = False, perc_floor: float = 0.15,
                 band: tuple = (R_LO, R_HI), active_ch: tuple | None = None):
        assert n_bits <= N_BLOCKS * N_CH
        self.pm = PhaseMark(vae_key, device, gamma=gamma, perceptual=perceptual,
                            perc_floor=perc_floor, band=band, active_ch=active_ch)
        self.strength = float(strength)
        self.master_key = master_key
        self.method_name = method_name
        self.n_bits = n_bits
        self.detection_threshold = detection_threshold
        self.codec = BCHCodec()

    def get_perm_M(self, image_id: str):
        return derive_method_keyed_constants(self.master_key, image_id, self.method_name, self.n_bits)

    def _codeword_target(self, image_id: str) -> np.ndarray:
        cw = self.codec.encode(image_id_to_payload(image_id, n_bits=self.codec.data_bits))[:self.n_bits]
        perm, M = self.get_perm_M(image_id)
        return apply_crypto(cw, perm, M)

    def embed_with_target(self, pil: Image.Image, target_bits: np.ndarray) -> Image.Image:
        return self.pm.embed(pil, np.asarray(target_bits, np.uint8)[:self.n_bits], strength=self.strength)

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        return self.embed_with_target(pil, self._codeword_target(image_id))

    def raw_scores(self, pil: Image.Image) -> np.ndarray:
        """Signed phase-sum per bit (sign = bit, |.| = confidence) -> soft-fusion ready."""
        return self.pm.soft_scores(pil, self.n_bits)

    def detect(self, pil: Image.Image, image_id: str) -> dict:
        scores = self.raw_scores(pil)
        hard = (scores > 0).astype(np.float64)
        perm, M = self.get_perm_M(image_id)
        recovered = undo_crypto(hard, perm, M)
        expected = self.codec.encode(image_id_to_payload(image_id, n_bits=self.codec.data_bits))[:self.n_bits]
        bit_acc = float(np.mean(recovered == expected))
        return {"detected": bool(bit_acc >= self.detection_threshold), "bit_accuracy": bit_acc,
                "method": self.method_name, "scores": scores.tolist(),
                "perm": perm.tolist(), "M": M.tolist()}
