"""Quantization-domain (block-mean QIM) encoder + decoder, VINE-style.

This is the quantization analog of src/dft_kred_modules.py. Where the DFT
fragment does QIM on FFT magnitudes, this fragment does QIM on the MEAN of
b x b spatial blocks of the green channel:

  - The encoder partitions the green channel into b x b blocks, picks a set of
    crypto-keyed blocks per bit, and shifts each selected block by a constant
    so its mean lands on the bit-0 lattice {k*delta} or the bit-1 lattice
    {k*delta + delta/2}. A constant per-block shift is the least visible
    perturbation and survives JPEG/blur/noise because those operations
    approximately preserve local means (the DC component).
  - The decoder reads block means at the same keyed positions and uses the
    differentiable soft-QIM feature cos(2*pi*mu/delta) (+1 on bit-0 lattice,
    -1 on bit-1) fed to a small shared MLP, exactly like FFTAwareDecoder.

Only the quantization step delta is learnable in the encoder, so the encoder
cannot collapse to identity. The decoder is the learned, reference-free part.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.sign_envelope import _hkdf_uint64_stream


def setup_block_carriers(master_key: bytes, image_id: str, n_bits: int,
                         n_blocks_total: int, n_pos: int):
    """Pick n_pos unique block indices per bit + a per-bit sign-flip mask.

    Returns:
        carriers: (n_bits, n_pos) long tensor of flat block indices.
        bit_flip: (n_bits,) uint8 array.
    """
    needed = n_bits * n_pos
    if needed > n_blocks_total:
        raise ValueError(
            f"need {needed} unique blocks but only {n_blocks_total} exist; "
            f"reduce n_pos or block_size."
        )
    pos_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/quant_position").encode("utf-8"),
        needed * 6,
    )
    used = set()
    carriers = []
    cur = 0
    for _ in range(n_bits):
        row = []
        while len(row) < n_pos:
            if cur >= len(pos_stream):
                raise RuntimeError("HKDF stream exhausted picking blocks")
            j = int(pos_stream[cur] % n_blocks_total); cur += 1
            if j in used:
                continue
            used.add(j)
            row.append(j)
        carriers.append(row)
    sign_stream = _hkdf_uint64_stream(
        master_key, (image_id + "/quant_sign").encode("utf-8"), n_bits,
    )
    bit_flip = (sign_stream % 2).astype(np.uint8)
    return torch.tensor(carriers, dtype=torch.long), bit_flip


class QuantQIMEncoder(nn.Module):
    """Block-mean QIM embedding with a single learnable base_delta parameter."""

    def __init__(
        self,
        n_bits: int = 100,
        n_pos: int = 8,
        block_size: int = 8,
        resolution: int = 256,
        canonical_key: bytes = b"vine_training_canonical_key",
        canonical_id: str = "quant_qim_train_canonical",
        init_delta: float = 0.06,
        channel: int = 1,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.n_pos = n_pos
        self.block_size = block_size
        self.resolution = resolution
        self.channel = channel
        self.n_by = resolution // block_size
        self.n_bx = resolution // block_size
        self.n_blocks = self.n_by * self.n_bx
        # softplus(base_delta) gives delta directly (delta is small, ~0.06).
        self.base_delta = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(init_delta))))
        )

        carriers, bit_flip = setup_block_carriers(
            canonical_key, canonical_id, n_bits, self.n_blocks, n_pos,
        )
        self.register_buffer("carriers", carriers)            # (n_bits, n_pos)
        self.register_buffer(
            "bit_flip", torch.tensor(bit_flip, dtype=torch.long)
        )

    def _delta(self):
        return F.softplus(self.base_delta) + 1e-4

    def forward(self, image: torch.Tensor, secret_bits: torch.Tensor) -> torch.Tensor:
        """image: (B,3,H,W) in [0,1]; secret_bits: (B,n_bits) in {0,1}."""
        B, _, H, W = image.shape
        assert H == W == self.resolution
        b = self.block_size
        delta = self._delta()

        green = image[:, self.channel]                        # (B, H, W)
        # Block means via average pooling.
        mu = F.avg_pool2d(green.unsqueeze(1), b).squeeze(1)    # (B, n_by, n_bx)
        mu_flat = mu.reshape(B, -1)                            # (B, n_blocks)

        secret_int = secret_bits.long()
        eff_bits = (secret_int ^ self.bit_flip.unsqueeze(0)).float()  # (B, n_bits)

        idx = self.carriers.reshape(-1)                       # (n_bits*n_pos,)
        eff_per_pos = (
            eff_bits.unsqueeze(-1).expand(-1, -1, self.n_pos).reshape(B, -1)
        )                                                     # (B, n_bits*n_pos)

        mu_sel = mu_flat[:, idx]                               # (B, n_carriers)
        target = torch.round(mu_sel / delta) * delta + (delta / 2.0) * eff_per_pos
        shift = target - mu_sel                                # (B, n_carriers)

        # Scatter shifts back into a per-block shift map (B, n_blocks).
        shift_map = torch.zeros(B, self.n_blocks, device=image.device, dtype=shift.dtype)
        shift_map = shift_map.index_add(
            1, idx, shift,
        )
        # If a block is selected by multiple bits (shouldn't happen — carriers
        # are globally unique), index_add would sum; uniqueness guarantees one.
        shift_map = shift_map.reshape(B, 1, self.n_by, self.n_bx)
        shift_up = F.interpolate(shift_map, scale_factor=b, mode="nearest")[:, 0]

        green_w = green + shift_up
        out = image.clone()
        out[:, self.channel] = torch.clamp(green_w, 0.0, 1.0)
        return out


class QuantQIMDecoder(nn.Module):
    """Reads block means at keyed positions; soft-QIM cos feature + MLP."""

    def __init__(
        self,
        n_bits: int = 100,
        n_pos: int = 8,
        block_size: int = 8,
        resolution: int = 256,
        canonical_key: bytes = b"vine_training_canonical_key",
        canonical_id: str = "quant_qim_train_canonical",
        init_delta: float = 0.06,
        channel: int = 1,
        hidden: int = 32,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.n_pos = n_pos
        self.block_size = block_size
        self.resolution = resolution
        self.channel = channel
        self.n_by = resolution // block_size
        self.n_bx = resolution // block_size
        self.n_blocks = self.n_by * self.n_bx
        self.decode_delta = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(init_delta))))
        )

        carriers, bit_flip = setup_block_carriers(
            canonical_key, canonical_id, n_bits, self.n_blocks, n_pos,
        )
        self.register_buffer("carriers", carriers)
        sign = (1 - 2 * torch.tensor(bit_flip, dtype=torch.long)).float()
        self.register_buffer("flip_sign", sign)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.bit_head = nn.Linear(hidden, 1)

    def _delta(self):
        return F.softplus(self.decode_delta) + 1e-4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        b = self.block_size
        green = image[:, self.channel]
        mu = F.avg_pool2d(green.unsqueeze(1), b).squeeze(1).reshape(B, -1)

        idx = self.carriers.reshape(-1)
        mu_sel = mu[:, idx]                                    # (B, n_carriers)
        delta = self._delta()
        phase = 2.0 * torch.pi * mu_sel / delta
        mu_norm = mu_sel / (mu_sel.mean(dim=1, keepdim=True) + 1e-6)
        feats = torch.stack(
            [torch.cos(phase), torch.sin(phase), torch.tanh(mu_norm)], dim=-1
        )                                                     # (B, n_carriers, 3)
        feats = feats.view(B, self.n_bits, self.n_pos, 3)
        h = self.pos_mlp(feats).mean(dim=2)                   # (B, n_bits, hidden)
        eff_logit = self.bit_head(h).squeeze(-1)              # (B, n_bits)
        return eff_logit * self.flip_sign.unsqueeze(0)
