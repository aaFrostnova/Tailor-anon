"""Key-conditioned learned encoder for v5 latent-domain watermark.

Takes (host_image, S_lat) where S_lat is the HKDF-derived ±1 sign envelope,
and outputs a watermarked image with per-pixel optimized perturbation.

The encoder weights are PUBLIC (like AES algorithm); the SECRET is master_key
which determines S_lat. Attacker with encoder weights but without master_key
cannot produce a valid watermark for a target image_id.

Architecture: pixel-domain ResNet-style CNN conditioned on upsampled S_lat.
  input:  concat(x, S_pixel)  →  (B, 4, 256, 256)
  body:   stem + N ResBlocks at 64 channels
  output: tanh residual × learnable scale  →  (B, 3, 256, 256)
  final:  x_w = clip(x + residual, -1, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class KeyConditionedEncoder(nn.Module):
    """Learned watermark encoder with deterministic sign modulation.

    The encoder learns a per-image "unsigned magnitude template" from the host
    image content. The sign comes from S_pixel (crypto-determined) via
    external multiplication, NOT as a conditioning channel.

    This is structurally T(x) × S where T(x) is learned and S is fixed by key.
    The decoder only needs to read the sign, which is always clearly structured.

    Args:
        in_ch_image: image channels (3 for RGB).
        base_ch: width of ResBlocks.
        n_blocks: number of ResBlocks in the body.
        init_scale: initial value of the learnable output scale.
    """

    def __init__(
        self,
        in_ch_image: int = 3,
        base_ch: int = 64,
        n_blocks: int = 8,
        init_scale: float = 0.05,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch_image, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
        )

        self.body = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_blocks)])

        self.to_rgb = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, in_ch_image, 3, padding=1),
        )

        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x: torch.Tensor, S_pixel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) host image in [-1, 1].
            S_pixel: (B, 1, H, W) sign envelope in {-1, +1}.

        Returns:
            x_w: (B, 3, H, W) watermarked image in [-1, 1].
        """
        feat = self.stem(x)
        feat = self.body(feat)
        unsigned_residual = torch.abs(self.to_rgb(feat)) * self.scale
        signed_residual = unsigned_residual * S_pixel
        return (x + signed_residual).clamp(-1.0, 1.0)


def s_lat_to_pixel(
    S_lat: torch.Tensor,
    target_h: int = 256,
    target_w: int = 256,
) -> torch.Tensor:
    """Convert (B, 4, 32, 32) latent sign envelope to (B, 1, H, W) pixel conditioning.

    Takes the channel-mean of S_lat then nearest-upsamples to pixel resolution.
    """
    if S_lat.dim() == 3:
        S_lat = S_lat.unsqueeze(0)
    s_mean = S_lat.mean(dim=1, keepdim=True)
    return F.interpolate(s_mean, size=(target_h, target_w), mode="nearest")


class SimpleDecoder(nn.Module):
    """Lightweight CNN decoder: (B, 3, 256, 256) → (B, n_bits) logits.

    Reference-free: takes only the suspect image, outputs region-sign logits.
    Does NOT use σ or M — those are applied externally after the CNN.
    """

    def __init__(self, n_bits: int = 127, base_ch: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4 * 4 * 4, 512),
            nn.GELU(),
            nn.Linear(512, n_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    enc = KeyConditionedEncoder(n_blocks=8, base_ch=64)
    print(f"encoder params: {count_params(enc):,}")

    x = torch.randn(2, 3, 256, 256)
    S = torch.randint(0, 2, (2, 1, 256, 256)).float() * 2 - 1
    y = enc(x, S)
    print(f"input: {x.shape}, output: {y.shape}")
    print(f"residual RMS: {(y - x).pow(2).mean().sqrt().item():.4f}")
