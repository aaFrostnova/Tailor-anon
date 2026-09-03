"""Pixel-domain VINE-style learned fragment.

Encoder takes (image, payload) -> watermarked image; decoder takes image ->
recovered payload. The payload is a per-image random 100-bit string at training
time, so the decoder must look at the image to recover it (no degenerate
constant solution). The perturbation is hard-bounded to [-epsilon, +epsilon] via
tanh, so PSNR is bounded above ~35 dB by construction at epsilon = 8/255.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PixelEncoder(nn.Module):
    """Project payload to a spatial feature map, concat with image, predict delta."""

    def __init__(
        self,
        n_bits: int = 100,
        epsilon: float = 8.0 / 255.0,
        hidden: int = 64,
        payload_hw: int = 8,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.epsilon = epsilon
        self.hidden = hidden
        self.payload_hw = payload_hw

        self.payload_proj = nn.Linear(n_bits, hidden * payload_hw * payload_hw)

        self.net = nn.Sequential(
            nn.Conv2d(3 + hidden, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        # Initialize the final conv with large weights so the encoder produces
        # a non-trivial perturbation from step 1. Without this, default init
        # produces near-zero delta; combined with sigmoid-saturated decoder
        # gradients, training converges to the trivial zero-perturbation /
        # zero-logit fixed point and never escapes.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, img: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) in [0, 1]; payload: (B, n_bits) in {0., 1.}."""
        B, _, H, W = img.shape
        feat = self.payload_proj(payload).view(B, self.hidden, self.payload_hw, self.payload_hw)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([img, feat], dim=1)
        delta = self.net(x)
        # Scale before tanh so we saturate easily — initial outputs of ±0.5
        # become ±tanh(2.5)≈±0.99, so the encoder uses essentially the full ε
        # budget from the start. Once the decoder picks up the signal, BCE
        # gradient can shape the *content* of the perturbation within budget.
        delta = torch.tanh(delta * 5.0) * self.epsilon
        return torch.clamp(img + delta, 0.0, 1.0)


class PixelDecoder(nn.Module):
    """EfficientNet-B0 backbone + MLP head -> n_bits LOGITS (no sigmoid).

    Outputting raw logits avoids the sigmoid saturation that traps BCE gradient
    flow at random init (when decoder output ≈ 0.5 → ∂BCE/∂out ≈ 0, encoder
    receives no signal and collapses to zero perturbation). Apply
    torch.sigmoid() at inference time for the bit decision.
    """

    def __init__(self, n_bits: int = 100):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, n_bits),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(img))
