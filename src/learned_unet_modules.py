"""U-Net residual encoder + from-scratch CNN decoder, jointly trained (StegaStamp/HiDDeN-style).

This is the "learned encoder" route the analytical QIM fragments deliberately
avoided. The earlier pixel attempt failed mainly because its decoder was an
ImageNet-pretrained classifier (EfficientNet/ConvNeXt) — those are trained to
be INVARIANT to small perturbations, the opposite of what a watermark reader
needs. Here the decoder is trained FROM SCRATCH so it can co-adapt to whatever
the encoder writes, and the encoder is a proper U-Net (not a 4-conv toy).

Encoder:  (image, payload) -> bounded residual -> watermarked image.
Decoder:  watermarked/attacked image -> per-bit logits (from scratch).
Pair with BCEWithLogitsLoss. Per-image random payload prevents constant collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ResidualUNetEncoder(nn.Module):
    """U-Net that maps (image, payload) to a bounded additive residual.

    Payload is projected to a small spatial map, upsampled to full resolution,
    and concatenated with the image as extra input channels (StegaStamp-style).
    The output residual is bounded by tanh * epsilon so PSNR is controlled.
    """

    def __init__(self, n_bits: int = 100, epsilon: float = 8.0 / 255.0,
                 base: int = 32, payload_hw: int = 32, payload_ch: int = 16):
        super().__init__()
        self.n_bits = n_bits
        self.epsilon = epsilon
        self.payload_hw = payload_hw
        self.payload_ch = payload_ch
        # Project payload to a payload_ch-channel spatial map (higher rank than
        # the original 3-channel 16x16 injection, which was too low-capacity for
        # 100 bits).
        self.payload_fc = nn.Linear(n_bits, payload_ch * payload_hw * payload_hw)

        # Encoder path (input = 3 image + payload_ch payload channels).
        self.enc1 = _conv_block(3 + payload_ch, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = _conv_block(base * 4, base * 4)

        # Decoder path with skip connections (upsample via bilinear + conv).
        self.up3 = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.dec3 = _conv_block(base * 4 + base * 4, base * 2)
        self.up2 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec2 = _conv_block(base * 2 + base * 2, base)
        self.up1 = nn.Conv2d(base, base, 3, padding=1)
        self.dec1 = _conv_block(base + base, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, image: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        p = self.payload_fc(payload).view(B, self.payload_ch, self.payload_hw, self.payload_hw)
        p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([image, p], dim=1)

        e1 = self.enc1(x)                      # (B, base,   H,   W)
        e2 = self.enc2(self.pool(e1))          # (B, 2base,  H/2, W/2)
        e3 = self.enc3(self.pool(e2))          # (B, 4base,  H/4, W/4)
        b = self.bott(self.pool(e3))           # (B, 4base,  H/8, W/8)

        d3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(d3), e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))

        residual = torch.tanh(self.out(d1)) * self.epsilon
        return torch.clamp(image + residual, 0.0, 1.0)


class ConvDecoder(nn.Module):
    """From-scratch CNN reader: image -> n_bits logits.

    Crucially NOT pretrained on ImageNet — pretrained classifiers are invariant
    to the small perturbations a watermark relies on.

    Two fixes over the first (collapsed) version, prompted by the diagnosis that
    BCE stayed flat even under pressure-free warmup:
      - The head FLATTENS a small spatial map (downsample to a fixed size) rather
        than global-average-pooling to 1x1. Global pooling destroys the spatial
        localisation a learned encoder uses to place bits, so the decoder could
        never read a spatially-distributed code.
      - GroupNorm instead of BatchNorm: at batch_size 8, BatchNorm statistics are
        noisy and fight the tiny watermark signal.
    """

    def __init__(self, n_bits: int = 100, base: int = 32, out_hw: int = 4):
        super().__init__()
        self.out_hw = out_hw

        def down(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.ReLU(inplace=True),
            )
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 (six downs); AdaptiveAvgPool to
        # a fixed out_hw x out_hw map preserves coarse spatial layout.
        self.net = nn.Sequential(
            down(3, base),          # H/2
            down(base, base),       # H/4
            down(base, base * 2),   # H/8
            down(base * 2, base * 2),  # H/16
            down(base * 2, base * 4),  # H/32
            down(base * 4, base * 4),  # H/64
            nn.AdaptiveAvgPool2d(out_hw),
            nn.Flatten(),
        )
        feat = base * 4 * out_hw * out_hw
        self.head = nn.Sequential(
            nn.Linear(feat, base * 8),
            nn.ReLU(inplace=True),
            nn.Linear(base * 8, n_bits),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(image))
