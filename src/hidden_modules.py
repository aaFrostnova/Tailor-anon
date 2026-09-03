"""HiDDeN-style learned spatial watermark (Zhu et al. 2018) with a geometric+compression
noise layer -- the route to a deployment-grade fragment robust to COMPOUND
geometry+compression, which the frequency carrier cannot do (EVALUATION_REPORT 6.1).

Why HiDDeN and not the U-Net we tried before: the message is replicated across every
spatial location and concatenated with image features (no bottleneck that starves the
channel), and the decoder is conv->global-avg-pool->FC (geometry-tolerant by pooling).
This architecture is known to train on modest budgets where the U-Net residual encoder
collapsed to chance.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())


class HiDDeNEncoder(nn.Module):
    def __init__(self, n_bits=64, ch=64, blocks=4, strength=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.strength = strength
        self.image_pre = nn.Sequential(_conv_bn(3, ch), *[_conv_bn(ch, ch) for _ in range(blocks - 1)])
        self.after_cat = _conv_bn(ch + n_bits + 3, ch)
        self.final = nn.Conv2d(ch, 3, 1)

    def forward(self, image, msg):           # image (B,3,H,W) in [0,1], msg (B,L) in {0,1}
        B, _, H, W = image.shape
        feat = self.image_pre(image)
        msg_map = msg.view(B, self.n_bits, 1, 1).expand(B, self.n_bits, H, W)
        x = torch.cat([feat, msg_map, image], dim=1)
        resid = self.final(self.after_cat(x))
        return torch.clamp(image + self.strength * resid, 0, 1)


class HiDDeNDecoder(nn.Module):
    def __init__(self, n_bits=64, ch=64, blocks=7):
        super().__init__()
        self.convs = nn.Sequential(_conv_bn(3, ch), *[_conv_bn(ch, ch) for _ in range(blocks - 1)],
                                   _conv_bn(ch, n_bits))
        self.fc = nn.Linear(n_bits, n_bits)

    def forward(self, image):
        x = self.convs(image).mean(dim=(2, 3))   # global avg pool -> geometry tolerance
        return self.fc(x)                          # logits


class NoiseLayer(nn.Module):
    """Differentiable attack layer: per-batch random geometric + compression + photometric.
    Compound (geometry AND compression together) is sampled to match deployment."""

    def __init__(self, resolution=128):
        super().__init__()
        self.res = resolution

    def _affine(self, x, smin, smax, rot, trans):
        B, dev = x.shape[0], x.device
        s = torch.empty(B, device=dev).uniform_(smin, smax)
        a = torch.empty(B, device=dev).uniform_(-rot, rot) * math.pi / 180
        tx = torch.empty(B, device=dev).uniform_(-trans, trans)
        ty = torch.empty(B, device=dev).uniform_(-trans, trans)
        ca, sa = torch.cos(a), torch.sin(a)
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = ca / s; th[:, 0, 1] = -sa / s; th[:, 0, 2] = tx
        th[:, 1, 0] = sa / s; th[:, 1, 1] = ca / s; th[:, 1, 2] = ty
        grid = F.affine_grid(th, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")

    def _jpeg_proxy(self, x):
        f = float(np.random.uniform(0.5, 0.9))
        h = max(8, int(x.shape[-1] * f))
        x = F.interpolate(F.interpolate(x, size=h, mode="bilinear", align_corners=False),
                          size=x.shape[-1], mode="bilinear", align_corners=False)
        q = float(np.random.choice([16, 24, 32]))
        return torch.round(x * q) / q

    def forward(self, x, geometric=True, compress=True):
        if geometric and torch.rand(()) < 0.85:
            x = self._affine(x, 0.8, 1.25, 12.0, 0.05)
        if compress and torch.rand(()) < 0.7:
            x = self._jpeg_proxy(x)
        if torch.rand(()) < 0.5:
            k = int(np.random.choice([3, 5]))
            x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)
        if torch.rand(()) < 0.6:
            x = x + torch.randn_like(x) * float(np.random.uniform(0, 0.05))
        return x.clamp(0, 1)
