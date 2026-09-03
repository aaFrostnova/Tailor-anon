"""Resize/rotation-tolerant ref-free fragment: log-polar magnitude decoder.

The fixed-bin DFT reader dies under resize because scaling moves the carrier off
its bin; augmentation alone cannot fix a reader pinned to absolute bins. This
decoder instead resamples the Fourier magnitude to LOG-POLAR coordinates, where a
spatial scaling is a SHIFT along the log-radius axis and a rotation is a circular
shift along the angle axis (Fourier-Mellin). A CNN over that log-polar image, with
circular padding on the angle axis and trained under resize/rotation augmentation,
learns a geometry-tolerant readout. The encoder stays the analytical DFT-magnitude
QIM carrier (frozen, no collapse); only this decoder is trained.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAwareFFTDecoder(nn.Module):
    """FFTAware-style known-position reader that samples carriers at a given
    (scale, rotation), so it can be TRAINED to read the value-changed magnitudes
    after a geometric transform (value-robust via per-image magnitude norm), and
    at inference scale/rotation-searched. Carriers + bit_flip come from the
    analytical encoder; only this decoder trains.
    """

    def __init__(self, carriers, bit_flip, resolution=256, channel=1, hidden=64,
                 init_delta=50.0):
        super().__init__()
        self.resolution = resolution
        self.channel = channel
        self.n_bits, self.n_pos = carriers.shape[0], carriers.shape[1]
        self.register_buffer("carriers", carriers.long())              # (n_bits,n_pos,2) raw bins
        sign = (1 - 2 * bit_flip.long()).float()
        self.register_buffer("flip_sign", sign)                        # (n_bits,)
        # signed frequency coords (DC-centered), normalized for grid_sample
        fy = carriers[..., 0].reshape(-1).float()
        fx = carriers[..., 1].reshape(-1).float()
        sy = torch.where(fy > resolution / 2, fy - resolution, fy)
        sx = torch.where(fx > resolution / 2, fx - resolution, fx)
        self.register_buffer("sy", sy)                                 # (n_carriers,)
        self.register_buffer("sx", sx)
        self.decode_delta = nn.Parameter(torch.tensor(float(init_delta - 1.0)))
        self.pos_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.bit_head = nn.Linear(hidden, 1)

    def forward(self, image, scale=1.0, angle=0.0):
        B = image.shape[0]
        res = self.resolution
        dev = image.device
        if not torch.is_tensor(scale):
            scale = torch.full((B,), float(scale), device=dev)
        if not torch.is_tensor(angle):
            angle = torch.full((B,), float(angle), device=dev)
        scale = scale.to(dev).view(B, 1)
        ca = torch.cos(angle).to(dev).view(B, 1)
        sa = torch.sin(angle).to(dev).view(B, 1)
        green = image[:, self.channel]
        mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(green), dim=(-2, -1)))
        mag = mag.unsqueeze(1)                                         # (B,1,res,res)
        sy = self.sy.view(1, -1); sx = self.sx.view(1, -1)            # (1,n_carriers)
        ry = (ca * sy - sa * sx) / scale                              # (B,n_carriers)
        rx = (sa * sy + ca * sx) / scale
        c = res / 2.0
        gx = (c + rx) / (res - 1) * 2 - 1
        gy = (c + ry) / (res - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)             # (B,1,n_carriers,2)
        m = F.grid_sample(mag, grid, mode="bilinear", align_corners=True,
                          padding_mode="reflection")[:, 0, 0, :]       # (B,n_carriers)
        delta = F.softplus(self.decode_delta) + 1.0
        phase = 2.0 * math.pi * m / delta
        mnorm = m / (m.mean(dim=1, keepdim=True) + 1e-6)
        feats = torch.stack([torch.cos(phase), torch.sin(phase), torch.tanh(mnorm)], dim=-1)
        feats = feats.view(B, self.n_bits, self.n_pos, 3)
        h = self.pos_mlp(feats).mean(dim=2)
        eff = self.bit_head(h).squeeze(-1)
        return eff * self.flip_sign.unsqueeze(0)


def _logpolar_grid(n_rho, n_theta, r_lo, r_hi, resolution):
    """Sampling grid (normalized for grid_sample) over the fftshifted magnitude."""
    rhos = r_lo * (r_hi / r_lo) ** np.linspace(0.0, 1.0, n_rho)        # log-spaced radius
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    c = resolution / 2.0
    R, T = np.meshgrid(rhos, thetas, indexing="ij")                    # (n_rho, n_theta)
    fx = R * np.cos(T)
    fy = R * np.sin(T)
    x = (c + fx) / (resolution - 1) * 2.0 - 1.0                        # width  -> x
    y = (c + fy) / (resolution - 1) * 2.0 - 1.0                        # height -> y
    grid = np.stack([x, y], axis=-1).astype(np.float32)                # (n_rho, n_theta, 2)
    return torch.from_numpy(grid).unsqueeze(0)                         # (1, n_rho, n_theta, 2)


class _ConvBlock(nn.Module):
    """Conv with circular padding on the angle (theta, width) axis and reflect on rho."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=0)
        self.norm = nn.GroupNorm(8, cout)
        self.act = nn.GELU()

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")   # theta wrap
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")  # rho edges
        return self.act(self.norm(self.conv(x)))


class LogPolarDecoder(nn.Module):
    def __init__(self, n_bits=100, n_rho=48, n_theta=96, r_lo=8.0, r_hi=110.0,
                 resolution=256, channel=1, width=64, pool_hw=4):
        super().__init__()
        self.n_bits = n_bits
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.resolution = resolution
        self.channel = channel
        self.register_buffer("grid", _logpolar_grid(n_rho, n_theta, r_lo, r_hi, resolution))
        self.stem = nn.Conv2d(1, width, 1)
        self.blocks = nn.Sequential(
            _ConvBlock(width, width),
            _ConvBlock(width, width * 2, stride=2),
            _ConvBlock(width * 2, width * 2),
            _ConvBlock(width * 2, width * 4, stride=2),
            _ConvBlock(width * 4, width * 4),
        )
        # Adaptive pool keeps a little spatial structure for bit routing while
        # giving translation tolerance (scale=rho shift, rotation=theta shift).
        self.pool = nn.AdaptiveAvgPool2d((pool_hw, pool_hw))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * pool_hw * pool_hw, width * 4), nn.GELU(),
            nn.Linear(width * 4, n_bits),
        )

    def _logpolar(self, image):
        green = image[:, self.channel]
        Fc = torch.fft.fft2(green)
        mag = torch.abs(torch.fft.fftshift(Fc, dim=(-2, -1)))          # (B, H, W), DC center
        mag = torch.log1p(mag).unsqueeze(1)                            # compress dynamic range
        B = image.shape[0]
        lp = F.grid_sample(mag, self.grid.expand(B, -1, -1, -1),
                           mode="bilinear", align_corners=True)        # (B,1,n_rho,n_theta)
        m = lp.mean(dim=(2, 3), keepdim=True)
        s = lp.std(dim=(2, 3), keepdim=True) + 1e-5
        return (lp - m) / s

    def forward(self, image):
        x = self._logpolar(image)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)                                            # (B, n_bits) logits
