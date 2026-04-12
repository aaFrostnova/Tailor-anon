"""
Minimal DDPM implementation for memorization testing.

Lightweight UNet + linear noise schedule, designed for CIFAR-10 (32x32).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Time embedding ---

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --- UNet building blocks ---

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# --- Simple UNet ---

class SimpleUNet(nn.Module):
    """Minimal UNet for 32×32 images. ~4M parameters."""

    def __init__(self, in_ch=3, base_ch=64, ch_mults=(1, 2, 4), time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        chs = [base_ch]
        ch = base_ch
        for mult in ch_mults:
            out_ch = base_ch * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_dim),
                ResBlock(out_ch, out_ch, time_dim),
            ]))
            self.downsamples.append(Downsample(out_ch))
            chs.append(out_ch)
            ch = out_ch

        # Middle
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.upsamples.append(Upsample(ch))
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch + out_ch, out_ch, time_dim),  # skip connection
                ResBlock(out_ch, out_ch, time_dim),
            ]))
            ch = out_ch

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        # Encoder
        skips = [x]
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                x = block(x, t_emb)
            skips.append(x)
            x = down(x)

        # Middle
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Decoder
        for blocks, up in zip(self.up_blocks, self.upsamples):
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            for block in blocks:
                x = block(x, t_emb)

        return self.final_conv(F.silu(self.final_norm(x)))


# --- DDPM noise schedule and sampling ---

class DDPM:
    """DDPM with linear noise schedule."""

    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.model = model
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1_ab * noise, noise

    def p_loss(self, x0):
        """Compute training loss (epsilon prediction)."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, shape, progress=False):
        """Generate samples via reverse diffusion."""
        self.model.eval()
        x = torch.randn(shape, device=self.device)

        steps = range(self.T - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Sampling")

        for t_val in steps:
            t = torch.full((shape[0],), t_val, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t)

            alpha = self.alphas[t_val]
            alpha_bar = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            mean = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise)

            if t_val > 0:
                z = torch.randn_like(x)
                x = mean + beta.sqrt() * z
            else:
                x = mean

        self.model.train()
        return x.clamp(-1, 1)
