"""Learned region-sign decoder for v4 latent-domain watermark.

Replaces the matched-filter operation in `recover_codeword_from_residual`
with a small region-aware CNN. Crypto-envelope descrambling (sigma^-1 +
XOR M) and BCH decoding stay external -- the CNN only reads (residual,
T_lat) and emits 127 region-sign logits in *region order*, never touches
the secret key.

Architecture is "matched-filter with learnable bells and whistles":
  - Input: concat(r_lat, T_lat, r_lat * T_lat)  -> (B, 12, 32, 32)
  - 2 conv layers @ 32x32 (no spatial reduction)
  - Per-region masked mean over feature map  -> (B, feat_ch, 127)
  - 1x1 conv (feat_ch -> 1) per region        -> (B, 127)

The per-region pooling gives the network a strong inductive bias: each
output position can only see the features within its corresponding 32x32
region. This matches the matched-filter structure (sum within region) and
makes the optimisation tractable on tens of training images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDecoderXR(nn.Module):
    """Region-aware decoder with cross-region (transformer-style) head.

    Same per-voxel stem as LatentDecoder, same per-region masked mean pool,
    but the resulting (B, n_bits, feat_ch) token sequence is then refined
    by a small transformer encoder before the per-region linear head. This
    lets each region read evidence from other regions -- the one piece of
    information matched filter explicitly cannot use.

    The transformer is applied over the region axis (sequence length =
    n_bits). Positional embedding is learned, one per region index.
    """

    def __init__(
        self,
        in_ch_residual: int = 4,
        n_bits: int = 127,
        feat_ch: int = 64,
        region_masks: torch.Tensor = None,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        in_ch = 3 * in_ch_residual

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_ch),
            nn.GELU(),
        )

        # region masks for per-region pooling — spatial-only
        if region_masks is None:
            raise ValueError("LatentDecoderXR requires region_masks")
        rm = region_masks.float()
        if rm.dim() == 4:
            rm = rm[:, 0]
        n, H, W = rm.shape
        rm_flat = rm.view(n, H * W)
        rm_flat = rm_flat / (rm_flat.sum(dim=1, keepdim=True) + 1e-12)
        self.register_buffer("region_masks_flat", rm_flat)

        # learned positional embedding per region
        self.pos_embed = nn.Parameter(torch.zeros(n_bits, feat_ch))
        nn.init.normal_(self.pos_embed, std=0.02)

        # transformer encoder over region tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_ch,
            nhead=n_heads,
            dim_feedforward=feat_ch * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # per-region linear head
        self.head_weight = nn.Parameter(torch.zeros(n_bits, feat_ch))
        nn.init.kaiming_uniform_(self.head_weight, a=5 ** 0.5)
        self.head_bias = nn.Parameter(torch.zeros(n_bits))

        self.n_bits = n_bits
        self.feat_ch = feat_ch

    def forward(self, r_lat: torch.Tensor, T_lat: torch.Tensor) -> torch.Tensor:
        if T_lat.dim() == 3:
            T_lat = T_lat.unsqueeze(0)
        B = r_lat.shape[0]
        T_b = T_lat.expand(B, -1, -1, -1)
        x = torch.cat([r_lat, T_b, r_lat * T_b], dim=1)
        feat = self.stem(x)                                      # (B, feat_ch, H, W)

        # per-region masked mean -> (B, n_bits, feat_ch)
        flat = feat.view(B, self.feat_ch, -1)
        pooled = flat @ self.region_masks_flat.T                  # (B, feat_ch, n_bits)
        tokens = pooled.transpose(1, 2)                           # (B, n_bits, feat_ch)
        tokens = tokens + self.pos_embed.unsqueeze(0)             # broadcast

        refined = self.transformer(tokens)                        # (B, n_bits, feat_ch)

        # per-region linear head
        logits = (refined * self.head_weight.unsqueeze(0)).sum(dim=-1) + self.head_bias
        return logits


class LatentDecoder(nn.Module):
    """Region-aware learned decoder.

    Args:
        in_ch_residual: latent channels (4 for SD-v1-5 VAE).
        n_bits: number of regions = number of output logits (127).
        feat_ch: width of internal feature map.
        region_masks: (n_bits, latent_C, latent_H, latent_W) bool / float.
                      If provided at construction, registered as a buffer
                      and used for per-region masked-mean pooling.
                      If None, pooling is global mean -> linear (fallback).
    """

    def __init__(
        self,
        in_ch_residual: int = 4,
        n_bits: int = 127,
        feat_ch: int = 64,
        region_masks: torch.Tensor = None,
    ):
        super().__init__()
        # Input has 3 streams: r_lat, T_lat, r_lat * T_lat -> 3 * in_ch_residual
        in_ch = 3 * in_ch_residual

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_ch),
            nn.GELU(),
        )

        # Per-region linear head: each region gets its own (feat_ch -> 1)
        # implemented as one parameter tensor (n_bits, feat_ch + 1) for
        # weight + bias.
        self.head_weight = nn.Parameter(torch.zeros(n_bits, feat_ch))
        nn.init.kaiming_uniform_(self.head_weight, a=5 ** 0.5)
        self.head_bias = nn.Parameter(torch.zeros(n_bits))

        # region masks for per-region pooling — spatial-only (drop channel dim)
        if region_masks is not None:
            rm = region_masks.float()
            if rm.dim() == 4:
                # (n_bits, C, H, W) -> (n_bits, H, W) by taking channel-0
                # (all channels share the same spatial mask in our setup)
                rm = rm[:, 0]
            # rm shape: (n_bits, H, W)
            n, H, W = rm.shape
            rm_flat = rm.view(n, H * W)
            # normalize to mean (each row sums to 1)
            rm_flat = rm_flat / (rm_flat.sum(dim=1, keepdim=True) + 1e-12)
            self.register_buffer("region_masks_flat", rm_flat)   # (n_bits, H*W)
            self.has_regions = True
        else:
            self.has_regions = False

        self.n_bits = n_bits
        self.feat_ch = feat_ch

    def forward(self, r_lat: torch.Tensor, T_lat: torch.Tensor) -> torch.Tensor:
        """r_lat: (B, C, H, W); T_lat: (C, H, W) or (1, C, H, W);
        returns (B, n_bits) logits.
        """
        if T_lat.dim() == 3:
            T_lat = T_lat.unsqueeze(0)
        B = r_lat.shape[0]
        T_b = T_lat.expand(B, -1, -1, -1)
        x = torch.cat([r_lat, T_b, r_lat * T_b], dim=1)   # (B, 3C, H, W)

        feat = self.stem(x)                                # (B, feat_ch, H, W)

        if not self.has_regions:
            # global mean fallback -> linear
            pooled = feat.flatten(2).mean(dim=2)            # (B, feat_ch)
            return pooled @ self.head_weight.T + self.head_bias

        # per-region masked mean: (B, feat_ch, V) @ (n_bits, V).T = (B, feat_ch, n_bits)
        flat = feat.view(B, self.feat_ch, -1)
        # region-pooled features: (B, feat_ch, n_bits)
        pooled = flat @ self.region_masks_flat.T
        # logits[b, i] = sum_c head_weight[i, c] * pooled[b, c, i] + head_bias[i]
        logits = (pooled * self.head_weight.T.unsqueeze(0)).sum(dim=1) + self.head_bias
        return logits


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick shape check
    n_bits = 127
    region_masks = torch.zeros(n_bits, 4, 32, 32, dtype=torch.bool)
    # fake region mask: 12x12 grid
    g = 12
    cell_h = 32 // g
    for i in range(n_bits):
        r, c = i // g, i % g
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < g - 1 else 32
        x0 = c * cell_h
        x1 = (c + 1) * cell_h if c < g - 1 else 32
        region_masks[i, :, y0:y1, x0:x1] = True

    m = LatentDecoder(region_masks=region_masks)
    print(f"params: {count_params(m):,}")
    r_lat = torch.randn(2, 4, 32, 32)
    T_lat = torch.randn(4, 32, 32)
    logits = m(r_lat, T_lat)
    print(f"output shape: {logits.shape}")
