"""DFT-Kred encoder + decoder modules for VINE-style training.

The encoder is the analytical DFT-magnitude carrier (with K-redundancy on M
radii bands, exactly as in scripts/prototype_dft_kredundant.py) lifted into
PyTorch with a single learnable parameter: the magnitude quantization step
delta. Carrier POSITIONS are fixed analytically from a canonical key — there
is no spatial pattern to learn, so the encoder cannot collapse to identity.

The decoder is a standard ConvNeXt-Base classifier head producing per-bit
logits, to be paired with BCEWithLogitsLoss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# NOTE: prototype_dft_carrier.enumerate_annulus_bins returns (fy, fx) tuples,
# whereas prototype_dft_kredundant.enumerate_annulus_bins returns
# (fy, fx, r) tuples with the radius (required by setup_k_redundant_carriers
# for binning by radius). We use the kred variant for carrier setup; the
# simple variant is imported here for parity with the design intent.
from scripts.prototype_dft_carrier import enumerate_annulus_bins  # noqa: F401
from scripts.prototype_dft_kredundant import enumerate_annulus_bins as enum_annulus_kred
from scripts.prototype_dft_kredundant import setup_k_redundant_carriers


class DFTKredEncoder(nn.Module):
    """Analytical FFT embedding with a single learnable base_delta parameter.

    The carrier positions are FIXED at construction time using a canonical
    training key. Only the magnitude quantization step delta is learned, so
    the model has no way to escape into a zero-perturbation collapse — the
    FFT modifications happen at fixed bins regardless of input image content.
    """

    def __init__(
        self,
        n_bits: int = 100,
        K: int = 5,
        M: int = 2,
        resolution: int = 256,
        r_lo: float = 20.0,
        r_hi: float = 60.0,
        canonical_key: bytes = b"vine_training_canonical_key",
        canonical_id: str = "dft_kred_train_canonical",
        init_delta: float = 50.0,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.K = K
        self.M = M
        self.resolution = resolution
        self.r_lo = r_lo
        self.r_hi = r_hi
        # softplus(base_delta) + 1 gives the actual delta used; init so that
        # softplus(base_delta) + 1 ~= init_delta. For init_delta=50 this is
        # simply base_delta = init_delta - 1 (softplus(x) ~ x for large x).
        self.base_delta = nn.Parameter(torch.tensor(float(init_delta - 1.0)))

        # Build fixed canonical carriers via the prototype's setup function.
        candidates = enum_annulus_kred(resolution, resolution, r_lo, r_hi)
        carriers_list, bit_flip = setup_k_redundant_carriers(
            canonical_key, canonical_id, n_bits, candidates, K, M,
        )
        # carriers_list: list of n_bits lists, each of K*M (fy, fx) tuples.
        # Flatten to a tensor (n_bits, K*M, 2) for fast indexed access.
        carriers_t = torch.tensor(
            [[(fy, fx) for (fy, fx) in pos] for pos in carriers_list],
            dtype=torch.long,
        )
        self.register_buffer("carriers", carriers_t)  # (n_bits, K*M, 2)
        self.register_buffer(
            "bit_flip", torch.tensor(bit_flip, dtype=torch.long)  # (n_bits,)
        )

    def forward(self, image: torch.Tensor, secret_bits: torch.Tensor) -> torch.Tensor:
        """Embed bits into the green channel via FFT-magnitude QIM.

        Args:
            image: (B, 3, H, W) float in [0, 1].
            secret_bits: (B, n_bits) float in {0., 1.}.

        Returns:
            Watermarked image (B, 3, H, W) clamped to [0, 1].
        """
        B, _, H, W = image.shape
        assert H == W == self.resolution, (
            f"DFTKredEncoder expects HxW={self.resolution}; got {H}x{W}"
        )
        delta = F.softplus(self.base_delta) + 1.0  # > 1

        green = image[:, 1]  # (B, H, W)
        Fc = torch.fft.fft2(green)  # complex (B, H, W)

        # Effective bits after sign-mask XOR. (B, n_bits)
        secret_int = secret_bits.long()
        eff_bits = (secret_int ^ self.bit_flip.unsqueeze(0)).float()

        # Vectorised carrier modification.
        # Flatten (n_bits, K*M) -> (n_carriers,).
        fy = self.carriers[..., 0].reshape(-1)  # (n_carriers,)
        fx = self.carriers[..., 1].reshape(-1)
        n_pos_per_bit = self.K * self.M
        # Expand eff_bits per-position: (B, n_bits, K*M) -> (B, n_carriers).
        eff_per_carrier = (
            eff_bits.unsqueeze(-1)
            .expand(-1, -1, n_pos_per_bit)
            .reshape(B, -1)
        )

        # Gather current complex values: (B, n_carriers).
        Fc_sel = Fc[:, fy, fx]
        mag = torch.abs(Fc_sel)
        phase = torch.angle(Fc_sel)
        target_mag = (
            torch.round(mag / delta) * delta + (delta / 2.0) * eff_per_carrier
        )
        new_val = target_mag * torch.exp(1j * phase)
        # Avoid in-place autograd issues by cloning before scatter.
        Fc = Fc.clone()
        Fc[:, fy, fx] = new_val
        # Hermitian conjugate pair so ifft2 stays real.
        fy_c = (-fy) % H
        fx_c = (-fx) % W
        Fc[:, fy_c, fx_c] = torch.conj(new_val)

        green_w = torch.fft.ifft2(Fc).real

        out = image.clone()
        out[:, 1] = torch.clamp(green_w, 0.0, 1.0)
        return out


class DFTKredDecoder(nn.Module):
    """ConvNeXt-Base classifier learning to recover bits from the watermarked
    (possibly attacked) image. Outputs per-bit logits; pair with
    BCEWithLogitsLoss."""

    def __init__(self, n_bits: int = 100, pretrained: bool = False):
        super().__init__()
        weights = (
            models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.convnext_base(weights=weights)
        # ConvNeXt's classifier: [LayerNorm2d, Flatten, Linear].
        in_feat = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_feat, n_bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class FFTAwareDecoder(nn.Module):
    """Frequency-domain decoder that reads the DFT-Kred carriers directly.

    The frozen-encoder diagnostic showed a spatial CNN cannot recover the
    frequency-domain watermark. This decoder instead takes the same fixed
    carrier positions the encoder used, computes the FFT magnitude of the green
    channel, and reads a differentiable soft-QIM feature at each carrier. The
    discriminative feature cos(2*pi*m/delta) equals +1 when the magnitude sits
    on the bit-0 lattice (m mod delta ~ 0) and -1 on the bit-1 lattice
    (m mod delta ~ delta/2). A small shared MLP turns the per-position features
    into per-bit evidence, aggregates over the K*M redundant positions, and the
    sign mask (bit_flip) is applied analytically. Decode delta is learnable so
    the decoder can adapt to attack-induced magnitude rescaling.
    """

    def __init__(
        self,
        n_bits: int = 100,
        K: int = 5,
        M: int = 2,
        resolution: int = 256,
        r_lo: float = 20.0,
        r_hi: float = 60.0,
        canonical_key: bytes = b"vine_training_canonical_key",
        canonical_id: str = "dft_kred_train_canonical",
        init_delta: float = 50.0,
        channel: int = 1,
        hidden: int = 32,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.K = K
        self.M = M
        self.resolution = resolution
        self.channel = channel
        self.n_pos = K * M
        self.decode_delta = nn.Parameter(torch.tensor(float(init_delta - 1.0)))

        # Build the SAME canonical carriers + bit_flip as the encoder.
        candidates = enum_annulus_kred(resolution, resolution, r_lo, r_hi)
        carriers_list, bit_flip = setup_k_redundant_carriers(
            canonical_key, canonical_id, n_bits, candidates, K, M,
        )
        carriers_t = torch.tensor(
            [[(fy, fx) for (fy, fx) in pos] for pos in carriers_list],
            dtype=torch.long,
        )
        self.register_buffer("carriers", carriers_t)  # (n_bits, K*M, 2)
        # Sign factor (1 - 2*bit_flip) in {+1, -1} for analytical un-flip.
        sign = (1 - 2 * torch.tensor(bit_flip, dtype=torch.long)).float()
        self.register_buffer("flip_sign", sign)  # (n_bits,)

        # Per-position MLP: [cos, sin, mag_norm] -> hidden -> hidden.
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        # Per-bit head applied after aggregation over positions.
        self.bit_head = nn.Linear(hidden, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) in [0, 1]. Returns (B, n_bits) logits."""
        B = image.shape[0]
        H = W = self.resolution
        green = image[:, self.channel]
        Fc = torch.fft.fft2(green)

        fy = self.carriers[..., 0].reshape(-1)
        fx = self.carriers[..., 1].reshape(-1)
        mag = torch.abs(Fc[:, fy, fx])             # (B, n_carriers)

        delta = F.softplus(self.decode_delta) + 1.0
        phase = 2.0 * torch.pi * mag / delta
        # Normalise magnitude per image for the third feature (scale-robust).
        mag_norm = mag / (mag.mean(dim=1, keepdim=True) + 1e-6)
        feats = torch.stack(
            [torch.cos(phase), torch.sin(phase), torch.tanh(mag_norm)], dim=-1
        )                                          # (B, n_carriers, 3)

        feats = feats.view(B, self.n_bits, self.n_pos, 3)
        h = self.pos_mlp(feats)                    # (B, n_bits, K*M, hidden)
        h = h.mean(dim=2)                          # (B, n_bits, hidden)
        eff_logit = self.bit_head(h).squeeze(-1)   # (B, n_bits) eff-bit logit
        # Recover secret logit from effective-bit logit via the sign mask.
        secret_logit = eff_logit * self.flip_sign.unsqueeze(0)
        return secret_logit
