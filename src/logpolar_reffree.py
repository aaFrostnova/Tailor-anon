"""Reference-free RST-tolerant readout for the DFT-magnitude carrier fragment.

The fixed-bin DFT fragment dies under resize because scaling the image scales the
Fourier coordinates: a carrier embedded at frequency-radius r moves to r/a after a
spatial enlargement by a. In the log-polar map of the Fourier magnitude that is a
SHIFT along the log-radius axis (and a rotation is a shift along the angle axis) --
the Fourier-Mellin invariance. We exploit it WITHOUT retraining: at decode we search
over the unknown scale (and optionally rotation), sampling the FFT magnitude at the
carrier positions transformed by the candidate (scale, angle), and pick the transform
under which the learned decoder is most confident (largest mean |logit|). This is
self-synchronizing and reference-free -- it needs neither the original image nor a
separate template, only the carrier geometry the decoder already owns.

Reuses an FFTAwareDecoder (carriers, decode_delta, pos_mlp, bit_head, flip_sign) loaded
from a trained DFT-Kred checkpoint.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import map_coordinates
    _MODE = "grid-wrap"
except Exception:  # pragma: no cover
    map_coordinates = None


def _sample_carrier_mag(mag_full, sy, sx, a, theta):
    """Bilinear-sample the (periodic) FFT magnitude at carrier positions transformed
    by scale a and rotation theta (radians), around the DC bin."""
    res = mag_full.shape[0]
    ca, sa = np.cos(theta), np.sin(theta)
    # rotate then scale the signed frequency coordinates, then sample radius/a
    ry = (ca * sy - sa * sx) / a
    rx = (sa * sy + ca * sx) / a
    yy = ry % res
    xx = rx % res
    try:
        return map_coordinates(mag_full, [yy, xx], order=1, mode="grid-wrap")
    except Exception:
        return map_coordinates(mag_full, [yy, xx], order=1, mode="wrap")


def mellin_analytical(carriers, bit_flip, delta, green, scales, angles=(0.0,)):
    """Fully analytical Fourier-Mellin decode with redundancy-consensus sync.

    Needs only the (analytical) encoder's carriers + bit_flip + delta -- no trained
    decoder. For each candidate (scale, rotation), sample the magnitude at the
    transformed carriers, take the per-carrier QIM sign, fold in the bit-flip mask
    to get a per-carrier SECRET vote, and measure the cross-carrier consensus. The
    transform whose redundant carriers most agree is the recovered geometry; decode
    the secret by majority vote there.

    carriers: (n_bits, n_pos, 2) int; bit_flip: (n_bits,) uint8; green: (res,res).
    Returns {bits, sync, scale, angle, consensus_mean}.
    """
    green = np.asarray(green, dtype=np.float64)
    res = green.shape[0]
    mag_full = np.abs(np.fft.fft2(green))
    carriers = np.asarray(carriers)
    n_bits, n_pos = carriers.shape[0], carriers.shape[1]
    fy = carriers[..., 0].reshape(-1).astype(np.float64)
    fx = carriers[..., 1].reshape(-1).astype(np.float64)
    sy = np.where(fy > res / 2, fy - res, fy)
    sx = np.where(fx > res / 2, fx - res, fx)
    flip_sign = (1.0 - 2.0 * np.asarray(bit_flip, dtype=np.float64))   # (n_bits,)
    best = None
    for th in angles:
        for a in scales:
            mag = _sample_carrier_mag(mag_full, sy, sx, a, th).reshape(n_bits, n_pos)
            cosv = np.cos(2.0 * np.pi * mag / delta)
            secret_vote = np.sign(cosv) * flip_sign[:, None]          # +1 -> bit0, -1 -> bit1
            consensus = np.abs(secret_vote.mean(axis=1))              # (n_bits,)
            sync = float(consensus.mean())
            if best is None or sync > best["sync"]:
                bits = (secret_vote.sum(axis=1) < 0).astype(np.uint8)
                best = {"bits": bits, "sync": sync, "scale": float(a),
                        "angle": float(th), "consensus_mean": sync}
    return best


@torch.no_grad()
def scale_search_logits(
    dec,
    green: np.ndarray,
    scales: Sequence[float],
    angles: Sequence[float] = (0.0,),
    device: str = "cuda",
):
    """Return (best_logits[n_bits], best_scale, best_angle, best_conf).

    dec: a loaded FFTAwareDecoder. green: (res,res) float in [0,1].
    """
    res = dec.resolution
    Fc = np.fft.fft2(green.astype(np.float64))
    mag_full = np.abs(Fc)                                  # unshifted, periodic
    carr = dec.carriers.cpu().numpy().reshape(-1, 2).astype(np.float64)
    fy, fx = carr[:, 0], carr[:, 1]
    sy = np.where(fy > res / 2, fy - res, fy)              # signed frequency
    sx = np.where(fx > res / 2, fx - res, fx)
    delta = float(F.softplus(dec.decode_delta) + 1.0)
    n_bits, n_pos = dec.n_bits, dec.n_pos
    flip = dec.flip_sign.to(device)

    best = None
    for theta in angles:
        for a in scales:
            mag = _sample_carrier_mag(mag_full, sy, sx, a, theta)
            mag_t = torch.tensor(mag, dtype=torch.float32, device=device).unsqueeze(0)
            phase = 2.0 * np.pi * mag_t / delta
            mag_norm = mag_t / (mag_t.mean(dim=1, keepdim=True) + 1e-6)
            feats = torch.stack(
                [torch.cos(phase), torch.sin(phase), torch.tanh(mag_norm)], dim=-1)
            feats = feats.view(1, n_bits, n_pos, 3)
            h = dec.pos_mlp(feats).mean(dim=2)
            eff = dec.bit_head(h).squeeze(-1)
            logit = (eff * flip.unsqueeze(0)).squeeze(0)
            conf = float(logit.abs().mean())
            if best is None or conf > best[3]:
                best = (logit.cpu().numpy(), float(a), float(theta), conf)
    return best
