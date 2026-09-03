"""Just-Noticeable-Difference (JND) perceptual mask for fingerprint embedding.

Modulates the per-pixel perturbation budget so that high-energy regions
(textures, edges, dark areas) can absorb more perturbation than low-energy
flat regions where artifacts are more visible. Used by
`embed_multi_domain_signed` to keep PSNR high under heavy signed embedding.

Reference: simplified Watson-style spatial masking — luminance + local
variance, normalized to [jnd_min, jnd_max]. No DCT-domain CSF weighting
(those would require a different formulation per strategy).
"""

import numpy as np
from typing import Tuple


def compute_jnd_mask(
    image: np.ndarray,
    jnd_min: float = 0.3,
    jnd_max: float = 2.5,
    sigma: float = 4.0,
) -> np.ndarray:
    """Compute a per-pixel perturbation scale in [jnd_min, jnd_max].

    Heuristic:
      1. Convert (C, H, W) float[0,1] image to luminance.
      2. Compute local luminance and local variance via box filters.
      3. mask ∝ luminance_factor × variance_factor, then rescale to
         [jnd_min, jnd_max] using percentile clamping.

    Areas with high texture and mid-luminance get the largest scale; flat
    bright or flat dark regions get the smallest.

    Args:
        image: (C, H, W) float in [0, 1].
        jnd_min: lower clamp of the output mask (0 < jnd_min < 1).
        jnd_max: upper clamp (jnd_max > 1).
        sigma: size (pixels) of the local box filter used for luminance /
            variance smoothing.

    Returns:
        Per-pixel scalar mask of shape (1, H, W) so it broadcasts onto (C,H,W).
    """
    if image.ndim != 3:
        raise ValueError(f"expected (C,H,W), got {image.shape}")
    C, H, W = image.shape

    # luminance approximation (Rec. 709 weights when C==3, else channel mean)
    if C == 3:
        lum = 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
    else:
        lum = image.mean(axis=0)
    lum = lum.astype(np.float32)

    # Box-filter via cumulative-sum trick for local mean + variance
    k = max(1, int(round(sigma)))
    mean_lum = _box_mean(lum, k)
    mean_sq = _box_mean(lum * lum, k)
    var = np.clip(mean_sq - mean_lum ** 2, 0, None)

    # Luminance factor: weakest at extremes (clipping risk), strongest at mid
    # Use a triangle peaking at 0.5
    lum_factor = 1.0 - 2.0 * np.abs(mean_lum - 0.5)
    lum_factor = np.clip(lum_factor, 0.1, 1.0)

    # Variance factor: scale by sqrt(var) (texture energy)
    tex_factor = np.sqrt(var + 1e-6)
    # Normalize per-image so the distribution lands roughly in [0, 1]
    p95 = np.percentile(tex_factor, 95)
    if p95 > 0:
        tex_factor = np.clip(tex_factor / p95, 0, 1.5)

    raw = (0.3 + 0.7 * tex_factor) * lum_factor

    # Rescale raw to [jnd_min, jnd_max]
    lo = np.percentile(raw, 5)
    hi = np.percentile(raw, 95)
    if hi - lo < 1e-6:
        return np.full((1, H, W), 1.0, dtype=np.float32)
    norm = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    mask = jnd_min + (jnd_max - jnd_min) * norm
    return mask[None, :, :].astype(np.float32)


def _box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """2D box-mean with radius k (window 2k+1) via scipy uniform_filter."""
    from scipy.ndimage import uniform_filter
    if k <= 0:
        return arr.copy()
    return uniform_filter(arr, size=2 * k + 1, mode="reflect").astype(arr.dtype)
