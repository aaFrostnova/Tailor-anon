"""
Per-fragment configuration for multi-domain embedding.

Each fragment is embedded in a different domain (pixel/DCT/DWT-DCT),
frequency band, or spatial region, so that no single attack can
destroy all fragments simultaneously.

Spatial regions (5):
  q1 (top-left), q2 (top-right), q3 (bottom-left), q4 (bottom-right), center

Frequency bands (3):
  low (1-12), mid (12-35), high (35-55)

Strategies (covering 4 input spaces per UNICORN taxonomy):
  Pixel Space:   pixel, dct, dwt-dct
  Signal Space:  dft-magnitude, color-filter
  Feature Space: warping (WaNet)
  Numerical Space: quantization (BppAttack)
"""

import numpy as np
from typing import Dict, List, Tuple


# --- Frequency band definitions (zigzag index ranges in 8×8 DCT block) ---
# Total 64 coefficients in zigzag order: index 0 = DC, 1-63 = AC
FREQ_LOW = [1, 12]     # Low frequency (coarse structure)
FREQ_MID = [12, 35]    # Mid frequency (textures, edges)
FREQ_HIGH = [35, 55]   # High frequency (fine detail, noise-like)

# DFT magnitude frequency bands (normalized, 0 to 0.5)
# These are annular rings in 2D Fourier space
DFT_LOW = [0.02, 0.12]    # Low freq ring (coarse structure)
DFT_MID = [0.12, 0.30]    # Mid freq ring (textures)
DFT_HIGH = [0.30, 0.45]   # High freq ring (fine detail)

# DFT low freq sub-bands (non-overlapping, for crop robustness)
DFT_LOW_1 = [0.03, 0.07]  # Sub-band 1
DFT_LOW_2 = [0.07, 0.12]  # Sub-band 2
DFT_LOW_3 = [0.12, 0.18]  # Sub-band 3 (low-mid transition)


def get_spatial_mask(shape: Tuple[int, ...], region: str) -> np.ndarray:
    """Create a binary spatial mask.

    Args:
        shape: (C, H, W) image shape.
        region: One of:
            "full"   - entire image
            "q1"     - top-left quadrant
            "q2"     - top-right quadrant
            "q3"     - bottom-left quadrant
            "q4"     - bottom-right quadrant
            "center" - center 50% area
            "top", "bottom", "left", "right" - halves

    Returns:
        Float32 binary mask of same shape.
    """
    C, H, W = shape
    mask = np.zeros(shape, dtype=np.float32)

    if region == "full":
        mask[:] = 1.0
    # --- Halves ---
    elif region == "top":
        mask[:, :H // 2, :] = 1.0
    elif region == "bottom":
        mask[:, H // 2:, :] = 1.0
    elif region == "left":
        mask[:, :, :W // 2] = 1.0
    elif region == "right":
        mask[:, :, W // 2:] = 1.0
    # --- Quadrants (四宫格) ---
    elif region == "q1":
        mask[:, :H // 2, :W // 2] = 1.0
    elif region == "q2":
        mask[:, :H // 2, W // 2:] = 1.0
    elif region == "q3":
        mask[:, H // 2:, :W // 2] = 1.0
    elif region == "q4":
        mask[:, H // 2:, W // 2:] = 1.0
    # --- Center (中间区域, 50% area) ---
    elif region == "center":
        h4, w4 = H // 4, W // 4
        mask[:, h4:H - h4, w4:W - w4] = 1.0
    else:
        raise ValueError(f"Unknown region: {region}")

    return mask


# ============================================================
# Default configs
# ============================================================

# K=17: Full configuration
# 5 spatial (pixel) + 3 freq × DCT + 3 freq × DWT-L1 + 3 freq × DWT-L2 + 3 DFT-mag bands
DEFAULT_FRAGMENT_CONFIGS_K17 = [
    # --- Pixel domain: 5 spatial regions (四宫格 + 中间) ---
    {"strategy": "pixel", "region": "q1"},       # 0
    {"strategy": "pixel", "region": "q2"},       # 1
    {"strategy": "pixel", "region": "q3"},       # 2
    {"strategy": "pixel", "region": "q4"},       # 3
    {"strategy": "pixel", "region": "center"},   # 4

    # --- DCT domain: 3 frequency bands ---
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_LOW},                    # 5
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_MID},                    # 6
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_HIGH},                   # 7

    # --- DWT-DCT level 1: 3 frequency bands ---
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 1, "block_size": 8, "freq_range": FREQ_LOW},   # 8
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 1, "block_size": 8, "freq_range": FREQ_MID},   # 9
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 1, "block_size": 8, "freq_range": FREQ_HIGH},  # 10

    # --- DWT-DCT level 2: 3 frequency bands ---
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_LOW},   # 11
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_MID},   # 12
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_HIGH},  # 13

    # --- DFT magnitude: 3 frequency bands (shift-invariant, anti-crop) ---
    {"strategy": "dft-magnitude", "region": "full",
     "freq_band": DFT_LOW},                     # 14
    {"strategy": "dft-magnitude", "region": "full",
     "freq_band": DFT_MID},                     # 15
    {"strategy": "dft-magnitude", "region": "full",
     "freq_band": DFT_HIGH},                    # 16

    # --- Radial profile: 3 bands (shift + scale invariant, strongest anti-crop) ---
    {"strategy": "radial-profile", "region": "full",
     "freq_band": DFT_LOW, "num_bins": 64},     # 17
    {"strategy": "radial-profile", "region": "full",
     "freq_band": DFT_MID, "num_bins": 64},     # 18
    {"strategy": "radial-profile", "region": "full",
     "freq_band": DFT_HIGH, "num_bins": 64},    # 19
]

# K=8: Pixel + frequency diversity + quantization
DEFAULT_FRAGMENT_CONFIGS_K8 = [
    # Pixel: center
    {"strategy": "pixel", "region": "center"},

    # DCT: low + mid + high
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_LOW},
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_MID},
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_HIGH},

    # DWT-DCT L2: low + mid + high
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_LOW},
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_MID},
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_HIGH},

    # Numerical Space: quantization
    {"strategy": "quantization", "region": "full",
     "bits": 6},
]

# K=11: Full pixel/signal coverage + all new spaces
DEFAULT_FRAGMENT_CONFIGS_K11 = [
    # Pixel Space: 四宫格 + center
    {"strategy": "pixel", "region": "q1"},
    {"strategy": "pixel", "region": "q2"},
    {"strategy": "pixel", "region": "q3"},
    {"strategy": "pixel", "region": "q4"},
    {"strategy": "pixel", "region": "center"},

    # Pixel Space: DCT low + mid
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_LOW},
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_MID},

    # Pixel Space: DWT-DCT L2 low
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_LOW},

    # Signal Space: color filter
    {"strategy": "color-filter", "region": "full",
     "intensity": 0.03},

    # Feature Space: warping (WaNet)
    {"strategy": "warping", "region": "full",
     "strength": 0.5, "grid_k": 4},

    # Numerical Space: quantization (BppAttack)
    {"strategy": "quantization", "region": "full",
     "bits": 6},
]

# K=14: Full coverage — all strategies × key parameters
DEFAULT_FRAGMENT_CONFIGS_K14 = [
    # Pixel Space: spatial
    {"strategy": "pixel", "region": "q1"},
    {"strategy": "pixel", "region": "q4"},
    {"strategy": "pixel", "region": "center"},

    # Pixel Space: DCT low + mid + high
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_LOW},
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_MID},
    {"strategy": "dct", "region": "full", "block_size": 8,
     "freq_range": FREQ_HIGH},

    # Pixel Space: DWT-DCT L2 low + mid
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_LOW},
    {"strategy": "dwt-dct", "region": "full", "wavelet": "haar",
     "level": 2, "block_size": 8, "freq_range": FREQ_MID},

    # Signal Space: color filter (2 intensities)
    {"strategy": "color-filter", "region": "full",
     "intensity": 0.02},
    {"strategy": "color-filter", "region": "full",
     "intensity": 0.04},

    # Feature Space: warping (2 strengths)
    {"strategy": "warping", "region": "full",
     "strength": 0.3, "grid_k": 4},
    {"strategy": "warping", "region": "full",
     "strength": 0.5, "grid_k": 8},

    # Numerical Space: quantization (2 bit depths)
    {"strategy": "quantization", "region": "full",
     "bits": 7},
    {"strategy": "quantization", "region": "full",
     "bits": 5},
]

# K=4: Minimal — one per UNICORN input space
DEFAULT_FRAGMENT_CONFIGS_K4 = [
    {"strategy": "pixel", "region": "center"},
    {"strategy": "color-filter", "region": "full", "intensity": 0.03},
    {"strategy": "warping", "region": "full", "strength": 0.5, "grid_k": 4},
    {"strategy": "quantization", "region": "full", "bits": 6},
]


def get_default_configs(num_fragments: int) -> List[Dict]:
    """Get default per-fragment configs for a given K.

    Args:
        num_fragments: Number of fragments.
            4  → minimal (1 per UNICORN input space)
            8  → balanced (all spaces with frequency diversity)
            11 → extended (full pixel coverage + all new spaces)
            14 → full (all strategies × multiple parameters)
            other → cycles through K=14

    Returns:
        List of K config dicts.
    """
    if num_fragments == 4:
        return [dict(c) for c in DEFAULT_FRAGMENT_CONFIGS_K4]
    elif num_fragments == 8:
        return [dict(c) for c in DEFAULT_FRAGMENT_CONFIGS_K8]
    elif num_fragments == 11:
        return [dict(c) for c in DEFAULT_FRAGMENT_CONFIGS_K11]
    elif num_fragments == 14:
        return [dict(c) for c in DEFAULT_FRAGMENT_CONFIGS_K14]
    elif num_fragments <= len(DEFAULT_FRAGMENT_CONFIGS_K17):
        return [dict(c) for c in DEFAULT_FRAGMENT_CONFIGS_K17[:num_fragments]]
    else:
        base = DEFAULT_FRAGMENT_CONFIGS_K14
        return [dict(base[i % len(base)]) for i in range(num_fragments)]
