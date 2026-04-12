"""
Embedding strategies for fingerprint fragments into images.

Seven strategies:
1. Pixel-space addition (baseline)
2. DCT mid-frequency embedding
3. DWT-DCT hybrid embedding
4. DFT magnitude embedding (shift-invariant, robust to crop+resize)
5. Radial profile embedding (shift + scale invariant)
6. Warping embedding (feature space, based on WaNet)
7. Color filter embedding (signal space)
8. Quantization/dithering embedding (numerical space)
"""

import numpy as np
from typing import Tuple, Optional

try:
    import pywt
except ImportError:
    pywt = None


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II using scipy or manual implementation."""
    from scipy.fftpack import dct
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    """2D inverse DCT-II."""
    from scipy.fftpack import idct
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _zigzag_indices(n: int) -> list:
    """Generate zigzag scan indices for an n x n block."""
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                indices.append((i, s - i))
        else:
            for i in range(max(0, s - n + 1), min(s + 1, n)):
                indices.append((i, s - i))
    return indices


# --- Strategy 1: Pixel-space embedding ---

def embed_pixel(
    image: np.ndarray,
    perturbation: np.ndarray,
) -> np.ndarray:
    """Embed fingerprint via direct pixel-space addition.

    Args:
        image: Original image as float32 in [0, 1], shape (C, H, W) or (H, W).
        perturbation: Aggregated perturbation, same shape as image.

    Returns:
        Fingerprinted image, clipped to [0, 1].
    """
    return np.clip(image + perturbation, 0.0, 1.0).astype(np.float32)


def extract_pixel(
    original: np.ndarray,
    fingerprinted: np.ndarray,
) -> np.ndarray:
    """Extract perturbation from pixel-space embedding (for verification)."""
    return (fingerprinted - original).astype(np.float32)


# --- Strategy 2: DCT mid-frequency embedding ---

def embed_dct(
    image: np.ndarray,
    perturbation: np.ndarray,
    block_size: int = 8,
    freq_range: Tuple[int, int] = (10, 40),
) -> np.ndarray:
    """Embed fingerprint in DCT mid-frequency coefficients.

    Processes each channel independently in 8x8 blocks.

    Args:
        image: Float32 image in [0, 1], shape (C, H, W).
        perturbation: Same shape as image.
        block_size: DCT block size (default 8).
        freq_range: (low, high) zigzag indices for mid-frequency band.

    Returns:
        Fingerprinted image in [0, 1].
    """
    result = image.copy()
    C, H, W = image.shape
    zigzag = _zigzag_indices(block_size)
    low, high = freq_range
    mid_indices = zigzag[low:high]

    for c in range(C):
        for i in range(0, H - block_size + 1, block_size):
            for j in range(0, W - block_size + 1, block_size):
                block = image[c, i : i + block_size, j : j + block_size]
                pert_block = perturbation[c, i : i + block_size, j : j + block_size]

                dct_block = _dct2(block)
                dct_pert = _dct2(pert_block)

                # Only add perturbation to mid-frequency coefficients
                for (r, col) in mid_indices:
                    dct_block[r, col] += dct_pert[r, col]

                result[c, i : i + block_size, j : j + block_size] = _idct2(dct_block)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_dct(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    block_size: int = 8,
    freq_range: Tuple[int, int] = (10, 40),
) -> np.ndarray:
    """Extract mid-frequency DCT perturbation for verification."""
    C, H, W = original.shape
    extracted = np.zeros_like(original)
    zigzag = _zigzag_indices(block_size)
    low, high = freq_range
    mid_indices = zigzag[low:high]

    for c in range(C):
        for i in range(0, H - block_size + 1, block_size):
            for j in range(0, W - block_size + 1, block_size):
                orig_dct = _dct2(original[c, i : i + block_size, j : j + block_size])
                fp_dct = _dct2(fingerprinted[c, i : i + block_size, j : j + block_size])

                diff = np.zeros((block_size, block_size))
                for (r, col) in mid_indices:
                    diff[r, col] = fp_dct[r, col] - orig_dct[r, col]

                extracted[c, i : i + block_size, j : j + block_size] = _idct2(diff)

    return extracted.astype(np.float32)


# --- Strategy 3: DWT-DCT hybrid embedding (recommended) ---

def embed_dwt_dct(
    image: np.ndarray,
    perturbation: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    block_size: int = 8,
    freq_range: Tuple[int, int] = (10, 40),
) -> np.ndarray:
    """Embed fingerprint using DWT-DCT hybrid strategy.

    1. Apply multi-level DWT to get LL subband
    2. Apply DCT to LL subband blocks
    3. Embed perturbation in mid-frequency DCT coefficients
    4. Inverse DCT + inverse DWT to reconstruct

    Args:
        image: Float32 image in [0, 1], shape (C, H, W).
        perturbation: Same shape as image.
        wavelet: Wavelet type (default "haar").
        level: DWT decomposition level (default 2).
        block_size: DCT block size.
        freq_range: Mid-frequency band for DCT embedding.

    Returns:
        Fingerprinted image in [0, 1].
    """
    if pywt is None:
        raise ImportError("PyWavelets is required for DWT-DCT embedding: pip install PyWavelets")

    C, H, W = image.shape
    result = image.copy()
    zigzag = _zigzag_indices(block_size)
    low, high = freq_range
    mid_indices = zigzag[low:high]

    for c in range(C):
        channel = image[c]

        # Forward DWT
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        ll = coeffs[0]  # LL subband at deepest level

        # Also decompose the perturbation to match LL subband size
        pert_coeffs = pywt.wavedec2(perturbation[c], wavelet, level=level)
        pert_ll = pert_coeffs[0]

        # Embed in DCT mid-frequency of LL subband
        ll_h, ll_w = ll.shape
        for i in range(0, ll_h - block_size + 1, block_size):
            for j in range(0, ll_w - block_size + 1, block_size):
                block = ll[i : i + block_size, j : j + block_size]
                pert_block = pert_ll[i : i + block_size, j : j + block_size]

                dct_block = _dct2(block)
                dct_pert = _dct2(pert_block)

                for (r, col) in mid_indices:
                    dct_block[r, col] += dct_pert[r, col]

                ll[i : i + block_size, j : j + block_size] = _idct2(dct_block)

        coeffs[0] = ll

        # Inverse DWT
        reconstructed = pywt.waverec2(coeffs, wavelet)

        # Handle size mismatch from DWT padding
        result[c] = reconstructed[:H, :W]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_dwt_dct(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    block_size: int = 8,
    freq_range: Tuple[int, int] = (10, 40),
) -> np.ndarray:
    """Extract DWT-DCT embedded perturbation for verification."""
    if pywt is None:
        raise ImportError("PyWavelets is required: pip install PyWavelets")

    C, H, W = original.shape
    extracted = np.zeros_like(original)
    zigzag = _zigzag_indices(block_size)
    low, high = freq_range
    mid_indices = zigzag[low:high]

    for c in range(C):
        orig_coeffs = pywt.wavedec2(original[c], wavelet, level=level)
        fp_coeffs = pywt.wavedec2(fingerprinted[c], wavelet, level=level)

        orig_ll = orig_coeffs[0]
        fp_ll = fp_coeffs[0]
        ll_h, ll_w = orig_ll.shape

        diff_ll = np.zeros_like(orig_ll)

        for i in range(0, ll_h - block_size + 1, block_size):
            for j in range(0, ll_w - block_size + 1, block_size):
                orig_dct = _dct2(orig_ll[i : i + block_size, j : j + block_size])
                fp_dct = _dct2(fp_ll[i : i + block_size, j : j + block_size])

                diff = np.zeros((block_size, block_size))
                for (r, col) in mid_indices:
                    diff[r, col] = fp_dct[r, col] - orig_dct[r, col]

                diff_ll[i : i + block_size, j : j + block_size] = _idct2(diff)

        # Put difference only in LL, zeros elsewhere
        diff_coeffs = [diff_ll] + [
            tuple(np.zeros_like(d) for d in detail)
            for detail in orig_coeffs[1:]
        ]
        rec = pywt.waverec2(diff_coeffs, wavelet)
        extracted[c] = rec[:H, :W]

    return extracted.astype(np.float32)


# --- Strategy 4: DFT magnitude embedding (shift-invariant) ---

def _build_freq_mask(H: int, W: int, freq_band: Tuple[float, float] = (0.05, 0.4)) -> np.ndarray:
    """Build a binary mask selecting an annular frequency band in 2D DFT space.

    Args:
        H, W: Image dimensions.
        freq_band: (low, high) as fraction of max frequency (0 to 0.5).
            low=0.05 skips DC/very low freq, high=0.4 avoids highest freq.

    Returns:
        Boolean mask of shape (H, W).
    """
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    # Normalized distance from center (0 to 0.5)
    dist = np.sqrt((Y - cy) ** 2 / H ** 2 + (X - cx) ** 2 / W ** 2)
    return (dist >= freq_band[0]) & (dist <= freq_band[1])


def embed_dft_magnitude(
    image: np.ndarray,
    perturbation: np.ndarray,
    freq_band: Tuple[float, float] = (0.05, 0.4),
) -> np.ndarray:
    """Embed fingerprint in the DFT magnitude spectrum using signed modulation.

    The magnitude of a 2D DFT is shift-invariant: |F{f(x-a)}| = |F{f(x)}|.
    This makes it robust to crop + resize (which is translation + scaling).

    We use the SIGN of the perturbation's DFT to modulate the image's
    magnitude: positive perturbation → increase magnitude, negative → decrease.
    This creates a signed pattern that is key-dependent and shift-invariant.

    Args:
        image: Float32 image [0, 1], shape (C, H, W).
        perturbation: Same shape as image.
        freq_band: (low, high) normalized frequency range.

    Returns:
        Fingerprinted image clipped to [0, 1].
    """
    C, H, W = image.shape
    freq_mask = _build_freq_mask(H, W, freq_band)
    result = image.copy()

    for c in range(C):
        fft = np.fft.fft2(image[c])
        fft_shifted = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        # Use perturbation's DFT as signed modulation
        pert_fft = np.fft.fftshift(np.fft.fft2(perturbation[c]))
        # Real part gives signed (+/-) pattern, normalize to unit variance
        pert_real = pert_fft.real
        band_vals = pert_real[freq_mask]
        std = np.std(band_vals)
        if std > 0:
            band_vals = band_vals / std

        # Scale: modulate magnitude by a small signed amount
        # Use fraction of local magnitude to keep perturbation proportional
        mag_band = magnitude[freq_mask]
        alpha = 0.02  # 2% modulation depth
        magnitude[freq_mask] = mag_band * (1.0 + alpha * band_vals)

        modified_fft = magnitude * np.exp(1j * phase)
        result[c] = np.fft.ifft2(np.fft.ifftshift(modified_fft)).real

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_dft_magnitude(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    freq_band: Tuple[float, float] = (0.05, 0.4),
) -> np.ndarray:
    """Extract signed DFT magnitude difference for verification.

    Computes the signed magnitude change between original and fingerprinted
    images within the selected frequency band. Because magnitude is
    shift-invariant, this works even after crop+resize.

    Args:
        original: Original image (C, H, W).
        fingerprinted: Fingerprinted (possibly cropped+resized) image.
        freq_band: Same frequency band used during embedding.

    Returns:
        Signed magnitude difference (for correlation with known fingerprint).
    """
    C, H, W = original.shape
    freq_mask = _build_freq_mask(H, W, freq_band)
    extracted = np.zeros_like(original)

    for c in range(C):
        orig_mag = np.abs(np.fft.fftshift(np.fft.fft2(original[c])))
        fp_mag = np.abs(np.fft.fftshift(np.fft.fft2(fingerprinted[c])))

        # Signed difference: positive where magnitude increased, negative where decreased
        diff = np.zeros((H, W), dtype=np.float32)
        orig_band = orig_mag[freq_mask]
        # Normalize by original magnitude to get relative change
        safe_orig = np.maximum(orig_band, 1e-10)
        diff[freq_mask] = (fp_mag[freq_mask] - orig_band) / safe_orig
        extracted[c] = diff

    return extracted.astype(np.float32)


# --- Strategy 5: Radial profile embedding (shift + scale invariant) ---

def _compute_radial_profile(magnitude_2d: np.ndarray, num_bins: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial average profile of a 2D magnitude spectrum.

    Args:
        magnitude_2d: 2D array (H, W), centered DFT magnitude.
        num_bins: Number of radial bins. If 0, use min(H, W) // 2.

    Returns:
        (profile, bin_edges): 1D radial profile and bin edge values.
    """
    H, W = magnitude_2d.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    # Distance in pixels from center
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32)

    if num_bins == 0:
        num_bins = min(H, W) // 2

    max_r = np.sqrt(cy ** 2 + cx ** 2)
    bin_edges = np.linspace(0, max_r, num_bins + 1)
    profile = np.zeros(num_bins, dtype=np.float64)

    for i in range(num_bins):
        ring = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if ring.any():
            profile[i] = np.mean(magnitude_2d[ring])

    return profile.astype(np.float32), bin_edges


def _radial_profile_to_2d(profile: np.ndarray, H: int, W: int, bin_edges: np.ndarray) -> np.ndarray:
    """Expand a 1D radial profile back to 2D for embedding.

    Each pixel gets the profile value of its radial bin.
    """
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32)
    result = np.zeros((H, W), dtype=np.float32)
    num_bins = len(profile)

    for i in range(num_bins):
        ring = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        result[ring] = profile[i]

    return result


def embed_radial_profile(
    image: np.ndarray,
    perturbation: np.ndarray,
    freq_band: Tuple[float, float] = (0.05, 0.4),
    num_bins: int = 64,
) -> np.ndarray:
    """Embed fingerprint via radial profile modulation of DFT magnitude.

    Robust to both shift (translation) AND scale (resize), because the
    radial profile captures energy distribution across frequency rings,
    which is invariant to translation and only shifts (not destroys)
    under scaling.

    Process:
    1. Compute DFT magnitude
    2. Compute radial profile of perturbation's DFT
    3. Use the signed radial profile to modulate image magnitude per ring
    4. Reconstruct

    Args:
        image: Float32 image [0, 1], shape (C, H, W).
        perturbation: Same shape.
        freq_band: Normalized frequency range for embedding.
        num_bins: Number of radial bins.

    Returns:
        Fingerprinted image.
    """
    C, H, W = image.shape
    freq_mask = _build_freq_mask(H, W, freq_band)
    result = image.copy()

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    bin_edges = np.linspace(0, max_r, num_bins + 1)

    for c in range(C):
        fft = np.fft.fft2(image[c])
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        # Compute radial profile of perturbation's DFT (signed, from real part)
        pert_fft = np.fft.fftshift(np.fft.fft2(perturbation[c]))
        pert_radial, _ = _compute_radial_profile(pert_fft.real, num_bins)

        # Normalize to unit variance
        std = np.std(pert_radial)
        if std > 0:
            pert_radial = pert_radial / std

        # Modulate magnitude ring-by-ring within freq band
        alpha = 0.03  # 3% modulation depth
        for i in range(num_bins):
            ring = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1]) & freq_mask
            if ring.any():
                magnitude[ring] *= (1.0 + alpha * pert_radial[i])

        modified_fft = magnitude * np.exp(1j * phase)
        result[c] = np.fft.ifft2(np.fft.ifftshift(modified_fft)).real

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_radial_profile(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    freq_band: Tuple[float, float] = (0.05, 0.4),
    num_bins: int = 64,
) -> np.ndarray:
    """Extract radial profile difference for verification.

    Returns a 1D-like signal (stored as (C, num_bins, 1) array) representing
    the per-ring magnitude change. This is compared with the known fingerprint's
    radial profile.

    Because radial profiles average over each ring, small scale changes from
    crop+resize only shift the profile slightly rather than destroying it.
    """
    C, H, W = original.shape
    freq_mask = _build_freq_mask(H, W, freq_band)

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    bin_edges = np.linspace(0, max_r, num_bins + 1)

    # Output: store radial profile as (C, num_bins) then pad to (C, num_bins, 1)
    profiles = np.zeros((C, num_bins), dtype=np.float32)

    for c in range(C):
        orig_mag = np.abs(np.fft.fftshift(np.fft.fft2(original[c])))
        fp_mag = np.abs(np.fft.fftshift(np.fft.fft2(fingerprinted[c])))

        for i in range(num_bins):
            ring = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1]) & freq_mask
            if ring.any():
                orig_mean = np.mean(orig_mag[ring])
                fp_mean = np.mean(fp_mag[ring])
                if orig_mean > 1e-10:
                    profiles[c, i] = (fp_mean - orig_mean) / orig_mean

    return profiles


# --- Strategy 6: Warping embedding (feature space, based on WaNet) ---

def _generate_warping_grid(
    H: int, W: int,
    perturbation_2d: np.ndarray,
    strength: float = 0.5,
    grid_k: int = 8,
) -> np.ndarray:
    """Generate a smooth warping grid from a perturbation seed.

    Based on WaNet (Nguyen & Tran, ICLR 2021):
    1. Sample control points from the perturbation on a k×k grid
    2. Upsample to image size via bicubic interpolation
    3. Add to identity grid, clamp to [-1, 1]

    Args:
        H, W: Image dimensions.
        perturbation_2d: 2D array used as seed for control points.
        strength: Warping strength (s parameter from WaNet).
        grid_k: Control grid size.

    Returns:
        Warping grid of shape (H, W, 2) in [-1, 1] range.
    """
    from scipy.ndimage import zoom

    # Extract control points from perturbation (use first grid_k×grid_k values)
    ctrl = perturbation_2d[:grid_k, :grid_k].copy() if perturbation_2d.shape[0] >= grid_k else \
        zoom(perturbation_2d, (grid_k / perturbation_2d.shape[0], grid_k / perturbation_2d.shape[1]))

    # Normalize like WaNet: divide by mean absolute value
    mean_abs = np.mean(np.abs(ctrl))
    if mean_abs > 0:
        ctrl = ctrl / mean_abs

    # Upsample control points to image size (bicubic)
    flow_x = zoom(ctrl, (H / grid_k, W / grid_k), order=3)
    # Use a different slice for y-direction
    ctrl_y = perturbation_2d[grid_k:2*grid_k, :grid_k].copy() if perturbation_2d.shape[0] >= 2*grid_k else \
        zoom(perturbation_2d[::-1, :], (grid_k / perturbation_2d.shape[0], grid_k / perturbation_2d.shape[1]))
    mean_abs_y = np.mean(np.abs(ctrl_y))
    if mean_abs_y > 0:
        ctrl_y = ctrl_y / mean_abs_y
    flow_y = zoom(ctrl_y, (H / grid_k, W / grid_k), order=3)

    # Identity grid [-1, 1]
    gy = np.linspace(-1, 1, H)
    gx = np.linspace(-1, 1, W)
    grid_x, grid_y = np.meshgrid(gx, gy)

    # Add scaled flow to identity
    grid_x = grid_x + strength * flow_x / W
    grid_y = grid_y + strength * flow_y / H

    # Clamp
    grid_x = np.clip(grid_x, -1, 1)
    grid_y = np.clip(grid_y, -1, 1)

    return np.stack([grid_x, grid_y], axis=-1).astype(np.float32)


def _apply_warping(image_2d: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Apply warping to a 2D image using grid sampling (bilinear)."""
    from scipy.interpolate import RegularGridInterpolator

    H, W = image_2d.shape
    gy = np.linspace(-1, 1, H)
    gx = np.linspace(-1, 1, W)

    interp = RegularGridInterpolator((gy, gx), image_2d, method='linear',
                                      bounds_error=False, fill_value=None)
    # grid[:,:,0] = x coords, grid[:,:,1] = y coords
    points = np.stack([grid[:, :, 1], grid[:, :, 0]], axis=-1)  # (H, W, 2) as (y, x)
    return interp(points.reshape(-1, 2)).reshape(H, W).astype(np.float32)


def embed_warping(
    image: np.ndarray,
    perturbation: np.ndarray,
    strength: float = 0.5,
    grid_k: int = 8,
) -> np.ndarray:
    """Embed fingerprint via elastic image warping.

    Based on WaNet: uses the perturbation as a seed to generate a
    deterministic warping field. The image pixels are moved, not added to.

    Args:
        image: Float32 image [0, 1], shape (C, H, W).
        perturbation: Used as seed for warping field generation.
        strength: Warping strength (higher = more visible but more robust).
        grid_k: Control grid resolution.

    Returns:
        Warped image.
    """
    C, H, W = image.shape
    # Generate warping grid from the perturbation pattern
    grid = _generate_warping_grid(H, W, perturbation[0], strength, grid_k)

    result = np.zeros_like(image)
    for c in range(C):
        result[c] = _apply_warping(image[c], grid)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_warping(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    strength: float = 0.5,
    grid_k: int = 8,
) -> np.ndarray:
    """Extract warping verification signal.

    Since warping is non-additive (x' = warp(x), not x + δ), we can't
    extract a residual that correlates with the fragment. Instead, return
    the pixel difference which will be compared against the expected
    warping result (re-warped original) in verify.
    """
    return (fingerprinted - original).astype(np.float32)


# --- Strategy 7: Color filter embedding (signal space) ---

def embed_color_filter(
    image: np.ndarray,
    perturbation: np.ndarray,
    intensity: float = 0.02,
) -> np.ndarray:
    """Embed fingerprint via per-channel nonlinear color transformation.

    Uses the perturbation to generate a deterministic per-channel
    polynomial color curve, creating a subtle "filter" effect.

    Based on UNICORN's signal space trigger analysis (Instagram filters).

    Args:
        image: Float32 image [0, 1], shape (C, H, W).
        perturbation: Used as seed for color curve coefficients.
        intensity: Strength of the color shift.

    Returns:
        Color-filtered image.
    """
    C, H, W = image.shape
    result = image.copy()

    for c in range(C):
        # Derive polynomial coefficients from perturbation
        # Use mean values from different spatial regions as coefficients
        flat = perturbation[c].flatten()
        n = len(flat)
        # 3rd-order polynomial: a*x^3 + b*x^2 + c*x + d
        a = np.mean(flat[:n//4])
        b = np.mean(flat[n//4:n//2])
        coeff_c = np.mean(flat[n//2:3*n//4])
        d = np.mean(flat[3*n//4:])

        # Normalize coefficients to unit variance
        coeffs = np.array([a, b, coeff_c, d])
        std = np.std(coeffs)
        if std > 0:
            coeffs = coeffs / std

        # Apply polynomial color curve: pixel_out = pixel + intensity * poly(pixel)
        x = result[c]
        poly = coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
        result[c] = x + intensity * poly

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_color_filter(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    intensity: float = 0.02,
) -> np.ndarray:
    """Extract color filter residual."""
    return (fingerprinted - original).astype(np.float32)


# --- Strategy 8: Quantization/dithering embedding (numerical space) ---

def embed_quantization(
    image: np.ndarray,
    perturbation: np.ndarray,
    bits: int = 5,
) -> np.ndarray:
    """Embed fingerprint via controlled quantization with key-dependent dithering.

    Based on BppAttack: reduces bit depth then restores, with the
    dithering pattern controlled by the cryptographic perturbation.

    Args:
        image: Float32 image [0, 1], shape (C, H, W).
        perturbation: Used as dithering pattern seed.
        bits: Target bit depth for quantization (lower = more visible).

    Returns:
        Quantized image with key-dependent dithering.
    """
    C, H, W = image.shape
    levels = 2 ** bits
    step = 1.0 / levels

    # Normalize perturbation to [-0.5, 0.5] * step as dithering noise
    dither = perturbation.copy()
    for c in range(C):
        std = np.std(dither[c])
        if std > 0:
            dither[c] = dither[c] / std
    dither = dither * 0.5 * step  # Scale to half a quantization step

    # Add dithering, quantize, remove dithering bias
    dithered = image + dither
    quantized = np.round(dithered * levels) / levels
    # The residual (quantized - image) contains the key-dependent pattern
    result = quantized

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_quantization(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    bits: int = 5,
) -> np.ndarray:
    """Extract quantization residual."""
    return (fingerprinted - original).astype(np.float32)


# --- Unified interface ---

STRATEGIES = {
    "pixel": (embed_pixel, extract_pixel),
    "dct": (embed_dct, extract_dct),
    "dwt-dct": (embed_dwt_dct, extract_dwt_dct),
    "dft-magnitude": (embed_dft_magnitude, extract_dft_magnitude),
    "radial-profile": (embed_radial_profile, extract_radial_profile),
    "warping": (embed_warping, extract_warping),
    "color-filter": (embed_color_filter, extract_color_filter),
    "quantization": (embed_quantization, extract_quantization),
}


def embed(
    image: np.ndarray,
    perturbation: np.ndarray,
    strategy: str = "dwt-dct",
    **kwargs,
) -> np.ndarray:
    """Embed fingerprint using the specified strategy.

    Args:
        image: Float32 image in [0, 1], shape (C, H, W).
        perturbation: Aggregated perturbation, same shape.
        strategy: One of "pixel", "dct", "dwt-dct".
        **kwargs: Strategy-specific parameters.

    Returns:
        Fingerprinted image.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES.keys())}")
    embed_fn, _ = STRATEGIES[strategy]
    return embed_fn(image, perturbation, **kwargs)


def extract(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    strategy: str = "dwt-dct",
    **kwargs,
) -> np.ndarray:
    """Extract perturbation using the specified strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES.keys())}")
    _, extract_fn = STRATEGIES[strategy]
    return extract_fn(original, fingerprinted, **kwargs)


# --- Multi-domain interface ---

def _get_strategy_kwargs(config: dict) -> dict:
    """Extract strategy-specific kwargs from a fragment config."""
    kwargs = {}
    strategy = config["strategy"]
    if strategy == "dct":
        kwargs["block_size"] = config.get("block_size", 8)
        kwargs["freq_range"] = tuple(config.get("freq_range", [10, 40]))
    elif strategy == "dwt-dct":
        kwargs["wavelet"] = config.get("wavelet", "haar")
        kwargs["level"] = config.get("level", 2)
        kwargs["block_size"] = config.get("block_size", 8)
        kwargs["freq_range"] = tuple(config.get("freq_range", [10, 40]))
    elif strategy == "dft-magnitude":
        kwargs["freq_band"] = tuple(config.get("freq_band", [0.05, 0.4]))
    elif strategy == "radial-profile":
        kwargs["freq_band"] = tuple(config.get("freq_band", [0.05, 0.4]))
        kwargs["num_bins"] = config.get("num_bins", 64)
    elif strategy == "warping":
        kwargs["strength"] = config.get("strength", 0.5)
        kwargs["grid_k"] = config.get("grid_k", 8)
    elif strategy == "color-filter":
        kwargs["intensity"] = config.get("intensity", 0.02)
    elif strategy == "quantization":
        kwargs["bits"] = config.get("bits", 5)
    return kwargs


def embed_single_fragment(
    image: np.ndarray,
    fragment: np.ndarray,
    config: dict,
    weight: float = 1.0,
) -> np.ndarray:
    """Embed one fragment into an image according to its config.

    Applies spatial masking (for pixel strategy) and routes to the
    correct frequency-domain strategy.

    Args:
        image: Current image state, float32 [0, 1], (C, H, W).
        fragment: The fragment perturbation, same shape.
        config: Fragment config dict with "strategy", "region", etc.
        weight: Scaling weight for this fragment.

    Returns:
        Image with this fragment embedded.
    """
    from .fragment_config import get_spatial_mask

    strategy = config["strategy"]
    region = config.get("region", "full")
    kwargs = _get_strategy_kwargs(config)

    # Apply spatial mask and weight
    mask = get_spatial_mask(fragment.shape, region)
    masked_fragment = fragment * mask * weight

    embed_fn, _ = STRATEGIES[strategy]
    return embed_fn(image, masked_fragment, **kwargs)


def extract_single_fragment(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Extract residual for one fragment's domain.

    Uses the same strategy/region/freq_range as embedding so we only
    look at the domain where this fragment lives.

    Args:
        original: Original image, float32 [0, 1], (C, H, W).
        fingerprinted: Fingerprinted image, same shape.
        config: Fragment config dict.

    Returns:
        Extracted residual in this fragment's domain, masked to its region.
    """
    from .fragment_config import get_spatial_mask

    strategy = config["strategy"]
    region = config.get("region", "full")
    kwargs = _get_strategy_kwargs(config)

    _, extract_fn = STRATEGIES[strategy]
    residual = extract_fn(original, fingerprinted, **kwargs)

    # These strategies don't need spatial masking
    if strategy == "radial-profile":
        return residual

    # Apply spatial mask so we only correlate within this fragment's region
    mask = get_spatial_mask(residual.shape, region)
    return residual * mask


def embed_multi_domain(
    image: np.ndarray,
    fragments: list,
    configs: list,
    weights: list = None,
) -> np.ndarray:
    """Embed K fragments, each in its own domain/region.

    Args:
        image: Original image, float32 [0, 1], (C, H, W).
        fragments: List of K fragment arrays.
        configs: List of K config dicts.
        weights: Per-fragment weights (uniform 1/K if None).

    Returns:
        Fingerprinted image with all fragments embedded.
    """
    K = len(fragments)
    if weights is None:
        weights = [1.0 / K] * K

    result = image.copy()
    for k in range(K):
        result = embed_single_fragment(result, fragments[k], configs[k], weights[k])

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_multi_domain(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    configs: list,
) -> list:
    """Extract per-fragment residuals, each in its own domain.

    Args:
        original: Original image.
        fingerprinted: Fingerprinted image.
        configs: List of K config dicts.

    Returns:
        List of K residual arrays, one per fragment.
    """
    return [
        extract_single_fragment(original, fingerprinted, config)
        for config in configs
    ]
