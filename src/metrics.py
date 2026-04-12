"""
Image quality metrics for evaluating fingerprint imperceptibility.

Supports: PSNR, SSIM, MS-SSIM, LPIPS.
All functions accept numpy arrays in (C, H, W) format with values in [0, 1].
"""

import numpy as np
from typing import Dict, Optional


def psnr(original: np.ndarray, fingerprinted: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Original image, float32 in [0, 1].
        fingerprinted: Fingerprinted image, float32 in [0, 1].
        max_val: Maximum pixel value.

    Returns:
        PSNR in dB.
    """
    mse = np.mean((original - fingerprinted) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val**2 / mse))


def ssim(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    use_gpu: bool = False,
) -> float:
    """Compute Structural Similarity Index.

    Uses pytorch_msssim if available (GPU-accelerated), falls back to skimage.

    Args:
        original: (C, H, W) float32 in [0, 1].
        fingerprinted: (C, H, W) float32 in [0, 1].
        use_gpu: Whether to use GPU via pytorch_msssim.

    Returns:
        SSIM value in [0, 1].
    """
    if use_gpu:
        import torch
        from pytorch_msssim import ssim as torch_ssim

        device = "cuda" if torch.cuda.is_available() else "cpu"
        orig_t = torch.from_numpy(original).unsqueeze(0).float().to(device)
        fp_t = torch.from_numpy(fingerprinted).unsqueeze(0).float().to(device)
        return torch_ssim(orig_t, fp_t, data_range=1.0).item()

    from skimage.metrics import structural_similarity

    # skimage expects (H, W, C) for multichannel
    if original.ndim == 3:
        orig_hwc = np.transpose(original, (1, 2, 0))
        fp_hwc = np.transpose(fingerprinted, (1, 2, 0))
        return float(structural_similarity(orig_hwc, fp_hwc, channel_axis=2, data_range=1.0))
    return float(structural_similarity(original, fingerprinted, data_range=1.0))


def ms_ssim(original: np.ndarray, fingerprinted: np.ndarray) -> float:
    """Compute Multi-Scale SSIM using pytorch_msssim."""
    import torch
    from pytorch_msssim import ms_ssim as torch_ms_ssim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_t = torch.from_numpy(original).unsqueeze(0).float().to(device)
    fp_t = torch.from_numpy(fingerprinted).unsqueeze(0).float().to(device)
    return torch_ms_ssim(orig_t, fp_t, data_range=1.0).item()


def lpips_score(original: np.ndarray, fingerprinted: np.ndarray, net: str = "alex") -> float:
    """Compute LPIPS perceptual distance.

    Args:
        original: (C, H, W) float32 in [0, 1].
        fingerprinted: (C, H, W) float32 in [0, 1].
        net: Backbone network ("alex", "vgg", "squeeze").

    Returns:
        LPIPS distance (lower = more similar).
    """
    import torch
    import lpips

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=net).to(device)

    # LPIPS expects [-1, 1] range
    orig_t = torch.from_numpy(original).unsqueeze(0).float().to(device) * 2 - 1
    fp_t = torch.from_numpy(fingerprinted).unsqueeze(0).float().to(device) * 2 - 1
    with torch.no_grad():
        return loss_fn(orig_t, fp_t).item()


def compute_all_metrics(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    use_gpu: bool = False,
    include_lpips: bool = True,
) -> Dict[str, float]:
    """Compute all quality metrics.

    Args:
        original: (C, H, W) float32 in [0, 1].
        fingerprinted: (C, H, W) float32 in [0, 1].
        use_gpu: Use GPU-accelerated metrics where possible.
        include_lpips: Whether to include LPIPS (requires torch + lpips package).

    Returns:
        Dict with metric names and values.
    """
    results = {
        "psnr_db": psnr(original, fingerprinted),
        "ssim": ssim(original, fingerprinted, use_gpu=use_gpu),
    }

    if use_gpu:
        try:
            results["ms_ssim"] = ms_ssim(original, fingerprinted)
        except Exception:
            pass

    if include_lpips:
        try:
            results["lpips"] = lpips_score(original, fingerprinted)
        except Exception:
            pass

    return results
