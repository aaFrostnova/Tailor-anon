"""
Fingerprint verification via statistical correlation testing.

Uses Pearson correlation between extracted residuals and known fingerprint
fragments, with Fisher's method to combine p-values across fragments.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

from .keygen import derive_all_subkeys
from .fragment import generate_fragment


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation between two flattened arrays.

    Returns:
        (correlation coefficient, p-value)
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    r, p = stats.pearsonr(a_flat, b_flat)
    return float(r), float(p)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def fisher_combine_pvalues(pvalues: List[float]) -> Tuple[float, float]:
    """Combine p-values using Fisher's method.

    Test statistic: T = -2 * sum(ln(p_i)) ~ chi2(2K) under H0.

    Args:
        pvalues: List of K p-values from individual fragment tests.

    Returns:
        (chi2_statistic, combined_p_value)
    """
    pvalues = [max(p, 1e-300) for p in pvalues]  # avoid log(0)
    T = -2.0 * sum(np.log(p) for p in pvalues)
    df = 2 * len(pvalues)
    combined_p = 1.0 - stats.chi2.cdf(T, df)
    return float(T), float(combined_p)


def verify_fingerprint(
    residual: np.ndarray,
    master_key: bytes,
    image_id: str,
    num_fragments: int = 8,
    epsilon: float = 4.0 / 255.0,
    alpha: float = 0.001,
) -> Dict:
    """Verify whether a residual contains the expected fingerprint.

    Regenerates all fragment patterns from the master key and computes
    per-fragment Pearson correlation, then combines via Fisher's method.

    Args:
        residual: Extracted residual from image, shape (C, H, W).
        master_key: The master secret key.
        image_id: Image identifier used during embedding.
        num_fragments: Number of fragments K.
        epsilon: Perturbation budget used during embedding.
        alpha: Significance level for detection (default 0.001).

    Returns:
        Dict with per-fragment correlations, p-values, combined statistics,
        and detection decision.
    """
    subkeys = derive_all_subkeys(master_key, image_id, num_fragments)

    correlations = []
    pvalues = []
    cosine_sims = []

    for k in range(num_fragments):
        fragment = generate_fragment(subkeys[k], residual.shape, epsilon, fragment_index=k)

        r, p = pearson_correlation(residual, fragment)
        cos = cosine_similarity(residual, fragment)

        correlations.append(r)
        pvalues.append(p)
        cosine_sims.append(cos)

    chi2_stat, combined_p = fisher_combine_pvalues(pvalues)

    return {
        "detected": combined_p < alpha,
        "combined_p_value": combined_p,
        "chi2_statistic": chi2_stat,
        "significance_level": alpha,
        "per_fragment": {
            "correlations": correlations,
            "p_values": pvalues,
            "cosine_similarities": cosine_sims,
        },
        "mean_correlation": float(np.mean(correlations)),
        "num_fragments": num_fragments,
    }


def verify_multi_domain(
    original: np.ndarray,
    fingerprinted: np.ndarray,
    master_key: bytes,
    image_id: str,
    configs: List[Dict],
    epsilon: float = 4.0 / 255.0,
    weights: List[float] = None,
    alpha: float = 0.001,
) -> Dict:
    """Verify fingerprint using per-fragment domain-aware extraction.

    Each fragment is extracted from its own domain (pixel region,
    DCT freq band, DWT-DCT level) and correlated with the known pattern.

    Args:
        original: Original image, (C, H, W) float32 [0, 1].
        fingerprinted: Fingerprinted (possibly transformed) image.
        master_key: Master secret key.
        image_id: Image identifier.
        configs: List of K fragment config dicts.
        epsilon: Perturbation budget.
        weights: Per-fragment weights (uniform 1/K if None).
        alpha: Significance level.

    Returns:
        Dict with per-fragment results and combined detection.
    """
    from .embed import extract_single_fragment
    from .fragment_config import get_spatial_mask

    K = len(configs)
    if weights is None:
        weights = [1.0 / K] * K

    subkeys = derive_all_subkeys(master_key, image_id, K)

    correlations = []
    pvalues = []
    cosine_sims = []
    fragment_details = []

    for k in range(K):
        config = configs[k]
        region = config.get("region", "full")

        # Extract residual in this fragment's domain
        residual_k = extract_single_fragment(original, fingerprinted, config)

        # Regenerate the fragment and apply mask + weight
        fragment = generate_fragment(subkeys[k], original.shape, epsilon, fragment_index=k)
        strategy = config.get("strategy", "pixel")

        # Radial-profile: skip spatial masking (output is 1D)
        if strategy == "radial-profile":
            masked_fragment = fragment * weights[k]
        else:
            mask = get_spatial_mask(fragment.shape, region)
            masked_fragment = fragment * mask * weights[k]

        # Transform reference fragment to match extraction domain

        if strategy in ("warping", "color-filter"):
            # Non-additive strategies: re-embed on original, compare result
            # Compare fingerprinted image directly with expected fingerprinted image
            from .embed import embed_single_fragment
            expected_fp = embed_single_fragment(original, fragment, config, weights[k])
            # Use MSE between actual and expected as test statistic
            # Under correct key: MSE ≈ 0 (identical transform)
            # Under wrong key: MSE >> 0 (different transform)
            actual_residual = (fingerprinted - original).astype(np.float32)
            expected_residual = (expected_fp - original).astype(np.float32)
            mse = np.mean((actual_residual - expected_residual) ** 2)
            # Convert MSE to a correlation-like score and p-value
            # Normalize: r = 1 - mse / baseline_mse (where baseline = var of expected)
            baseline = np.var(expected_residual) + 1e-10
            r = max(0.0, 1.0 - mse / baseline)
            # Approximate p-value: use the MSE ratio as chi-squared-like statistic
            n_eff = min(expected_residual.size, 10000)  # effective sample size
            if r > 0.5:
                p = float(stats.chi2.sf(n_eff * (1 - r), df=n_eff))
            else:
                p = 1.0
            cos = float(r)
            correlations.append(r)
            pvalues.append(p)
            cosine_sims.append(cos)
            fragment_details.append({
                "index": k, "strategy": strategy, "region": region,
                "freq_range": config.get("freq_range"),
                "level": config.get("level"), "correlation": r, "p_value": p,
            })
            continue

        elif strategy == "dft-magnitude":
            # DFT-magnitude: use signed DFT real part as reference
            from .embed import _build_freq_mask
            freq_band = tuple(config.get("freq_band", [0.05, 0.4]))
            C, H, W = original.shape
            fmask = _build_freq_mask(H, W, freq_band)
            ref = np.zeros_like(masked_fragment)
            for c in range(C):
                pert_fft = np.fft.fftshift(np.fft.fft2(masked_fragment[c]))
                pert_real = pert_fft.real
                band_vals = pert_real[fmask]
                std = np.std(band_vals)
                if std > 0:
                    band_vals = band_vals / std
                ref_c = np.zeros((H, W), dtype=np.float32)
                ref_c[fmask] = band_vals
                ref[c] = ref_c
            masked_fragment = ref

        elif strategy == "radial-profile":
            # Radial-profile: compute radial profile of fragment's DFT
            from .embed import _compute_radial_profile, _build_freq_mask
            freq_band = tuple(config.get("freq_band", [0.05, 0.4]))
            num_bins = config.get("num_bins", 64)
            C_img = original.shape[0]
            ref = np.zeros((C_img, num_bins), dtype=np.float32)
            for c in range(C_img):
                pert_fft = np.fft.fftshift(np.fft.fft2(masked_fragment[c]))
                radial, _ = _compute_radial_profile(pert_fft.real, num_bins)
                std = np.std(radial)
                if std > 0:
                    radial = radial / std
                ref[c] = radial
            masked_fragment = ref

        # Correlate
        if strategy in ("dft-magnitude", "radial-profile"):
            # Use SIGN correlation: sign(residual) vs sign(reference)
            # Sign pattern is key-dependent; amplitude has structural bias
            res_flat = residual_k.flatten()
            ref_flat = masked_fragment.flatten()
            # Only compare at non-zero positions
            nonzero = (res_flat != 0) & (ref_flat != 0)
            if nonzero.sum() > 2:
                r, p = stats.pearsonr(
                    np.sign(res_flat[nonzero]),
                    np.sign(ref_flat[nonzero])
                )
                r, p = float(r), float(p)
            else:
                r, p = 0.0, 1.0
            cos = cosine_similarity(np.sign(residual_k), np.sign(masked_fragment))
        elif region != "full":
            nonzero = mask.flatten() > 0
            res_vals = residual_k.flatten()[nonzero]
            frag_vals = masked_fragment.flatten()[nonzero]
            if len(res_vals) > 2:
                r, p = stats.pearsonr(res_vals, frag_vals)
                r, p = float(r), float(p)
            else:
                r, p = 0.0, 1.0
            cos = cosine_similarity(residual_k, masked_fragment)
        else:
            r, p = pearson_correlation(residual_k, masked_fragment)
            cos = cosine_similarity(residual_k, masked_fragment)

        correlations.append(r)
        pvalues.append(p)
        cosine_sims.append(cos)
        fragment_details.append({
            "index": k,
            "strategy": config["strategy"],
            "region": region,
            "freq_range": config.get("freq_range"),
            "level": config.get("level"),
            "correlation": r,
            "p_value": p,
        })

    chi2_stat, combined_p = fisher_combine_pvalues(pvalues)

    return {
        "detected": combined_p < alpha,
        "combined_p_value": combined_p,
        "chi2_statistic": chi2_stat,
        "significance_level": alpha,
        "per_fragment": {
            "correlations": correlations,
            "p_values": pvalues,
            "cosine_similarities": cosine_sims,
            "details": fragment_details,
        },
        "mean_correlation": float(np.mean(correlations)),
        "num_fragments": K,
    }


def verify_batch(
    residuals: List[np.ndarray],
    master_key: bytes,
    image_ids: List[str],
    **kwargs,
) -> List[Dict]:
    """Verify fingerprints for a batch of images.

    Args:
        residuals: List of residual arrays.
        master_key: Master secret key.
        image_ids: List of image identifiers.
        **kwargs: Additional arguments for verify_fingerprint.

    Returns:
        List of verification result dicts.
    """
    results = []
    for residual, image_id in zip(residuals, image_ids):
        result = verify_fingerprint(residual, master_key, image_id, **kwargs)
        results.append(result)
    return results
