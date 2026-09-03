"""v3 hybrid pipeline: image-provenance via template + crypto envelope.

Provides embed_v3 / verify_v3 functions that use:
  - a shared template T (constructed via VAE or learned, see template_v3.py)
  - a per-image keyed sign envelope S (see sign_envelope.py)
  - BCH(127, 64, t=10) ECC over the codeword (see payload.py)
  - optional JND mask (see jnd.py)

All helpers are pure-numpy at the boundary; the trainable path uses torch
internally.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .jnd import compute_jnd_mask
from .payload import BCHCodec, image_id_to_payload, payload_to_hex
from .sign_envelope import (
    build_sign_pattern,
    derive_keyed_constants,
    make_region_masks,
    recover_codeword_from_residual,
)


# --------------------------------------------------------- alignment helpers

def _estimate_scale_logpolar(orig_chw: np.ndarray, suspect_chw: np.ndarray) -> float:
    """Log-polar DFT scale estimation between two same-size images.

    Returns scale factor (>1 means suspect is a zoomed-in / crop+resize of orig).
    """
    from scipy.ndimage import map_coordinates

    C, H, W = orig_chw.shape
    gray_orig = orig_chw.mean(axis=0)
    gray_susp = suspect_chw.mean(axis=0)

    mag_orig = np.abs(np.fft.fftshift(np.fft.fft2(gray_orig)))
    mag_susp = np.abs(np.fft.fftshift(np.fft.fft2(gray_susp)))

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    hp = 1.0 - np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * (min(H, W) * 0.05) ** 2))
    mag_orig = mag_orig * hp
    mag_susp = mag_susp * hp

    max_r = min(H, W) // 2
    num_angles = 360
    num_radii = 200
    log_base = np.exp(np.log(max_r) / num_radii)

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    radii = log_base ** np.arange(num_radii)
    ag, rg = np.meshgrid(angles, radii)
    yc = cy + rg * np.sin(ag)
    xc = cx + rg * np.cos(ag)

    lp_o = map_coordinates(mag_orig, [yc, xc], order=1, mode="constant")
    lp_s = map_coordinates(mag_susp, [yc, xc], order=1, mode="constant")

    f1 = np.fft.fft2(lp_o)
    f2 = np.fft.fft2(lp_s)
    cross = f1 * np.conj(f2)
    cross /= np.maximum(np.abs(cross), 1e-10)
    corr = np.fft.ifft2(cross).real

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift_r = peak[0]
    if shift_r > num_radii // 2:
        shift_r -= num_radii
    return float(log_base ** shift_r)


def _phase_correlation_offset(orig_chw: np.ndarray, crop_chw: np.ndarray) -> tuple:
    """Find (top, left) offset where `crop_chw` was located inside `orig_chw`.

    Both arrays are (C, *, *); crop_chw must have smaller H/W than orig_chw.
    """
    _, th, tw = crop_chw.shape
    _, oh, ow = orig_chw.shape
    padded = np.zeros_like(orig_chw)
    padded[:, :th, :tw] = crop_chw
    best_pos = (0, 0)
    best_val = -1.0
    for c in range(orig_chw.shape[0]):
        f_o = np.fft.fft2(orig_chw[c])
        f_p = np.fft.fft2(padded[c])
        cross = f_o * np.conj(f_p)
        cross /= np.maximum(np.abs(cross), 1e-10)
        corr_map = np.fft.ifft2(cross).real
        valid = corr_map[: oh - th + 1, : ow - tw + 1]
        peak = np.unravel_index(np.argmax(valid), valid.shape)
        if valid[peak] > best_val:
            best_val = valid[peak]
            best_pos = peak
    return int(best_pos[0]), int(best_pos[1])


def align_suspect_to_orig(
    orig: np.ndarray, suspect: np.ndarray,
    scale_min: float = 1.01, scale_max: float = 2.5,
) -> tuple:
    """Estimate crop+resize transform and undo it.

    Pipeline:
      1. log-polar DFT estimates scale s = orig_size / crop_size.
      2. If s ≈ 1 → return suspect unchanged.
      3. If s ∈ [scale_min, scale_max] → resize suspect down to crop_size,
         find offset via phase correlation, splice back onto an orig-shaped
         canvas (non-crop region copied from orig so residual is 0 there).

    Returns:
        (aligned_suspect, valid_mask) where valid_mask is (1, H, W) bool array
        marking pixels with real suspect content.
    """
    if orig.shape != suspect.shape:
        return suspect, np.ones((1,) + orig.shape[1:], dtype=bool)

    C, H, W = orig.shape
    try:
        scale = _estimate_scale_logpolar(orig, suspect)
    except Exception:
        return suspect, np.ones((1, H, W), dtype=bool)

    # Small scale → no crop+resize; pass through.
    if scale < scale_min or scale > scale_max:
        return suspect, np.ones((1, H, W), dtype=bool)

    crop_h = max(16, int(round(H / scale)))
    crop_w = max(16, int(round(W / scale)))
    if crop_h >= H or crop_w >= W:
        return suspect, np.ones((1, H, W), dtype=bool)

    # Resize suspect 256×256 → crop_size to undo the resize part of crop+resize
    from PIL import Image as _PIL
    susp_pil = _PIL.fromarray(
        (np.clip(suspect.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)), "RGB",
    )
    susp_resized_pil = susp_pil.resize((crop_w, crop_h), _PIL.BILINEAR)
    susp_resized = (np.array(susp_resized_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0)

    # Find offset
    top, left = _phase_correlation_offset(orig, susp_resized)

    # Splice: outside crop region → copy orig (residual = 0); inside → susp_resized
    aligned = orig.copy()
    aligned[:, top:top + crop_h, left:left + crop_w] = susp_resized

    valid_mask = np.zeros((1, H, W), dtype=bool)
    valid_mask[:, top:top + crop_h, left:left + crop_w] = True

    return aligned, valid_mask


# ----------------------------------------------------------------- embed

def embed_v3(
    image: np.ndarray,
    template_T: np.ndarray,
    master_key: bytes,
    image_id: str,
    codec: Optional[BCHCodec] = None,
    use_jnd: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Embed v3 template + crypto envelope into image.

    Args:
        image: (C, H, W) float32 in [0, 1].
        template_T: (C, H, W) float32, magnitude ≤ ε.
        master_key: master secret key.
        image_id: per-image identifier (any string).
        codec: optional BCHCodec; default BCH(127, 64, t=10).
        use_jnd: apply JND perceptual mask.

    Returns:
        (watermarked_image, info_dict)
    """
    if codec is None:
        codec = BCHCodec()
    if image.shape != template_T.shape:
        raise ValueError(f"image {image.shape} ≠ template {template_T.shape}")

    payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    codeword = codec.encode(payload)
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    region_masks, gn, gn2 = make_region_masks(image.shape, n_bits=codec.n)
    sign_map = build_sign_pattern(codeword, perm, M, region_masks, image.shape)

    if use_jnd:
        jnd = compute_jnd_mask(image)  # (1, H, W) in [jnd_min, jnd_max]
    else:
        jnd = np.ones((1, image.shape[1], image.shape[2]), dtype=np.float32)

    delta = jnd * template_T * sign_map
    watermarked = np.clip(image + delta, 0.0, 1.0).astype(np.float32)
    info = {
        "payload_hex": payload_to_hex(payload),
        "codeword_n_ones": int(codeword.sum()),
        "grid": (gn, gn2),
    }
    return watermarked, info


# ----------------------------------------------------------------- verify (Plan A/B)

def verify_v3(
    original: np.ndarray,
    suspect: np.ndarray,
    template_T: np.ndarray,
    master_key: bytes,
    image_id: str,
    codec: Optional[BCHCodec] = None,
    alpha: float = 1e-3,
    align: bool = True,
) -> Dict:
    """Reference-needed statistical verify for v3 hybrid template.

    Pipeline:
      1. residual = suspect − original.
      2. For each of 127 regions, inner product residual_region · T_region;
         sign gives recovered region-sign.
      3. Reverse keyed envelope (perm + M) → recovered 127-bit codeword.
      4. BCH decode → recovered 64-bit payload.
      5. Compare to expected image_id_to_payload(image_id) → match/no-match.

    Returns:
        dict with keys: detected, image_id_match, recovered_codeword,
        recovered_payload_hex, expected_payload_hex, n_err, region_scores,
        raw_bit_accuracy.
    """
    if codec is None:
        codec = BCHCodec()
    if original.shape != suspect.shape:
        raise ValueError(f"orig {original.shape} ≠ suspect {suspect.shape}")
    if original.shape != template_T.shape:
        raise ValueError(f"orig {original.shape} ≠ T {template_T.shape}")

    # Optional crop+resize alignment via log-polar DFT
    if align:
        suspect_aligned, _valid = align_suspect_to_orig(original, suspect)
    else:
        suspect_aligned = suspect

    residual = (suspect_aligned - original).astype(np.float32)
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    region_masks, _, _ = make_region_masks(original.shape, n_bits=codec.n)

    recovered_cw, scores = recover_codeword_from_residual(
        residual, template_T, perm, M, region_masks,
    )

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)

    # raw bit accuracy against expected codeword (pre-BCH)
    raw_bit_acc = float(np.mean(recovered_cw == expected_cw))

    # BCH decode
    payload, n_err = codec.decode(recovered_cw)
    image_id_match = (
        payload is not None and np.array_equal(payload, expected_payload)
    )

    return {
        "detected": bool(image_id_match and n_err >= 0 and n_err <= codec.t),
        "image_id_match": bool(image_id_match),
        "n_err": int(n_err),
        "raw_bit_accuracy": raw_bit_acc,
        "recovered_codeword": recovered_cw.tolist(),
        "recovered_payload_hex": (
            None if payload is None else payload_to_hex(payload)
        ),
        "expected_payload_hex": payload_to_hex(expected_payload),
        "region_score_mean": float(np.mean(scores)),
        "region_score_std": float(np.std(scores)),
        "significance_level": alpha,
    }
