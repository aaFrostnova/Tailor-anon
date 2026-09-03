"""Component C.2/C.3: tamper detection and localization.

Two ref-keyed signals (the detector knows image_id + master_key, so it can place
the carriers and the intended bits, but does NOT need the original image):

  - margin-distribution shift: the per-method soft signal (|LLR|) collapses
    relative to a clean calibration -> the image was degraded after embedding,
    even when the payload still BCH-decodes correctly;
  - survivor clustering: surviving carriers are spectrally clustered (DFT radial
    bands) or spatially clustered (QIM blocks) rather than uniform -> the damage
    is localized (blur eats high frequencies; a local edit eats one region).

Localization:
  - DFT-Kred: the radial frequency band with the largest survival drop;
  - Quant-QIM: the bounding box of the largest connected low-survival block
    region (e.g. an inpainted / locally edited area).

SCOPE: this REPAIRS / LOCATES the payload and the damaged region. It does NOT
restore image content.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.vine_crypto_wrapper import apply_crypto


# ---------------------------------------------------------- array-level helpers

def carrier_eff_decode(mag: np.ndarray, delta: float) -> np.ndarray:
    """Per-carrier effective-bit decode (0 on bit-0 lattice, 1 near delta/2)."""
    return (np.cos(2.0 * np.pi * np.asarray(mag) / delta) < 0).astype(np.uint8)


def degradation_score(mean_abs_llr: float, clean_mean_abs_llr: float) -> float:
    """How much the soft signal collapsed vs clean (0 = intact, ->1 = gone)."""
    c = max(clean_mean_abs_llr, 1e-9)
    return float(np.clip((c - mean_abs_llr) / c, 0.0, 1.0))


def radial_nonuniformity(radial_survival) -> float:
    """Spread of survival across DFT radial bands; large = damage concentrated."""
    s = np.asarray(radial_survival, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    return float(np.max(s) - np.min(s))


def localize_dft_band(radial_survival, rad_bins) -> Dict:
    """Band with the lowest survival (the most-damaged frequency range)."""
    s = np.asarray(radial_survival, dtype=np.float64)
    b = int(np.nanargmin(s))
    return {"band": b, "r_lo": float(rad_bins[b]), "r_hi": float(rad_bins[b + 1]),
            "survival": float(s[b])}


def qim_low_region(block_survival_map: np.ndarray, thresh: float = 0.6):
    """Largest connected low-survival block region -> (clustering, bbox).

    clustering = fraction of low blocks inside the largest connected component
    (high for a contiguous local edit, low for scattered global damage).
    bbox = (by0, bx0, by1, bx1) inclusive block coords, or None.
    """
    low = np.asarray(block_survival_map) < thresh
    n_low = int(low.sum())
    if n_low == 0:
        return 0.0, None
    try:
        from scipy.ndimage import label
        lab, n = label(low)
        if n == 0:
            return 0.0, None
        sizes = [(lab == k).sum() for k in range(1, n + 1)]
        kbig = int(np.argmax(sizes)) + 1
        comp = lab == kbig
        clustering = float(sizes[kbig - 1] / n_low)
    except Exception:
        comp = low
        clustering = 1.0
    ys, xs = np.where(comp)
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    return clustering, bbox


# ---------------------------------------------------------- high-level analyze

def _per_carrier_survival(method, pil, image_id, tx):
    """(survival_flat, ) using the detector's key to know the intended eff bits."""
    perm, M = method.get_perm_M(image_id)
    secret = apply_crypto(np.asarray(tx, np.uint8), perm, M).astype(np.uint8)
    eff_emb = (secret ^ method.bit_flip.astype(np.uint8))
    if method.carrier_kind == "dft_freq":
        mag = method.carrier_magnitudes(pil).reshape(-1)
        rep = method.K * method.M
    else:
        mag = method.carrier_means(pil).reshape(-1)
        rep = method.n_pos
    eff_hat = carrier_eff_decode(mag, method.decode_delta())
    eff_full = np.repeat(eff_emb, rep)
    return (eff_hat == eff_full).astype(np.float32)


def analyze(
    fd,
    pil,
    image_id: str,
    calib: Optional[Dict] = None,
    degr_thresh: float = 0.35,
    clustering_thresh: float = 0.5,
) -> Dict:
    """Ref-keyed tamper detection + localization for one suspect image.

    `calib`: {method: {"mean_abs_llr_clean": float}} from Component B. If None,
    degradation is reported but the boolean flag falls back to clustering only.
    """
    tx = fd.codeword(image_id)
    aligned, _ = fd.aligned_llrs(pil, image_id)

    # degradation per method (margin collapse vs clean calibration)
    degr = {}
    for m, a in aligned.items():
        mean_abs = float(np.mean(np.abs(a)))
        if calib and m in calib:
            degr[m] = degradation_score(mean_abs, calib[m]["mean_abs_llr_clean"])
        else:
            degr[m] = None
    degr_vals = [v for v in degr.values() if v is not None]
    mean_degr = float(np.mean(degr_vals)) if degr_vals else None

    out = {"degradation": degr, "mean_degradation": mean_degr}

    # DFT spectral localization
    if "dft_kred" in fd.methods:
        dft = fd.methods["dft_kred"]
        surv = _per_carrier_survival(dft, pil, image_id, tx)
        radii = dft.carrier_radii().reshape(-1)
        rad_bins = np.linspace(radii.min(), radii.max() + 1e-6, 9)
        idx = np.clip(np.digitize(radii, rad_bins) - 1, 0, len(rad_bins) - 2)
        band_surv = [float(surv[idx == b].mean()) if (idx == b).any() else np.nan
                     for b in range(len(rad_bins) - 1)]
        out["dft_radial_survival"] = band_surv
        out["dft_nonuniformity"] = radial_nonuniformity(band_surv)
        out["dft_damaged_band"] = localize_dft_band(band_surv, rad_bins)

    # QIM spatial localization
    if "quant_qim" in fd.methods:
        qim = fd.methods["quant_qim"]
        surv = _per_carrier_survival(qim, pil, image_id, tx)
        coords = qim.carrier_block_coords().reshape(-1, 2)
        smap = np.full((qim.n_by, qim.n_bx), np.nan)
        cnt = np.zeros((qim.n_by, qim.n_bx))
        acc = np.zeros((qim.n_by, qim.n_bx))
        for (by, bx), s in zip(coords, surv):
            acc[by, bx] += s; cnt[by, bx] += 1
        nz = cnt > 0
        smap[nz] = acc[nz] / cnt[nz]
        # Smooth the sparse/noisy per-block survival before localization so a
        # contiguous damaged region (e.g. a local edit) stands out from per-block
        # decode noise.
        filled = np.nan_to_num(smap, nan=1.0)
        try:
            from scipy.ndimage import uniform_filter
            filled = uniform_filter(filled, size=3, mode="nearest")
        except Exception:
            pass
        clustering, bbox = qim_low_region(filled, thresh=0.75)
        out["qim_block_survival"] = smap.tolist()
        out["qim_clustering"] = clustering
        out["qim_tamper_bbox_blocks"] = bbox
        if bbox is not None:
            bs = qim.block_size
            out["qim_tamper_bbox_px"] = [bbox[0] * bs, bbox[1] * bs,
                                         (bbox[2] + 1) * bs, (bbox[3] + 1) * bs]

    # tamper decision
    flag = False
    if mean_degr is not None and mean_degr > degr_thresh:
        flag = True
    if out.get("qim_clustering", 0.0) > clustering_thresh and \
            out.get("dft_nonuniformity", 0.0) > 0.25:
        flag = True
    out["is_tampered"] = bool(flag)
    out["tamper_score"] = float(
        (mean_degr or 0.0) + 0.5 * out.get("qim_clustering", 0.0)
        + 0.5 * out.get("dft_nonuniformity", 0.0))
    return out
