"""Multi-method redundant cryptographic fingerprint.

Stacks multiple watermarking methods onto the same image, each embedding
the same 100-bit crypto payload. Uses existing embed/extract pipeline from
src/embed.py and src/fragment.py.

Methods (based on Weekly Report findings):
  0. VINE pretrained (ref-free, ConvNeXt decoder) — classical attacks
  1. DCT low-freq (ref-needed, Pearson correlation) — JPEG, resize robust
  2. DWT-DCT L2 mid-freq (ref-needed) — multi-scale robust
  3. Quantization bits=6 (ref-needed) — strongest fragment, nearly all attacks
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.vine_crypto_wrapper import VineCryptoWrapper
from src.payload import BCHCodec, image_id_to_payload
from src.keygen import derive_all_subkeys
from src.fragment import generate_fragment
from src.embed import embed_single_fragment, extract_single_fragment, STRATEGIES
from src.verify import pearson_correlation, fisher_combine_pvalues


def _pil_to_chw(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0


def _chw_to_pil(arr: np.ndarray) -> Image.Image:
    hwc = np.clip(arr.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, "RGB")


# Fragment configs matching the Weekly Report retained methods
# ============================================================ log-polar alignment for crop+resize

def estimate_scale_logpolar(orig_chw: np.ndarray, susp_chw: np.ndarray) -> float:
    """Estimate scale factor between two same-size images using log-polar DFT."""
    from scipy.ndimage import map_coordinates

    C, H, W = orig_chw.shape
    gray_orig = np.mean(orig_chw, axis=0)
    gray_susp = np.mean(susp_chw, axis=0)

    mag_orig = np.abs(np.fft.fftshift(np.fft.fft2(gray_orig)))
    mag_susp = np.abs(np.fft.fftshift(np.fft.fft2(gray_susp)))

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    hp = 1.0 - np.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * (min(H, W) * 0.05)**2))
    mag_orig *= hp
    mag_susp *= hp

    max_r = min(H, W) // 2
    num_angles, num_radii = 360, 200
    log_base = np.exp(np.log(max_r) / num_radii)

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    radii = log_base ** np.arange(num_radii)
    angle_grid, radius_grid = np.meshgrid(angles, radii)
    y_coords = cy + radius_grid * np.sin(angle_grid)
    x_coords = cx + radius_grid * np.cos(angle_grid)

    lp_orig = map_coordinates(mag_orig, [y_coords, x_coords], order=1, mode='constant')
    lp_susp = map_coordinates(mag_susp, [y_coords, x_coords], order=1, mode='constant')

    f1 = np.fft.fft2(lp_orig)
    f2 = np.fft.fft2(lp_susp)
    cross = f1 * np.conj(f2)
    cross /= np.maximum(np.abs(cross), 1e-10)
    corr = np.fft.ifft2(cross).real

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift_r = peak[0]
    if shift_r > num_radii // 2:
        shift_r -= num_radii
    return log_base ** shift_r


def align_cropped(orig_chw: np.ndarray, cropped_chw: np.ndarray):
    """Find offset of cropped image within original using phase correlation."""
    _, th, tw = cropped_chw.shape
    _, oh, ow = orig_chw.shape
    if th >= oh and tw >= ow:
        return 0, 0

    padded = np.zeros_like(orig_chw)
    padded[:, :th, :tw] = cropped_chw
    best_pos, best_val = (0, 0), -1
    for c in range(orig_chw.shape[0]):
        f_orig = np.fft.fft2(orig_chw[c])
        f_pad = np.fft.fft2(padded[c])
        cross = f_orig * np.conj(f_pad)
        cross /= np.maximum(np.abs(cross), 1e-10)
        corr_map = np.fft.ifft2(cross).real
        valid = corr_map[:oh - th + 1, :ow - tw + 1]
        peak = np.unravel_index(np.argmax(valid), valid.shape)
        if valid[peak] > best_val:
            best_val = valid[peak]
            best_pos = peak

    ct, cl = best_pos
    best_mse = float("inf")
    for dt in range(max(0, ct - 3), min(oh - th + 1, ct + 4)):
        for dl in range(max(0, cl - 3), min(ow - tw + 1, cl + 4)):
            mse = np.mean((cropped_chw - orig_chw[:, dt:dt + th, dl:dl + tw]) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_pos = (dt, dl)
    return best_pos


def try_align_and_detect(orig_chw, susp_chw, subkeys, configs, epsilon, n_classical):
    """Try log-polar alignment then detect classical fragments on aligned pair."""
    C, H, W = orig_chw.shape
    scale = estimate_scale_logpolar(orig_chw, susp_chw)
    if scale < 1.01 or scale > 2.5:
        return None

    crop_h, crop_w = int(H / scale), int(W / scale)
    if crop_h < 16 or crop_w < 16:
        return None

    susp_pil = _chw_to_pil(susp_chw)
    susp_resized_pil = susp_pil.resize((crop_w, crop_h), Image.BILINEAR)
    susp_resized = _pil_to_chw(susp_resized_pil)

    top, left = align_cropped(orig_chw, susp_resized)
    _, th, tw = susp_resized.shape
    orig_crop = orig_chw[:, top:top + th, left:left + tw]

    results = {}
    for k, config in enumerate(configs):
        method_name = f"{config['strategy']}_{config.get('freq_range', config.get('bits', ''))}"
        ref_fragment = generate_fragment(
            subkeys[k], orig_chw.shape, epsilon=epsilon, fragment_index=k,
        )
        frag_crop = ref_fragment[:, top:top + th, left:left + tw] / n_classical
        extracted = extract_single_fragment(orig_crop, susp_resized, config)

        if frag_crop.shape == extracted.shape:
            corr, p_val = pearson_correlation(frag_crop, extracted)
        else:
            min_len = min(frag_crop.size, extracted.size)
            corr, p_val = pearson_correlation(
                frag_crop.flatten()[:min_len], extracted.flatten()[:min_len],
            )
        results[method_name] = {
            "detected": bool(p_val < 0.001),
            "correlation": float(corr),
            "p_value": float(p_val),
            "method": method_name,
            "aligned": True,
            "estimated_scale": float(scale),
        }
    return results


FRAGMENT_CONFIGS = [
    {"strategy": "dct", "region": "full", "freq_range": [1, 12]},
    {"strategy": "dwt-dct", "region": "full", "level": 2, "block_size": 8, "freq_range": [12, 35]},
    {"strategy": "quantization", "region": "full", "bits": 6},
]


class MultiMethodFingerprint:
    """Embed and detect using multiple redundant watermarking methods."""

    def __init__(
        self,
        master_key: bytes = b"v5_key_encoder_master",
        epsilon: float = 8.0 / 255.0,
        device: str = "cuda",
        use_vine: bool = True,
        dft_ckpt: str = "results/dft_fftaware_baseline/ckpt.pt",
        qim_ckpt: str = "results/quant_qim_frozen_d006/ckpt.pt",
    ):
        self.master_key = master_key
        self.epsilon = epsilon
        self.device = device
        self.codec = BCHCodec()
        self.configs = FRAGMENT_CONFIGS
        self.n_classical = len(self.configs)
        self.dft_ckpt = dft_ckpt
        self.qim_ckpt = qim_ckpt
        self._fused = None  # lazily built FusedDetector

        self.vine = None
        if use_vine:
            self.vine = VineCryptoWrapper(
                master_key=master_key, method_name="vine",
                n_bits=100, device=device,
            )

    def _get_subkeys(self, image_id: str):
        return derive_all_subkeys(self.master_key, image_id, num_fragments=self.n_classical)

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        # Step 1: VINE (learned encoder)
        if self.vine is not None:
            pil = self.vine.embed(pil, image_id)

        # Step 2: Classical fragments (DCT, DWT-DCT, Quantization)
        img = _pil_to_chw(pil)
        subkeys = self._get_subkeys(image_id)

        weight = 1.0 / self.n_classical
        for k, config in enumerate(self.configs):
            fragment = generate_fragment(
                subkeys[k], img.shape, epsilon=self.epsilon, fragment_index=k,
            )
            img = embed_single_fragment(img, fragment, config, weight=weight)

        return _chw_to_pil(img)

    def get_classical_reference(self, pil_original: Image.Image, image_id: str) -> Image.Image:
        """Get the reference image for classical fragment detection.

        If VINE is used, classical fragments are embedded AFTER VINE,
        so the reference for classical detection is the VINE-embedded image.
        """
        if self.vine is not None:
            return self.vine.embed(pil_original, image_id)
        return pil_original

    def _vine_aligned_detect(
        self, pil_original: Image.Image, pil_suspect: Image.Image, image_id: str,
    ) -> Optional[dict]:
        """VINE detection with log-polar alignment for crop+resize attacks.

        Uses correlation between VINE's perturbation pattern and the aligned
        suspect residual, rather than the neural decoder (which can't survive
        the double interpolation from crop alignment).
        """
        wm_pil = self.vine.embed(pil_original, image_id)
        wm_chw = _pil_to_chw(wm_pil)
        orig_chw = _pil_to_chw(pil_original)
        susp_chw = _pil_to_chw(pil_suspect)

        scale = estimate_scale_logpolar(wm_chw, susp_chw)
        if scale < 1.01 or scale > 2.5:
            return None

        C, H, W = wm_chw.shape
        crop_h, crop_w = int(H / scale), int(W / scale)
        if crop_h < 16 or crop_w < 16:
            return None

        susp_resized_pil = pil_suspect.resize((crop_w, crop_h), Image.BILINEAR)
        susp_resized_chw = _pil_to_chw(susp_resized_pil)

        top, left = align_cropped(wm_chw, susp_resized_chw)

        vine_perturbation = wm_chw - orig_chw
        vine_pert_crop = vine_perturbation[:, top:top + crop_h, left:left + crop_w]
        orig_crop = orig_chw[:, top:top + crop_h, left:left + crop_w]
        susp_residual = susp_resized_chw - orig_crop

        corr, p_val = pearson_correlation(vine_pert_crop, susp_residual)

        return {
            "detected": bool(p_val < 0.001),
            "correlation": float(corr),
            "p_value": float(p_val),
            "method": "vine_aligned",
            "aligned": True,
            "estimated_scale": float(scale),
        }

    def detect(
        self, pil_suspect: Image.Image, image_id: str,
        pil_original: Optional[Image.Image] = None,
    ) -> dict:
        results = {}

        # Method 0: VINE (ref-free)
        if self.vine is not None:
            results["vine"] = self.vine.detect(pil_suspect, image_id)

        # Methods 1-3: Classical fragments (ref-needed)
        # Reference is the VINE-embedded image (not raw original),
        # because classical fragments are embedded after VINE.
        if pil_original is not None:
            ref_for_classical = self.get_classical_reference(pil_original, image_id)
            orig = _pil_to_chw(ref_for_classical)
            susp = _pil_to_chw(pil_suspect)
            subkeys = self._get_subkeys(image_id)

            for k, config in enumerate(self.configs):
                method_name = f"{config['strategy']}_{config.get('freq_range', config.get('bits', ''))}"
                ref_fragment = generate_fragment(
                    subkeys[k], orig.shape, epsilon=self.epsilon, fragment_index=k,
                )
                extracted = extract_single_fragment(orig, susp, config)

                # Pearson correlation between reference fragment and extracted residual
                corr, p_val = pearson_correlation(ref_fragment, extracted)

                results[method_name] = {
                    "detected": bool(p_val < 0.001),
                    "correlation": float(corr),
                    "p_value": float(p_val),
                    "method": method_name,
                }

        # If no classical method detected, try log-polar alignment (crop+resize recovery)
        classical_detected = any(
            r.get("detected", False) for k, r in results.items() if k != "vine"
        )
        if not classical_detected and pil_original is not None:
            ref_for_classical = self.get_classical_reference(pil_original, image_id)
            orig_for_align = _pil_to_chw(ref_for_classical)
            susp_for_align = _pil_to_chw(pil_suspect)
            subkeys = self._get_subkeys(image_id)
            aligned_results = try_align_and_detect(
                orig_for_align, susp_for_align, subkeys,
                self.configs, self.epsilon, self.n_classical,
            )
            if aligned_results is not None:
                for k, v in aligned_results.items():
                    results[k + "_aligned"] = v

        # VINE + log-polar alignment (ref-needed fallback for crop attacks)
        vine_detected = results.get("vine", {}).get("detected", False)
        if self.vine is not None and not vine_detected and pil_original is not None:
            vine_aligned = self._vine_aligned_detect(pil_original, pil_suspect, image_id)
            if vine_aligned is not None:
                results["vine_aligned"] = vine_aligned

        # Aggregate
        any_detected = any(r.get("detected", False) for r in results.values())
        surviving = [k for k, r in results.items() if r.get("detected", False)]

        return {
            "detected": any_detected,
            "surviving_methods": surviving,
            "n_methods_survived": len(surviving),
            "n_methods_tried": len(results),
            "per_method": results,
        }

    # ----------------------------------------------- soft-fusion path (Component A)
    def _ensure_fused(self):
        """Lazily build the FusedDetector (VINE + DFT-Kred + QIM, shared codeword)."""
        if self._fused is None:
            from src.fused_detector import FusedDetector
            self._fused = FusedDetector(
                master_key=self.master_key,
                dft_ckpt=str(REPO / self.dft_ckpt),
                qim_ckpt=str(REPO / self.qim_ckpt),
                device=self.device,
                use_vine=self.vine is not None,
            )
        return self._fused

    def embed_fused(self, pil: Image.Image, image_id: str) -> Image.Image:
        """Embed the shared shortened-BCH codeword with all bit-carrying methods."""
        return self._ensure_fused().embed(pil, image_id)

    def detect_fused(self, pil_suspect: Image.Image, image_id: str, **kwargs) -> dict:
        """Soft per-bit LLR fusion + Chase soft-BCH decode across bit methods.

        Returns {detected, payload_bits, or_detected, per_method, fused_llr, ...}.
        kwargs: weights, chase_p, erasure_idx, erasure_abs_thresh.
        """
        return self._ensure_fused().detect(pil_suspect, image_id, **kwargs)
