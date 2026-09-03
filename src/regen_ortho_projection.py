"""Reactive projection onto the regen-PRESERVED subspace (the orthogonal complement of what regen kills).

Idea: regen (VAE-roundtrip + denoise) erases some directions of an image and preserves others. Per image,
PROBE regen to estimate a per-frequency survival ratio S(rho), then keep the watermark only where regen
preserves energy. This is "reactive" (measured per image) and a projection (drop the volatile directions).

Public API (all numpy, FFT on CPU is fine at 512):
  measure_survival_spectrum(x, regen_seeded, probe_seed, attack_unused, n_probes, probe_std)
      -> (rho_norm, S)   radial survival ratio |F(regen(x+p)-regen(x))| / |F(p)| , averaged over probes/channels
  build_kept_mask(S, H, W, mode='hard', thresh=0.5)
      -> 2D fftshifted mask in [0,1]  (the regen-kept frequency support)
  project(delta_hw3, mask, renorm=True)
      -> delta with volatile freqs removed, energy renormalized to the original ||delta||
  survival_ratio(delta_hw3, base_arr, attacked_arr)
      -> scalar in ~[0,1]: fraction of delta that survives regen, measured along delta itself
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def _fft2c(a):
    return np.fft.fftshift(np.fft.fft2(a))


def _ifft2c(A):
    return np.real(np.fft.ifft2(np.fft.ifftshift(A)))


def _radial(H, W):
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.hypot(y - cy, x - cx)
    return r, min(cy, cx)


def measure_survival_spectrum(x, regen_seeded, probe_seed=1234, n_probes=8, probe_std=3.0, rng_seed=0):
    """x: HxWx3 float[0,255]. regen_seeded(arr_float[0,255], seed)->HxWx3 float[0,255].

    Estimate regen's per-frequency LINEAR transfer gain via cross-spectral density:
        T(f) = sum_k conj(P_k) S_k / sum_k |P_k|^2      (least-squares transfer estimate)
    where P_k=FFT(probe_k), S_k=FFT(regen(x+probe_k)-regen(x)) with a SHARED seed (so regen's own
    stochastic noise cancels). regen's CHAOTIC divergence is uncorrelated with the probe, so it averages
    toward zero in the cross-spectrum (unlike a naive |S|/|P| ratio, which sums its positive floor).
    Returns (rho_norm, S) with S = radial-mean |T| ~ fraction of perturbation amplitude regen retains."""
    H, W, _ = x.shape
    r, maxr = _radial(H, W)
    base = regen_seeded(x, probe_seed)                       # regen(x), fixed seed
    rng = np.random.RandomState(rng_seed)
    Pss = np.zeros((H, W)); PSc = np.zeros((H, W), dtype=np.complex128)
    for k in range(n_probes):
        p = rng.randn(H, W, 3).astype(np.float32) * probe_std
        g1 = regen_seeded(np.clip(x + p, 0, 255), probe_seed)
        surv = g1 - base
        for c in range(3):
            P = _fft2c(p[:, :, c]); Sf = _fft2c(surv[:, :, c])
            Pss += np.abs(P) ** 2
            PSc += np.conj(P) * Sf
    gain = np.abs(PSc / (Pss + 1e-6))                        # |complex transfer|, divergence cancels
    ri = r.ravel().astype(int); g = gain.ravel()
    acc = np.bincount(ri, g, minlength=maxr)[:maxr]; cnt = np.bincount(ri, minlength=maxr)[:maxr]
    S = acc / np.maximum(cnt, 1)
    S = ndimage.uniform_filter1d(S, 5)                       # smooth radial profile
    rho = np.arange(maxr) / maxr
    return rho, S


def build_kept_mask(S, H, W, mode="hard", thresh=0.5):
    r, maxr = _radial(H, W)
    ri = np.clip(r.astype(int), 0, len(S) - 1)
    Smap = S[ri]
    if mode == "hard":
        mask = (Smap >= thresh).astype(np.float64)
    elif mode == "soft":                                    # Wiener-like keep weight
        mask = np.clip(Smap, 0, 1)
    else:
        raise ValueError(mode)
    return mask


def project(delta, mask, renorm=True):
    """delta: HxWx3 float. Keep only mask-selected frequencies; optionally renormalize to ||delta||."""
    out = np.zeros_like(delta, dtype=np.float64)
    for c in range(delta.shape[2]):
        D = _fft2c(delta[:, :, c]) * mask
        out[:, :, c] = _ifft2c(D)
    if renorm:
        n0 = np.linalg.norm(delta); n1 = np.linalg.norm(out)
        if n1 > 1e-9:
            out *= n0 / n1
    return out


def survival_ratio(delta, base_arr, attacked_arr):
    """How much of delta survives regen, measured ALONG delta (predicts correlation-decode SNR).
    base_arr=regen(x), attacked_arr=regen(x+delta), same seed. <surv,delta>/<delta,delta>."""
    surv = (attacked_arr - base_arr).ravel()
    d = delta.ravel()
    return float(np.dot(surv, d) / (np.dot(d, d) + 1e-9))
