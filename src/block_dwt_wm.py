"""Faithful reimplementation of the zero-bit, interpretable block-DWT watermark of
Bulychev, Marchant & Rubinstein, "Where is the Watermark? Interpretable Watermark
Detection at the Block Level", WACV 2026 (arXiv:2512.14994).

Method (training-free, classical signal processing):
  * Per RGB channel, tile the image into non-overlapping m=96 blocks, each split into
    k=8 sub-blocks (144 per block).
  * Each 8x8 sub-block gets a d=3-level DWT; only the single level-3 LL coefficient is
    touched. For the orthonormal Haar wavelet the level-3 LL of an 8x8 block equals
    8 * mean(block), so modifying LL3 == adding a uniform offset (LL3'-LL3)/8 to the
    sub-block (verified in the self-test). We use that analytic equivalence -> fully
    vectorised, no per-block pywt loop.
  * A per-block secret partition of the coefficient range [-R, R] into length-l
    intervals coloured green/red is derived from PRF(secret_key, seed) where
    seed = round(block_mean / round_val) * round_val (robust to small mean shifts).
    EMBED: project every LL3 in a red interval to the centre of the nearest green
    interval, capped at |delta| <= 3l (else leave it).
  * DETECT: re-derive the same per-block partition from the (possibly attacked) image,
    count LL3 coefficients landing in green intervals per block (N=144), and run the
    one-sided test  green_count > c* = z_{0.95}/2 * sqrt(N) + N/2  ~= 82.
    Outputs a block-level detection map and a global score = fraction of green blocks.

Zero-bit: presence/localisation only, NO payload. (See the eval scripts for how it is
benchmarked against the VINE+TrustMark composite and under regeneration attacks.)
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.sign_envelope import _hkdf_uint64_stream

Z_095 = 1.6448536269514722  # Phi^-1(0.95)


class BlockDWTWatermark:
    def __init__(self, master_key: bytes = b"v5_key_encoder_master", m: int = 96, k: int = 8,
                 levels: int = 3, l: float = 8.0, R: float = 3000.0, round_val: int = 30,
                 entropy_adaptive: bool = False):
        assert m % k == 0, "block must divide into sub-blocks"
        self.key = master_key
        self.m, self.k, self.levels, self.l, self.R, self.round_val = m, k, levels, l, R, round_val
        self.scale = float(2 ** levels)                 # Haar LL_d = scale * mean(sub-block)
        self.n_int = int(np.ceil(2 * R / l))            # number of intervals over [-R, R]
        self.N = (m // k) ** 2                           # LL coeffs per block (one per sub-block)
        self.cstar = Z_095 / 2.0 * np.sqrt(self.N) + self.N / 2.0
        self.entropy_adaptive = entropy_adaptive

    # ---- per-block green/red interval partition from PRF(key, seed) ----
    def _green_mask(self, seed: int) -> np.ndarray:
        salt = f"blockdwt/{int(seed)}".encode()
        n_words = (self.n_int + 63) // 64
        words = _hkdf_uint64_stream(self.key, salt, n_words)
        bits = np.unpackbits(np.asarray(words, dtype=">u8").view(np.uint8))[: self.n_int]
        return bits.astype(bool)                         # True = green interval

    def _interval_idx(self, coeff: np.ndarray) -> np.ndarray:
        idx = np.floor((coeff + self.R) / self.l).astype(np.int64)
        return np.clip(idx, 0, self.n_int - 1)

    def _green_centers(self, green: np.ndarray) -> np.ndarray:
        gi = np.where(green)[0]
        return -self.R + (gi + 0.5) * self.l, gi          # centres, indices

    # ---- entropy-adaptive block mask (paper App. B.5/C.1: embed/detect only blocks
    #      whose Shannon entropy exceeds the per-image median) ----
    def _high_entropy_mask(self, img: np.ndarray):
        lum = img.astype(np.float64).mean(axis=2)
        m = self.m; nBy, nBx = lum.shape[0] // m, lum.shape[1] // m
        H = np.zeros((nBy, nBx))
        for by in range(nBy):
            for bx in range(nBx):
                b = np.clip(lum[by * m:(by + 1) * m, bx * m:(bx + 1) * m], 0, 255).astype(np.uint8)
                hist = np.bincount(b.ravel(), minlength=256).astype(np.float64)
                p = hist / hist.sum(); p = p[p > 0]
                H[by, bx] = -np.sum(p * np.log2(p))
        return H > np.median(H)                          # True = embed/score this block

    # ---- sub-block mean grid (vectorised) ----
    def _subblock_means(self, chan: np.ndarray):
        H = (chan.shape[0] // self.k) * self.k; W = (chan.shape[1] // self.k) * self.k
        c = chan[:H, :W]
        sb = c.reshape(H // self.k, self.k, W // self.k, self.k)
        return sb.mean(axis=(1, 3)), H, W                 # (nby, nbx) sub-block means

    def embed(self, img: np.ndarray) -> np.ndarray:
        """img float [0,255] HxWx3 -> watermarked float [0,255]."""
        out = img.astype(np.float64).copy()
        m, k = self.m, self.k
        nb = m // k
        emask = self._high_entropy_mask(img) if self.entropy_adaptive else None
        for ch in range(3):
            chan = out[:, :, ch]
            means, H, W = self._subblock_means(chan)      # (H/k, W/k)
            ll3 = means * self.scale                       # Haar level-3 LL
            nby, nbx = ll3.shape
            nB_y, nB_x = nby // nb, nbx // nb              # number of m-blocks
            offset = np.zeros_like(ll3)
            for by in range(nB_y):
                for bx in range(nB_x):
                    if emask is not None and not emask[by, bx]:
                        continue                              # skip low-entropy block (adaptive)
                    blk = ll3[by * nb:(by + 1) * nb, bx * nb:(bx + 1) * nb]   # nb x nb LL3
                    # seed from the m-block mean (this channel)
                    bm = chan[by * m:(by + 1) * m, bx * m:(bx + 1) * m].mean()
                    seed = int(round(bm / self.round_val) * self.round_val)
                    green = self._green_mask(seed)
                    idx = self._interval_idx(blk)
                    is_red = ~green[idx]
                    if not is_red.any():
                        continue
                    centers, gidx = self._green_centers(green)
                    # nearest green centre to each red coeff
                    red_vals = blk[is_red]
                    nearest = centers[np.abs(centers[None, :] - red_vals[:, None]).argmin(axis=1)]
                    delta = nearest - red_vals
                    movable = np.abs(delta) <= 3 * self.l
                    new_vals = red_vals.copy(); new_vals[movable] = nearest[movable]
                    blk_new = blk.copy(); blk_new[is_red] = new_vals
                    offset[by * nb:(by + 1) * nb, bx * nb:(bx + 1) * nb] = (blk_new - blk) / self.scale
            # apply uniform per-sub-block offset (inverse Haar of an LL3-only change)
            off_full = np.repeat(np.repeat(offset, k, axis=0), k, axis=1)
            chan[:off_full.shape[0], :off_full.shape[1]] += off_full
        return np.clip(out, 0, 255)

    def detect(self, img: np.ndarray):
        """-> (global_score in [0,1], block-green map HbxWb in {0,1}).

        Faithful to the paper (Algorithm 4): per channel keep the RAW green COUNT per
        block; average the counts across R,G,B (S_mean); apply the c* test ONCE to the
        channel-averaged count. (NOT threshold-per-channel-then-majority-vote.)
        """
        x = img.astype(np.float64)
        m, k = self.m, self.k; nb = m // k
        count_maps = []
        for ch in range(3):
            chan = x[:, :, ch]
            means, H, W = self._subblock_means(chan); ll3 = means * self.scale
            nby, nbx = ll3.shape; nB_y, nB_x = nby // nb, nbx // nb
            cmap = np.zeros((nB_y, nB_x))
            for by in range(nB_y):
                for bx in range(nB_x):
                    blk = ll3[by * nb:(by + 1) * nb, bx * nb:(bx + 1) * nb]
                    bm = chan[by * m:(by + 1) * m, bx * m:(bx + 1) * m].mean()
                    seed = int(round(bm / self.round_val) * self.round_val)
                    green = self._green_mask(seed)
                    cmap[by, bx] = int(green[self._interval_idx(blk)].sum())   # raw green count
            count_maps.append(cmap)
        S_mean = np.mean(count_maps, axis=0)               # average green COUNT across channels
        block_green = (S_mean > self.cstar).astype(float)  # one-sided test, applied once
        if self.entropy_adaptive:
            em = self._high_entropy_mask(img)              # score only high-entropy blocks
            sel = em[:block_green.shape[0], :block_green.shape[1]]
            score = float(block_green[sel].mean()) if sel.any() else float(block_green.mean())
        else:
            score = float(block_green.mean())
        return score, block_green


if __name__ == "__main__":
    import glob
    from PIL import Image
    import pywt
    REPO = os.path.dirname(os.path.dirname(__file__))

    # (0) verify Haar LL3 == 8 * mean for an 8x8 block
    rng = np.random.RandomState(0); blk = rng.uniform(0, 255, (8, 8))
    ll3_pywt = pywt.wavedec2(blk, "haar", level=3)[0][0, 0]
    print(f"[check] Haar LL3 {ll3_pywt:.3f} vs 8*mean {8*blk.mean():.3f}  diff {abs(ll3_pywt-8*blk.mean()):.2e}")

    wm = BlockDWTWatermark(l=8.0)
    print(f"[cfg] n_int={wm.n_int} N={wm.N} cstar={wm.cstar:.1f}")
    files = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
             sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:188]
    def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
    def psnr(a, b):
        e = np.mean((a - b) ** 2); return 99.0 if e < 1e-9 else 10 * np.log10(255 * 255 / e)
    wm_scores, clean_scores, ps = [], [], []
    for fp in files:
        orig = np.asarray(to512(Image.open(fp).convert("RGB")), np.float64)
        w = wm.embed(orig)
        wm_scores.append(wm.detect(w)[0]); clean_scores.append(wm.detect(orig)[0]); ps.append(psnr(orig, w))
    print(f"[roundtrip] watermarked global score = {np.mean(wm_scores):.3f} (want ~1.0)")
    print(f"[FPR]       clean (unwm)  global score = {np.mean(clean_scores):.3f} (want ~0.0)")
    print(f"[fidelity]  PSNR = {np.mean(ps):.1f} dB")
    print("BLOCKDWT_SELFTEST_DONE")
