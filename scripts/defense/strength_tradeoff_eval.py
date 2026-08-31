"""Strength vs quality trade-off for the VINE+TrustMark composite.

Scale BOTH fragments' residuals by alpha (alpha=1 = default strength). Lower
alpha -> higher PSNR/SSIM (better quality) but weaker mark. For each alpha,
measure quality + REAL composite dual-detection (zerobit ba>=tau OR crypto-ID)
under clean / JPEG25 / centercrop0.75 / regen. Find the sweet spot: smallest
alpha that still detects ~1.0 -> best quality with an effective watermark.
"""
import os, sys, glob, io
import numpy as np, torch
from PIL import Image
from scipy.stats import binom
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
N = 16
dev = "cuda"
imgs = sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png")))[:N]

sb = ShortenedBCH(); tau = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
print(f"n_codeword={sb.n} data_bits={sb.data_bits} tau={tau:.3f}", flush=True)
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
regen_pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

def scale_resid(orig_pil, wm_pil, alpha):
    o = np.asarray(to512(orig_pil), np.float64); w = np.asarray(to512(wm_pil), np.float64)
    return Image.fromarray(np.clip(o + alpha * (w - o), 0, 255).astype(np.uint8))

def jpeg(pil, q=25):
    b = io.BytesIO(); pil.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")

def centercrop(pil, area=0.75):
    s = int(round(512 * np.sqrt(area))); off = (512 - s) // 2
    return pil.crop((off, off, off + s, off + s)).resize((512, 512))

ATTACKS = {
    "clean": lambda im, sd: im,
    "jpeg25": lambda im, sd: jpeg(im, 25),
    "crop75": lambda im, sd: centercrop(im, 0.75),
    "regen": lambda im, sd: stable_regen(regen_pipe, im, seed=sd),
}

def detect(att_pil, image_id, tx, pv, Mv, pt, Mt):
    av = method_soft_to_codeword_llr(vine.raw_probs(att_pil), pv, Mv, kind="prob", n_codeword=sb.n)
    at = method_soft_to_codeword_llr(tm.raw_logits(att_pil), pt, Mt, kind="logit", n_codeword=sb.n)
    fused = fuse_llrs({"vine": av, "tm": at}, n_codeword=sb.n)
    fba = float(np.mean(llr_to_bits(fused) == tx))
    fver = bool(decode_and_verify(fused, image_id, codec=sb)["detected"])
    return (1.0 if (fver or fba >= tau) else 0.0), fba

# results[alpha] = {psnr, ssim, det[attack], ba[attack]}
res = {a: {"psnr": [], "ssim": [], "det": {k: [] for k in ATTACKS}, "ba": {k: [] for k in ATTACKS}} for a in ALPHAS}

for i, fp in enumerate(imgs):
    image_id = f"trade_{i:05d}"
    tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(image_id); pt, Mt = tm.get_perm_M(image_id)
    tv = apply_crypto(tx, pv, Mv); tt = apply_crypto(tx, pt, Mt)
    # full-strength fragment outputs (scale later, linear in residual)
    v_full = to512(vine.embed_with_target(orig, tv))
    for a in ALPHAS:
        v_a = scale_resid(orig, v_full, a)                 # VINE at strength a
        tm_full_on_v = to512(tm.embed_with_target(v_a, tt))
        comp = scale_resid(v_a, tm_full_on_v, a)           # TM at strength a on top
        o = np.asarray(orig, np.float64); c = np.asarray(comp, np.float64)
        mse = np.mean((o - c) ** 2)
        res[a]["psnr"].append(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
        res[a]["ssim"].append(float(ssim_fn(o, c, channel_axis=2, data_range=255)))
        for k, fn in ATTACKS.items():
            att = to512(fn(comp, 1000 + i))
            d, ba = detect(att, image_id, tx, pv, Mv, pt, Mt)
            res[a]["det"][k].append(d); res[a]["ba"][k].append(ba)
    print(f"  [{i+1}/{N}] {os.path.basename(fp)} done", flush=True)

def m(x): return float(np.mean(x)) if x else float("nan")
print("\n=== STRENGTH TRADE-OFF (n=%d, tau=%.3f) ===" % (N, tau))
hdr = "alpha | PSNR  SSIM  | " + "  ".join(f"{k}(det/ba)" for k in ATTACKS)
print(hdr)
for a in ALPHAS:
    r = res[a]
    cells = "  ".join(f"{m(r['det'][k]):.2f}/{m(r['ba'][k]):.2f}" for k in ATTACKS)
    print(f" {a:.2f} | {m(r['psnr']):5.2f} {m(r['ssim']):.3f} | {cells}")

# plot
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ps = [m(res[a]["psnr"]) for a in ALPHAS]; ss = [m(res[a]["ssim"]) for a in ALPHAS]
ax2 = ax[0].twinx()
ax[0].plot(ALPHAS, ps, "-o", color="tab:blue", label="PSNR")
ax2.plot(ALPHAS, ss, "-s", color="tab:green", label="SSIM")
ax[0].set_xlabel("watermark strength alpha"); ax[0].set_ylabel("PSNR (dB)", color="tab:blue")
ax2.set_ylabel("SSIM", color="tab:green"); ax[0].set_title("Quality vs strength"); ax[0].grid(alpha=.3)
for k in ATTACKS:
    ax[1].plot(ALPHAS, [m(res[a]["det"][k]) for a in ALPHAS], "-o", label=k, lw=2)
ax[1].axhline(1.0, color="grey", ls=":", lw=1)
ax[1].set_xlabel("watermark strength alpha"); ax[1].set_ylabel("composite detection rate")
ax[1].set_title("Detection vs strength (real dual-detect)"); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_ylim(-0.03, 1.05)
plt.tight_layout()
out = os.path.join(REPO, "results/defense/strength_tradeoff.png")
plt.savefig(out, dpi=120); print(f"[plot] -> {out}")
print("TRADEOFF_DONE")
