"""Two residual figures:
 (A) fig_residual_fragments: per-fragment residual signatures — original | VINE | TrustMark | VideoSeal | composite (ALPHA-scaled, as deployed).
 (B) fig_residual_central: full-VINE vs central-75%-VINE residual — shows the outer 25% left unmarked (quality gain), matching the +2.8 dB PSNR claim."""
import os, sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as _psnr
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.syncseal_frontend import load_sync, sync_embed
KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; AMP = 20
sb = ShortenedBCH(); n = sb.n
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def tx_of(iid): return sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
def frag_alone(orig, iid, nm):
    p, M = F[nm].get_perm_M(iid); return scale(orig, to512(F[nm].embed_with_target(orig, apply_crypto(tx_of(iid), p, M))))
def composite(orig, iid):
    x = orig
    for nm in ["vine", "trustmark", "videoseal"]:
        p, M = F[nm].get_perm_M(iid); x = scale(x, to512(F[nm].embed_with_target(x, apply_crypto(tx_of(iid), p, M))))
    return to512(sync_embed(sync, to512(x), dev))
def vine_full(orig, iid):
    p, M = F["vine"].get_perm_M(iid); return to512(F["vine"].embed_with_target(orig, apply_crypto(tx_of(iid), p, M)))
def vine_central(orig, iid, keep=0.75):
    p, M = F["vine"].get_perm_M(iid); s = int(512 * keep); o = (512 - s) // 2
    center = orig.crop((o, o, o + s, o + s)); marked = F["vine"].embed_with_target(center, apply_crypto(tx_of(iid), p, M))
    out = orig.copy(); out.paste(marked.resize((s, s)), (o, o)); return out
def resid(orig, wm):
    O = np.asarray(orig, np.float64); W = np.asarray(to512(wm), np.float64)
    r = np.clip((W - O) * AMP + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(r), _psnr(O, W, data_range=255)

OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/paper_figs"
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:2]

# ---- Figure A: per-fragment residuals ----
cols = ["Original", "VINE", "TrustMark", "VideoSeal", "Composite"]
figA, ax = plt.subplots(len(srcs), 5, figsize=(15, 3.05 * len(srcs)))
if len(srcs) == 1: ax = ax[None, :]
for r, fp in enumerate(srcs):
    iid = f"frg_{r:03d}"; orig = to512(Image.open(fp).convert("RGB"))
    views = [(orig, None)]
    for nm in ["vine", "trustmark", "videoseal"]:
        rr, ps = resid(orig, frag_alone(orig, iid, nm)); views.append((rr, ps))
    rr, ps = resid(orig, composite(orig, iid)); views.append((rr, ps))
    for c, (img, ps) in enumerate(views):
        ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(cols[c] + ("" if c == 0 else f" residual ×{AMP}"), fontsize=11, fontweight="bold")
        if ps is not None: ax[r, c].set_xlabel(f"PSNR {ps:.1f} dB", fontsize=9)
figA.suptitle("Per-fragment residual signatures (each ALPHA-scaled as deployed) vs the composite", fontsize=12.5, y=0.998)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_residual_fragments.png", dpi=140, bbox_inches="tight"); plt.close()
print("FIG_A_DONE fragments", flush=True)

# ---- Figure B: full-VINE vs central-75%-VINE residual ----
colsB = ["Original", "Full-VINE residual", "Central-75% VINE residual\n(+ crop-search decode)"]
figB, ax = plt.subplots(len(srcs), 3, figsize=(9.6, 3.2 * len(srcs)))
if len(srcs) == 1: ax = ax[None, :]
for r, fp in enumerate(srcs):
    iid = f"cvr_{r:03d}"; orig = to512(Image.open(fp).convert("RGB"))
    rf, pf = resid(orig, vine_full(orig, iid)); rc, pc = resid(orig, vine_central(orig, iid, 0.75))
    for c, (img, ps) in enumerate([(orig, None), (rf, pf), (rc, pc)]):
        ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(colsB[c], fontsize=10.5, fontweight="bold")
        if ps is not None: ax[r, c].set_xlabel(f"PSNR {ps:.1f} dB", fontsize=9)
figB.suptitle("VINE residual: full-frame vs central-75% (outer 25% left unmarked → quality gain)", fontsize=12, y=0.998)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_residual_central.png", dpi=140, bbox_inches="tight"); plt.close()
print("FIG_B_DONE central", flush=True)
print("RESIDUAL_FRAGS_DONE")
