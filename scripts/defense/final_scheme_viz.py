"""FINAL scheme visualization: 3 fragments where VINE is CENTRAL-75% (TrustMark / VideoSeal full-frame),
+ SyncSeal sync. One comprehensive figure per row:
  Original | Watermarked (final) | VINE(central-75%) residual | TrustMark residual | VideoSeal residual | Total residual
Each residual ×20; watermarked col annotated PSNR/SSIM; each fragment col annotated its own PSNR."""
import os, sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.syncseal_frontend import load_sync, sync_embed
KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; AMP = 20; KEEP = 0.75
sb = ShortenedBCH(); n = sb.n
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def txc(iid, nm):
    p, M = F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def vine_central_marked(x, iid):
    """VINE mark in the central KEEP% square only; border unchanged."""
    s = int(512 * KEEP); o = (512 - s) // 2
    center = to512(x).crop((o, o, o + s, o + s))
    marked = F["vine"].embed_with_target(center, txc(iid, "vine"))
    out = to512(x).copy(); out.paste(marked.resize((s, s)), (o, o)); return out
def frag_alone(orig, iid, nm):
    if nm == "vine": return scale(orig, vine_central_marked(orig, iid))         # central-75%, ALPHA-blended
    return scale(orig, to512(F[nm].embed_with_target(orig, txc(iid, nm))))       # full-frame
def final_composite(orig, iid):
    x = scale(orig, vine_central_marked(orig, iid))                              # 1) VINE central-75%
    x = scale(x, to512(F["trustmark"].embed_with_target(x, txc(iid, "trustmark"))))   # 2) TrustMark
    x = scale(x, to512(F["videoseal"].embed_with_target(x, txc(iid, "videoseal"))))   # 3) VideoSeal
    return to512(sync_embed(sync, to512(x), dev))                               # 4) SyncSeal
def resid(orig, wm):
    O = np.asarray(to512(orig), np.float64); W = np.asarray(to512(wm), np.float64)
    return Image.fromarray(np.clip((W - O) * AMP + 128, 0, 255).astype(np.uint8)), _psnr(O, W, data_range=255)

OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/paper_figs"
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:3]
cols = ["Original", "Watermarked (final)", "VINE residual\n(central-75%)", "TrustMark residual", "VideoSeal residual", "Total residual"]
fig, ax = plt.subplots(len(srcs), 6, figsize=(18, 3.1 * len(srcs)))
if len(srcs) == 1: ax = ax[None, :]
for r, fp in enumerate(srcs):
    iid = f"fin_{r:03d}"; orig = to512(Image.open(fp).convert("RGB"))
    wm = final_composite(orig, iid)
    O, W = np.asarray(orig, np.float64), np.asarray(wm, np.float64)
    ps_c, ss_c = _psnr(O, W, data_range=255), _ssim(O, W, channel_axis=2, data_range=255)
    rv, pv = resid(orig, frag_alone(orig, iid, "vine"))
    rt, pt = resid(orig, frag_alone(orig, iid, "trustmark"))
    rs, psv = resid(orig, frag_alone(orig, iid, "videoseal"))
    rtot, _ = resid(orig, wm)
    imgs = [(orig, None), (wm, f"PSNR {ps_c:.1f} dB / SSIM {ss_c:.3f}"),
            (rv, f"PSNR {pv:.1f} dB"), (rt, f"PSNR {pt:.1f} dB"), (rs, f"PSNR {psv:.1f} dB"), (rtot, "composite")]
    for c, (img, lab) in enumerate(imgs):
        ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(cols[c], fontsize=10.5, fontweight="bold")
        if lab: ax[r, c].set_xlabel(lab, fontsize=8.5)
    print(f"row {r}: composite PSNR={ps_c:.2f} SSIM={ss_c:.4f} | VINE-c75 {pv:.1f} TM {pt:.1f} VS {psv:.1f}", flush=True)
    orig.save(f"{OUT}/final_{r}_orig.png"); wm.save(f"{OUT}/final_{r}_wm.png")
fig.suptitle("FINAL scheme (3 fragments; VINE = central-75%) — original, watermarked, per-fragment & total residual (×20)", fontsize=13, y=0.998)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_final_residual.png", dpi=140, bbox_inches="tight"); plt.close()
print("FINAL_SCHEME_VIZ_DONE")
