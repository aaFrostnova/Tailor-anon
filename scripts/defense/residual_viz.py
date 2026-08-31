"""Before/after + residual visualization for the DEPLOYED composite (ours = vine+trustmark+
videoseal + SyncSeal sync mark). Makes a montage: rows=images, cols=[original, watermarked, residual x AMP]."""
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
KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; AMP = 20
sb = ShortenedBCH(); n = sb.n
frag = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
        "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
        "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def embed(orig, iid):
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); x = orig
    for nm in ["vine", "trustmark", "videoseal"]:
        p, M = frag[nm].get_perm_M(iid); x = scale(x, to512(frag[nm].embed_with_target(x, apply_crypto(tx, p, M))))
    x = sync_embed(sync, to512(x), dev)  # deployed geo_cascade adds the SyncSeal sync mark
    return to512(x)

OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/paper_figs"
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:4]
rows = len(srcs)
fig, axes = plt.subplots(rows, 3, figsize=(9.6, 3.2 * rows))
if rows == 1: axes = axes[None, :]
col_titles = ["Original", "Watermarked (ours)", f"Residual ×{AMP}"]
for r, fp in enumerate(srcs):
    iid = f"viz_{r:03d}"; orig = to512(Image.open(fp).convert("RGB"))
    wm = embed(orig, iid)
    O = np.asarray(orig, np.float64); W = np.asarray(wm, np.float64)
    res = np.clip((W - O) * AMP + 128, 0, 255).astype(np.uint8)
    ps = _psnr(O, W, data_range=255); ss = _ssim(O, W, channel_axis=2, data_range=255)
    for c, (img, ttl) in enumerate(zip([orig, wm, Image.fromarray(res)], col_titles)):
        ax = axes[r, c]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(ttl, fontsize=12, fontweight="bold")
    axes[r, 1].set_ylabel(f"PSNR {ps:.1f} dB\nSSIM {ss:.3f}", fontsize=9, rotation=0, ha="right", va="center", labelpad=2)
    # also save individual full-res triptych images
    orig.save(f"{OUT}/resid_{r}_orig.png"); wm.save(f"{OUT}/resid_{r}_wm.png"); Image.fromarray(res).save(f"{OUT}/resid_{r}_residual.png")
    print(f"row {r}: PSNR={ps:.2f} SSIM={ss:.4f}", flush=True)
fig.suptitle("Deployed composite watermark — before / after / residual (VINE + TrustMark + VideoSeal + SyncSeal)", fontsize=12.5, y=0.995)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_residual.png", dpi=140, bbox_inches="tight"); plt.close()
print("RESIDUAL_VIZ_DONE", f"{OUT}/fig_residual.png")
