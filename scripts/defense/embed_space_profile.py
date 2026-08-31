"""Measure WHERE each fragment puts its mark. For VINE / TrustMark / VideoSeal residuals:
  1. radial frequency profile  -> energy fraction in 4 bands of Nyquist (low..high)
  2. per-channel energy        -> RGB + YCbCr (luma vs chroma hiding)
  3. texture correlation       -> corr(|residual|, local image gradient) : content-adaptive vs global template
  4. spatial locality          -> center-75% vs border energy density
  5. magnitude                 -> mean|r|, PSNR
  6. cross-fragment residual correlation -> do the three occupy the SAME space (overlap) or orthogonal?
"""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from skimage.metrics import peak_signal_noise_ratio as _psnr
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def embed_alone(orig, iid, nm):
    p, M = F[nm].get_perm_M(iid)
    tx = apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
    return scale(orig, to512(F[nm].embed_with_target(orig, tx)))

def radial_bands(r_gray, nb=4):
    """energy fraction per radial frequency band (0=DC .. 1=Nyquist)."""
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(r_gray))) ** 2
    h, w = Fm.shape; cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    rad = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)   # 0..~1.41, normalize by Nyquist=1
    tot = Fm.sum() + 1e-12; out = []
    for b in range(nb):
        lo, hi = b / nb, (b + 1) / nb
        m = (rad >= lo) & (rad < hi if b < nb - 1 else rad >= lo)
        out.append(float(Fm[m].sum() / tot))
    return out

def rgb2ycbcr(a):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
    return np.stack([Y, Cb, Cr], -1)

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:24]
ACC = {nm: {"bands": [], "rgb": [], "ycc": [], "tex": [], "center": [], "mag": [], "psnr": []} for nm in F}
XCORR = {k: [] for k in ["vine|trustmark", "vine|videoseal", "trustmark|videoseal"]}
for j, fp in enumerate(srcs):
    iid = f"sp_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
    O = np.asarray(orig, np.float64)
    # local image gradient magnitude = texture/busyness map
    gy, gx = np.gradient(O.mean(-1)); tex = np.sqrt(gx ** 2 + gy ** 2)
    res = {}
    for nm in F:
        W = np.asarray(embed_alone(orig, iid, nm), np.float64); r = W - O; res[nm] = r
        rg = r.mean(-1)
        ACC[nm]["bands"].append(radial_bands(rg))
        e_rgb = (r ** 2).sum((0, 1)); ACC[nm]["rgb"].append((e_rgb / (e_rgb.sum() + 1e-12)).tolist())
        ycc = rgb2ycbcr(r); e_y = (ycc ** 2).sum((0, 1)); ACC[nm]["ycc"].append((e_y / (e_y.sum() + 1e-12)).tolist())
        am = np.abs(rg)
        ACC[nm]["tex"].append(float(np.corrcoef(am.ravel(), tex.ravel())[0, 1]))
        s = int(512 * 0.75); o = (512 - s) // 2
        c_e = (am[o:o + s, o:o + s] ** 2).sum(); t_e = (am ** 2).sum() + 1e-12
        ACC[nm]["center"].append(float((c_e / t_e) / 0.5625))          # 1.0 = uniform (center75 = 56.25% area)
        ACC[nm]["mag"].append(float(np.abs(r).mean())); ACC[nm]["psnr"].append(_psnr(O, W, data_range=255))
    for k, (a, b) in [("vine|trustmark", ("vine", "trustmark")), ("vine|videoseal", ("vine", "videoseal")),
                      ("trustmark|videoseal", ("trustmark", "videoseal"))]:
        XCORR[k].append(float(np.corrcoef(res[a].ravel(), res[b].ravel())[0, 1]))

out = {"n": len(srcs), "fragments": {}}
for nm in F:
    A = ACC[nm]
    out["fragments"][nm] = {
        "freq_bands_lo_to_hi": np.mean(A["bands"], 0).round(4).tolist(),
        "rgb_energy": np.mean(A["rgb"], 0).round(4).tolist(),
        "ycbcr_energy": np.mean(A["ycc"], 0).round(4).tolist(),
        "texture_corr": round(float(np.mean(A["tex"])), 4),
        "center_density_ratio": round(float(np.mean(A["center"])), 4),
        "mean_abs_residual": round(float(np.mean(A["mag"])), 4),
        "psnr_alone": round(float(np.mean(A["psnr"])), 2)}
    f = out["fragments"][nm]
    print(f"{nm:10} freq(lo->hi)={f['freq_bands_lo_to_hi']}  Y/Cb/Cr={f['ycbcr_energy']}  "
          f"tex_corr={f['texture_corr']:+.3f}  center={f['center_density_ratio']:.3f}  "
          f"|r|={f['mean_abs_residual']:.2f}  PSNR={f['psnr_alone']:.1f}", flush=True)
out["cross_fragment_residual_corr"] = {k: round(float(np.mean(v)), 4) for k, v in XCORR.items()}
print("cross-fragment residual corr:", out["cross_fragment_residual_corr"], flush=True)
json.dump(out, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/embed_space_profile.json", "w"), indent=2)
print("EMBED_SPACE_PROFILE_DONE")
