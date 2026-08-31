"""Probe HOW VINE embeds: is the mark content-adaptive / texture-locked, or a fixed additive template?

Four independent probes (all in pixel space, the VINE encoder output):
  P1 fixed-template?   same payload+key into N images -> cross-image residual correlation.
                       fixed additive template (e.g. Gaussian Shading) -> ~1.0; content-adaptive -> low.
                       Also: fixed-component energy fraction ||mean R||^2 / mean||R||^2.
  P2 texture-locked?   per-image corr(|residual|, local texture[grad,std]); residual energy by texture quintile.
                       texture-locked -> positive corr + monotone rising profile.
  P3 payload vs content: same image, K payloads -> payload-independent ("carrier shaping") energy fraction
                       vs payload-driven energy. high content fraction = residual shape set by image, not bits.
  P4 frequency:        radial power spectrum of residual + ratio to image spectrum. rising-with-freq ratio
                       = mark sits on detail/texture relative to content; flat = mirrors content; falling = global/low-freq.
Outputs JSON to results/defense/vine_embedding_structure.json + prints a summary.
"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
OUTJ = os.path.join(REPO, "results/defense/vine_embedding_structure.json")
DEV = "cuda"; RES = 512


def arr(pil):
    return np.asarray(pil.convert("RGB").resize((RES, RES)), np.float32)


def gray(a):
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def texture_maps(g):
    gx = ndimage.sobel(g, axis=1); gy = ndimage.sobel(g, axis=0)
    gradmag = np.hypot(gx, gy)
    mean = ndimage.uniform_filter(g, 8)
    var = ndimage.uniform_filter(g * g, 8) - mean * mean
    localstd = np.sqrt(np.clip(var, 0, None))
    return gradmag, localstd


def resid_mag(R):  # R: HxWx3 signed -> per-pixel L2 over channels
    return np.sqrt((R ** 2).sum(2))


def radial_spectrum(img2d):
    F = np.fft.fftshift(np.fft.fft2(img2d - img2d.mean()))
    P = np.abs(F) ** 2
    h, w = P.shape; cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(y - cy, x - cx).astype(int)
    tbin = np.bincount(r.ravel(), P.ravel())
    nbin = np.bincount(r.ravel())
    prof = tbin / np.maximum(nbin, 1)
    rn = np.arange(len(prof)) / (min(cy, cx))  # normalized radial freq, 1 ~ Nyquist
    return rn[:min(cy, cx)], prof[:min(cy, cx)]


def band_fracs(rn, prof):
    tot = prof.sum() + 1e-12
    lo = prof[rn < 0.10].sum() / tot
    mid = prof[(rn >= 0.10) & (rn < 0.40)].sum() / tot
    hi = prof[rn >= 0.40].sum() / tot
    return float(lo), float(mid), float(hi)


def main():
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=DEV)
    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + 40]

    def embed(C_pil, iid):
        W = vine.embed(C_pil, iid)
        if W.size != (RES, RES): W = W.resize((RES, RES))
        return W

    out = {}

    # ---------- P1: fixed-template test (same payload+key across N images) ----------
    N1 = 12; resids = []; covers = []
    for i in range(N1):
        C = Image.open(files[i]).convert("RGB").resize((RES, RES))
        W = embed(C, "FIXED_PAYLOAD_ID")           # identical message+key for every cover
        R = arr(W) - arr(C); resids.append(R); covers.append(arr(C))
    Rs = np.stack(resids)                            # (N,H,W,3)
    flat = Rs.reshape(N1, -1)
    flat = flat - flat.mean(1, keepdims=True)
    cc = np.corrcoef(flat)
    off = cc[np.triu_indices(N1, 1)]
    Rbar = Rs.mean(0)
    fixed_frac = float((Rbar ** 2).sum() / (Rs ** 2).sum(axis=(1, 2, 3)).mean())  # ||meanR||^2 / mean_i ||R_i||^2
    out["P1_fixed_template"] = {
        "n_images": N1,
        "cross_image_residual_corr_mean": float(off.mean()),
        "cross_image_residual_corr_std": float(off.std()),
        "fixed_component_energy_fraction": fixed_frac,
        "note": "corr~1 & frac~1 => fixed additive template; corr~0 & frac~0 => content-adaptive",
    }

    # ---------- P2: texture-locking (use the P1 set) ----------
    pear_g, pear_s, spear_g, prof_quint = [], [], [], []
    for C, R in zip(covers, resids):
        g = gray(C); gm, ls = texture_maps(g); rm = resid_mag(R)
        a, b = rm.ravel(), gm.ravel()
        pear_g.append(float(np.corrcoef(a, b)[0, 1]))
        pear_s.append(float(np.corrcoef(a, ls.ravel())[0, 1]))
        # spearman via rank
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        spear_g.append(float(np.corrcoef(ra, rb)[0, 1]))
        # residual energy by texture quintile
        q = np.quantile(b, [0.2, 0.4, 0.6, 0.8])
        idx = np.digitize(b, q)
        prof_quint.append([float(a[idx == k].mean()) for k in range(5)])
    pq = np.array(prof_quint).mean(0)
    out["P2_texture_locked"] = {
        "pearson_residmag_vs_gradmag_mean": float(np.mean(pear_g)),
        "pearson_residmag_vs_localstd_mean": float(np.mean(pear_s)),
        "spearman_residmag_vs_gradmag_mean": float(np.mean(spear_g)),
        "residmag_by_texture_quintile": [float(x) for x in pq],
        "quintile_ratio_top_over_bottom": float(pq[-1] / max(pq[0], 1e-9)),
        "note": "positive corr + rising quintile profile => mark concentrates on textured/edge regions",
    }

    # ---------- P3: payload vs content (same image, K payloads) ----------
    K = 8; img_idx = [0, 5, 10, 15, 20]; fracs_content = []; fracs_payload = []
    for ii in img_idx:
        C = Image.open(files[ii]).convert("RGB").resize((RES, RES)); Ca = arr(C)
        Rk = np.stack([arr(embed(C, f"payload_{ii}_{k}")) - Ca for k in range(K)])
        Rb = Rk.mean(0)
        content_e = float((Rb ** 2).sum())                          # ||mean_k R_k||^2
        total_e = float((Rk ** 2).sum(axis=(1, 2, 3)).mean())       # mean_k ||R_k||^2
        payload_e = float(((Rk - Rb) ** 2).sum(axis=(1, 2, 3)).mean())
        fracs_content.append(content_e / total_e); fracs_payload.append(payload_e / total_e)
    out["P3_payload_vs_content"] = {
        "n_images": len(img_idx), "n_payloads": K,
        "content_shaped_energy_fraction_mean": float(np.mean(fracs_content)),
        "payload_driven_energy_fraction_mean": float(np.mean(fracs_payload)),
        "note": "content fraction high => residual SHAPE set by image (carrier shaping); "
                "payload fraction = how much per-pixel residual is message modulation",
    }

    # ---------- P4: frequency content of residual vs image ----------
    lo_r, mid_r, hi_r, ratio_curve = [], [], [], []
    for C, R in zip(covers, resids):
        prs = [radial_spectrum(R[:, :, c]) for c in range(3)]       # SIGNED residual per channel
        rn = prs[0][0]; pr = np.mean([p[1] for p in prs], axis=0)
        _, pc = radial_spectrum(gray(C))
        l, m, h = band_fracs(rn, pr); lo_r.append(l); mid_r.append(m); hi_r.append(h)
        ratio = pr / (pc + 1e-9)
        # summarize ratio slope: mean ratio in high band vs low band (normalized)
        rl = ratio[rn < 0.10].mean(); rh = ratio[rn >= 0.40].mean()
        ratio_curve.append(float(rh / (rl + 1e-12)))
    out["P4_frequency"] = {
        "residual_band_energy_low_<0.1": float(np.mean(lo_r)),
        "residual_band_energy_mid_0.1-0.4": float(np.mean(mid_r)),
        "residual_band_energy_high_>0.4": float(np.mean(hi_r)),
        "resid_to_image_spectrum_ratio_high_over_low": float(np.mean(ratio_curve)),
        "note": "ratio>1 => residual is MORE high-freq than the image (rides on detail/texture); "
                "~1 => mirrors content; <1 => more global/low-freq than content",
    }

    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    json.dump(out, open(OUTJ, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("VINE_STRUCTURE_DONE ->", OUTJ)


if __name__ == "__main__":
    main()
