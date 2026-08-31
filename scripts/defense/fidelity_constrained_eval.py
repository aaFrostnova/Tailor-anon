"""Fidelity-constrained attack eval: detection vs attack-induced quality loss.

For each attack we sweep strength, recording (a) attack-induced quality (SSIM/PSNR
of attacked vs clean watermarked) and (b) composite detection. Rotation is INCLUDED
(with resync). Regen strength axis = number of regeneration ROUNDS (x1/x2/x4), the
correct monotonic axis (noise_step does NOT control strength in stable_regen).
Composite = VINE+TrustMark @ alpha=0.7, dual detection (+resync for rotation).
"""
import os, sys, glob, io, json
import numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import binom
from skimage.metrics import structural_similarity as ssim_fn

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; ALPHA = 0.70; N = 24; dev = "cuda"
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[:N]
sb = ShortenedBCH(); tau = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
regen_pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))

def att_jpeg(im, r): q = int(round(90 - 80 * r)); b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def att_blur(im, r): return im.filter(ImageFilter.GaussianBlur(radius=0.5 + 7.5 * r))
def att_noise(im, r):
    a = np.asarray(im, np.float64) + np.random.RandomState(int(r * 1e5) + 7).normal(0, (0.02 + 0.10 * r) * 255, (512, 512, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def att_bright(im, r): return ImageEnhance.Brightness(im).enhance(1 + r)
def att_contrast(im, r): return ImageEnhance.Contrast(im).enhance(1 + r)
def att_crop(im, r):
    area = 1.0 - 0.45 * r; s = int(round(512 * np.sqrt(area))); off = (512 - s) // 2
    return im.crop((off, off, off + s, off + s)).resize((512, 512))
def rot(im, deg): return im.rotate(deg, resample=Image.BILINEAR, expand=False)
def regen_rounds(im, k):
    out = im
    for r in range(k): out = stable_regen(regen_pipe, out, seed=r)
    return out

SWEEP = {
    "jpeg": [("%.1f" % r, (lambda im, rr=r: att_jpeg(im, rr))) for r in [0.2, 0.4, 0.6, 0.8, 1.0]],
    "blur": [("%.1f" % r, (lambda im, rr=r: att_blur(im, rr))) for r in [0.2, 0.4, 0.6, 0.8, 1.0]],
    "noise": [("%.1f" % r, (lambda im, rr=r: att_noise(im, rr))) for r in [0.2, 0.4, 0.6, 0.8, 1.0]],
    "bright": [("%.1f" % r, (lambda im, rr=r: att_bright(im, rr))) for r in [0.2, 0.4, 0.6, 0.8, 1.0]],
    "contrast": [("%.1f" % r, (lambda im, rr=r: att_contrast(im, rr))) for r in [0.2, 0.4, 0.6, 0.8, 1.0]],
    "crop": [("%.1f" % r, (lambda im, rr=r: att_crop(im, rr))) for r in [0.2, 0.4, 0.6, 0.8, 1.0]],
    "rotate": [("%ddeg" % a, (lambda im, aa=a: rot(im, aa))) for a in [9, 18, 27]],
    "regen": [("x%d" % k, (lambda im, kk=k: regen_rounds(im, kk))) for k in [1, 2, 4]],
}
RESYNC_ATTACKS = {"rotate"}
ANGLES = np.arange(-30, 30.01, 3.0)

def _fused(att, pv, Mv, pt, Mt):
    av = method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
    at = method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n)
    return fuse_llrs({"v": av, "t": at}, n_codeword=sb.n)

def detect(att, pv, Mv, pt, Mt, tx, iid):
    fused = _fused(att, pv, Mv, pt, Mt); ba = float(np.mean(llr_to_bits(fused) == tx))
    ver = bool(decode_and_verify(fused, iid, codec=sb)["detected"])
    return (1.0 if (ver or ba >= tau) else 0.0), ba

def detect_resync(att, pv, Mv, pt, Mt, tx, iid):
    det, ba = detect(att, pv, Mv, pt, Mt, tx, iid)
    if det: return det, ba
    for d in ANGLES:                                   # crypto-verify gate per candidate angle
        fused = _fused(rot(att, d), pv, Mv, pt, Mt)
        if bool(decode_and_verify(fused, iid, codec=sb)["detected"]):
            return 1.0, float(np.mean(llr_to_bits(fused) == tx))
    return 0.0, ba

def q(a, b):
    A = np.asarray(a, np.float64); B = np.asarray(b, np.float64)
    mse = np.mean((A - B) ** 2); psnr = 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
    return float(ssim_fn(A, B, channel_axis=2, data_range=255)), float(psnr)

rec = {}
for i, fp in enumerate(imgs):
    image_id = f"fc_{i:05d}"; tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(image_id); pt, Mt = tm.get_perm_M(image_id)
    v_a = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(tx, pv, Mv))), ALPHA)
    comp = scale_resid(v_a, to512(tm.embed_with_target(v_a, apply_crypto(tx, pt, Mt))), ALPHA)
    cnp = np.asarray(comp, np.float64)
    for atk, variants in SWEEP.items():
        dfn = detect_resync if atk in RESYNC_ATTACKS else detect
        for lab, fn in variants:
            att = to512(fn(comp)); ss, ps = q(cnp, att)
            det, ba = dfn(att, pv, Mv, pt, Mt, tx, image_id)
            k = (atk, lab); rec.setdefault(k, {"det": [], "ssim": [], "psnr": [], "ba": []})
            rec[k]["det"].append(det); rec[k]["ssim"].append(ss); rec[k]["psnr"].append(ps); rec[k]["ba"].append(ba)
    if (i + 1) % 6 == 0: print(f"  [{i+1}/{N}]", flush=True)

out = {"alpha": ALPHA, "n": N, "tau": tau, "rows": []}
print(f"\n=== FIDELITY-CONSTRAINED EVAL v2 (n={N}, alpha={ALPHA}, tau={tau:.3f}) ===")
print(f"{'attack':9s} {'lvl':6s} | {'SSIM':>6s} {'PSNR':>6s} | {'det':>5s} {'ba':>5s}")
for (atk, lab), v in rec.items():
    row = {"attack": atk, "level": lab, "ssim": float(np.mean(v["ssim"])), "psnr": float(np.mean(v["psnr"])),
           "det": float(np.mean(v["det"])), "ba": float(np.mean(v["ba"]))}
    out["rows"].append(row)
    print(f"{atk:9s} {lab:6s} | {row['ssim']:6.3f} {row['psnr']:6.2f} | {row['det']:5.2f} {row['ba']:5.2f}")

print("\n=== detection within an attack-quality budget (photometric+regen; rotation excluded as pixel-SSIM understates it) ===")
for bud in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
    passing = [r for r in out["rows"] if r["ssim"] >= bud and r["attack"] != "rotate"]
    det = np.mean([r["det"] for r in passing]) if passing else float("nan")
    print(f"  attacked SSIM >= {bud:.2f} : {len(passing):2d} levels, mean detection = {det:.3f}")
json.dump(out, open(os.path.join(REPO, "results/defense/fidelity_constrained.json"), "w"), indent=2)
print("FIDELITY_DONE")
