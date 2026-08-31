"""Fidelity-constrained baseline comparison: OUR composite vs single fragments vs classical.

Methods:
  VINE-frag   : VINE only (our crypto fragment @alpha0.7, carries full BCH codeword -> dual-detect + resync)
  TM-frag     : TrustMark only (same)
  Composite   : VINE+TrustMark fused (ours, dual-detect + resync)
  DwtDct / DwtDctSvd / RivaGAN : classical baselines (native, zero-bit detect, no crypto/resync)

For each method x attack x strength: record attack-induced SSIM, bit-acc, detection.
Crypto methods (VINE/TM/Composite) get resync on rotation; classical do not.
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
from wbench.methods import InvisibleWMMethod, VineMethod, TrustMarkMethod
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; ALPHA = 0.70; N = int(sys.argv[1]) if len(sys.argv) > 1 else 10; dev = "cuda"
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[:N]
sb = ShortenedBCH(); TAU100 = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
def tau_of(n): return float(binom.ppf(0.99, n, 0.5) + 1) / n

vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
regen_pipe = build_regen_pipe()
CLASSICAL = {}
for nm, key in [("DwtDct", "dwtDct"), ("DwtDctSvd", "dwtDctSvd"), ("RivaGAN", "rivaGan")]:
    try: CLASSICAL[nm] = InvisibleWMMethod(key); print(f"loaded {nm}", flush=True)
    except Exception as e: print(f"SKIP {nm}: {str(e)[:80]}", flush=True)
# learned baselines, NATIVE pipeline (no crypto, no resync) — the fair "raw method" comparison vs our Composite
for nm, ctor in [("VINE-B", lambda: VineMethod("B")), ("VINE-R", lambda: VineMethod("R")), ("TrustMark-Q", lambda: TrustMarkMethod("Q"))]:
    try: CLASSICAL[nm] = ctor(); print(f"loaded {nm}", flush=True)
    except Exception as e: print(f"SKIP {nm}: {str(e)[:80]}", flush=True)

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))

# ---- attacks ----
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

# (attack, level_label, fn, kind) ; kind in {plain, rotate}
ATTACKS = []
for r in [0.3, 0.6, 1.0]:
    ATTACKS += [("jpeg", "%.1f" % r, (lambda im, rr=r: att_jpeg(im, rr)), "plain"),
                ("blur", "%.1f" % r, (lambda im, rr=r: att_blur(im, rr)), "plain"),
                ("noise", "%.1f" % r, (lambda im, rr=r: att_noise(im, rr)), "plain"),
                ("bright", "%.1f" % r, (lambda im, rr=r: att_bright(im, rr)), "plain"),
                ("contrast", "%.1f" % r, (lambda im, rr=r: att_contrast(im, rr)), "plain"),
                ("crop", "%.1f" % r, (lambda im, rr=r: att_crop(im, rr)), "plain")]
ATTACKS += [("rotate", "%ddeg" % a, (lambda im, aa=a: rot(im, aa)), "rotate") for a in [12, 24]]
ATTACKS += [("regen", "x%d" % k, (lambda im, kk=k: regen_rounds(im, kk)), "plain") for k in [1, 2]]
ANGLES = np.arange(-30, 30.01, 3.0)

def ssim(a, b): return float(ssim_fn(np.asarray(a, np.float64), np.asarray(b, np.float64), channel_axis=2, data_range=255))

# crypto-method detection from an aligned-LLR fn (handles resync)
def crypto_detect(llr_fn, att, tx, iid, kind):
    llr = llr_fn(att); ba = float(np.mean(llr_to_bits(llr) == tx))
    det = bool(decode_and_verify(llr, iid, codec=sb)["detected"]) or ba >= TAU100
    if not det and kind == "rotate":
        for d in ANGLES:
            l2 = llr_fn(rot(att, d))
            if bool(decode_and_verify(l2, iid, codec=sb)["detected"]):
                return 1.0, float(np.mean(llr_to_bits(l2) == tx))
    return (1.0 if det else 0.0), ba

rows = []
for i, fp in enumerate(imgs):
    iid = f"bl_{i:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    # crypto fragment watermarked images
    v_only = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(tx, pv, Mv))), ALPHA)
    comp = scale_resid(v_only, to512(tm.embed_with_target(v_only, apply_crypto(tx, pt, Mt))), ALPHA)
    t_only = scale_resid(orig, to512(tm.embed_with_target(orig, apply_crypto(tx, pt, Mt))), ALPHA)
    # classical watermarked images (native, random bits)
    cwm = {}; cbits = {}
    for nm, mobj in CLASSICAL.items():
        b = np.random.RandomState(1000 + i).randint(0, 2, mobj.n_bits).astype(np.uint8)
        cbits[nm] = b; cwm[nm] = to512(mobj.embed(orig, b))
    llr_v = lambda att: method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
    llr_t = lambda att: method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n)
    llr_c = lambda att: fuse_llrs({"v": llr_v(att), "t": llr_t(att)}, n_codeword=sb.n)

    for atk, lab, fn, kind in ATTACKS:
        # crypto methods
        a_v = to512(fn(v_only)); a_t = to512(fn(t_only)); a_c = to512(fn(comp))
        for nm, src, wm, lf in [("VINE-frag", a_v, v_only, llr_v), ("TM-frag", a_t, t_only, llr_t), ("Composite", a_c, comp, llr_c)]:
            det, ba = crypto_detect(lf, src, tx, iid, kind)
            rows.append({"method": nm, "attack": atk, "level": lab, "ssim": ssim(np.asarray(wm, np.float64), src), "ba": ba, "det": det})
        # classical methods
        for nm, mobj in CLASSICAL.items():
            att = to512(fn(cwm[nm])); rec = mobj.decode(att)
            nb = min(len(rec), len(cbits[nm])); ba = float(np.mean(rec[:nb] == cbits[nm][:nb]))
            det = 1.0 if ba >= tau_of(mobj.n_bits) else 0.0
            rows.append({"method": nm, "attack": atk, "level": lab, "ssim": ssim(np.asarray(cwm[nm], np.float64), att), "ba": ba, "det": det})
    print(f"  [{i+1}/{N}] {os.path.basename(fp)} done", flush=True)

out = {"alpha": ALPHA, "n": N, "tau100": TAU100, "rows": rows,
       "methods": ["VINE-frag", "TM-frag", "Composite"] + list(CLASSICAL.keys())}
json.dump(out, open(os.path.join(REPO, "results/defense/baseline_comparison.json"), "w"), indent=2)

# console summary: mean det + ba per method (overall + within SSIM>=0.8)
methods = out["methods"]
print(f"\n=== BASELINE COMPARISON (n={N}, alpha={ALPHA}) ===")
print(f"{'method':12s} | {'det(all)':>8s} {'ba(all)':>8s} | {'det(SSIM>=.8)':>13s} {'ba(SSIM>=.8)':>12s}")
for m in methods:
    rs = [r for r in rows if r["method"] == m]
    rs8 = [r for r in rs if r["ssim"] >= 0.80]
    print(f"{m:12s} | {np.mean([r['det'] for r in rs]):8.3f} {np.mean([r['ba'] for r in rs]):8.3f} | "
          f"{(np.mean([r['det'] for r in rs8]) if rs8 else float('nan')):13.3f} {(np.mean([r['ba'] for r in rs8]) if rs8 else float('nan')):12.3f}")
print("BASELINE_DONE")
