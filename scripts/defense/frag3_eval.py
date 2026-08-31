"""Does adding MaskWM-ED as a 3rd fragment improve the COMBINED detection/bit-acc?
(Answering the right question: fragment value = lift to the fused result, not whether
ED individually beats VINE/TM.)

Embed VINE+TM+ED composite (alpha=0.70) once per image, apply the full attack suite,
decode all 3 fragments, and compare per attack:
  2-frag (VINE+TM)  vs  3-frag (VINE+TM+ED),  each under equal-MRC AND reliability-
  weighted (conf) fusion (conf down-weights a dead fragment so it can't poison),
  plus oracle3 (upper bound) and per-fragment det.  Reports detection + mean bit-acc.

ED LLR = K*(raw_score-0.5), K=5.5 (calibrated by maskwm_ed_probe).
"""
import os, sys, glob, io, json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.maskwm_wrapper import MaskWMWrapper
from src.soft_fusion import method_soft_to_codeword_llr
from scipy.stats import binom

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0; K_ED = 5.5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
ed = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/ED_128bits.pth"),
                   master_key=KEY, method_name="maskwm_ed", n_bits=sb.n, msg_len=128,
                   jnd_factor=1.75, device=dev, use_self_mask=False, config_name="ED_128bits")
# regen attacks intentionally OMITTED here: the ED-regen check + CtrlRegen+ sweep already
# show ED (pixel-space) DIES under regen like TrustMark, so it can't add regen value.
# This suite targets where ED CAN contribute independent signal: signal-proc + geometric.

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def psnr(a, b):
    m = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if m < 1e-9 else 10 * np.log10(255 * 255 / m)

# ---- attack suite (fixed strengths for a readable table) ----
def a_jpeg(im, q): b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def a_blur(im, r): return im.filter(ImageFilter.GaussianBlur(radius=r))
def a_noise(im, s):
    x = np.asarray(im, np.float64) + np.random.RandomState(7).normal(0, s * 255, (512, 512, 3))
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
def a_bright(im, f): return ImageEnhance.Brightness(im).enhance(f)
def a_contrast(im, f): return ImageEnhance.Contrast(im).enhance(f)
def a_crop(im, area):
    s = int(round(512 * np.sqrt(area))); off = (512 - s) // 2
    return im.crop((off, off, off + s, off + s)).resize((512, 512))
def a_rot(im, deg): return im.rotate(deg, resample=Image.BILINEAR, expand=False)
ATTACKS = [("clean", lambda im: im), ("jpeg50", lambda im: a_jpeg(im, 50)), ("jpeg25", lambda im: a_jpeg(im, 25)),
           ("blur2", lambda im: a_blur(im, 2.0)), ("noise.05", lambda im: a_noise(im, 0.05)),
           ("bright1.4", lambda im: a_bright(im, 1.4)), ("contr1.4", lambda im: a_contrast(im, 1.4)),
           ("crop0.90", lambda im: a_crop(im, 0.90)), ("crop0.85", lambda im: a_crop(im, 0.85)),
           ("rot5", lambda im: a_rot(im, 5)), ("rot10", lambda im: a_rot(im, 10))]

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:180 + N]

# accumulate aligned LLRs: per (attack) -> lists of (av, at, ae, tx, iid)
store = {nm: {"av": [], "at": [], "ae": [], "tx": [], "id": []} for nm, _ in ATTACKS}
psnr3 = []; psnr2 = []
for j, fp in enumerate(imgs):
    iid = f"sw_{180 + j:05d}"
    orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid); pe, Me = ed.get_perm_M(iid)
    v = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
    vt = scale_resid(v, to512(tm.embed_with_target(v, apply_crypto(cw, pt, Mt))), ALPHA)
    vte = scale_resid(vt, to512(ed.embed_with_target(vt, apply_crypto(cw, pe, Me))), ALPHA)
    psnr2.append(psnr(orig, vt)); psnr3.append(psnr(orig, vte))
    for nm, fn in ATTACKS:
        att = to512(fn(vte))   # ATTACK THE 3-FRAG IMAGE (2-frag decode just ignores ED)
        av = np.clip(method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
        at = np.clip(method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
        ae = np.clip(method_soft_to_codeword_llr(K_ED * (ed.raw_scores(att) - 0.5), pe, Me, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
        store[nm]["av"].append(av); store[nm]["at"].append(at); store[nm]["ae"].append(ae)
        store[nm]["tx"].append(cw.astype(np.uint8)); store[nm]["id"].append(iid)
    if (j + 1) % 10 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

# ---- detectors ----
e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def conf_w(*frags):  # reliability = own mean|LLR|
    return sum(np.abs(f).mean(1, keepdims=True) * norm(f) for f in frags)
def oracle_w(tx, *frags):
    f = 0
    for fr in frags:
        b = ((fr > 0).astype(np.uint8) == tx).mean(1)[:, None]; f = f + np.clip(2 * b - 1, 0, None) * norm(fr)
    return f
def metr(F, tx, ids):
    dets = []
    for i in range(len(F)):
        ver = bool(decode_and_verify(F[i], ids[i], codec=sb)["detected"])
        dets.append(1.0 if (ver or ((F[i] > 0).astype(np.uint8) == tx[i]).mean() >= TAU) else 0.0)
    return float(np.mean(dets)), float(((F > 0).astype(np.uint8) == tx).mean())

print(f"\nn={N}  tau={TAU:.3f}  PSNR: 2-frag={np.mean(psnr2):.1f}dB  3-frag={np.mean(psnr3):.1f}dB (cost {np.mean(psnr2)-np.mean(psnr3):.1f}dB)\n")
hdr = (f"{'attack':10s} | {'V':>4s} {'T':>4s} {'E':>4s} det || {'2f-eq':>9s} {'2f-cf':>9s} | "
       f"{'3f-eq':>9s} {'3f-cf':>9s} | {'orc3':>9s}   (det/ba)")
print(hdr); print("-" * len(hdr))
rows = []
for nm, _ in ATTACKS:
    s = store[nm]; av = np.array(s["av"]); at = np.array(s["at"]); ae = np.array(s["ae"]); tx = np.array(s["tx"]); ids = s["id"]
    dV = metr(av, tx, ids)[0]; dT = metr(at, tx, ids)[0]; dE = metr(ae, tx, ids)[0]
    f2e = metr(av + at, tx, ids); f2c = metr(conf_w(av, at), tx, ids)
    f3e = metr(av + at + ae, tx, ids); f3c = metr(conf_w(av, at, ae), tx, ids); o3 = metr(oracle_w(tx, av, at, ae), tx, ids)
    def c(x): return f"{x[0]:.2f}/{x[1]:.2f}"
    print(f"{nm:10s} | {dV:4.2f} {dT:4.2f} {dE:4.2f}     || {c(f2e):>9s} {c(f2c):>9s} | {c(f3e):>9s} {c(f3c):>9s} | {c(o3):>9s}")
    rows.append({"attack": nm, "detV": dV, "detT": dT, "detE": dE,
                 "f2_eq": {"det": f2e[0], "ba": f2e[1]}, "f2_conf": {"det": f2c[0], "ba": f2c[1]},
                 "f3_eq": {"det": f3e[0], "ba": f3e[1]}, "f3_conf": {"det": f3c[0], "ba": f3c[1]},
                 "oracle3": {"det": o3[0], "ba": o3[1]}})
json.dump({"n": N, "psnr2": float(np.mean(psnr2)), "psnr3": float(np.mean(psnr3)), "rows": rows},
          open(os.path.join(REPO, "results/defense/frag3_eval.json"), "w"), indent=2)
print("\nFRAG3_DONE")
