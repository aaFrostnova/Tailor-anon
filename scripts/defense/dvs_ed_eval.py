"""Which 3rd fragment integrates better: MaskWM-D (global, jnd1.3) or MaskWM-ED
(mask-adaptive, jnd1.75)?  Compare 2-frag (VINE+TM) vs +D vs +ED on the COMBINED
det/ba across an attack suite that INCLUDES regen (the decisive regime: both D & ED
are pixel-space and die, so the question is whether a dead 3rd fragment POISONS the
equal-MRC sum -- which depends on whether its dead LLRs are confidently-wrong &
high-magnitude or degrade gracefully toward 0).

Reports per attack: 2f / +D / +ED under equal-MRC AND reliability-weighted (conf)
fusion; PSNR cost of each 3rd fragment; and the dead-fragment |LLR| of D vs ED under
regen (explains any poisoning). LLR scale k calibrated per model from clean scores.
Runs on the SLURM-allocated GPU (CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS).
"""
import os, sys, glob, io, json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, undo_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.maskwm_wrapper import MaskWMWrapper
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen
from scipy.stats import binom

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
d_wm = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/D_128bits.pth"),
                     master_key=KEY, method_name="maskwm_d", n_bits=sb.n, msg_len=128,
                     jnd_factor=1.3, device=dev, use_self_mask=False, config_name="D_128bits")
ed_wm = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/ED_128bits.pth"),
                      master_key=KEY, method_name="maskwm_ed", n_bits=sb.n, msg_len=128,
                      jnd_factor=1.75, device=dev, use_self_mask=False, config_name="ED_128bits")
pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def psnr(a, b):
    m = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if m < 1e-9 else 10 * np.log10(255 * 255 / m)

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:180 + N]

# ---- calibrate LLR scale k per MaskWM model from clean scores ----
def calib_k(wm):
    s1 = []
    for fp in imgs[:5]:
        iid = "cal_" + os.path.basename(fp); orig = to512(Image.open(fp).convert("RGB"))
        cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); perm, M = wm.get_perm_M(iid)
        w = scale_resid(orig, to512(wm.embed_with_target(orig, apply_crypto(cw, perm, M))), ALPHA)
        sc = wm.raw_scores(w); s1 += sc[np.asarray(apply_crypto(cw, perm, M)) > 0.5].tolist()
    m1 = float(np.mean(s1)); return 3.0 / max(abs(m1 - 0.5), 1e-3), m1
K_D, m1d = calib_k(d_wm); K_ED, m1e = calib_k(ed_wm)
print(f"calib: D k={K_D:.1f} (mean1 {m1d:.3f})  ED k={K_ED:.1f} (mean1 {m1e:.3f})", flush=True)

# ---- attacks (include regen: the decisive poisoning regime) ----
def a_jpeg(im, q): b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def a_crop(im, area):
    s = int(round(512 * np.sqrt(area))); o = (512 - s) // 2; return im.crop((o, o, o + s, o + s)).resize((512, 512))
ATTACKS = [("clean", lambda im: im), ("jpeg25", lambda im: a_jpeg(im, 25)),
           ("blur2", lambda im: im.filter(ImageFilter.GaussianBlur(2.0))),
           ("noise.05", lambda im: Image.fromarray(np.clip(np.asarray(im, np.float64) + np.random.RandomState(7).normal(0, .05 * 255, (512, 512, 3)), 0, 255).astype(np.uint8))),
           ("crop0.85", lambda im: a_crop(im, 0.85)), ("rot5", lambda im: im.rotate(5, resample=Image.BILINEAR)),
           ("regen_x1", lambda im: to512(stable_regen(pipe, im, seed=11))),
           ("regen_x2", lambda im: to512(stable_regen(pipe, to512(stable_regen(pipe, im, seed=11)), seed=12)))]

def edl(wm, K, att, perm, M):
    return np.clip(method_soft_to_codeword_llr(K * (wm.raw_scores(att) - 0.5), perm, M, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)

store = {nm: {"av": [], "at": [], "ad": [], "ae": [], "tx": [], "id": []} for nm, _ in ATTACKS}
ps2, psD, psE = [], [], []
for j, fp in enumerate(imgs):
    iid = f"sw_{180 + j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid); pd, Md = d_wm.get_perm_M(iid); pe, Me = ed_wm.get_perm_M(iid)
    v = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
    vt = scale_resid(v, to512(tm.embed_with_target(v, apply_crypto(cw, pt, Mt))), ALPHA)
    vtd = scale_resid(vt, to512(d_wm.embed_with_target(vt, apply_crypto(cw, pd, Md))), ALPHA)
    vte = scale_resid(vt, to512(ed_wm.embed_with_target(vt, apply_crypto(cw, pe, Me))), ALPHA)
    ps2.append(psnr(orig, vt)); psD.append(psnr(orig, vtd)); psE.append(psnr(orig, vte))
    for nm, fn in ATTACKS:
        aD = to512(fn(vtd)); aE = to512(fn(vte))   # +D suite on vtd, +ED suite on vte
        s = store[nm]
        s["av"].append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(aD), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        s["at"].append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(aD), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
        s["ad"].append(edl(d_wm, K_D, aD, pd, Md))
        # ED branch: recompute V/T on vte for the +ED fusion (kept separately via ae + a second V/T pair)
        s["ae"].append(edl(ed_wm, K_ED, aE, pe, Me))
        s["av2_t2"] = s.get("av2_t2", [])  # placeholder to keep both branches' V/T
        s["tx"].append(cw.astype(np.uint8)); s["id"].append(iid)
        # store V/T on the ED-branch image too (for a fair +ED fusion)
        s.setdefault("avE", []).append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(aE), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        s.setdefault("atE", []).append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(aE), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
    if (j + 1) % 8 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

# ---- detectors ----
e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def conf(*F): return sum(np.abs(f).mean(1, keepdims=True) * norm(f) for f in F)
def metr(F, tx, ids):
    det = []
    for i in range(len(F)):
        ver = bool(decode_and_verify(F[i], ids[i], codec=sb)["detected"])
        det.append(1.0 if (ver or ((F[i] > 0).astype(np.uint8) == tx[i]).mean() >= TAU) else 0.0)
    return float(np.mean(det)), float(((F > 0).astype(np.uint8) == tx).mean())

print(f"\nn={N}  PSNR: 2f={np.mean(ps2):.1f}  +D={np.mean(psD):.1f} (cost {np.mean(ps2)-np.mean(psD):.1f})  +ED={np.mean(psE):.1f} (cost {np.mean(ps2)-np.mean(psE):.1f}) dB\n")
hdr = f"{'attack':9s} | {'2f-eq':>9s} {'2f-cf':>9s} || {'+D-eq':>9s} {'+D-cf':>9s} || {'+ED-eq':>9s} {'+ED-cf':>9s} | {'D|LLR|':>6s} {'E|LLR|':>6s}"
print(hdr); print("-" * len(hdr))
rows = []
for nm, _ in ATTACKS:
    s = store[nm]; av = np.array(s["av"]); at = np.array(s["at"]); ad = np.array(s["ad"])
    avE = np.array(s["avE"]); atE = np.array(s["atE"]); ae = np.array(s["ae"]); tx = np.array(s["tx"]); ids = s["id"]
    f2 = metr(av + at, tx, ids); f2c = metr(conf(av, at), tx, ids)
    fD = metr(av + at + ad, tx, ids); fDc = metr(conf(av, at, ad), tx, ids)
    fE = metr(avE + atE + ae, tx, ids); fEc = metr(conf(avE, atE, ae), tx, ids)
    dmag = float(np.abs(ad).mean()); emag = float(np.abs(ae).mean())
    def c(x): return f"{x[0]:.2f}/{x[1]:.2f}"
    print(f"{nm:9s} | {c(f2):>9s} {c(f2c):>9s} || {c(fD):>9s} {c(fDc):>9s} || {c(fE):>9s} {c(fEc):>9s} | {dmag:6.2f} {emag:6.2f}")
    rows.append({"attack": nm, "f2_eq": f2, "f2_cf": f2c, "D_eq": fD, "D_cf": fDc, "ED_eq": fE, "ED_cf": fEc, "Dmag": dmag, "Emag": emag})
json.dump({"n": N, "k_D": K_D, "k_ED": K_ED, "psnr": {"f2": float(np.mean(ps2)), "D": float(np.mean(psD)), "ED": float(np.mean(psE))},
           "rows": [{"attack": r["attack"], **{k: {"det": v[0], "ba": v[1]} for k, v in r.items() if k.endswith(("_eq", "_cf"))},
                     "Dmag": r["Dmag"], "Emag": r["Emag"]} for r in rows]},
          open(os.path.join(REPO, "results/defense/dvs_ed_eval.json"), "w"), indent=2)
print("DVSED_DONE")
