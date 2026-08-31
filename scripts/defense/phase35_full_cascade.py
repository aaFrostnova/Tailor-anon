"""PHASE 3 (geo/regen strength sweeps to failure) + PHASE 5 (2-vs-3 fragment ablation) + Phase 4 data.

Configs (VINE ALWAYS nested{1,.75,.5} + dense scale search; fusion = equal-weight per gate G2):
  A = vine
  B = vine + trustmark
  C = vine + videoseal
  D = vine + trustmark + videoseal   (deployed 3-fragment)
For each config, sweep each attack's strength until detection dies; record detection curve + breakpoint
(the strength at which detection first drops below 0.5). Also PSNR/SSIM per config.
Attacks here: rotation, centered crop-zoom, random regen. (CtrlRegen+/UnMarker are separate-env jobs.)"""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
from _regen_util import build_regen_pipe, stable_regen
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scl(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def txc(iid, nm):
    p, M = F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def vine_nested(x, iid):
    x = to512(x)
    for K in (1.0, 0.75, 0.5):
        if K >= 0.999: x = to512(F["vine"].embed_with_target(x, txc(iid, "vine")))
        else:
            s = int(512*K); o = (512-s)//2
            m = F["vine"].embed_with_target(x.crop((o, o, o+s, o+s)), txc(iid, "vine"))
            out = x.copy(); out.paste(m.resize((s, s)), (o, o)); x = out
    return x
def embed_cfg(orig, iid, frags):
    x = to512(orig)
    for nm in frags:
        if nm == "vine": x = vine_nested(x, iid)
        else: x = scl(x, to512(F[nm].embed_with_target(x, txc(iid, nm))))
    return to512(sync_embed(sync, to512(x), dev))
def llr(nm, pil, iid):
    kind, getter = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], getter)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def rot(img, deg):
    a = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(a, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0]-512)//2; return big.crop((l, l, l+512, l+512))
VS = np.round(np.arange(0.34, 1.0001, 0.03), 3).tolist()
def vine_search(pil, iid, tx):
    P = to512(pil)
    for v in VS:
        s = int(512*v); o = (512-s)//2; view = P if v >= 0.999 else P.crop((o, o, o+s, o+s))
        rl = llr("vine", view, iid)
        if cv(rl, iid) or ((rl > 0).astype(np.uint8) == tx).mean() >= tau: return True
    return False
def geo_cascade_full(att, iid, tx, frags):
    """FAITHFUL port of the deployed geo_cascade (composite_external_eval.py:175-206):
       A) SyncSeal one-shot rectify -> every fragment crypto-verify
       B) local +-3/6 deg refine on the rectified frame (TM carrier)
       C) blind coarse (+-30, step 10, TM-guided) with early-abort gate 0.60 -> fine (+-10, step 3, TM+VS)
       plus our new VINE dense-scale stage for the nested layers."""
    rect, _ = sync_rectify(sync, att, dev)                                  # A
    for nm in frags:
        if cv(llr(nm, rect, iid), iid): return 1.0
    if "vine" in frags and vine_search(att, iid, tx): return 1.0            # new crop stage (nested layers)
    if "trustmark" in frags:                                                # B
        for d in (-3.0, 3.0, -6.0, 6.0):
            if cv(llr("trustmark", rot(rect, d), iid), iid): return 1.0
    best_d, best_ba = 0.0, -1.0
    if "trustmark" in frags:                                                # C coarse + gate
        for d in np.arange(-30, 30.01, 10.0):
            rl = llr("trustmark", rot(att, float(d)), iid)
            if cv(rl, iid): return 1.0
            ba = float(((rl > 0).astype(np.uint8) == tx).mean())
            if ba > best_ba: best_ba, best_d = ba, d
        if best_ba < 0.60: return 0.0                                       # early-abort: info loss, not misalignment
    search = [nm for nm in ("trustmark", "videoseal") if nm in frags]       # C fine
    for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
        rimg = rot(att, float(d))
        for nm in search:
            if cv(llr(nm, rimg, iid), iid): return 1.0
    return 0.0

def detect(att, iid, tx, frags):
    al = {nm: llr(nm, att, iid) for nm in frags}
    fused = fuse_llrs(al, weights=None, n_codeword=n) if len(al) > 1 else next(iter(al.values()))
    if cv(fused, iid) or ((fused > 0).astype(np.uint8) == tx).mean() >= tau: return 1.0
    if any(cv(al[nm], iid) for nm in frags): return 1.0
    return geo_cascade_full(att, iid, tx, frags)

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:30]
pipe = build_regen_pipe()
def a_crop(x, c): s = int(512*c); o = (512-s)//2; return to512(x).crop((o, o, o+s, o+s)).resize((512, 512))
GRID = {"rotate": [3, 9, 15, 22, 30, 45, 60, 90], "crop_zoom": [0.9, 0.8, 0.75, 0.65, 0.6, 0.5, 0.45, 0.4],
        "regen": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
CFG = {"A_vine": ["vine"], "B_vine_tm": ["vine", "trustmark"], "C_vine_vs": ["vine", "videoseal"],
       "D_3frag": ["vine", "trustmark", "videoseal"]}
out = {}
for cname, frags in CFG.items():
    wms = []; ps = []; ss = []
    for j, fp in enumerate(srcs):
        iid = f"p35_{cname}_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed_cfg(orig, iid, frags); wms.append((iid, wm, tx, j))
        O = np.asarray(orig, np.float64); W = np.asarray(wm, np.float64)
        ps.append(_psnr(O, W, data_range=255)); ss.append(_ssim(O, W, channel_axis=2, data_range=255))
    curves = {}
    for atk, grid in GRID.items():
        curves[atk] = {"strength": grid, "detect": []}
        for s in grid:
            dets = []
            for iid, wm, tx, j in wms:
                if atk == "rotate": att = rot(wm, s)
                elif atk == "crop_zoom": att = a_crop(wm, s)
                else: att = to512(stable_regen(pipe, wm, 7000+j, noise_step=int(s*100)))
                dets.append(detect(att, iid, tx, frags))
            curves[atk]["detect"].append(round(float(np.mean(dets)), 3))
        print(f"{cname} {atk}: " + " ".join(f"{g}:{d:.2f}" for g, d in zip(grid, curves[atk]["detect"])), flush=True)
    out[cname] = {"frags": frags, "psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4), "curves": curves}
json.dump(out, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/phase35_full_cascade.json", "w"), indent=2)
print("PHASE35_SWEEP_DONE")
