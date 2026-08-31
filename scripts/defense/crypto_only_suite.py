"""Re-decode the full attack suite under PURE crypto-verify (NO zerobit anywhere -> empirical 0% FPR).
Reuses the saved n=50 3-ring embeds (final_embed_n50). Reports clean / rotation-sweep / crop-staircase.
Regen + UnMarker crypto numbers come from regen_fpr_safe.json (crypto_only column). This makes every
reported detection backed by a 2^-37 cryptographic check (union <= 2^-30 across the geometric search)."""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0, "scripts/defense")
import torch
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_rectify
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"; EMB = f"{SC}/final_embed_n50"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
FR = ["vine", "trustmark", "videoseal"]; sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def llr(nm, pil, iid):
    kind, g = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], g)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def view_at(P, f):
    if f >= 0.999: return P
    s = int(round(512 * float(f))); o = (512 - s) // 2; return P.crop((o, o, o + s, o + s))
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VS = np.arange(0.34, 1.0001, 0.005)
def detect_crypto(att, iid):
    """CRYPTO-VERIFY ONLY: no zerobit anywhere. primary fused/bestpath -> syncseal -> refine -> vine_scale -> rot."""
    P = to512(att)
    al = {nm: llr(nm, P, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n)
    if cv(fused, iid) or any(cv(al[nm], iid) for nm in FR): return 1.0
    rect, _ = sync_rectify(sync, P, dev)
    if any(cv(llr(nm, rect, iid), iid) for nm in FR): return 1.0
    for d in (-3.0, 3.0, -6.0, 6.0):
        if cv(llr("trustmark", rot(rect, d), iid), iid): return 1.0
    for f in VS:
        if cv(llr("vine", view_at(P, float(f)), iid), iid): return 1.0
    best_d, best_ba = 0.0, -1.0
    for d in np.arange(-180, 180.01, 10.0):
        rl = llr("trustmark", rot(P, float(d)), iid)
        if cv(rl, iid): return 1.0
        ba = float(((rl > 0).astype(np.uint8) == tx_of(iid)).mean())
        if ba > best_ba: best_ba, best_d = ba, d
    if best_ba < 0.60: return 0.0
    for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
        for nm in ("trustmark", "videoseal"):
            if cv(llr(nm, rot(P, float(d)), iid), iid): return 1.0
    return 0.0
_txcache = {}
def tx_of(iid):
    if iid not in _txcache: _txcache[iid] = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    return _txcache[iid]
meta = json.load(open(f"{EMB}/meta.json"))["items"]
wms = [(m["iid"], to512(Image.open(f"{EMB}/img_{m['i']:05d}.png").convert("RGB"))) for m in meta
       if os.path.exists(f"{EMB}/img_{m['i']:05d}.png")]
print(f"[loaded] {len(wms)} embeds", flush=True)
def crop(im, c):
    s = int(512 * c); o = (512 - s) // 2; return im.crop((o, o, o + s, o + s)).resize((512, 512))
report = {"rule": "crypto-verify only (no zerobit; empirical FPR 0%)", "n": len(wms)}
report["clean"] = round(float(np.mean([detect_crypto(wm, iid) for iid, wm in wms])), 3)
print(f"clean: {report['clean']}", flush=True)
report["rotation"] = {}
for a in [15, 30, 45, 60, 90, 120, 150, 180]:
    report["rotation"][a] = round(float(np.mean([detect_crypto(rot(wm, a), iid) for iid, wm in wms])), 3)
    print(f"  rot{a:3d}: {report['rotation'][a]:.3f}", flush=True)
report["crop"] = {}
for c in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]:
    report["crop"][c] = round(float(np.mean([detect_crypto(crop(wm, c), iid) for iid, wm in wms])), 3)
    print(f"  crop{int(c*100)}: {report['crop'][c]:.3f}", flush=True)
json.dump(report, open(f"{OUT}/crypto_only_suite.json", "w"), indent=2)
print("CRYPTO_ONLY_SUITE_DONE", flush=True)
