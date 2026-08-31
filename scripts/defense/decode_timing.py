"""Decode-latency profile of the DEPLOYED pipeline across attacks of increasing strength.
Decode time is dominated by HOW DEEP the crypto-verify cascade must go before a hit:
  primary (3 fwd) < scale-search (<=133 VINE fwd) < rotation-search (<=37 TM fwd + fine).
Times each stage with torch.cuda.synchronize() for true wall-clock; records the carrying stage
and the number of fragment forward passes per image. Uses the deployed 3-ring embeds (p3embed_D)
+ existing regen/unmarker sets + locally-generated classical/geometric attacks."""
import os, sys, glob, json, time, io, numpy as np
from PIL import Image, ImageFilter
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
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
FR = ["vine", "trustmark", "videoseal"]; sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
_ctr = {"nfwd": 0}
def llr(nm, pil, iid):
    _ctr["nfwd"] += 1
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
def detect_timed(att, iid, tx):
    """Return (det, stage, elapsed_ms, n_forwards). Mirrors the DEPLOYED geo_cascade EXACTLY:
    primary -> A syncseal -> B refine -> B2 vine_scale (VERIFY-ONLY, FPR-safe) -> C blind rotation(+gate)."""
    _ctr["nfwd"] = 0; torch.cuda.synchronize(); t0 = time.perf_counter()
    P = to512(att)
    al = {nm: llr(nm, P, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n)
    det, stage = 0.0, "none"
    if cv(fused, iid) or ((fused > 0).astype(np.uint8) == tx).mean() >= tau: det, stage = 1.0, "primary_fused"
    elif any(cv(al[nm], iid) for nm in FR): det, stage = 1.0, "primary_bestpath"
    else:
        hit = False
        rect, _ = sync_rectify(sync, P, dev)                              # A. syncseal one-shot unwarp
        if any(cv(llr(nm, rect, iid), iid) for nm in FR): det, stage, hit = 1.0, "syncseal", True
        if not hit:                                                        # B. local rotation refine on rectified frame
            for d in (-3.0, 3.0, -6.0, 6.0):
                if cv(llr("trustmark", rot(rect, d), iid), iid): det, stage, hit = 1.0, "refine", True; break
        if not hit:                                                        # B2. vine scale search (VERIFY-ONLY -> FPR-safe)
            for f in VS:
                if cv(llr("vine", view_at(P, float(f)), iid), iid): det, stage, hit = 1.0, "vine_scale", True; break
        if not hit:                                                        # C. blind full-circle rotation, TM-guided
            best_d, best_ba = 0.0, -1.0
            for d in np.arange(-180, 180.01, 10.0):
                rl = llr("trustmark", rot(P, float(d)), iid)
                if cv(rl, iid): det, stage, hit = 1.0, "rot_blind", True; break
                ba = float(((rl > 0).astype(np.uint8) == tx).mean())
                if ba > best_ba: best_ba, best_d = ba, d
            if not hit and best_ba < 0.60:
                stage = "gated"                                            # info-loss early-abort (crop/regen)
            elif not hit:
                for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
                    for nm in ("trustmark", "videoseal"):
                        if cv(llr(nm, rot(P, float(d)), iid), iid): det, stage, hit = 1.0, "rot_fine", True; break
                    if hit: break
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000.0
    return det, stage, ms, _ctr["nfwd"]

# attack generators / loaders (base = deployed 3-ring embeds p3embed_D)
meta = json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]
def base_imgs():
    return [(m["iid"], to512(Image.open(f"{SC}/p3embed_D/{m['fname']}").convert("RGB"))) for m in meta
            if os.path.exists(f"{SC}/p3embed_D/{m['fname']}")]
def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def blur(im, s): return im.filter(ImageFilter.GaussianBlur(s))
def noise(im, sd):
    a = np.asarray(im, np.float32) + np.random.RandomState(0).randn(512, 512, 3) * sd * 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def crop(im, c):
    s = int(512 * c); o = (512 - s) // 2; return im.crop((o, o, o + s, o + s)).resize((512, 512))
def load_set(embed_meta, adir):
    """attacked dirs carry no meta of their own -> take iids from the EMBED meta, map to attacked fnames."""
    return [(m["iid"], to512(Image.open(f"{adir}/{m['fname']}").convert("RGB")))
            for m in json.load(open(embed_meta))["items"] if os.path.exists(f"{adir}/{m['fname']}")]

ATTACKS = {
    "clean":        lambda: base_imgs(),
    "jpeg_q40":     lambda: [(i, jpeg(im, 40)) for i, im in base_imgs()],
    "blur_s2":      lambda: [(i, blur(im, 2.0)) for i, im in base_imgs()],
    "noise_.05":    lambda: [(i, noise(im, 0.05)) for i, im in base_imgs()],
    "crop75":       lambda: [(i, crop(im, 0.75)) for i, im in base_imgs()],
    "crop50":       lambda: [(i, crop(im, 0.50)) for i, im in base_imgs()],
    "rot30":        lambda: [(i, rot(im, 30)) for i, im in base_imgs()],
    "rot90":        lambda: [(i, rot(im, 90)) for i, im in base_imgs()],
    "ctrlregen_s05": lambda: load_set(f"{SC}/p3embed_D/meta.json", f"{SC}/p3cr_D_s05") if os.path.isdir(f"{SC}/p3cr_D_s05") else [],
    "ctrlregen_s09": lambda: load_set(f"{SC}/p3embed_D/meta.json", f"{SC}/p3cr_D_s09") if os.path.isdir(f"{SC}/p3cr_D_s09") else [],
    "unmarker":     lambda: load_set(f"{SC}/p3embed_Dum/meta.json", f"{SC}/p3um_D") if os.path.isdir(f"{SC}/p3um_D") else [],
}
# warm up (first GPU call is slow)
for i, im in base_imgs()[:2]:
    tx = sb.encode(image_id_to_payload(i, n_bits=sb.data_bits)).astype(np.uint8); detect_timed(im, i, tx)
rows = {}
for aname, gen in ATTACKS.items():
    items = gen()
    if not items: print(f"  {aname}: (no images)"); continue
    ts, nf, dets, stages = [], [], [], {}
    for iid, im in items:
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        d, stg, ms, nfwd = detect_timed(im, iid, tx)
        ts.append(ms); nf.append(nfwd); dets.append(d); stages[stg] = stages.get(stg, 0) + 1
    ts = np.array(ts); dom = max(stages, key=stages.get)
    rows[aname] = {"n": len(items), "detect": round(float(np.mean(dets)), 3),
                   "ms_mean": round(float(ts.mean()), 1), "ms_median": round(float(np.median(ts)), 1),
                   "ms_p95": round(float(np.percentile(ts, 95)), 1),
                   "fwd_mean": round(float(np.mean(nf)), 1), "dominant_stage": dom, "stages": stages}
    r = rows[aname]
    print(f"  {aname:15} det={r['detect']:.2f}  time(ms) mean={r['ms_mean']:7.1f} median={r['ms_median']:7.1f} "
          f"p95={r['ms_p95']:7.1f}  fwd={r['fwd_mean']:6.1f}  via={dom}", flush=True)
json.dump({"tau": tau, "attacks": rows}, open(f"{OUT}/decode_timing.json", "w"), indent=2)
print("DECODE_TIMING_DONE", flush=True)
