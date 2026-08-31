"""Unified table: for every attack report (1) detection under crypto-verify-only (0% FPR), (2) RAW bit-acc =
fused codeword bit-acc at scale 1.0 (undefended), (3) RECOVERED bit-acc = best fused/fragment bit-acc at the
view the pipeline selects (defended). bit-acc is the continuous signal-survival metric that binary detection
hides — esp. for regeneration, where crypto detection collapses but ~0.6 of the payload still survives.
Geometry/crop use the saved n=50 3-ring embeds; regen/UnMarker reuse the existing attacked sets."""
import os, sys, glob, json, numpy as np
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
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"; EMB = f"{SC}/final_embed_n50"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB  tau={tau:.4f}", flush=True)
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
def ba(rl, tx): return float(((rl > 0).astype(np.uint8) == tx).mean())
def view_at(P, f):
    if f >= 0.999: return P
    s = int(round(512 * float(f))); o = (512 - s) // 2; return P.crop((o, o, o + s, o + s))
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VS = np.arange(0.34, 1.0001, 0.005)
def measure(att, iid, tx):
    """Return (det_crypto, raw_ba, recovered_ba). Stops the search at the first crypto hit (recovered_ba then
    read at that view); if crypto never fires, scans the full search to report the max recoverable bit-acc."""
    P = to512(att)
    al = {nm: llr(nm, P, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n)
    raw = ba(fused, tx)
    rec = max(raw, max(ba(al[nm], tx) for nm in FR))
    if cv(fused, iid) or any(cv(al[nm], iid) for nm in FR): return 1.0, raw, max(rec, 0.90)
    # A syncseal
    rect, _ = sync_rectify(sync, P, dev)
    for nm in FR:
        rl = llr(nm, rect, iid); rec = max(rec, ba(rl, tx))
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    # B refine
    for d in (-3.0, 3.0, -6.0, 6.0):
        rl = llr("trustmark", rot(rect, d), iid); rec = max(rec, ba(rl, tx))
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    # B2 scale search (VINE)
    for f in VS:
        rl = llr("vine", view_at(P, float(f)), iid); rec = max(rec, ba(rl, tx))
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    # C blind rotation (TM)
    best_ba_rot = -1.0; best_d = 0.0
    for d in np.arange(-180, 180.01, 10.0):
        rl = llr("trustmark", rot(P, float(d)), iid); b = ba(rl, tx); rec = max(rec, b)
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
        if b > best_ba_rot: best_ba_rot, best_d = b, d
    if best_ba_rot >= 0.60:
        for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
            for nm in ("trustmark", "videoseal"):
                rl = llr(nm, rot(P, float(d)), iid); rec = max(rec, ba(rl, tx))
                if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    return 0.0, raw, rec

def tx_of(iid): return sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
# geometry/clean: saved n=50 3-ring embeds
meta = json.load(open(f"{EMB}/meta.json"))["items"]
wms = [(m["iid"], to512(Image.open(f"{EMB}/img_{m['i']:05d}.png").convert("RGB"))) for m in meta
       if os.path.exists(f"{EMB}/img_{m['i']:05d}.png")]
def crop(im, c):
    s = int(512 * c); o = (512 - s) // 2; return im.crop((o, o, o + s, o + s)).resize((512, 512))
def run_local(fn, tag):
    D, RAW, REC = [], [], []
    for iid, wm in wms:
        tx = tx_of(iid); d, r, rc = measure(fn(wm), iid, tx); D.append(d); RAW.append(r); REC.append(rc)
    row = {"n": len(wms), "detect_crypto": round(float(np.mean(D)), 3), "raw_ba": round(float(np.mean(RAW)), 3),
           "recovered_ba": round(float(np.mean(REC)), 3)}
    print(f"  {tag:16} det={row['detect_crypto']:.2f}  raw_ba={row['raw_ba']:.3f}  recovered_ba={row['recovered_ba']:.3f}", flush=True)
    return row
def run_set(embed_meta, adir, tag):
    mm = json.load(open(embed_meta))["items"]; D, RAW, REC = [], [], []
    for m in mm:
        p = f"{adir}/{m['fname']}"
        if not os.path.exists(p): continue
        iid = m["iid"]; tx = tx_of(iid)
        d, r, rc = measure(to512(Image.open(p).convert("RGB")), iid, tx); D.append(d); RAW.append(r); REC.append(rc)
    row = {"n": len(D), "detect_crypto": round(float(np.mean(D)), 3), "raw_ba": round(float(np.mean(RAW)), 3),
           "recovered_ba": round(float(np.mean(REC)), 3)}
    print(f"  {tag:16} det={row['detect_crypto']:.2f}  raw_ba={row['raw_ba']:.3f}  recovered_ba={row['recovered_ba']:.3f}", flush=True)
    return row
report = {"rule": "crypto-verify only; bit-acc = fused/best-fragment codeword agreement", "tau": tau}
print("=== clean + geometry (n=50 3-ring embeds) ===", flush=True)
report["clean"] = run_local(lambda w: w, "clean")
for a in [30, 45, 90, 180]:
    report[f"rot{a}"] = run_local(lambda w, a=a: rot(w, a), f"rot{a}")
for c in [0.75, 0.6, 0.5]:
    report[f"crop{int(c*100)}"] = run_local(lambda w, c=c: crop(w, c), f"crop{int(c*100)}")
print("=== regeneration + UnMarker (existing attacked sets) ===", flush=True)
for s in ["s03", "s05", "s07", "s09"]:
    ad = f"{SC}/p3cr_D_{s}"
    if os.path.isdir(ad): report[f"ctrlregen_{s}"] = run_set(f"{SC}/p3embed_D/meta.json", ad, f"ctrlregen_{s}")
for tag, ad in [("default", f"{SC}/p3um_D"), ("strong", f"{SC}/p3um_strong")]:
    if os.path.isdir(ad): report[f"unmarker_{tag}"] = run_set(f"{SC}/p3embed_Dum/meta.json", ad, f"unmarker_{tag}")
json.dump(report, open(f"{OUT}/unified_bitacc.json", "w"), indent=2)
print("UNIFIED_BITACC_DONE", flush=True)
