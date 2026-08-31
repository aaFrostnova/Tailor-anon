"""Decode ONE attacked dir with the deployed pipeline; report det(crypto) + raw_ba + recovered_ba.
Same measure() as unified_bitacc_suite. Usage: --embed_meta <meta.json> --attacked_dir <dir> --tag <name>"""
import os, sys, json, argparse, numpy as np
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
ap = argparse.ArgumentParser(); ap.add_argument("--embed_meta", required=True); ap.add_argument("--attacked_dir", required=True)
ap.add_argument("--tag", default="set"); a = ap.parse_args()
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
    P = to512(att); al = {nm: llr(nm, P, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n); raw = ba(fused, tx)
    rec = max(raw, max(ba(al[nm], tx) for nm in FR))
    if cv(fused, iid) or any(cv(al[nm], iid) for nm in FR): return 1.0, raw, max(rec, 0.90)
    rect, _ = sync_rectify(sync, P, dev)
    for nm in FR:
        rl = llr(nm, rect, iid); rec = max(rec, ba(rl, tx))
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    for d in (-3.0, 3.0, -6.0, 6.0):
        rl = llr("trustmark", rot(rect, d), iid); rec = max(rec, ba(rl, tx))
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    for f in VS:
        rl = llr("vine", view_at(P, float(f)), iid); rec = max(rec, ba(rl, tx))
        if cv(rl, iid): return 1.0, raw, max(rec, 0.90)
    best_ba_rot, best_d = -1.0, 0.0
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
mm = json.load(open(a.embed_meta))["items"]; D, RAW, REC = [], [], []
for m in mm:
    p = f"{a.attacked_dir}/{m['fname']}"
    if not os.path.exists(p): continue
    iid = m["iid"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    d, r, rc = measure(to512(Image.open(p).convert("RGB")), iid, tx); D.append(d); RAW.append(r); REC.append(rc)
print(f"  {a.tag:16} n={len(D)}  det={np.mean(D):.3f}  raw_ba={np.mean(RAW):.3f}  recovered_ba={np.mean(REC):.3f}", flush=True)
print("BITACC_ONE_DONE", flush=True)
