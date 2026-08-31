"""PHASE 3 advanced attacks (CtrlRegen+ / UnMarker) on the NEW nested-VINE composite.
--mode embed  --config {D,A}: embed n images (nested-VINE [+TM+VideoSeal+sync]) -> PNGs + meta.json
--mode decode --config {D,A} --embed_dir --attacked_dir --out: nested-VINE search + geo_cascade detect.
Hypothesis to test on UnMarker: its bundled 0.9 crop is now recoverable by the nested layers + scale
search, so nested-VINE may survive the crop+spectral compound that killed full-VINE (0.02)."""
import os, sys, glob, json, argparse, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
CFG = {"D": ["vine", "trustmark", "videoseal"], "A": ["vine"]}
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--out_dir", default=""); ap.add_argument("--embed_dir", default=""); ap.add_argument("--attacked_dir", default="")
ap.add_argument("--out", default=""); ap.add_argument("--n", type=int, default=25)
ap.add_argument("--src", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
ap.add_argument("--glob", default="*.jpg"); ap.add_argument("--start", type=int, default=6000)
a = ap.parse_args(); frags = CFG[a.config]
F = {nm: (VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev) if nm == "vine" else
          TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev) if nm == "trustmark" else
          VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)) for nm in frags}
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
def embed(orig, iid):
    x = to512(orig)
    for nm in frags: x = vine_nested(x, iid) if nm == "vine" else scl(x, to512(F[nm].embed_with_target(x, txc(iid, nm))))
    return to512(sync_embed(sync, to512(x), dev))
def llr(nm, pil, iid):
    kind, getter = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], getter)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0]-512)//2; return big.crop((l, l, l+512, l+512))
VS = np.round(np.arange(0.34, 1.0001, 0.03), 3).tolist()
def vine_search(pil, iid, tx):
    P = to512(pil)
    for v in VS:
        s = int(512*v); o = (512-s)//2; view = P if v >= 0.999 else P.crop((o, o, o+s, o+s))
        rl = llr("vine", view, iid)
        if cv(rl, iid) or ((rl > 0).astype(np.uint8) == tx).mean() >= tau: return True
    return False
def detect(att, iid, tx):
    al = {nm: llr(nm, att, iid) for nm in frags}
    fused = fuse_llrs(al, weights=None, n_codeword=n) if len(al) > 1 else next(iter(al.values()))
    if cv(fused, iid) or ((fused > 0).astype(np.uint8) == tx).mean() >= tau: return 1.0
    if any(cv(al[nm], iid) for nm in frags): return 1.0
    if "vine" in frags and vine_search(att, iid, tx): return 1.0
    rect, _ = sync_rectify(sync, att, dev)
    if any(cv(llr(nm, rect, iid), iid) for nm in frags): return 1.0
    if "trustmark" in frags and any(cv(llr("trustmark", rot(att, d), iid), iid) for d in (-30,-20,-10,10,20,30)): return 1.0
    return 0.0

if a.mode == "embed":
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, a.glob)))[a.start:a.start + a.n]
    meta = []
    for i, fp in enumerate(files):
        iid = f"p3_{a.config}_{i:05d}"; wm = embed(to512(Image.open(fp).convert("RGB")), iid)
        fn = f"img_{i:05d}.png"; wm.save(os.path.join(a.out_dir, fn)); meta.append({"i": i, "iid": iid, "fname": fn})
        if (i+1) % 8 == 0: print(f"embed {i+1}/{len(files)}", flush=True)
    json.dump({"config": a.config, "frags": frags, "items": meta}, open(os.path.join(a.out_dir, "meta.json"), "w"))
    print("EMBED_DONE", a.out_dir, len(meta))
else:
    meta = json.load(open(os.path.join(a.embed_dir, "meta.json")))["items"]
    dets = []
    for m in meta:
        ap_ = os.path.join(a.attacked_dir, m["fname"])
        if not os.path.exists(ap_): continue
        iid = m["iid"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        dets.append(detect(to512(Image.open(ap_).convert("RGB")), iid, tx))
    summ = {"composite_or": round(float(np.mean(dets)), 4) if dets else None, "n": len(dets)}
    json.dump({"config": a.config, "attacked_dir": a.attacked_dir, "summary": summ}, open(a.out, "w"), indent=2)
    print(f"COMPOSITE_or={summ['composite_or']} n={summ['n']} {a.attacked_dir}")
