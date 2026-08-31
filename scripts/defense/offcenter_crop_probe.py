"""OFF-CENTER crop robustness of nested concentric VINE embedding.

Theory: a centered layer of side K is contained in EVERY crop window of side c iff c >= (1+K)/2.
  K=0.5 -> survives ANY off-center crop with c >= 0.75
  K=0.75 -> only c >= 0.875
So nesting should buy off-center robustness too -- but ONLY if the decode search covers TRANSLATION,
because the surviving layer is no longer centered in the attacked frame.

Compares two decoders on the same attacked images:
  search_centered : centered crop/scale search only  (what we have today)
  search_xy       : scale x (dx,dy) translation search, crypto-verify gated
Attacks: crop fraction c in {0.9, 0.75, 0.6} x offset in {center, half-max, max(corner)}."""
import os, sys, glob, json, itertools, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_fusion import method_soft_to_codeword_llr
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def txc(iid):
    p, M = vine.get_perm_M(iid)
    return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def embed_at(x, iid, keep):
    x = to512(x)
    if keep >= 0.999: return to512(vine.embed_with_target(x, txc(iid)))
    s = int(512 * keep); o = (512 - s) // 2
    m = vine.embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid))
    out = x.copy(); out.paste(m.resize((s, s)), (o, o)); return out
def embed_scheme(orig, iid, keeps):
    x = to512(orig)
    for k in keeps: x = embed_at(x, iid, k)
    return x
def _try(view, iid, tx):
    rl = np.clip(method_soft_to_codeword_llr(vine.raw_probs(view), *vine.get_perm_M(iid),
                                             kind="prob", n_codeword=n), -CLAMP, CLAMP)
    return bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or (((rl > 0).astype(np.uint8) == tx).mean() >= tau)

SCALES = [1.0, 0.9, 0.8, 0.75, 0.67, 0.6, 0.5]
def search_centered(pil, iid, tx):
    for v in SCALES:
        s = int(512 * v); o = (512 - s) // 2
        if _try(to512(pil) if v >= 0.999 else to512(pil).crop((o, o, o + s, o + s)), iid, tx): return 1.0
    return 0.0
NOFF = 4   # offsets per axis
def search_xy(pil, iid, tx):
    P = to512(pil)
    for v in SCALES:
        s = int(512 * v); mx = 512 - s
        offs = [0] if mx == 0 else [int(round(t)) for t in np.linspace(0, mx, NOFF)]
        for dx, dy in itertools.product(sorted(set(offs)), repeat=2):
            if _try(P.crop((dx, dy, dx + s, dy + s)), iid, tx): return 1.0
    return 0.0

def crop_at(x, c, frac):
    """crop side c*512 at offset frac in [0,1] of the max offset (0=top-left, .5=center, 1=bottom-right)."""
    s = int(512 * c); mx = 512 - s; o = int(round(mx * frac))
    return to512(x).crop((o, o, o + s, o + s)).resize((512, 512))

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:12]
SCHEMES = {"full": [1.0], "nested_75_50": [0.75, 0.5], "nested_1_75_50": [1.0, 0.75, 0.5]}
ATT = {f"c{int(c*100)}_{nm}": (c, f) for c in (0.9, 0.75, 0.6)
       for nm, f in [("center", 0.5), ("half", 0.25), ("corner", 0.0)]}
res = {}
for name, keeps in SCHEMES.items():
    acc = {k: {"centered": [], "xy": []} for k in ATT}
    for j, fp in enumerate(srcs):
        iid = f"oc_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed_scheme(orig, iid, keeps)
        for k, (c, fr) in ATT.items():
            a = crop_at(wm, c, fr)
            acc[k]["centered"].append(search_centered(a, iid, tx))
            acc[k]["xy"].append(search_xy(a, iid, tx))
    res[name] = {k: {"search_centered": round(float(np.mean(v["centered"])), 3),
                     "search_xy": round(float(np.mean(v["xy"])), 3)} for k, v in acc.items()}
    print(f"--- {name} (keeps={keeps}) ---", flush=True)
    for k in ATT:
        r = res[name][k]; print(f"  {k:14} centered-search={r['search_centered']:.2f}   xy-search={r['search_xy']:.2f}", flush=True)
json.dump(res, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/offcenter_crop.json", "w"), indent=2)
print("OFFCENTER_PROBE_DONE")
