"""Refined test of the user's idea: VINE embedded in the CENTRAL keep% region + a
DECODE-SIDE crop/scale SEARCH (crypto-verify gated, like geo_cascade) that finds the
sub-window where the canonical template lives. Isolates the mark region BEFORE decoding.

The key questions my earlier (whole-image-decode) test could NOT answer:
  (a) does crop-first-search decode RESTORE clean detection? (prev 0.00 = decode artifact?)
  (b) does central-VINE survive REGEN under the correct crop decode? (its crown jewel)
  (c) does it break the crop75+REGEN COMPOUND corner? (the actual goal)
Compares full_vine vs central75_vine, both with the search decode for fairness."""
import os, sys, glob, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

def embed_full(orig, iid):
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); p, M = vine.get_perm_M(iid)
    return to512(vine.embed_with_target(orig, apply_crypto(tx, p, M))), tx.astype(np.uint8)

def embed_central(orig, iid, keep=0.75):
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); p, M = vine.get_perm_M(iid)
    s = int(512 * keep); o = (512 - s) // 2
    center = orig.crop((o, o, o + s, o + s))
    marked = vine.embed_with_target(center, apply_crypto(tx, p, M))
    out = orig.copy(); out.paste(marked.resize((s, s)), (o, o))
    return out, tx.astype(np.uint8)

def _cv_at(pil, iid, tx):
    """crypto-verify + ba on a single (already-cropped) view."""
    p, M = vine.get_perm_M(iid)
    rl = np.clip(method_soft_to_codeword_llr(vine.raw_probs(pil), p, M, kind="prob", n_codeword=n), -CLAMP, CLAMP)
    det = bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or (((rl > 0).astype(np.uint8) == tx).mean() >= tau)
    return det, float(((rl > 0).astype(np.uint8) == tx).mean())

# DECODE-SIDE crop/scale search: try centered crops at several keep ratios; accept if any crypto-verifies.
SEARCH_KEEPS = [1.0, 0.9375, 0.875, 0.8, 0.75, 0.70]
def decode_search(pil, iid, tx):
    best_ba = -1.0
    for k in SEARCH_KEEPS:
        if k >= 0.999:
            view = pil
        else:
            s = int(512 * k); o = (512 - s) // 2; view = pil.crop((o, o, o + s, o + s))
        det, ba = _cv_at(view, iid, tx)
        if det:
            return 1.0, ba
        best_ba = max(best_ba, ba)
    return 0.0, best_ba

def a_crop(x, keep):
    s = int(512 * keep); o = (512 - s) // 2; return x.crop((o, o, o + s, o + s)).resize((512, 512))

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:30]
pipe = build_regen_pipe()

def regen(x, j): return to512(stable_regen(pipe, x, 7000 + j, noise_step=30))
# attack builders: fn(wm, j) -> attacked PIL
ATT = {
    "clean":        lambda x, j: x,
    "crop75":       lambda x, j: a_crop(x, 0.75),
    "crop80":       lambda x, j: a_crop(x, 0.80),
    "crop60":       lambda x, j: a_crop(x, 0.60),
    "regen0.3":     lambda x, j: regen(x, j),
    "crop75+regen": lambda x, j: regen(a_crop(x, 0.75), j),   # COMPOUND: crop then regen
}
res = {}
for scheme, embed in [("full_vine", embed_full), ("central75_vine", lambda o, i: embed_central(o, i, 0.75))]:
    acc = {k: [] for k in ATT}; ba = {k: [] for k in ATT}
    for j, fp in enumerate(srcs):
        iid = f"cvs_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        wm, tx = embed(orig, iid)
        for k, f in ATT.items():
            d, b = decode_search(f(wm, j), iid, tx)   # SAME search decode for both schemes (fair)
            acc[k].append(d); ba[k].append(b)
    res[scheme] = {"detect": {k: float(np.mean(acc[k])) for k in ATT},
                   "bit_acc": {k: float(np.mean(ba[k])) for k in ATT}}
    print(f"{scheme} (search-decode): " + "  ".join(f"{k}={np.mean(acc[k]):.2f}(ba{np.mean(ba[k]):.2f})" for k in ATT), flush=True)
json.dump({"search_keeps": SEARCH_KEEPS, **res},
          open("/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks/central_vine_search.json", "w"), indent=2)
print("CENTRAL_VINE_SEARCH_DONE")
