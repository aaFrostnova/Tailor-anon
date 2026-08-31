"""Embed the 2-fragment composite (VINE+TrustMark @ alpha=0.70) on N FRESH images
(idx>=START, disjoint from fusion-head training fh_0..179) for the full advanced-
attack parameter sweep. Writes OUT/comp/img_%05d.png + meta.json.

  python sweep_embed.py N START OUT
"""
import os, sys, glob, json
import numpy as np
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.payload import image_id_to_payload

KEY = b"v5_key_encoder_master"; ALPHA = 0.70; dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
START = int(sys.argv[2]) if len(sys.argv) > 2 else 180
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(REPO, "results/defense/sweep")
PREFIX = "sw"
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[START:START + N]
sb = ShortenedBCH()
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
os.makedirs(os.path.join(OUT, "comp"), exist_ok=True)
meta = {"alpha": ALPHA, "start": START, "items": []}
for j, fp in enumerate(imgs):
    iid = f"{PREFIX}_{START + j:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    v_only = scale(orig, to512(vine.embed_with_target(orig, apply_crypto(tx, pv, Mv))), ALPHA)
    comp = scale(v_only, to512(tm.embed_with_target(v_only, apply_crypto(tx, pt, Mt))), ALPHA)
    comp.save(os.path.join(OUT, "comp", f"img_{j:05d}.png"))
    meta["items"].append({"i": j, "image_id": iid})
    if (j + 1) % 12 == 0: print(f"  embed [{j+1}/{len(imgs)}]", flush=True)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print(f"[sweep_embed] {len(imgs)} comp imgs @alpha={ALPHA} idx[{START},{START+N}) -> {OUT}\nSWEEP_EMBED_DONE")
