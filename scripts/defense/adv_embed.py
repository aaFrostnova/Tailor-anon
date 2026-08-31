"""Embed 3 watermark configs (composite / VINE-only / TM-only @alpha=0.70) for the
advanced-attack (CtrlRegen+/UnMarker) cross-env eval. Writes 3 dirs + meta.json."""
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "results/defense/adv")
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[:N]
sb = ShortenedBCH()
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
for sub in ["comp", "vine", "tm"]: os.makedirs(os.path.join(OUT, sub), exist_ok=True)
meta = {"alpha": ALPHA, "items": []}
for i, fp in enumerate(imgs):
    iid = f"adv_{i:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    v_only = scale(orig, to512(vine.embed_with_target(orig, apply_crypto(tx, pv, Mv))), ALPHA)
    t_only = scale(orig, to512(tm.embed_with_target(orig, apply_crypto(tx, pt, Mt))), ALPHA)
    comp = scale(v_only, to512(tm.embed_with_target(v_only, apply_crypto(tx, pt, Mt))), ALPHA)
    comp.save(os.path.join(OUT, "comp", f"img_{i:05d}.png"))
    v_only.save(os.path.join(OUT, "vine", f"img_{i:05d}.png"))
    t_only.save(os.path.join(OUT, "tm", f"img_{i:05d}.png"))
    meta["items"].append({"i": i, "image_id": iid})
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print(f"[adv_embed] {len(imgs)} imgs x3 configs @alpha={ALPHA} -> {OUT}\nADV_EMBED_DONE")
