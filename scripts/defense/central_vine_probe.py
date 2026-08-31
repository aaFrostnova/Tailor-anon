"""Test the user's idea: embed VINE only in the CENTRAL keep% region so a centered
crop-keep% preserves the whole mark. Compare full-VINE vs central-VINE on:
  clean / crop75 / crop60 / crop80 (mismatched ratios) / regen(s0.3)  + PSNR.
Detection = crypto-verify (real metric). n small for speed."""
import os, sys, glob, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio as _psnr
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
    """Embed VINE into ONLY the central keep% sub-region; border stays original."""
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); p, M = vine.get_perm_M(iid)
    s = int(512 * keep); o = (512 - s) // 2
    center = orig.crop((o, o, o + s, o + s))                     # central s x s content
    marked = vine.embed_with_target(center, apply_crypto(tx, p, M))  # embed (resizes to 256 canonical, back to s)
    out = orig.copy(); out.paste(marked.resize((s, s)), (o, o))  # paste marked center, border unmarked
    return out, tx.astype(np.uint8)

def cv_ba(pil, iid, tx):
    p, M = vine.get_perm_M(iid)
    rl = np.clip(method_soft_to_codeword_llr(vine.raw_probs(pil), p, M, kind="prob", n_codeword=n), -CLAMP, CLAMP)
    det = bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or (((rl > 0).astype(np.uint8) == tx).mean() >= tau)
    return (1.0 if det else 0.0), float(((rl > 0).astype(np.uint8) == tx).mean())

def a_crop(x, keep):
    s = int(512 * keep); o = (512 - s) // 2; return x.crop((o, o, o + s, o + s)).resize((512, 512))

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:30]
pipe = build_regen_pipe()
ATT = {"clean": lambda x: x, "crop75": lambda x: a_crop(x, 0.75), "crop60": lambda x: a_crop(x, 0.60),
       "crop80": lambda x: a_crop(x, 0.80), "regen0.3": None}
res = {}
for scheme, embed in [("full_vine", embed_full), ("central75_vine", lambda o, i: embed_central(o, i, 0.75))]:
    acc = {k: [] for k in ATT}; ba = {k: [] for k in ATT}; ps = []
    for j, fp in enumerate(srcs):
        iid = f"cv_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        wm, tx = embed(orig, iid)
        ps.append(_psnr(np.asarray(orig), np.asarray(wm), data_range=255))
        for k, f in ATT.items():
            att = to512(stable_regen(pipe, wm, 7000 + j, noise_step=30)) if k == "regen0.3" else f(wm)
            d, b = cv_ba(att, iid, tx); acc[k].append(d); ba[k].append(b)
    res[scheme] = {"psnr": float(np.mean(ps)),
                   "detect": {k: float(np.mean(acc[k])) for k in ATT},
                   "bit_acc": {k: float(np.mean(ba[k])) for k in ATT}}
    print(f"{scheme}: PSNR={np.mean(ps):.1f}  " + "  ".join(f"{k}={np.mean(acc[k]):.2f}(ba{np.mean(ba[k]):.2f})" for k in ATT), flush=True)
json.dump(res, open("/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks/central_vine_probe.json", "w"), indent=2)
print("CENTRAL_VINE_PROBE_DONE")
