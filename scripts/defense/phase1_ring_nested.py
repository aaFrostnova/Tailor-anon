"""PHASE 1 — ring-nested VINE (E1.1) + dense scale/translation decode search (E1.2).

Gate G1 passed: a single VINE embed's center is strippable to a ~2% ring for free (robustness flat,
PSNR up). So build a NESTED stack where each layer keeps ONLY its own ring -> layers stack without
paying for center overlap, and the crop staircase gets more rungs cheaply.

Construction embed_ring_nested(Ks, ring_strip): for each K (outer->inner), embed at central K on the
running image, then revert the central K*ring_strip square back to the pre-layer image (keep this
layer's ring, leave the interior clean for the next inner layer).

Decode: DENSE centered scale search (E1.2) over a fine grid covering all K/c ratios; crypto-verify
gated (accept first hit), FPR-safe. (Off-center handled separately in Phase 3.)"""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
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
def txc(iid):
    p, M = vine.get_perm_M(iid)
    return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def embed_at(x, iid, K):
    x = to512(x)
    if K >= 0.999: return to512(vine.embed_with_target(x, txc(iid)))
    s = int(512 * K); o = (512 - s) // 2
    m = vine.embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid))
    out = x.copy(); out.paste(m.resize((s, s)), (o, o)); return out
def embed_ring_nested(orig, iid, Ks, ring_strip):
    x = to512(orig)
    for K in Ks:
        wm = np.asarray(embed_at(x, iid, K)).copy(); base = np.asarray(x)
        s = int(512 * K * ring_strip); o = (512 - s) // 2
        if s > 0: wm[o:o + s, o:o + s] = base[o:o + s, o:o + s]   # keep this layer's ring, clean interior
        x = Image.fromarray(wm)
    return x
def embed_plain_nested(orig, iid, Ks):
    x = to512(orig)
    for K in Ks: x = embed_at(x, iid, K)
    return x

SCALES = np.round(np.arange(0.34, 1.0001, 0.02), 3).tolist()   # dense scale grid, covers all K/c
def decode_search(pil, iid, tx):
    P = to512(pil)
    for v in SCALES:
        if v >= 0.999: view = P
        else:
            s = int(512 * v); o = (512 - s) // 2; view = P.crop((o, o, o + s, o + s))
        rl = np.clip(method_soft_to_codeword_llr(vine.raw_probs(view), *vine.get_perm_M(iid),
                                                 kind="prob", n_codeword=n), -CLAMP, CLAMP)
        if bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or (((rl > 0).astype(np.uint8) == tx).mean() >= tau):
            return 1.0
    return 0.0
def a_crop(x, c):
    s = int(512 * c); o = (512 - s) // 2; return to512(x).crop((o, o, o + s, o + s)).resize((512, 512))

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:24]
pipe = build_regen_pipe()
def regen(x, j): return to512(stable_regen(pipe, x, 7000 + j, noise_step=30))
ATT = {"clean": lambda x, j: x, "crop90": lambda x, j: a_crop(x, .90), "crop80": lambda x, j: a_crop(x, .80),
       "crop75": lambda x, j: a_crop(x, .75), "crop65": lambda x, j: a_crop(x, .65), "crop60": lambda x, j: a_crop(x, .60),
       "crop50": lambda x, j: a_crop(x, .50), "crop40": lambda x, j: a_crop(x, .40), "regen0.3": lambda x, j: regen(x, j),
       "crop75+regen": lambda x, j: regen(a_crop(x, .75), j), "crop50+regen": lambda x, j: regen(a_crop(x, .50), j)}
SCHEMES = {
    "full": ("plain", [1.0]),
    "nested_plain_1_75_50": ("plain", [1.0, 0.75, 0.5]),
    "ring_1_75_50": ("ring", [1.0, 0.75, 0.5]),
    "ring6_dense": ("ring", [0.9, 0.8, 0.75, 0.65, 0.6, 0.5]),
}
res = {}
for name, (kind, Ks) in SCHEMES.items():
    acc = {k: [] for k in ATT}; ps = []; ss = []
    for j, fp in enumerate(srcs):
        iid = f"p1_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed_plain_nested(orig, iid, Ks) if kind == "plain" else embed_ring_nested(orig, iid, Ks, 0.85)
        O = np.asarray(orig, np.float64); W = np.asarray(wm, np.float64)
        ps.append(_psnr(O, W, data_range=255)); ss.append(_ssim(O, W, channel_axis=2, data_range=255))
        for k, f in ATT.items(): acc[k].append(decode_search(f(wm, j), iid, tx))
    res[name] = {"kind": kind, "keeps": Ks, "psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4),
                 "detect": {k: round(float(np.mean(acc[k])), 3) for k in ATT}}
    r = res[name]
    print(f"{name:22} PSNR={r['psnr']:.1f} SSIM={r['ssim']:.3f} | " +
          "  ".join(f"{k}={r['detect'][k]:.2f}" for k in ATT), flush=True)
json.dump(res, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/phase1_ring_nested.json", "w"), indent=2)
print("PHASE1_RING_NESTED_DONE")
