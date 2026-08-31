"""Region ablation: which part of the image actually CARRIES each fragment's bits?

Embed full-frame, then surgically REMOVE the watermark from one region by reverting those
pixels to the ORIGINAL (cleanest possible ablation: watermark deleted there, no new artifacts).

  keep_center(K) : watermark kept ONLY inside the central K square (outside reverted)
  keep_border(K) : watermark kept ONLY outside the central K square (center reverted = "destroy the middle")

Falsifiable prediction from the measured spatial density (center-75% density ratio):
  VINE      0.048 (energy in a BORDER RING) -> destroying the CENTER should barely hurt;
                                               keeping only the center should COLLAPSE it.
  TrustMark 1.49  (center-weighted)         -> destroying the center should hurt MORE.
  VideoSeal 1.06  (uniform)                 -> degradation roughly proportional to area.
Also records the retained residual-energy fraction so decodability can be plotted against energy."""
import os, sys, glob, json, numpy as np
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
from src.soft_fusion import method_soft_to_codeword_llr
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def embed_alone(orig, iid, nm):
    p, M = F[nm].get_perm_M(iid)
    tx = apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
    return scale(orig, to512(F[nm].embed_with_target(orig, tx)))
def dec(pil, iid, nm, tx):
    kind, getter = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    rl = np.clip(method_soft_to_codeword_llr(getattr(F[nm], getter)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP)
    ba = float(((rl > 0).astype(np.uint8) == tx).mean())
    det = bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or (ba >= tau)
    return (1.0 if det else 0.0), ba

def ablate(O, W, K, mode):
    """O,W = uint8 arrays. mode='keep_center' -> wm only inside central K; 'keep_border' -> wm only outside."""
    s = int(512 * K); o = (512 - s) // 2
    if mode == "keep_center":
        out = O.copy(); out[o:o + s, o:o + s] = W[o:o + s, o:o + s]
    else:
        out = W.copy(); out[o:o + s, o:o + s] = O[o:o + s, o:o + s]
    return out

KS = [0.9, 0.75, 0.5, 0.25]
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:20]
res = {nm: {} for nm in F}
for nm in F:
    conds = {"full": []}
    for K in KS: conds[f"keep_center{K}"] = []; conds[f"keep_border{K}"] = []
    ba_acc = {k: [] for k in conds}; en_acc = {k: [] for k in conds}
    for j, fp in enumerate(srcs):
        iid = f"ab_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed_alone(orig, iid, nm)
        O = np.asarray(orig); W = np.asarray(wm)
        full_e = ((W.astype(np.float64) - O) ** 2).sum() + 1e-12
        d, b = dec(wm, iid, nm, tx); conds["full"].append(d); ba_acc["full"].append(b); en_acc["full"].append(1.0)
        for K in KS:
            for mode in ("keep_center", "keep_border"):
                A = ablate(O, W, K, mode)
                e = ((A.astype(np.float64) - O) ** 2).sum() / full_e
                d, b = dec(Image.fromarray(A), iid, nm, tx)
                key = f"{mode}{K}"; conds[key].append(d); ba_acc[key].append(b); en_acc[key].append(e)
    res[nm] = {k: {"detect": round(float(np.mean(v)), 3), "bit_acc": round(float(np.mean(ba_acc[k])), 4),
                   "energy_kept": round(float(np.mean(en_acc[k])), 4)} for k, v in conds.items()}
    print(f"--- {nm} ---", flush=True)
    for k in conds:
        r = res[nm][k]; print(f"  {k:18} det={r['detect']:.2f}  ba={r['bit_acc']:.3f}  energy_kept={r['energy_kept']:.3f}", flush=True)
json.dump(res, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/region_ablation.json", "w"), indent=2)
print("REGION_ABLATION_DONE")
