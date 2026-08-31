"""Does NESTED multi-scale embedding generalize the central-75% trick into a crop STAIRCASE?

Hypothesis (from the measured VINE profile): VINE's residual energy sits in a BORDER RING of
whatever canvas it embeds into (measured center-75% density ratio 0.048). So nested concentric
copies may COEXIST — each copy's energy lives on its own ring, outside the next inner copy's canvas.
If so, embedding the same payload at K in {1.0, .75, .5} gives a staircase: any centered crop that
still contains one intact copy decodes, pushing the crop floor from .75 down to .5.

Schemes: full | central75 | nested{1,.75,.5} | nested{.75,.5}
Decode: centered crop/scale SEARCH, crypto-verify gated (accept first hit).
Measures detection across a crop sweep + regen + the crop+regen compound, and the PSNR cost."""
import os, sys, glob, json, numpy as np
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
def txc(iid):
    p, M = vine.get_perm_M(iid)
    return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)

def embed_at(x, iid, keep):
    """Embed the payload into the central `keep` square of x (keep=1.0 -> whole frame)."""
    x = to512(x)
    if keep >= 0.999:
        return to512(vine.embed_with_target(x, txc(iid)))
    s = int(512 * keep); o = (512 - s) // 2
    marked = vine.embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid))
    out = x.copy(); out.paste(marked.resize((s, s)), (o, o)); return out

def embed_scheme(orig, iid, keeps):
    x = to512(orig)
    for k in keeps: x = embed_at(x, iid, k)      # outer -> inner, sequential
    return x

SEARCH = [1.0, 0.9, 0.8, 0.75, 0.65, 0.55, 0.5, 0.4]
def decode_search(pil, iid, tx):
    best = -1.0
    for k in SEARCH:
        if k >= 0.999: view = to512(pil)
        else:
            s = int(512 * k); o = (512 - s) // 2; view = to512(pil).crop((o, o, o + s, o + s))
        rl = np.clip(method_soft_to_codeword_llr(vine.raw_probs(view), *vine.get_perm_M(iid),
                                                 kind="prob", n_codeword=n), -CLAMP, CLAMP)
        if bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or (((rl > 0).astype(np.uint8) == tx).mean() >= tau):
            return 1.0, k
        best = max(best, float(((rl > 0).astype(np.uint8) == tx).mean()))
    return 0.0, None

def a_crop(x, keep):
    s = int(512 * keep); o = (512 - s) // 2; return to512(x).crop((o, o, o + s, o + s)).resize((512, 512))

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:24]
pipe = build_regen_pipe()
def regen(x, j): return to512(stable_regen(pipe, x, 7000 + j, noise_step=30))
ATT = {"clean": lambda x, j: x, "crop90": lambda x, j: a_crop(x, .90), "crop75": lambda x, j: a_crop(x, .75),
       "crop60": lambda x, j: a_crop(x, .60), "crop50": lambda x, j: a_crop(x, .50),
       "crop40": lambda x, j: a_crop(x, .40), "regen0.3": lambda x, j: regen(x, j),
       "crop75+regen": lambda x, j: regen(a_crop(x, .75), j), "crop50+regen": lambda x, j: regen(a_crop(x, .50), j)}
SCHEMES = {"full": [1.0], "central75": [0.75], "nested_1_75_50": [1.0, 0.75, 0.5], "nested_75_50": [0.75, 0.5]}
res = {}
for name, keeps in SCHEMES.items():
    acc = {k: [] for k in ATT}; ps = []; hit_k = {k: [] for k in ATT}
    for j, fp in enumerate(srcs):
        iid = f"nst_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed_scheme(orig, iid, keeps)
        ps.append(_psnr(np.asarray(orig, np.float64), np.asarray(wm, np.float64), data_range=255))
        for k, f in ATT.items():
            d, kk = decode_search(f(wm, j), iid, tx); acc[k].append(d)
            if kk is not None: hit_k[k].append(kk)
    res[name] = {"keeps": keeps, "psnr": round(float(np.mean(ps)), 2),
                 "detect": {k: round(float(np.mean(acc[k])), 3) for k in ATT},
                 "winning_view": {k: (round(float(np.mean(hit_k[k])), 3) if hit_k[k] else None) for k in ATT}}
    print(f"{name:16} PSNR={res[name]['psnr']:.1f}  " +
          "  ".join(f"{k}={res[name]['detect'][k]:.2f}" for k in ATT), flush=True)
json.dump(res, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/nested_vine_probe.json", "w"), indent=2)
print("NESTED_VINE_PROBE_DONE")
