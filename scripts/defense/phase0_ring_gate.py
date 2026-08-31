"""PHASE 0 — ring-width scan (E0.1) + thin-ring attack robustness GATE G1 (E0.2).

Embed full-frame VINE, then keep ONLY the border ring outside the central `strip` fraction
(strip=0.0 -> keep full frame; strip=0.95 -> keep only the outer ~2.5%-per-side ring), by reverting
the interior pixels to the ORIGINAL. Test each stripped image under clean + removal attacks.

E0.1: how thin can the ring get before clean bit-acc drops?  (energy_kept + clean ba vs strip)
E0.2 / GATE G1: does a thin ring survive REGEN / jpeg / noise / blur, or does stripping the
  center trade away robustness?  Anchor: full-VINE regen0.3 detect ~0.92.  If a thin ring's
  regen collapses < 0.6, stripping the center is net-negative for that width."""
import os, sys, glob, json, io, numpy as np
from PIL import Image, ImageFilter
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
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def embed_full(orig, iid):
    p, M = vine.get_perm_M(iid)
    tx = apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
    return scale(orig, to512(vine.embed_with_target(orig, tx)))
def keep_ring(O, W, strip):
    """revert central `strip` square to original -> keep only the outer ring."""
    if strip <= 0: return W.copy()
    s = int(512 * strip); o = (512 - s) // 2
    out = W.copy(); out[o:o + s, o:o + s] = O[o:o + s, o:o + s]; return out
def dec(arr, iid, tx):
    rl = np.clip(method_soft_to_codeword_llr(vine.raw_probs(Image.fromarray(arr)), *vine.get_perm_M(iid),
                                             kind="prob", n_codeword=n), -CLAMP, CLAMP)
    ba = float(((rl > 0).astype(np.uint8) == tx).mean())
    return (1.0 if (bool(decode_and_verify(rl, iid, codec=sb)["detected"]) or ba >= tau) else 0.0), ba

def _jpeg(im, q): b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def _noise(im, s): return Image.fromarray(np.clip(np.asarray(im, np.float64) + np.random.RandomState(0).normal(0, s*255, (512,512,3)), 0, 255).astype(np.uint8))
pipe = build_regen_pipe()
def regen(im, j, st): return to512(stable_regen(pipe, im, 7000 + j, noise_step=st))
ATT = {"clean": lambda im, j: im, "regen0.3": lambda im, j: regen(im, j, 30), "regen0.5": lambda im, j: regen(im, j, 50),
       "jpeg25": lambda im, j: _jpeg(im, 25), "jpeg10": lambda im, j: _jpeg(im, 10),
       "noise0.05": lambda im, j: _noise(im, 0.05), "blur3": lambda im, j: im.filter(ImageFilter.GaussianBlur(3))}
STRIPS = [0.0, 0.90, 0.93, 0.95, 0.97, 0.99]
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:20]
res = {}
for strip in STRIPS:
    acc = {k: [] for k in ATT}; ba_acc = {k: [] for k in ATT}; ps = []; en = []
    for j, fp in enumerate(srcs):
        iid = f"p0_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        O = np.asarray(orig); W = np.asarray(embed_full(orig, iid))
        R = keep_ring(O, W, strip)
        full_e = ((W.astype(np.float64) - O) ** 2).sum() + 1e-12
        en.append(((R.astype(np.float64) - O) ** 2).sum() / full_e)
        ps.append(_psnr(O.astype(np.float64), R.astype(np.float64), data_range=255))
        Rpil = Image.fromarray(R)
        for k, f in ATT.items():
            att = f(Rpil, j)
            d, b = dec(np.asarray(to512(att)), iid, tx); acc[k].append(d); ba_acc[k].append(b)
    res[f"strip{strip}"] = {"psnr": round(float(np.mean(ps)), 2), "energy_kept": round(float(np.mean(en)), 4),
                            "detect": {k: round(float(np.mean(acc[k])), 3) for k in ATT},
                            "bit_acc": {k: round(float(np.mean(ba_acc[k])), 4) for k in ATT}}
    r = res[f"strip{strip}"]
    print(f"strip={strip:.2f} ring_area={1-strip**2:.3f} E={r['energy_kept']:.3f} PSNR={r['psnr']:.1f} | " +
          "  ".join(f"{k}={r['detect'][k]:.2f}(ba{r['bit_acc'][k]:.2f})" for k in ATT), flush=True)
json.dump(res, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/phase0_ring_gate.json", "w"), indent=2)
print("PHASE0_RING_GATE_DONE")
