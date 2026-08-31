"""PROVE the FPR gap: on UNMARKED cover images, how often does each scale-search rule falsely 'detect'?
For each image + a fixed target codeword tx (iid 'fpr_probe'), run the 133-scale VINE search and count
false accepts under (a) crypto-verify-only [FPR-safe] vs (b) zerobit ba>=tau at any scale [permissive]."""
import os, sys, glob, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.soft_fusion import method_soft_to_codeword_llr
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
V = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
iid = "fpr_probe"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
per_test_fpr = float(binom.sf(np.ceil(tau * n) - 1, n, 0.5))
print(f"[params] n={n}  tau={tau:.4f}  per-scale zerobit FPR (analytic) = {per_test_fpr*100:.2f}%", flush=True)
print(f"[analytic] union over 133 scales = {(1-(1-per_test_fpr)**133)*100:.1f}%   crypto union = 133*2^-37 ~ 2^-30", flush=True)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def llr(pil):
    p, M = V.get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(V.raw_probs(pil), p, M, kind="prob", n_codeword=n), -CLAMP, CLAMP)
def cv(rl): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def view(P, f):
    if f >= 0.999: return P
    s = int(round(512 * float(f))); o = (512 - s) // 2; return P.crop((o, o, o + s, o + s))
VS = np.arange(0.34, 1.0001, 0.005)
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:60]
safe_fp, perm_fp = [], []
for fp in srcs:
    P = to512(Image.open(fp).convert("RGB"))   # UNMARKED cover
    s_hit = p_hit = False
    for f in VS:
        rl = llr(view(P, float(f)))
        if not s_hit and cv(rl): s_hit = True
        if not p_hit and ((rl > 0).astype(np.uint8) == tx).mean() >= tau: p_hit = True
        if s_hit and p_hit: break
    safe_fp.append(1.0 if s_hit else 0.0); perm_fp.append(1.0 if p_hit else 0.0)
print(f"[EMPIRICAL on {len(srcs)} UNMARKED images]", flush=True)
print(f"  crypto-verify-only scale search  (FPR-SAFE)   false-detect rate = {np.mean(safe_fp)*100:.1f}%", flush=True)
print(f"  zerobit>=tau scale search        (PERMISSIVE)  false-detect rate = {np.mean(perm_fp)*100:.1f}%", flush=True)
print("FPR_DEMO_DONE", flush=True)
