"""Precisely locate the 2-frag composite's geometric crop gap before borrowing the
block-DWT spatial-redundancy idea. Embed VINE+TM @0.70, then apply several crop/shift
families at increasing severity and report VINE / TM / composite(equal-MRC) det+ba.

Families:
  cresize_a : centered crop to area fraction a, resize back to 512   (scale; alignment ~kept)
  corner_a  : top-left s x s window (s=512*sqrt(a)), resize to 512    (off-center crop)
  border_p  : remove p-fraction border, paste remainder into a 512 canvas at (0,0) (translation/occlusion, NO resize)
"""
import os, sys, glob, json
import numpy as np
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))

def cresize(im, a):
    s = int(round(512 * np.sqrt(a))); o = (512 - s) // 2
    return im.crop((o, o, o + s, o + s)).resize((512, 512))
def corner(im, a):
    s = int(round(512 * np.sqrt(a))); return im.crop((0, 0, s, s)).resize((512, 512))
def border(im, p):
    d = int(round(512 * p)); reg = im.crop((d, d, 512, 512))    # drop top-left border, keep native res
    canvas = Image.new("RGB", (512, 512), (0, 0, 0)); canvas.paste(reg, (0, 0)); return canvas

ATTACKS = ([("clean", lambda im: im)] +
           [(f"cresize{a}", (lambda a: lambda im: cresize(im, a))(a)) for a in [0.9, 0.75, 0.5]] +
           [(f"corner{a}", (lambda a: lambda im: corner(im, a))(a)) for a in [0.75, 0.5]] +
           [(f"border{p}", (lambda p: lambda im: border(im, p))(p)) for p in [0.1, 0.2, 0.3]])

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:180 + N]
store = {nm: {"av": [], "at": [], "tx": [], "id": []} for nm, _ in ATTACKS}
for j, fp in enumerate(imgs):
    iid = f"sw_{180 + j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    v = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
    vt = scale_resid(v, to512(tm.embed_with_target(v, apply_crypto(cw, pt, Mt))), ALPHA)
    for nm, fn in ATTACKS:
        att = to512(fn(vt)); s = store[nm]
        s["av"].append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        s["at"].append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
        s["tx"].append(cw.astype(np.uint8)); s["id"].append(iid)
    if (j + 1) % 8 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

def metr(F, tx, ids):
    det = []
    for i in range(len(F)):
        ver = bool(decode_and_verify(F[i], ids[i], codec=sb)["detected"])
        det.append(1.0 if (ver or ((F[i] > 0).astype(np.uint8) == tx[i]).mean() >= TAU) else 0.0)
    return float(np.mean(det)), float(((F > 0).astype(np.uint8) == tx).mean())

print(f"\nn={N} tau={TAU:.3f}   VINE | TM | composite(eq)   (det/ba)")
print(f"{'attack':10s} | {'VINE':>9s} | {'TM':>9s} | {'composite':>9s}")
print("-" * 48)
rows = []
for nm, _ in ATTACKS:
    s = store[nm]; av = np.array(s["av"]); at = np.array(s["at"]); tx = np.array(s["tx"]); ids = s["id"]
    V = metr(av, tx, ids); T = metr(at, tx, ids); C = metr(av + at, tx, ids)
    def c(x): return f"{x[0]:.2f}/{x[1]:.2f}"
    print(f"{nm:10s} | {c(V):>9s} | {c(T):>9s} | {c(C):>9s}")
    rows.append({"attack": nm, "vine": V, "tm": T, "comp": C})
json.dump(rows, open(os.path.join(REPO, "results/defense/crop_gap_diag.json"), "w"), indent=2)
print("CROPDIAG_DONE")
