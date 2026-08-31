"""Apply the tiling structure to VideoSeal (the fragment-agnostic TiledTrustMark class
takes any fragment with embed_with_target + raw_logits). VideoSeal is already natively
crop-robust (border0.2=0.88), so tiling targets the RESIDUAL heavy-crop gap that global
VideoSeal still misses: corner0.5 (0.12) and cresize0.5 (0.56, zoom).

Compares composite detection: VINE + global-VideoSeal  vs  VINE + tiled-VideoSeal, on a
crop-focused suite. VINE stays global (regen fragment).
"""
import os, sys, glob, json
import numpy as np
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts/defense")]:
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.videoseal_fragment import VideoSealFragment
from src.tiled_trustmark import TiledTrustMark
from src.soft_fusion import method_soft_to_codeword_llr

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
vsf = VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=sb.n, device=dev)
tt = TiledTrustMark(vsf, sb, G=2, step=32)

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def vine_llr(att, pv, Mv):
    return np.clip(method_soft_to_codeword_llr(vine.raw_probs(to512(att)), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
def vs_llr(att, ps, Ms):
    return np.clip(method_soft_to_codeword_llr(vsf.raw_logits(to512(att)), ps, Ms, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)

def cresize(im, a): s=int(round(512*np.sqrt(a))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def corner(im, a): s=int(round(512*np.sqrt(a))); return im.crop((0,0,s,s)).resize((512,512))
def border(im, p): d=int(round(512*p)); reg=im.crop((d,d,512,512)); cv=Image.new("RGB",(512,512)); cv.paste(reg,(0,0)); return cv
ATTACKS = [("clean",lambda im:im),("cresize0.5",lambda im:cresize(im,0.5)),("corner0.5",lambda im:corner(im,0.5)),
           ("corner0.6",lambda im:corner(im,0.6)),("border0.2",lambda im:border(im,0.2)),("border0.3",lambda im:border(im,0.3))]

imgs = (sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[180:180+N]
rows = {nm:{"g":[], "t":[]} for nm,_ in ATTACKS}
def det_global(av, asv, iid, tx):
    f = av + asv
    return 1.0 if (bool(decode_and_verify(f, iid, codec=sb)["detected"]) or ((f>0).astype(np.uint8)==tx).mean()>=TAU) else 0.0
for j, fp in enumerate(imgs):
    iid = f"sw_{180+j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); tx = cw.astype(np.uint8)
    pv,Mv = vine.get_perm_M(iid); ps,Ms = vsf.get_perm_M(iid); tgt = apply_crypto(cw, ps, Ms)
    v = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw,pv,Mv))), ALPHA)
    vg = scale_resid(v, to512(vsf.embed_with_target(v, tgt)), ALPHA)     # VINE + global VideoSeal
    vt = scale_resid(v, tt.embed(v, tgt), ALPHA)                          # VINE + tiled VideoSeal
    for nm, fn in ATTACKS:
        aG = to512(fn(vg)); aT = to512(fn(vt))
        rows[nm]["g"].append(det_global(vine_llr(aG,pv,Mv), vs_llr(aG,ps,Ms), iid, tx))
        rows[nm]["t"].append(1.0 if tt.detect(aT, iid, ps, Ms, vine_llr(aT,pv,Mv), TAU) else 0.0)
    if (j+1)%6==0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

print(f"\nn={N}  VINE+VideoSeal: global vs tiled  (composite detection)")
print(f"{'attack':12s} | {'global-VS':>9s} | {'tiled-VS':>8s}")
print("-"*36); out=[]
for nm,_ in ATTACKS:
    g=float(np.mean(rows[nm]["g"])); t=float(np.mean(rows[nm]["t"]))
    print(f"{nm:12s} | {g:9.2f} | {t:8.2f}"); out.append({"attack":nm,"global":g,"tiled":t})
json.dump({"n":N,"rows":out}, open(os.path.join(REPO,"results/defense/tiled_videoseal_eval.json"),"w"), indent=2)
print("TILEDVS_DONE")
