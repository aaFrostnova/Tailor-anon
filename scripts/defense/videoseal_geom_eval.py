"""Replace TrustMark with VideoSeal? Compare VINE+TM (current) vs VINE+VideoSeal on a
GEOMETRY-focused suite (the user's interest: VideoSeal's geometric robustness).

Both pixel fragments carry the same crypto codeword at alpha=0.70. Report per-fragment
(TM vs VideoSeal) and composite (VINE+TM vs VINE+VideoSeal) detection/bit-acc + PSNR,
across crop families + rotations (+ clean/jpeg/regen for context). Run with cwd =
external/videoseal (videoseal cards use repo-root-relative config paths).
"""
import os, sys, io, glob, json
import numpy as np, torch
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts/defense"), os.path.join(REPO, "external/videoseal")]:
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, derive_method_keyed_constants
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen
import videoseal

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
vs = videoseal.load("videoseal").to(dev).eval()
VSB = vs.get_random_msg().shape[-1]
pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def to_t(pil): return torch.from_numpy(np.asarray(to512(pil), np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
def to_pil(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255+0.5).astype(np.uint8))
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a*(W-C), 0, 255).astype(np.uint8))
def psnr(a, b):
    e = np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2); return 99 if e<1e-9 else 10*np.log10(255*255/e)

def vs_pm(iid): return derive_method_keyed_constants(KEY, iid, "videoseal", sb.n)
def vs_embed(base512, target100):
    msg = torch.zeros(1, VSB, device=dev); msg[0, :sb.n] = torch.tensor(np.asarray(target100, np.float32), device=dev)
    with torch.no_grad():
        out = vs.embed(to_t(base512), msgs=msg, is_video=False)
    return to_pil(out["imgs_w"])
def vs_llr(att, perm, M):
    with torch.no_grad():
        preds = vs.detect(to_t(att), is_video=False)["preds"][:, 1:][0, :sb.n].cpu().numpy()  # per-bit logits
    return np.clip(method_soft_to_codeword_llr(preds, perm, M, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)

# ---- geometry-focused attack suite ----
def a_jpeg(im,q): b=io.BytesIO(); im.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
def cresize(im,a): s=int(round(512*np.sqrt(a))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def corner(im,a): s=int(round(512*np.sqrt(a))); return im.crop((0,0,s,s)).resize((512,512))
def border(im,p): d=int(round(512*p)); reg=im.crop((d,d,512,512)); cv=Image.new("RGB",(512,512)); cv.paste(reg,(0,0)); return cv
def rot(im,d): return im.rotate(d, resample=Image.BILINEAR, expand=False)
ATTACKS = [("clean",lambda im:im),("jpeg25",lambda im:a_jpeg(im,25)),("regen_x1",lambda im:to512(stable_regen(pipe,im,seed=11))),
           ("cresize0.75",lambda im:cresize(im,0.75)),("cresize0.5",lambda im:cresize(im,0.5)),("corner0.5",lambda im:corner(im,0.5)),
           ("border0.1",lambda im:border(im,0.1)),("border0.2",lambda im:border(im,0.2)),
           ("rot5",lambda im:rot(im,5)),("rot10",lambda im:rot(im,10)),("rot30",lambda im:rot(im,30)),("rot90",lambda im:rot(im,90))]

imgs = (sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[180:180+N]
rec = {nm:{"av":[],"at":[],"as":[],"tx":[],"id":[]} for nm,_ in ATTACKS}
psTM, psVS = [], []
for j, fp in enumerate(imgs):
    iid = f"sw_{180+j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); tx = cw.astype(np.uint8)
    pv,Mv = vine.get_perm_M(iid); pt,Mt = tm.get_perm_M(iid); ps_,Ms_ = vs_pm(iid)
    v   = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw,pv,Mv))), ALPHA)
    vtm = scale_resid(v, to512(tm.embed_with_target(v, apply_crypto(cw,pt,Mt))), ALPHA)        # VINE+TM
    vvs = scale_resid(v, vs_embed(v, apply_crypto(cw,ps_,Ms_)), ALPHA)                          # VINE+VideoSeal
    psTM.append(psnr(orig,vtm)); psVS.append(psnr(orig,vvs))
    for nm, fn in ATTACKS:
        aT = to512(fn(vtm)); aS = to512(fn(vvs)); r = rec[nm]
        r["av"].append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(aT), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        r["at"].append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(aT), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
        r.setdefault("avS",[]).append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(aS), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        r["as"].append(vs_llr(aS, ps_, Ms_))
        r["tx"].append(tx); r["id"].append(iid)
    if (j+1)%4==0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

def metr(F, tx, ids):
    det=[]
    for i in range(len(F)):
        ver=bool(decode_and_verify(F[i], ids[i], codec=sb)["detected"])
        det.append(1.0 if (ver or ((F[i]>0).astype(np.uint8)==tx[i]).mean()>=TAU) else 0.0)
    return float(np.mean(det)), float(((F>0).astype(np.uint8)==tx).mean())

print(f"\nn={N}  PSNR VINE+TM={np.mean(psTM):.1f}  VINE+VideoSeal={np.mean(psVS):.1f}dB  (VideoSeal {VSB}-bit)\n")
hdr=f"{'attack':12s} | {'TM-only':>9s} {'VS-only':>9s} || {'VINE+TM':>9s} {'VINE+VS':>9s}"
print(hdr); print("-"*len(hdr)); out=[]
for nm,_ in ATTACKS:
    r=rec[nm]; av=np.array(r["av"]); at=np.array(r["at"]); avS=np.array(r["avS"]); as_=np.array(r["as"]); tx=np.array(r["tx"]); ids=r["id"]
    T=metr(at,tx,ids); S=metr(as_,tx,ids); cT=metr(av+at,tx,ids); cS=metr(avS+as_,tx,ids)
    def c(x): return f"{x[0]:.2f}/{x[1]:.2f}"
    print(f"{nm:12s} | {c(T):>9s} {c(S):>9s} || {c(cT):>9s} {c(cS):>9s}")
    out.append({"attack":nm,"tm":T,"vs":S,"vine_tm":cT,"vine_vs":cS})
json.dump({"n":N,"vsbits":int(VSB),"psnr_tm":float(np.mean(psTM)),"psnr_vs":float(np.mean(psVS)),"rows":out},
          open(os.path.join(REPO,"results/defense/videoseal_geom_eval.json"),"w"), indent=2)
print("VSGEOM_DONE")
