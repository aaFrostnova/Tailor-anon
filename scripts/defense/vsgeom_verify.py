"""Verify the surprising VideoSeal geometry result before acting on it:
(A) FPR under geometry — decode the VideoSeal crypto-fragment on UNWATERMARKED images
    (clean / rot90 / border0.2). If rotated unwatermarked images falsely detect, the
    rot90=1.00 is spurious. Must be ~0.
(B) Isolated rotation robustness — pure VideoSeal (no crypto/BCH): embed a random 256-bit
    message, rotate, decode RAW bit-acc. Confirms the model itself is rotation-robust and
    the eval pipeline isn't manufacturing the result.
"""
import os, sys, glob
import numpy as np, torch
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts"), os.path.join(REPO,"scripts/defense"), os.path.join(REPO,"external/videoseal")]:
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import apply_crypto, derive_method_keyed_constants
from src.soft_fusion import method_soft_to_codeword_llr
import videoseal

KEY=b"v5_key_encoder_master"; dev="cuda"; ALPHA=0.70; CLAMP=15.0
sb=ShortenedBCH(); TAU=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
vs=videoseal.load("videoseal").to(dev).eval(); VSB=vs.get_random_msg().shape[-1]
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def to_t(pil): return torch.from_numpy(np.asarray(to512(pil),np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
def to_pil(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255+0.5).astype(np.uint8))
def scale_resid(c,w,a): C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64); return Image.fromarray(np.clip(C+a*(W-C),0,255).astype(np.uint8))
def rot(im,d): return im.rotate(d,resample=Image.BILINEAR)
def border(im,p): d=int(round(512*p)); reg=im.crop((d,d,512,512)); cv=Image.new("RGB",(512,512)); cv.paste(reg,(0,0)); return cv
def vs_pm(iid): return derive_method_keyed_constants(KEY,iid,"videoseal",sb.n)
def vs_embed(b,t):
    m=torch.zeros(1,VSB,device=dev); m[0,:sb.n]=torch.tensor(np.asarray(t,np.float32),device=dev)
    with torch.no_grad(): return to_pil(vs.embed(to_t(b),msgs=m,is_video=False)["imgs_w"])
def vs_llr(att,perm,M):
    with torch.no_grad(): p=vs.detect(to_t(att),is_video=False)["preds"][:,1:][0,:sb.n].cpu().numpy()
    return np.clip(method_soft_to_codeword_llr(p,perm,M,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP)
def detected(llr,iid,tx): return bool(decode_and_verify(llr,iid,codec=sb)["detected"]) or ((llr>0).astype(np.uint8)==tx).mean()>=TAU

imgs=(sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png")))+sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))
# ---- (A) FPR under geometry on UNWATERMARKED images ----
neg=imgs[240:256]; fpr={"clean":0,"rot90":0,"border0.2":0}
for j,fp in enumerate(neg):
    iid=f"neg_{j:05d}"; pm=vs_pm(iid); tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
    orig=to512(Image.open(fp).convert("RGB"))                     # NO watermark embedded
    for nm,att in [("clean",orig),("rot90",rot(orig,90)),("border0.2",border(orig,0.2))]:
        fpr[nm]+=int(detected(vs_llr(att,*pm),iid,tx))
print(f"[A] FPR on UNWATERMARKED (n={len(neg)}): clean={fpr['clean']}  rot90={fpr['rot90']}  border0.2={fpr['border0.2']}  (want 0)")

# ---- (B) isolated rotation: pure VideoSeal, no crypto ----
print("[B] pure VideoSeal raw bit-acc (no crypto):")
for deg in [0,10,30,90]:
    bas=[]
    for fp in imgs[180:188]:
        x=to_t(Image.open(fp).convert("RGB"))
        with torch.no_grad():
            out=vs.embed(x,is_video=False); msg=out["msgs"][0].cpu().numpy().astype(int)
            xw=to_pil(out["imgs_w"]); att=to_t(rot(xw,deg)) if deg else to_t(xw)
            pr=vs.detect(att,is_video=False)["preds"][:,1:][0].cpu().numpy()
        bas.append(np.mean((pr>0).astype(int)==msg))
    print(f"   rot{deg:<3d}: bit-acc {np.mean(bas):.3f}")
print("VSVERIFY_DONE")
