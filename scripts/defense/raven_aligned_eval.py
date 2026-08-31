"""RAVEN/WAVES-aligned eval: composite (VINE+TrustMark fused, dual detect) on SD-2.1 generated images,
attacked with the EXACT WAVES distortions (ported from umd-huang-lab/WAVES distortions.py), swept over
relative strength. Reports composite detection + fused bit-acc. Also reports per-fragment to compare with
RAVEN Table-1 VINE/TrustMark numbers."""
import glob, io, os, sys, random, json, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch, torchvision.transforms as T, torchvision.transforms.functional as F
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
RES=512; dev="cuda"
sb=ShortenedBCH(); tau=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits")}
frag={"vine":VineCryptoWrapper(master_key=b"v5_key_encoder_master",method_name="vine",n_bits=sb.n,device=dev),
      "trustmark":TrustMarkFragment(master_key=b"v5_key_encoder_master",method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)}
# --- WAVES distortions (verbatim params) ---
PARAS=dict(rotation=(0,45),resizedcrop=(1,0.5),erasing=(0,0.25),brightness=(1,2),contrast=(1,2),blurring=(0,20),noise=(0,0.1),compression=(90,10))
def rel2abs(s,t):
    a,b=PARAS[t]; v=s*(b-a)+a; return min(max(v,min(a,b)),max(a,b))
def wattack(image,t,rel,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); s=rel2abs(rel,t)
    if t=="rotation": return F.rotate(image,s)
    if t=="resizedcrop":
        i,j,h,w=T.RandomResizedCrop.get_params(image,scale=(s,s),ratio=(1,1)); return F.resized_crop(image,i,j,h,w,list(image.size))
    if t=="erasing":
        x=F.to_tensor(image)[None]; i,j,h,w,v=T.RandomErasing.get_params(x,scale=(s,s),ratio=(1,1),value=[0]); return F.to_pil_image(F.erase(x,i,j,h,w,v)[0])
    if t=="brightness": return ImageEnhance.Brightness(image).enhance(s)
    if t=="contrast": return ImageEnhance.Contrast(image).enhance(s)
    if t=="blurring": return image.filter(ImageFilter.GaussianBlur(int(s)))
    if t=="noise":
        x=F.to_tensor(image); return F.to_pil_image((x+torch.randn_like(x)*s).clamp(0,1))
    if t=="compression":
        b=io.BytesIO(); image.save(b,format="JPEG",quality=int(s)); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
def main():
    N=int(sys.argv[1]) if len(sys.argv)>1 else 50
    rels=[0.25,0.5,0.75,1.0]
    ATKS=["rotation","resizedcrop","erasing","brightness","contrast","blurring","noise","compression"]
    files=(sorted(glob.glob(REPO+"/results/defense/raven_gen/img_*.png"))+sorted(glob.glob(REPO+"/results/defense/raven_gen2/img_*.png")))[:N]
    print(f"[raven-aligned] {len(files)} SD-2.1 images, WAVES attacks x relative strength {rels}",flush=True)
    R={(a,r):{"or":[],"fba":[],"vine":[],"tm":[]} for a in ATKS for r in rels}
    for idx,fp in enumerate(files):
        C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"rv_{idx:05d}"
        tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); img=C
        for nm in ["vine","trustmark"]:
            p,M=frag[nm].get_perm_M(iid); img=frag[nm].embed_with_target(img,apply_crypto(tx,p,M))
            if img.size!=(RES,RES): img=img.resize((RES,RES))
        for a in ATKS:
            for r in rels:
                At=wattack(img,a,r,1234+idx)
                if At.size!=(RES,RES): At=At.resize((RES,RES))
                al={};per={}
                for nm in ["vine","trustmark"]:
                    kind,getter=SPEC[nm]; p,M=frag[nm].get_perm_M(iid); aa=method_soft_to_codeword_llr(getattr(frag[nm],getter)(At),p,M,kind=kind,n_codeword=sb.n)
                    al[nm]=aa; per[nm]=float(np.mean(llr_to_bits(aa)==tx))
                fused=fuse_llrs(al,weights=None,n_codeword=sb.n); fba=float(np.mean(llr_to_bits(fused)==tx)); fver=bool(decode_and_verify(fused,iid,codec=sb)["detected"])
                R[(a,r)]["or"].append(1.0 if(fba>=tau or fver)else 0.0); R[(a,r)]["fba"].append(fba); R[(a,r)]["vine"].append(per["vine"]); R[(a,r)]["tm"].append(per["trustmark"])
        if (idx+1)%10==0: print(f"  [{idx+1}/{len(files)}]",flush=True)
    print(f"\n=== composite on SD-2.1 imgs, WAVES attacks (n={len(files)}; comp_or / fused_ba / VINE / TM bit-acc) ===")
    print(f"{'attack':<13}"+"".join(f"  rel={r}" for r in rels))
    for a in ATKS:
        row=f"{a:<13}"
        for r in rels: row+=f"  {np.mean(R[(a,r)]['or']):.2f}/{np.mean(R[(a,r)]['fba']):.2f}"
        print(row)
    print("\nper-fragment (VINE | TM bit-acc) at rel=1.0 (strongest) vs RAVEN table:")
    for a in ATKS: print(f"  {a:<13} VINE={np.mean(R[(a,1.0)]['vine']):.3f}  TM={np.mean(R[(a,1.0)]['tm']):.3f}")
    json.dump({str(k):{m:float(np.mean(v[m])) for m in v} for k,v in R.items()},open(os.path.join(REPO,"results/defense/raven_aligned.json"),"w"),indent=2)
    print("RAVEN_ALIGNED_DONE")
if __name__=="__main__": main()
