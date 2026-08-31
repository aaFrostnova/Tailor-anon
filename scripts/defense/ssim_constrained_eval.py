"""Detection under an SSIM>=0.8 fidelity constraint on the attack.
Embed composite (VINE-R + TrustMark-B, soft-fused, dual detect). For each attack: SSIM(attacked, clean),
fused bit-acc, crypto-verify, composite_or. Then report detection on the SUBSET of attacks with SSIM>=0.8
(i.e. attacks that keep the image recognizably the same -- excludes alignment-breaking geometry)."""
import glob, io, os, sys, tempfile, numpy as np
from PIL import Image
from scipy.stats import binom
from skimage.metrics import structural_similarity as ssim
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense"); sys.path.insert(0,os.path.join(REPO,"external","WatermarkAttacker"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
RES=512; dev="cuda"; N=15
sb=ShortenedBCH(); tau=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits")}
frag={"vine":VineCryptoWrapper(master_key=b"v5_key_encoder_master",method_name="vine",n_bits=sb.n,device=dev),
      "trustmark":TrustMarkFragment(master_key=b"v5_key_encoder_master",method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)}
pipe=build_regen_pipe()
from wmattacker import (VAEWMAttacker,GaussianBlurAttacker,GaussianNoiseAttacker,JPEGAttacker,BrightnessAttacker,ContrastAttacker,BM3DAttacker)
tmp=tempfile.mkdtemp()
sig={"jpeg":JPEGAttacker(quality=25),"blur":GaussianBlurAttacker(5,1),"noise":GaussianNoiseAttacker(std=0.05),
     "bright":BrightnessAttacker(0.2),"contrast":ContrastAttacker(0.2),"bm3d":BM3DAttacker()}
vae={}
for k,mn in [("vae_b","bmshj2018-hyperprior"),("vae_c","cheng2020-anchor")]:
    try: vae[k]=VAEWMAttacker(mn,quality=3,metric="mse",device=dev)
    except Exception as e: print("vae skip",k,e)
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
def sapply(a,pil):
    ip=os.path.join(tmp,"i.png"); op=os.path.join(tmp,"o.png"); pil.save(ip); a.attack([ip],[op]); return Image.open(op).convert("RGB").resize((RES,RES))
def geom(name,pil):
    if name=="hflip": return pil.transpose(Image.FLIP_LEFT_RIGHT)
    if name=="rs256": return pil.resize((256,256),Image.BICUBIC).resize((RES,RES),Image.BICUBIC)
    if name in("crop75","crop50"):
        r=0.75 if name=="crop75" else 0.5; cw=int(RES*r); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
    if name=="rot9":
        a=np.array(pil); pad=RES//2; big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),'reflect')).rotate(9,resample=Image.BICUBIC); bw,bh=big.size; l=(bw-RES)//2; return big.crop((l,l,l+RES,l+RES))
    if name=="crop_jpeg":
        c=geom("crop75",pil); b=io.BytesIO(); c.save(b,"JPEG",quality=25); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
ATKS=["clean","jpeg","blur","noise","bright","contrast","bm3d","regen","rinse2x","vae_b","vae_c","rs256","hflip","crop75","crop50","rot9","crop_jpeg"]
def attack(name,W,seed):
    if name=="clean": return W
    if name in sig: return sapply(sig[name],W)
    if name in vae: return sapply(vae[name],W)
    if name=="regen": return stable_regen(pipe,W,seed,denoise_steps=8)
    if name=="rinse2x":
        x=stable_regen(pipe,W,seed,denoise_steps=8); return stable_regen(pipe,x,seed+1,denoise_steps=8)
    return geom(name,W)
files=sorted(glob.glob('/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg'))[4000:4000+N]
R={a:{"ssim":[],"or":[],"fba":[]} for a in ATKS}
for i,fp in enumerate(files):
    C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"sc_{i:05d}"; Ca=arr(C)
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    img=C
    for nm in ["vine","trustmark"]:
        p,M=frag[nm].get_perm_M(iid); img=frag[nm].embed_with_target(img,apply_crypto(tx,p,M))
        if img.size!=(RES,RES): img=img.resize((RES,RES))
    for a in ATKS:
        At=attack(a,img,1234+i); Aa=arr(At)
        al={}
        for nm in ["vine","trustmark"]:
            kind,getter=SPEC[nm]; p,M=frag[nm].get_perm_M(iid); al[nm]=method_soft_to_codeword_llr(getattr(frag[nm],getter)(At),p,M,kind=kind,n_codeword=sb.n)
        fused=fuse_llrs(al,weights=None,n_codeword=sb.n); fba=float(np.mean(llr_to_bits(fused)==tx))
        fver=bool(decode_and_verify(fused,iid,codec=sb)["detected"])
        R[a]["ssim"].append(ssim(Aa,Ca,channel_axis=2,data_range=255)); R[a]["fba"].append(fba)
        R[a]["or"].append(1.0 if (fba>=tau or fver) else 0.0)
    print(f"[{i+1}/{N}] done",flush=True)
print(f"\n=== composite detection under attack fidelity (n={N}, tau={tau:.2f}) ===")
print(f"{'attack':<11}{'SSIM':>7}{'legal(>=.8)':>12}{'comp_or':>9}{'fused_ba':>9}")
legal_or=[]
for a in ATKS:
    s=np.mean(R[a]["ssim"]); o=np.mean(R[a]["or"]); leg=s>=0.8
    if leg and a!="clean": legal_or.append(o)
    print(f"{a:<11}{s:>7.3f}{('YES' if leg else 'no'):>12}{o:>9.3f}{np.mean(R[a]['fba']):>9.3f}")
print(f"\nLEGAL attacks (SSIM>=0.8, excl clean): mean composite detection = {np.mean(legal_or):.3f}  over {len(legal_or)} attacks")
print("SSIM_CONSTRAINED_DONE")
