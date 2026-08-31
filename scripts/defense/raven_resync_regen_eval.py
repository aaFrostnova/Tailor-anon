"""RAVEN-aligned eval part 2: regeneration attacks + resync post-processing.
Composite (VINE+TM) on SD-2.1 gen images. Attacks: geometry (WAVES rotation/resizedcrop) + regeneration
(Regen/Rinse2x/VAE-B/VAE-C). Detection reported BASELINE vs +RESYNC (angle-search + crypto-verify fallback)."""
import glob, io, os, sys, random, tempfile, numpy as np
from PIL import Image
import torch, torchvision.transforms as T, torchvision.transforms.functional as F
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense"); sys.path.insert(0,os.path.join(REPO,"external","WatermarkAttacker"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
RES=512; dev="cuda"; N=int(sys.argv[1]) if len(sys.argv)>1 else 50
sb=ShortenedBCH(); tau=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits")}
frag={"vine":VineCryptoWrapper(master_key=b"v5_key_encoder_master",method_name="vine",n_bits=sb.n,device=dev),
      "trustmark":TrustMarkFragment(master_key=b"v5_key_encoder_master",method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)}
pipe=build_regen_pipe()
from wmattacker import VAEWMAttacker
vae={}
for k,mn in [("vae_b","bmshj2018-hyperprior"),("vae_c","cheng2020-anchor")]:
    try: vae[k]=VAEWMAttacker(mn,quality=3,metric="mse",device=dev)
    except Exception as e: print("vae skip",k,flush=True)
tmp=tempfile.mkdtemp()
def rot(img,deg):
    a=np.array(img); pad=RES//2; big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),'reflect')).rotate(deg,resample=Image.BICUBIC); bw,bh=big.size; l=(bw-RES)//2; return big.crop((l,l,l+RES,l+RES))
def wattack(image,t,rel,seed):
    random.seed(seed); a,b=(0,45) if t=="rotation" else (1,0.5); s=rel*(b-a)+a
    if t=="rotation": return F.rotate(image,s)
    i,j,h,w=T.RandomResizedCrop.get_params(image,scale=(s,s),ratio=(1,1)); return F.resized_crop(image,i,j,h,w,[RES,RES])
def sapply(a,pil):
    ip=os.path.join(tmp,"i.png"); op=os.path.join(tmp,"o.png"); pil.save(ip); a.attack([ip],[op]); return Image.open(op).convert("RGB").resize((RES,RES))
def attack(name,W,seed):
    if name=="regen": return stable_regen(pipe,W,seed,denoise_steps=8)
    if name=="rinse2x":
        x=stable_regen(pipe,W,seed,denoise_steps=8); return stable_regen(pipe,x,seed+1,denoise_steps=8)
    if name in vae: return sapply(vae[name],W)
def decode(att,iid):
    al={}
    for nm in ["vine","trustmark"]:
        kind,getter=SPEC[nm]; p,M=frag[nm].get_perm_M(iid); al[nm]=method_soft_to_codeword_llr(getattr(frag[nm],getter)(att),p,M,kind=kind,n_codeword=sb.n)
    fused=fuse_llrs(al,weights=None,n_codeword=sb.n); return fused
def detect_base(att,iid,tx):
    f=decode(att,iid); return (float(np.mean(llr_to_bits(f)==tx))>=tau) or bool(decode_and_verify(f,iid,codec=sb)["detected"]), float(np.mean(llr_to_bits(f)==tx))
def detect_resync(att,iid,tx,base_ok):
    if base_ok: return True
    for d in np.arange(-30,30.01,3.0):   # angle search, crypto-verify selects (zero false-accept)
        f=decode(rot(att,d),iid)
        if bool(decode_and_verify(f,iid,codec=sb)["detected"]): return True
    return False
GEOM=[("rotation",0.2),("rotation",0.5),("resizedcrop",0.75),("resizedcrop",1.0)]; REGEN=["regen","rinse2x","vae_b","vae_c"]
files=(sorted(glob.glob(REPO+"/results/defense/raven_gen/img_*.png"))+sorted(glob.glob(REPO+"/results/defense/raven_gen2/img_*.png")))[:N]
R={}
for (a,r) in GEOM: R[f"{a}@{r}"]={"base":[],"resync":[],"fba":[]}
for a in REGEN: R[a]={"base":[],"resync":[],"fba":[]}
for idx,fp in enumerate(files):
    C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"rr_{idx:05d}"
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); img=C
    for nm in ["vine","trustmark"]:
        p,M=frag[nm].get_perm_M(iid); img=frag[nm].embed_with_target(img,apply_crypto(tx,p,M))
        if img.size!=(RES,RES): img=img.resize((RES,RES))
    for (a,r) in GEOM:
        At=wattack(img,a,r,1234+idx);  At=At if At.size==(RES,RES) else At.resize((RES,RES))
        ok,fba=detect_base(At,iid,tx); rs=detect_resync(At,iid,tx,ok)
        k=f"{a}@{r}"; R[k]["base"].append(1.0 if ok else 0.0); R[k]["resync"].append(1.0 if rs else 0.0); R[k]["fba"].append(fba)
    for a in REGEN:
        At=attack(a,img,1234+idx)
        ok,fba=detect_base(At,iid,tx); rs=detect_resync(At,iid,tx,ok)
        R[a]["base"].append(1.0 if ok else 0.0); R[a]["resync"].append(1.0 if rs else 0.0); R[a]["fba"].append(fba)
    print(f"[{idx+1}/{len(files)}]",flush=True)
print(f"\n=== RAVEN-aligned: regen attacks + resync (SD-2.1 imgs, n={len(files)}) ===")
print(f"{'attack':<16}{'baseline_or':>12}{'+resync_or':>12}{'fused_ba':>10}")
for k in [f"{a}@{r}" for (a,r) in GEOM]+REGEN:
    print(f"{k:<16}{np.mean(R[k]['base']):>12.3f}{np.mean(R[k]['resync']):>12.3f}{np.mean(R[k]['fba']):>10.3f}")
print("RESYNC_REGEN_DONE")
