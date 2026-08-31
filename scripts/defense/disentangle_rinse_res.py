"""Disentangle the VINE rinse ba gap (solo256 0.82 vs composite512 0.75): resolution vs stacking.
Measure VINE-solo rinse at a 256 pipeline and a 512 pipeline (everything at that res), same imgs/seeds.
solo256 vs solo512 = pure RESOLUTION effect; solo512 vs composite512 = pure STACKING effect."""
import os,sys,glob,numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts/defense")); sys.path.insert(0,os.path.join(REPO,"external/videoseal"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"
N=int(sys.argv[1]) if len(sys.argv)>1 else 30
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def R(im,res): return im.resize((res,res)) if im.size!=(res,res) else im
def sr(c,w,a,res):
    C=np.asarray(R(c,res),np.float64); W=np.asarray(R(w,res),np.float64); return Image.fromarray(np.clip(C+a*(W-C),0,255).astype(np.uint8))
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def al(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]
def vine_ba(img,iid,tx):
    pv,Mv=vine.get_perm_M(iid); F=al(np.clip(p2l(vine.raw_probs(img)),-15,15),pv,Mv); return float(((F>0).astype(int)==tx).mean())
def rinse(im,res,k):
    x=R(im,res)
    for j in range(2): x=R(stable_regen(pipe,x,seed=1234+k*10+j),res)
    return x
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
S={"solo256":[],"solo512":[],"comp512":[]}
for i,fp in enumerate(files):
    iid=f"dz_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vs.get_perm_M(iid)
    orig=Image.open(fp).convert("RGB")
    for res,key in [(256,"solo256"),(512,"solo512")]:
        o=R(orig,res); wm=sr(o,R(vine.embed_with_target(o,apply_crypto(tx,pv,Mv)),res),1.0,res)
        S[key].append(vine_ba(rinse(wm,res,i),iid,tx))
    # composite @512: VideoSeal->TM->VINE
    o=R(orig,512)
    a1=sr(o,R(vs.embed_with_target(o,apply_crypto(tx,ps,Ms)),512),0.70,512)
    a2=sr(a1,R(tm.embed_with_target(a1,apply_crypto(tx,pt,Mt)),512),0.70,512)
    comp=sr(a2,R(vine.embed_with_target(a2,apply_crypto(tx,pv,Mv)),512),1.00,512)
    S["comp512"].append(vine_ba(rinse(comp,512,i),iid,tx))
    if (i+1)%10==0: print(f"  [{i+1}/{N}]",flush=True)
def m(x): return float(np.mean(x))
print(f"\n=== VINE rinse bit-acc, disentangled (n={N}) ===")
print(f"  solo @256 pipeline : {m(S['solo256']):.3f}")
print(f"  solo @512 pipeline : {m(S['solo512']):.3f}   (Δ vs 256 = RESOLUTION effect {m(S['solo512'])-m(S['solo256']):+.3f})")
print(f"  composite @512     : {m(S['comp512']):.3f}   (Δ vs solo512 = STACKING effect {m(S['comp512'])-m(S['solo512']):+.3f})")
print("DZ_DONE")
