"""Definitively measure VINE-R on rinse (double regen), SOLO vs IN-COMPOSITE — the L3 ceiling / SMT UNSAT driver.
Was previously ESTIMATED (0.73) in the composite; measure it. Also regen(single) + rinse x3 for the curve."""
import os,sys,glob,numpy as np
from PIL import Image
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"; RES=512
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def sr(c,w,a): c=np.asarray(to(c),np.float64); w=np.asarray(to(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def al(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def regen_n(im,k,n):
    x=to(im)
    for j in range(n): x=to(stable_regen(pipe,x,1234+k*10+j))
    return x
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
S={"solo":{1:[],2:[],3:[]},"comp":{1:[],2:[],3:[]}}
for i,fp in enumerate(files):
    iid=f"rn_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    orig=to(Image.open(fp).convert("RGB"))
    pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
    solo=sr(orig,to(vine.embed_with_target(orig,apply_crypto(tx,pv_p,Mv))),1.00)
    a1=sr(orig,to(vs.embed_with_target(orig,apply_crypto(tx,ps_p,Ms))),0.70)
    a2=sr(a1,to(tm.embed_with_target(a1,apply_crypto(tx,pt_p,Mt))),0.70)
    comp=sr(a2,to(vine.embed_with_target(a2,apply_crypto(tx,pv_p,Mv))),1.00)
    for n in (1,2,3):
        for tag,im in (("solo",solo),("comp",comp)):
            x=regen_n(im,i,n)
            FV=al(np.clip(p2l(vine.raw_probs(x)),-15,15),pv_p,Mv)
            S[tag][n].append(float(((FV>0).astype(int)==tx).mean()))
    if (i+1)%8==0: print(f"  [{i+1}/{N}]",flush=True)
def m(x): return float(np.mean(x))
print(f"\nVINE-R bit-acc vs number of regen rounds (n={N}):")
print(f"{'':6s} {'regen x1':>9s} {'rinse x2':>9s} {'x3':>9s}")
print(f"{'solo':6s} {m(S['solo'][1]):9.3f} {m(S['solo'][2]):9.3f} {m(S['solo'][3]):9.3f}")
print(f"{'comp':6s} {m(S['comp'][1]):9.3f} {m(S['comp'][2]):9.3f} {m(S['comp'][3]):9.3f}")
print("RINSE_DONE")
