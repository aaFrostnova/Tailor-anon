"""Test nested-ring VINE under attack ORDER: crop->regen (known-good) vs regen->crop (the open corner).
Concern: regen first kills the high-freq inner rings before a crop can zoom them to low-freq; then the
crop removes the low-freq outer ring -> nothing left. Decode = blind central-scale search + crypto-verify."""
import os,sys,glob,numpy as np
from PIL import Image
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_bch import decode_and_verify
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
from composite_external_eval import nested_vine_embed
KEY=b"v5_key_encoder_master"; dev="cuda"
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
KEEP=float(sys.argv[2]) if len(sys.argv)>2 else 0.75   # crop keeps central KEEP, resizes back to 512
sb=ShortenedBCH(); vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def R512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def crop_zoom(im,keep): s=int(round(512*keep)); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def regen(im,i): return stable_regen(pipe,R512(im),seed=1234+i)
def scale_search(att,iid,pv,Mv,tx,step=0.005):
    det=0; best=0.0
    for f in np.arange(0.34,1.0001,step):
        if f>=0.999: view=att
        else: s=int(round(512*f)); o=(512-s)//2; view=att.crop((o,o,o+s,o+s))
        llr=method_soft_to_codeword_llr(vine.raw_probs(view),pv,Mv,kind="prob",n_codeword=sb.n)
        best=max(best,float(((llr>0).astype(int)==tx).mean()))
        if bool(decode_and_verify(llr,iid,codec=sb)["detected"]): det=1
    return det,best
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
COND=["clean","regen","crop","crop_then_regen","regen_then_crop"]
Rz={c:{"det":[],"ba":[]} for c in COND}
for i,fp in enumerate(files):
    iid=f"no_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); tgt=apply_crypto(tx,pv,Mv)
    cover=R512(Image.open(fp).convert("RGB"))
    wm=nested_vine_embed(vine,cover,tgt,scales=(1.0,0.75,0.5))
    atk={"clean":wm,"regen":regen(wm,i),"crop":crop_zoom(wm,KEEP),
         "crop_then_regen":regen(crop_zoom(wm,KEEP),i),
         "regen_then_crop":crop_zoom(regen(wm,i),KEEP)}
    for c in COND:
        d,b=scale_search(atk[c],iid,pv,Mv,tx); Rz[c]["det"].append(d); Rz[c]["ba"].append(b)
    if (i+1)%5==0: print(f"  [{i+1}/{N}]",flush=True)
m=lambda x:float(np.mean(x))
print(f"\n=== nested-ring VINE by attack ORDER (n={N}, crop keeps {KEEP}) ===")
print(f"{'condition':18s} {'det':>6s} {'best ba':>8s}")
for c in COND: print(f"{c:18s} {m(Rz[c]['det']):6.3f} {m(Rz[c]['ba']):8.3f}")
print("NESTED_ORDER_DONE")
