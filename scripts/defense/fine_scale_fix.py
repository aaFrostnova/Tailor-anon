"""END-TO-END FIX TEST: does a FINE scale grid recover the nested layers under full UnMarker?
Oracle probe showed the layers are intact (ba~0.98, verify 1.00) but our 0.02-0.03 grid misses them
(<1% scale tolerance). Test blind fine grids: step 0.02 (old) / 0.01 / 0.005 / 0.0025, crypto-verify gated."""
import os,sys,glob,json,numpy as np
from PIL import Image
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0,"scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.soft_fusion import method_soft_to_codeword_llr
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
sb=ShortenedBCH(); n=sb.n; tau=float(binom.ppf(0.99,n,0.5)+1)/n
SC="/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
V=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def view(pil,f):
    if f>=0.999: return to512(pil)
    s=int(round(512*f)); o=(512-s)//2; return to512(pil).crop((o,o,o+s,o+s))
def verify_at(pil,f,iid,tx):
    p,M=V.get_perm_M(iid)
    rl=np.clip(method_soft_to_codeword_llr(V.raw_probs(view(pil,f)),p,M,kind="prob",n_codeword=n),-CLAMP,CLAMP)
    ba=float(((rl>0).astype(np.uint8)==tx).mean())
    return bool(decode_and_verify(rl,iid,codec=sb)["detected"]) or ba>=tau
def blind_search(pil,iid,tx,step):
    grid=np.arange(0.34,1.0001,step)
    for f in grid:
        if verify_at(pil,float(f),iid,tx): return 1.0,len(grid)
    return 0.0,len(grid)
SETS={}
meta=json.load(open(f"{SC}/p3embed_Dum/meta.json"))["items"]
SETS["full_UnMarker"]=[(m["iid"],f"{SC}/p3um_D/{m['fname']}") for m in meta if os.path.exists(f"{SC}/p3um_D/{m['fname']}")][:15]
mD=json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]
SETS["crop09_only"]=[(m["iid"],f"{SC}/crop09_nested/{m['fname']}") for m in mD if os.path.exists(f"{SC}/crop09_nested/{m['fname']}")][:15]
SETS["spectral_only"]=[(m["iid"],f"{SC}/p3um_nested_nocrop/{m['fname']}") for m in mD if os.path.exists(f"{SC}/p3um_nested_nocrop/{m['fname']}")][:15]
res={}
for sname,items in SETS.items():
    res[sname]={}
    for step in (0.02,0.01,0.005,0.0025):
        d=[];npts=0
        for iid,path in items:
            tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
            r,npts=blind_search(to512(Image.open(path).convert("RGB")),iid,tx,step); d.append(r)
        res[sname][f"step{step}"]={"detect":round(float(np.mean(d)),3),"grid_points":int(npts)}
        print(f"{sname:16} step={step:<7} grid={npts:4d} pts  ->  VINE-only detect = {np.mean(d):.3f}",flush=True)
json.dump(res,open("results/defense/fine_scale_fix.json","w"),indent=2)
print("FINE_SCALE_FIX_DONE")
