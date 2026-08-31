"""CALIBRATION: how far off can the decode scale be before a nested VINE layer stops decoding?
Sweep relative offset d around the ORACLE scale of each layer; report ba(d) and verify(d).
The 'capture width' (widest |d| where verify still fires) sets the required blind grid step:
   grid_step <= capture_width  (so any oracle lands within capture of some grid point).
Two sets: CLEAN embeds (layers at exactly K) and UnMarker-attacked (layers at K*512/460)."""
import os,sys,json,numpy as np
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
def dec(pil,f,iid,tx):
    if f>=0.999: view=to512(pil)
    else:
        s=int(round(512*f)); o=(512-s)//2; view=to512(pil).crop((o,o,o+s,o+s))
    p,M=V.get_perm_M(iid)
    rl=np.clip(method_soft_to_codeword_llr(V.raw_probs(view),p,M,kind="prob",n_codeword=n),-CLAMP,CLAMP)
    ba=float(((rl>0).astype(np.uint8)==tx).mean())
    return ba, (bool(decode_and_verify(rl,iid,codec=sb)["detected"]) or ba>=tau)
ZOOM=512/460.0
DELTAS=np.round(np.arange(-0.020,0.02001,0.001),4)     # relative offset, -2%..+2% in 0.1% steps
SETS={"clean":        (f"{SC}/p3embed_D",  {0.75:0.75, 0.5:0.5}),
      "unmarker":     (f"{SC}/p3um_D",     {0.75:0.75*ZOOM, 0.5:0.5*ZOOM})}
metaD={m["fname"]:m["iid"] for m in json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]}
metaU={m["fname"]:m["iid"] for m in json.load(open(f"{SC}/p3embed_Dum/meta.json"))["items"]}
res={}
for sname,(d,oracles) in SETS.items():
    meta = metaD if sname=="clean" else metaU
    files=[(iid,f"{d}/{fn}") for fn,iid in meta.items() if os.path.exists(f"{d}/{fn}")][:12]
    for K,orc in oracles.items():
        ba_c=[];vf_c=[]
        for dd in DELTAS:
            f=orc*(1+dd); bas=[];vfs=[]
            for iid,path in files:
                tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
                b,v=dec(to512(Image.open(path).convert("RGB")),float(f),iid,tx); bas.append(b); vfs.append(1.0 if v else 0.0)
            ba_c.append(float(np.mean(bas))); vf_c.append(float(np.mean(vfs)))
        ok=[abs(float(DELTAS[i])) for i in range(len(DELTAS)) if vf_c[i]>=0.9]
        cap=max(ok) if ok else 0.0
        res[f"{sname}_K{K}"]={"oracle":round(orc,4),"deltas":DELTAS.tolist(),"ba":ba_c,"verify":vf_c,
                              "capture_halfwidth_rel":round(cap,4)}
        print(f"{sname:9} K={K}: oracle={orc:.4f}  capture |d|<= {cap*100:.2f}%  "
              f"(ba at d=0: {ba_c[len(ba_c)//2]:.3f}; at +0.5%: {ba_c[min(len(ba_c)-1,len(ba_c)//2+5)]:.3f}; "
              f"at +1.0%: {ba_c[min(len(ba_c)-1,len(ba_c)//2+10)]:.3f})",flush=True)
json.dump(res,open("results/defense/scale_tolerance.json","w"),indent=2)
caps=[v["capture_halfwidth_rel"] for v in res.values()]
print(f"MIN capture half-width across layers/sets = {min(caps)*100:.2f}%  -> recommended blind grid step <= {min(caps)*100:.2f}% (relative)")
print("SCALE_TOLERANCE_DONE")
