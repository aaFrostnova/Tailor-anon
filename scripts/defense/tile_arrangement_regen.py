"""Test: does STAGGERED (Olympic-ring interleave) tile arrangement survive regen better than ALIGNED grid?
VINE energy = border ring. Aligned grid => rings pile at shared borders (hotspots + gaps).
Staggered brick => each tile's ring falls over neighbors' low-energy centers => uniform ring coverage.
Embed native-256 VINE tiles (full strength, no taper) -> regen -> sliding-window crypto-verify decode.
Report detection rate + best bit-acc + PSNR for both arrangements (matched per-tile amplitude)."""
import os,sys,glob,numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_bch import decode_and_verify
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"; C=512; T=256; S=128
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def aligned():   return [(x,y) for y in range(0,C-T+1,S) for x in range(0,C-T+1,S)]
def staggered():
    pos=[]
    for i,y in enumerate(range(0,C-T+1,S)):
        off=(S//2) if (i%2) else 0
        pos += [(min(max(x,0),C-T),y) for x in range(off,C-T+1,S)]
        if off:  # fill left/right edge gaps created by the offset
            pos += [(0,y),(C-T,y)]
    return sorted(set(pos))
def embed_tiled(orig_arr, positions, tgt):
    resid=np.zeros((C,C,3))
    for (x,y) in positions:
        patch=Image.fromarray(orig_arr[y:y+T,x:x+T].astype(np.uint8))
        wm=vine.embed_with_target(patch,tgt)
        r=np.asarray(wm.resize((T,T)),np.float64)-np.asarray(patch,np.float64)
        resid[y:y+T,x:x+T]+=r                         # plain add (no taper -> full ring); hotspots show up in PSNR
    out=np.clip(orig_arr+resid,0,255)
    psnr=10*np.log10(255**2/max(np.mean((out-orig_arr)**2),1e-9))
    return Image.fromarray(out.astype(np.uint8)),psnr
def survival(canvas,iid,tx,pv,Mv,stride=128):
    a=canvas.resize((C,C)); det=0; best=0.0
    for y in range(0,C-T+1,stride):
        for x in range(0,C-T+1,stride):
            win=a.crop((x,y,x+T,y+T))
            llr=method_soft_to_codeword_llr(vine.raw_probs(win),pv,Mv,kind="prob",n_codeword=sb.n)
            best=max(best,float(((llr>0).astype(int)==tx).mean()))
            if decode_and_verify(llr,iid,codec=sb)["detected"]: det=1
    return det,best
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
ARR={"aligned":aligned(),"staggered":staggered()}
print("tiles:",{k:len(v) for k,v in ARR.items()},flush=True)
R={k:{"det":[],"ba":[],"psnr":[]} for k in ARR}
for i,fp in enumerate(files):
    iid=f"ta_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); tgt=apply_crypto(tx,pv,Mv)
    orig=np.asarray(Image.open(fp).convert("RGB").resize((C,C)),np.float64)
    for k,pos in ARR.items():
        wm,ps=embed_tiled(orig,pos,tgt)
        rg=stable_regen(pipe,wm,seed=1234+i)
        det,ba=survival(rg,iid,tx,pv,Mv)
        R[k]["det"].append(det); R[k]["ba"].append(ba); R[k]["psnr"].append(ps)
    if (i+1)%5==0: print(f"  [{i+1}/{N}]",flush=True)
def m(x): return float(np.mean(x))
print(f"\n=== tile arrangement under REGEN (n={N}, C={C}, T={T}) ===")
print(f"{'arrangement':12s} {'#tiles':>7s} {'PSNR':>7s} {'regen det':>10s} {'regen ba':>9s}")
for k in ARR:
    print(f"{k:12s} {len(ARR[k]):7d} {m(R[k]['psnr']):7.2f} {m(R[k]['det']):10.3f} {m(R[k]['ba']):9.3f}")
print("ARR_DONE")
