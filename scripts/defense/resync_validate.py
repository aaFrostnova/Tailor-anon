"""Clean end-to-end resync validation (the last unvalidated solver assumption).
Fresh composite embed (known-good) -> add SyncSeal sync mark -> geo attack -> SyncSeal detect+unwarp -> decode.
Confirms resync recovers rot/crop detection to ~0.96 (vs ~0 baseline), backing the solver's resync boost."""
import os,sys,glob,numpy as np,torch
from PIL import Image
import torchvision.transforms.functional as TF
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
KEY=b"v5_key_encoder_master"; dev="cuda"; RES=512; TAU=0.63
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def sr(c,w,a):
    c=np.asarray(to(c),np.float64); w=np.asarray(to(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def align1d(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
sync=load_sync()
def crypto_ok(hard,pay):
    try: return int(np.array_equal(sb.decode(hard.astype(np.uint8))[:sb.data_bits],pay[:sb.data_bits]))
    except Exception: return 0
def decode_det(pil,pv_p,Mv,pt_p,Mt,ps_p,Ms,tx,pay):
    LV=align1d(np.clip(p2l(vine.raw_probs(pil)),-15,15),pv_p,Mv)
    LT=align1d(np.clip(tm.raw_logits(pil),-15,15),pt_p,Mt)
    LS=method_soft_to_codeword_llr(vs.raw_logits(pil),ps_p,Ms,kind="logit",n_codeword=sb.n)
    bv=((LV>0).astype(int)==tx).mean();bt=((LT>0).astype(int)==tx).mean();bs=((LS>0).astype(int)==tx).mean()
    best=max(bv,bt,bs)
    det=int(best>=TAU or crypto_ok((LV>0).astype(int),pay) or crypto_ok((LT>0).astype(int),pay) or crypto_ok((LS>0).astype(int),pay))
    return best,det
def pt(pil): return TF.to_tensor(to(pil)).unsqueeze(0).to(dev)
def pil(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy())
ATT={"rot9":lambda t:TF.rotate(t,9.0),"rot30":lambda t:TF.rotate(t,30.0),
     "crop75":lambda t:TF.resize(t[:,:,64:64+384,64:64+384],[RES,RES],antialias=True)}
N=int(sys.argv[1]) if len(sys.argv)>1 else 16
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[2000:2000+N]
R={a:{"base":[],"sync":[],"best_b":[],"best_s":[]} for a in ATT}
for i,fp in enumerate(files):
    iid=f"rs_{i:05d}"; pay=image_id_to_payload(iid,n_bits=sb.data_bits); tx=sb.encode(pay)
    orig=to(Image.open(fp).convert("RGB"))
    pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
    a1=sr(orig,to(vs.embed_with_target(orig,apply_crypto(tx,ps_p,Ms))),0.70)
    a2=sr(a1,to(tm.embed_with_target(a1,apply_crypto(tx,pt_p,Mt))),0.70)
    comp=sr(a2,to(vine.embed_with_target(a2,apply_crypto(tx,pv_p,Mv))),1.00)
    Ws=pil(sync.embed(pt(comp))["imgs_w"])            # add sync mark on top of composite
    for a,fn in ATT.items():
        A=pil(fn(pt(Ws)))                             # geo attack
        d=sync.detect(pt(A)); Rimg=pil(sync.unwarp(pt(A),d["preds_pts"],(RES,RES)))   # detect corners + unwarp
        bb,db=decode_det(A,pv_p,Mv,pt_p,Mt,ps_p,Ms,tx,pay)          # baseline (no resync)
        bs,ds=decode_det(Rimg,pv_p,Mv,pt_p,Mt,ps_p,Ms,tx,pay)      # after resync
        R[a]["base"].append(db);R[a]["sync"].append(ds);R[a]["best_b"].append(bb);R[a]["best_s"].append(bs)
    if (i+1)%8==0: print(f"  [{i+1}/{N}]",flush=True)
print(f"\n{'attack':7s} | base_det -> sync_det | best-single ba: base -> resync")
for a in ATT:
    print(f"{a:7s} |  {np.mean(R[a]['base']):.3f}   ->  {np.mean(R[a]['sync']):.3f}   | {np.mean(R[a]['best_b']):.3f} -> {np.mean(R[a]['best_s']):.3f}")
print("RESYNC_VAL_DONE")
