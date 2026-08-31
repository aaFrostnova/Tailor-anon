"""Extend the capacity cache to the 3 MISSING attacks: rot9, rot30, vaeB, vaeC, rinse.
Same composite embed (VideoSeal.7->TM.7->VINE1.0) + same cache format as hidden_gendata_big.py,
so capacity_combos.py can concat hidden_big.npz (6 attacks) + this (5 new) -> full table.
N=1000 default (MI stabilizes; rinse=2x diffusion is the cost driver)."""
import os,sys,glob,numpy as np,torch
from PIL import Image
import torchvision.transforms.functional as TF
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts/defense")); sys.path.insert(0,os.path.join(REPO,"external/videoseal"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
KEY=b"v5_key_encoder_master"; dev="cuda"
N=int(sys.argv[1]) if len(sys.argv)>1 else 1000
OUT=sys.argv[2] if len(sys.argv)>2 else "/scratch/workspace/mingzhel_umass_edu-ablator/hidden_ext.npz"
imgs=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev)
VAEC=cheng2020_anchor(quality=3,pretrained=True).eval().to(dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def sr(c,w,a):
    c=np.asarray(to512(c),np.float64); w=np.asarray(to512(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def _p(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy())
def _t(im): return TF.to_tensor(to512(im)).unsqueeze(0)
def vae_att(model,im):
    with torch.no_grad(): return _p(model(_t(im).to(dev))["x_hat"].clamp(0,1).cpu())
def att(name,im,sd):
    if name=="rot9":  return _p(TF.rotate(_t(im),9.0))
    if name=="rot30": return _p(TF.rotate(_t(im),30.0))
    if name=="vaeB":  return vae_att(VAEB,im)
    if name=="vaeC":  return vae_att(VAEC,im)
    if name=="rinse": r1=stable_regen(pipe,to512(im),seed=sd); return stable_regen(pipe,to512(r1),seed=sd+1)
ATTS=["rot9","rot30","vaeB","vaeC","rinse"]
SV,MV,ST,MT,PV,PT,AS,TX,ATK,IMG=[],[],[],[],[],[],[],[],[],[]
for i,fp in enumerate(imgs):
    iid=f"big_{i:06d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    orig=to512(Image.open(fp).convert("RGB"))
    pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
    a1=sr(orig,to512(vs.embed_with_target(orig,apply_crypto(tx,ps_p,Ms))),0.70)
    a2=sr(a1,to512(tm.embed_with_target(a1,apply_crypto(tx,pt_p,Mt))),0.70)
    comp=sr(a2,to512(vine.embed_with_target(a2,apply_crypto(tx,pv_p,Mv))),1.00)
    for k in ATTS:
        a=to512(att(k,comp,9000+i))
        pv=vine.raw_probs(a); pt=tm.raw_logits(a)                                   # VINE prob (raw), TM logit (raw)
        as_=method_soft_to_codeword_llr(vs.raw_logits(a),ps_p,Ms,kind="logit",n_codeword=sb.n)  # VS cw LLR (aligned)
        SV.append(pv_p.astype(np.int32)); MV.append(np.asarray(Mv,np.int8)); ST.append(pt_p.astype(np.int32)); MT.append(np.asarray(Mt,np.int8))
        PV.append(np.asarray(pv,np.float32)); PT.append(np.asarray(pt,np.float32)); AS.append(np.asarray(as_,np.float32))
        TX.append(tx.astype(np.uint8)); ATK.append(k); IMG.append(i)
    if (i+1)%50==0: print(f"  [{i+1}/{len(imgs)}] samples={len(PV)}",flush=True)
np.savez_compressed(OUT,SV=np.array(SV),MV=np.array(MV),ST=np.array(ST),MT=np.array(MT),
                    PV=np.array(PV),PT=np.array(PT),AS=np.array(AS),TX=np.array(TX),ATK=np.array(ATK),IMG=np.array(IMG),n=sb.n)
print(f"[saved] {len(PV)} samples ({len(ATTS)} attacks x {N} imgs) -> {OUT}\nEXT_DONE")
