"""LARGE-scale hidden+LLR fusion data. Embed composite VideoSeal(.7)->TM(.7)->VINE(1.0) [VINE-last,
full strength] on N UltraEdit imgs, apply cheap attacks + regen, cache VINE(1000)+TM(2048) hidden
+ all 3 frozen LLRs + sigma/M + codeword. -> big npz for overfitting-free fusion-head training."""
import os,sys,glob,io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts/defense")); sys.path.insert(0,os.path.join(REPO,"external/videoseal"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from hidden_extract import HiddenTap
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"
N=int(sys.argv[1]) if len(sys.argv)>1 else 3000
OUT=sys.argv[2] if len(sys.argv)>2 else "/scratch/workspace/mingzhel_umass_edu-ablator/hidden_big.npz"
imgs=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
tap=HiddenTap(vine,tm,n_bits=sb.n)
pipe=build_regen_pipe()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def sr(c,w,a):
    c=np.asarray(to512(c),np.float64); w=np.asarray(to512(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def att(name,im,r,sd):
    if name=="clean": return im
    if name=="jpeg": q=int(round(90-80*r)); b=io.BytesIO(); im.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(0.5+7.5*r))
    if name=="noise": a=np.asarray(im,np.float64)+np.random.RandomState(sd).normal(0,(0.02+0.08*r)*255,(512,512,3)); return Image.fromarray(np.clip(a,0,255).astype(np.uint8))
    if name=="crop": s=int(512*(0.90-0.15*r)); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
    if name=="regen": return stable_regen(pipe,im,seed=sd)
ATTS=["clean","jpeg","blur","noise","crop","regen"]
HV,HT,SV,MV,ST,MT,PV,PT,AS,TX,ATK,IMG=[],[],[],[],[],[],[],[],[],[],[],[]
for i,fp in enumerate(imgs):
    iid=f"big_{i:06d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    orig=to512(Image.open(fp).convert("RGB"))
    pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
    a1=sr(orig,to512(vs.embed_with_target(orig,apply_crypto(tx,ps_p,Ms))),0.70)
    a2=sr(a1,to512(tm.embed_with_target(a1,apply_crypto(tx,pt_p,Mt))),0.70)
    comp=sr(a2,to512(vine.embed_with_target(a2,apply_crypto(tx,pv_p,Mv))),1.00)   # VINE last, full
    rng=np.random.RandomState(7000+i)
    for k in ATTS:
        a=to512(att(k,comp,float(rng.uniform(0.3,1.0)),9000+i))
        hv,ht,pv,pt=tap.extract(a)                      # VINE prob, TM logit, + hiddens
        as_=method_soft_to_codeword_llr(vs.raw_logits(a),ps_p,Ms,kind="logit",n_codeword=sb.n)  # VideoSeal cw LLR
        HV.append(hv.astype(np.float32)); HT.append(ht.astype(np.float32))
        SV.append(pv_p.astype(np.int32)); MV.append(np.asarray(Mv,np.int8)); ST.append(pt_p.astype(np.int32)); MT.append(np.asarray(Mt,np.int8))
        PV.append(pv.astype(np.float32)); PT.append(pt.astype(np.float32)); AS.append(as_.astype(np.float32))
        TX.append(tx.astype(np.uint8)); ATK.append(k); IMG.append(i)
    if (i+1)%100==0: print(f"  [{i+1}/{len(imgs)}] samples={len(HV)}",flush=True)
np.savez_compressed(OUT,HV=np.array(HV),HT=np.array(HT),SV=np.array(SV),MV=np.array(MV),ST=np.array(ST),MT=np.array(MT),
                    PV=np.array(PV),PT=np.array(PT),AS=np.array(AS),TX=np.array(TX),ATK=np.array(ATK),IMG=np.array(IMG),n=sb.n)
print(f"[saved] {len(HV)} samples -> {OUT}\nBIG_DONE")
