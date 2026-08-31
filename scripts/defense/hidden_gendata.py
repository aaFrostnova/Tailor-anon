"""Cache VINE(1000)+TM(2048) HIDDEN states + frozen logits + per-image sigma/M + codeword,
on attacked composite images. -> results/defense/hidden_data.npz
Attacks: clean + regen (applied to fh3_embed) + external dirs (ctrlregen/vae) passed as DIR:label."""
import os,sys,glob,argparse
import numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.payload import image_id_to_payload
from hidden_extract import HiddenTap
from _regen_util import build_regen_pipe, stable_regen
ap=argparse.ArgumentParser(); ap.add_argument("--embed",required=True); ap.add_argument("--dirs",nargs="*",default=[])
ap.add_argument("--out",required=True); a=ap.parse_args()
KEY=b"v5_key_encoder_master"; dev="cuda"
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
tap=HiddenTap(vine,tm,n_bits=sb.n)
pipe=build_regen_pipe()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
HV,HT,SV,MV,ST,MT,PV,PT,TX,ATK,IMG=[],[],[],[],[],[],[],[],[],[],[]
def add(iid,i,att,label):
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv_perm,Mv=vine.get_perm_M(iid); pt_perm,Mt=tm.get_perm_M(iid)
    hv,ht,pv,pt=tap.extract(att)
    HV.append(hv.astype(np.float32)); HT.append(ht.astype(np.float32))
    SV.append(pv_perm.astype(np.int32)); MV.append(np.asarray(Mv,np.int8)); ST.append(pt_perm.astype(np.int32)); MT.append(np.asarray(Mt,np.int8))
    PV.append(pv.astype(np.float32)); PT.append(pt.astype(np.float32)); TX.append(tx.astype(np.uint8)); ATK.append(label); IMG.append(i)
embs=sorted(glob.glob(os.path.join(a.embed,"*.png")))
for fp in embs:
    i=int(os.path.basename(fp)[:5]); iid=f"fh3_{i:05d}"; comp=to512(Image.open(fp).convert("RGB"))
    add(iid,i,comp,"clean")
    add(iid,i,to512(stable_regen(pipe,comp,seed=5000+i)),"regen")
    if (i+1)%20==0: print(f"  embed pass {i+1}",flush=True)
for spec in a.dirs:
    dd,label=spec.rsplit(":",1)
    for fp in sorted(glob.glob(os.path.join(dd,"*.png"))):
        i=int(os.path.basename(fp)[:5]); iid=f"fh3_{i:05d}"
        add(iid,i,to512(Image.open(fp).convert("RGB")),label)
    print(f"  [{label}] done",flush=True)
np.savez_compressed(a.out,HV=np.array(HV),HT=np.array(HT),SV=np.array(SV),MV=np.array(MV),ST=np.array(ST),MT=np.array(MT),
                    PV=np.array(PV),PT=np.array(PT),TX=np.array(TX),ATK=np.array(ATK),IMG=np.array(IMG),n=sb.n)
print(f"[saved] {len(HV)} samples -> {a.out}\nHGEN_DONE")
