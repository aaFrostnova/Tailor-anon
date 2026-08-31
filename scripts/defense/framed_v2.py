"""Scheme A v2: framed payload (magic-BCH + id-BCH, UEP) + soft-chase + formalized resync detect().
magic: BCH(m=6,t=5) 16-bit -> 48 tx; id: BCH(m=6,t=4) 24-bit -> 48 tx; +4 pad = 100.
Blind detection (no image_id): chase-decode magic seg, compare HMAC magic. ID: chase-decode id seg.
resync formalized into FramedDetector.detect(resync=True): angle-search + magic-verify (zero false-accept)."""
import bchlib, hmac, hashlib, io, glob, os, sys, numpy as np
from itertools import combinations
from PIL import Image
import torchvision.transforms.functional as TF
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense")
RES=512; MKEY=b"v5_key_encoder_master"

class SegBCH:
    def __init__(self,m,t,data_bits):
        self.bch=bchlib.BCH(t=t,m=m); self.t=t; self.data_bits=data_bits; self.data_bytes=data_bits//8
        self.ecc_bytes=len(self.bch.encode(bytes(self.data_bytes))); self.ecc_bits=self.ecc_bytes*8
        self.tx=self.data_bits+self.ecc_bits
    def encode(self,d):
        ecc=self.bch.encode(bytes(np.packbits(d.astype(np.uint8)).tobytes()))
        return np.concatenate([d.astype(np.uint8), np.unpackbits(np.frombuffer(ecc,np.uint8))])
    def _hard(self,bits):
        d=bytearray(np.packbits(bits[:self.data_bits].astype(np.uint8)).tobytes())
        e=bytearray(np.packbits(bits[self.data_bits:self.tx].astype(np.uint8)).tobytes())
        res=self.bch.decode(d,e)
        if isinstance(res,tuple): nerr,d,e=res
        else: nerr=res; self.bch.correct(d,e)
        return None if nerr<0 else np.unpackbits(np.frombuffer(bytes(d),np.uint8))[:self.data_bits]
    def chase(self,llr,n_flip=10):
        hard=(llr>0).astype(np.uint8); d=self._hard(hard)
        if d is not None: return d
        order=np.argsort(np.abs(llr[:self.tx]))
        for r in (1,2,3):
            for c in combinations(order[:n_flip],r):
                h=hard.copy(); h[list(c)]^=1; d=self._hard(h)
                if d is not None: return d
        return None

MAGIC_SB=SegBCH(6,5,16); ID_SB=SegBCH(6,4,24); NC=100
MAGIC=np.unpackbits(np.frombuffer(hmac.new(MKEY,b"magic",hashlib.sha256).digest()[:2],np.uint8))
def id_bits(image_id): 
    v=int(hashlib.sha256(image_id.encode()).hexdigest(),16)%(1<<24); return np.array([(v>>i)&1 for i in range(24)],np.uint8)
def encode_cw(image_id):
    cw=np.concatenate([MAGIC_SB.encode(MAGIC), ID_SB.encode(id_bits(image_id)), np.zeros(NC-MAGIC_SB.tx-ID_SB.tx,np.uint8)])
    assert len(cw)==NC, len(cw); return cw

def _selftest():
    print(f"[dims] magic tx={MAGIC_SB.tx}(data16,ecc{MAGIC_SB.ecc_bits},t5) id tx={ID_SB.tx}(data24,ecc{ID_SB.ecc_bits},t4) total={MAGIC_SB.tx+ID_SB.tx}+pad{NC-MAGIC_SB.tx-ID_SB.tx}",flush=True)
    cw=encode_cw("test_img"); 
    for ne in [0,4,5,8]:
        rng=np.random.RandomState(ne); llr=np.where(cw>0,4.0,-4.0)
        flip=rng.choice(NC,ne,replace=False); llr[flip]*=-1; llr[flip]*=0.3  # flipped + low-conf
        m=MAGIC_SB.chase(llr[:MAGIC_SB.tx]); i=ID_SB.chase(llr[MAGIC_SB.tx:MAGIC_SB.tx+ID_SB.tx])
        mok=m is not None and np.array_equal(m,MAGIC); iok=i is not None and np.array_equal(i,id_bits("test_img"))
        print(f"  selftest flips={ne}: magic_ok={mok} id_ok={iok}",flush=True)
    print("SELFTEST_DONE",flush=True)

if __name__=="__main__" and len(sys.argv)>1 and sys.argv[1]=="selftest":
    _selftest(); sys.exit()

# ---- detector (needs GPU fragments) ----
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, derive_method_keyed_constants
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from _regen_util import build_regen_pipe, stable_regen
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits")}
class FramedDetector:
    def __init__(self,dev="cuda"):
        self.frag={"vine":VineCryptoWrapper(master_key=MKEY,method_name="vine",n_bits=NC,device=dev),
                   "trustmark":TrustMarkFragment(master_key=MKEY,method_name="trustmark",n_bits=NC,model_type="B",device=dev)}
        self.PM={nm:derive_method_keyed_constants(MKEY,"GLOBAL",nm,NC) for nm in self.frag}
    def embed(self,pil,image_id):
        cw=encode_cw(image_id); img=pil
        for nm in ["vine","trustmark"]:
            p,M=self.PM[nm]; img=self.frag[nm].embed_with_target(img,apply_crypto(cw,p,M)); img=img if img.size==(RES,RES) else img.resize((RES,RES))
        return img
    def _llr(self,att):
        al={}
        for nm,(kind,getter) in SPEC.items():
            p,M=self.PM[nm]; al[nm]=method_soft_to_codeword_llr(getattr(self.frag[nm],getter)(att),p,M,kind=kind,n_codeword=NC)
        return fuse_llrs(al,None,NC)
    def _parse(self,llr):
        m=MAGIC_SB.chase(llr[:MAGIC_SB.tx]); present=m is not None and np.array_equal(m,MAGIC)
        idv=None
        if present: idv=ID_SB.chase(llr[MAGIC_SB.tx:MAGIC_SB.tx+ID_SB.tx])
        return present,idv
    def detect(self,pil,resync=True,angles=None):
        if angles is None: angles=np.arange(-30,30.01,3.0)
        present,idv=self._parse(self._llr(pil))
        if present: return {"present":True,"id":idv,"angle":0.0}
        if resync:
            for d in angles:
                a=np.array(pil); pad=RES//2; big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),'reflect')).rotate(d,resample=Image.BICUBIC)
                l=(big.width-RES)//2; rp=big.crop((l,l,l+RES,l+RES))
                pr,iv=self._parse(self._llr(rp))
                if pr: return {"present":True,"id":iv,"angle":float(d)}
        return {"present":False,"id":None,"angle":None}
