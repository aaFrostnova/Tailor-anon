"""Hybrid + detection-first sweep: magic=soft-rep (strong blind detect, TOL) + id=BCH (strong recovery).
Configs:  H1 magic16x3 + id-BCHt4 ;  H2 magic16x4(64)+id-BCHt2(40) [detection-first].
Reuses framed_v2.SegBCH/FramedDetector machinery; only codeword layout + magic parse change."""
import io, glob, os, sys, numpy as np
from PIL import Image
sys.path.insert(0,os.path.join("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint","scripts/defense"))
import framed_v2 as f2
from framed_v2 import SegBCH
from src.vine_crypto_wrapper import apply_crypto
from _regen_util import build_regen_pipe, stable_regen
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; RES=512; N=40
MAGIC=f2.MAGIC
CONFIGS={
 "DF_rep4_bcht1":dict(mrep=4, mbits=16, id_sb=SegBCH(6,1,24), tol=1),
 "H1_rep3_bcht4":dict(mrep=3, mbits=16, id_sb=SegBCH(6,4,24), tol=1),
}
det=f2.FramedDetector(); pipe=build_regen_pipe()
def make_codec(cfg):
    mtx=cfg["mbits"]*cfg["mrep"]; idtx=cfg["id_sb"].tx; pad=f2.NC-mtx-idtx
    def enc(iid): return np.concatenate([np.repeat(MAGIC[:cfg["mbits"]],cfg["mrep"]), cfg["id_sb"].encode(f2.id_bits(iid)), np.zeros(pad,np.uint8)])
    def parse(llr):
        d=llr[:mtx].reshape(cfg["mbits"],cfg["mrep"]).sum(1); m=(d>0).astype(np.uint8)
        present=int((m!=MAGIC[:cfg["mbits"]]).sum())<=cfg["tol"]
        idv=cfg["id_sb"].chase(llr[mtx:mtx+idtx]) if present else None
        return present,idv
    return enc,parse,mtx,idtx,pad
def llr_of(att):
    from framed_v2 import SPEC
    al={}
    for nm,(kind,getter) in SPEC.items():
        p,M=det.PM[nm]; al[nm]=f2.method_soft_to_codeword_llr(getattr(det.frag[nm],getter)(att),p,M,kind=kind,n_codeword=f2.NC)
    return f2.fuse_llrs(al,None,f2.NC)
def embed(C,iid,enc):
    img=C
    for nm in ["vine","trustmark"]:
        p,M=det.PM[nm]; img=det.frag[nm].embed_with_target(img,apply_crypto(enc(iid),p,M)); img=img if img.size==(RES,RES) else img.resize((RES,RES))
    return img
def rot(pil,deg):
    a=np.array(pil); pad=RES//2; big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),'reflect')).rotate(deg,resample=Image.BICUBIC); l=(big.width-RES)//2; return big.crop((l,l,l+RES,l+RES))
def ccr(pil,r): cw=int(RES*r); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
def jpeg(pil): b=io.BytesIO(); pil.save(b,"JPEG",quality=25); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def attack(name,W,seed):
    return {"clean":W,"jpeg":jpeg(W),"crop75":ccr(W,0.75),"rot22":rot(W,22.5)}.get(name) if name!="regen" else stable_regen(pipe,W,seed,denoise_steps=8)
def detect_resync(att,parse,angles=np.arange(-30,30.01,3.0)):
    p,iv=parse(llr_of(att))
    if p: return True,iv
    for d in angles:
        p2,iv2=parse(llr_of(rot(att,d)))
        if p2: return True,iv2
    return False,None
ATKS=["clean","jpeg","crop75","rot22","regen"]
files=sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen","*.png")))[:N]
for cname,cfg in CONFIGS.items():
    enc,parse,mtx,idtx,pad=make_codec(cfg)
    if pad<0: print(f"{cname}: OVER BUDGET (mtx{mtx}+idtx{idtx}={mtx+idtx}) skip",flush=True); continue
    R={a:{"d":[],"id":[]} for a in ATKS}; fpr=[]
    for idx,fp in enumerate(files):
        C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"v3_{idx:05d}"; idt=f2.id_bits(iid); W=embed(C,iid,enc)
        fpr.append(1.0 if detect_resync(C,parse)[0] else 0.0)
        for a in ATKS:
            At=attack(a,W,1234+idx); At=At if At.size==(RES,RES) else At.resize((RES,RES))
            d,iv=detect_resync(At,parse)
            R[a]["d"].append(1.0 if d else 0.0); R[a]["id"].append(1.0 if (iv is not None and np.array_equal(iv,idt)) else 0.0)
    print(f"\n=== {cname} (magic {cfg['mbits']}x{cfg['mrep']}={mtx}, id-BCH tx={idtx}, pad={pad}) FPR(+resync)={np.mean(fpr):.4f} ===",flush=True)
    print(f"{'attack':<9}{'detect(+resync)':>16}{'ID-recover':>12}",flush=True)
    for a in ATKS: print(f"{a:<9}{np.mean(R[a]['d']):>16.3f}{np.mean(R[a]['id']):>12.3f}",flush=True)
print("FRAMED_V3_DONE")
