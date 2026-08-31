"""Scheme A prototype: framed payload (UEP) for blind-detection-first + best-effort ID recovery.
codeword(100) = [magic16 x3 = 48 | id24 x2 = 48 | pad 4].  magic=HMAC(master_key,'magic'); id=hash(image_id)%2^24.
crypto perm/M from master_key ONLY (fixed 'GLOBAL') so blind detection needs no image_id.
Stage1 (blind): soft-combine magic copies -> 16b -> compare HMAC magic (FPR~2^-16).
Stage2 (id): soft-combine id copies -> 24b -> image ID. UEP: magic x3 (strong) > id x2.
Measures: blind detection rate + FPR(no-watermark) + ID-recovery rate, vs attacks, on SD-2.1 imgs."""
import glob, io, os, sys, hmac, hashlib, tempfile, numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense")
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, derive_method_keyed_constants
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from _regen_util import build_regen_pipe, stable_regen
RES=512; dev="cuda"; N=40; MKEY=b"v5_key_encoder_master"; NC=100
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits")}
frag={"vine":VineCryptoWrapper(master_key=MKEY,method_name="vine",n_bits=NC,device=dev),
      "trustmark":TrustMarkFragment(master_key=MKEY,method_name="trustmark",n_bits=NC,model_type="B",device=dev)}
# master-key-only perm/M per fragment (fixed id 'GLOBAL')
PM={nm:derive_method_keyed_constants(MKEY,"GLOBAL",nm,NC) for nm in ["vine","trustmark"]}
def bits_of(x,nb): return np.array([(x>>i)&1 for i in range(nb)],np.uint8)
MAGIC=np.unpackbits(np.frombuffer(hmac.new(MKEY,b"magic",hashlib.sha256).digest()[:2],np.uint8))  # 16 bits
def id24(image_id): return bits_of(int(hashlib.sha256(image_id.encode()).hexdigest(),16)%(1<<24),24)
def encode(image_id):
    cw=np.concatenate([np.repeat(MAGIC,3), np.repeat(id24(image_id),2), np.zeros(4,np.uint8)])  # 48+48+4=100
    assert len(cw)==NC; return cw
pipe=None
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
def ccr(pil,r): cw=int(RES*r); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
def jpeg(pil): b=io.BytesIO(); pil.save(b,"JPEG",quality=25); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def attack(name,W,seed):
    global pipe
    if name=="clean": return W
    if name=="jpeg": return jpeg(W)
    if name=="crop75": return ccr(W,0.75)
    if name=="regen":
        if pipe is None: pipe=build_regen_pipe()
        return stable_regen(pipe,W,seed,denoise_steps=8)
def fused_cw_llr(att):
    al={}
    for nm in ["vine","trustmark"]:
        kind,getter=SPEC[nm]; p,M=PM[nm]; al[nm]=method_soft_to_codeword_llr(getattr(frag[nm],getter)(att),p,M,kind=kind,n_codeword=NC)
    return fuse_llrs(al,weights=None,n_codeword=NC)
def detect_blind(llr, tol=0):   # stage1: magic, no image_id needed
    det=llr[0:48].reshape(16,3).sum(1)  # soft-combine 3 copies
    magic_dec=(det>0).astype(np.uint8); return int((magic_dec!=MAGIC).sum())<=tol
def recover_id(llr):            # stage2
    idl=llr[48:96].reshape(24,2).sum(1); return (idl>0).astype(np.uint8)
def main():
    ATKS=["clean","jpeg","crop75","regen"]; TOL=1   # allow <=1 magic-bit error
    files=sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen","*.png")))[:N]
    R={a:{"det":[],"idok":[]} for a in ATKS}; FP=[]
    for idx,fp in enumerate(files):
        C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"fr_{idx:05d}"; cw=encode(iid); idtrue=id24(iid)
        img=C
        for nm in ["vine","trustmark"]:
            p,M=PM[nm]; img=frag[nm].embed_with_target(img,apply_crypto(cw,p,M)); img=img if img.size==(RES,RES) else img.resize((RES,RES))
        # no-watermark FPR: decode clean cover (no embed)
        FP.append(1.0 if detect_blind(fused_cw_llr(C),TOL) else 0.0)
        for a in ATKS:
            At=attack(a,img,1234+idx); At=At if At.size==(RES,RES) else At.resize((RES,RES))
            llr=fused_cw_llr(At)
            R[a]["det"].append(1.0 if detect_blind(llr,TOL) else 0.0)
            R[a]["idok"].append(1.0 if np.array_equal(recover_id(llr),idtrue) else 0.0)
        if (idx+1)%10==0: print(f"[{idx+1}/{N}]",flush=True)
    print(f"\n=== Scheme A framed payload (n={N}, magic16x3 + id24x2, TOL={TOL}) ===")
    print(f"blind-detection FPR (no-watermark covers): {np.mean(FP):.4f}")
    print(f"{'attack':<9}{'blind-detect':>14}{'ID-recover':>12}")
    for a in ATKS: print(f"{a:<9}{np.mean(R[a]['det']):>14.3f}{np.mean(R[a]['idok']):>12.3f}")
    print("FRAMED_PROTO_DONE")
if __name__=="__main__": main()
