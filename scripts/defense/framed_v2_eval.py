import io, glob, os, sys, numpy as np
from PIL import Image
sys.path.insert(0,os.path.join("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint","scripts/defense"))
from framed_v2 import FramedDetector, ID_SB, MAGIC_SB, id_bits, encode_cw
from _regen_util import build_regen_pipe, stable_regen
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; RES=512; N=40
det=FramedDetector(); pipe=build_regen_pipe()
def ccr(pil,r): cw=int(RES*r); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
def jpeg(pil): b=io.BytesIO(); pil.save(b,"JPEG",quality=25); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def rot(pil,deg):
    a=np.array(pil); pad=RES//2; big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),'reflect')).rotate(deg,resample=Image.BICUBIC); l=(big.width-RES)//2; return big.crop((l,l,l+RES,l+RES))
def attack(name,W,seed):
    if name=="clean": return W
    if name=="jpeg": return jpeg(W)
    if name=="crop75": return ccr(W,0.75)
    if name=="rot22": return rot(W,22.5)
    if name=="regen": return stable_regen(pipe,W,seed,denoise_steps=8)
ATKS=["clean","jpeg","crop75","rot22","regen"]
files=sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen","*.png")))[:N]
R={a:{"d0":[],"dr":[],"id":[]} for a in ATKS}; fp0=[]; fpr=[]
for idx,fp in enumerate(files):
    C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"f2_{idx:05d}"; idt=id_bits(iid)
    W=det.embed(C,iid)
    # FPR on no-watermark cover
    fp0.append(1.0 if det._parse(det._llr(C))[0] else 0.0)
    fpr.append(1.0 if det.detect(C,resync=True)["present"] else 0.0)
    for a in ATKS:
        At=attack(a,W,1234+idx); At=At if At.size==(RES,RES) else At.resize((RES,RES))
        p0,_=det._parse(det._llr(At))                       # no-resync
        res=det.detect(At,resync=True)                      # +resync
        R[a]["d0"].append(1.0 if p0 else 0.0); R[a]["dr"].append(1.0 if res["present"] else 0.0)
        R[a]["id"].append(1.0 if (res["id"] is not None and np.array_equal(res["id"],idt)) else 0.0)
    if (idx+1)%10==0: print(f"[{idx+1}/{N}]",flush=True)
print(f"\n=== Scheme A v2 (BCH magic-t5 + id-t4 + chase + resync, n={N}) ===")
print(f"blind-detection FPR: resync-off={np.mean(fp0):.4f}  resync-on={np.mean(fpr):.4f}")
print(f"{'attack':<9}{'detect(no-rsync)':>18}{'detect(+resync)':>17}{'ID-recover':>12}")
for a in ATKS: print(f"{a:<9}{np.mean(R[a]['d0']):>18.3f}{np.mean(R[a]['dr']):>17.3f}{np.mean(R[a]['id']):>12.3f}")
print("FRAMED_V2_DONE")
