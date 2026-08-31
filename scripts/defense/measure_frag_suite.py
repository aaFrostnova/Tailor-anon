"""Per-fragment coverage matrix: TrustMark-B + VINE-R on the SAME full attack suite (PIL interface via build_methods).
Completes the fragment x attack matrix used to derive per-scenario minimal watermark sets."""
import os,sys,io,glob,numpy as np,torch
from PIL import Image, ImageEnhance, ImageFilter
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts")
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from wbench.methods import build_methods
dev="cuda"; RES=256
def to(pil): return pil.resize((RES,RES)) if pil.size!=(RES,RES) else pil
def pil2t(im): return torch.from_numpy(np.asarray(im.convert("RGB"),np.float32)/255).permute(2,0,1)[None].to(dev)
def t2pil(x): return Image.fromarray((x[0].permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy())
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev)
VAEC=cheng2020_anchor(quality=3,pretrained=True).eval().to(dev)
def vae(im,m):
    with torch.no_grad(): return t2pil(m(pil2t(to(im)))["x_hat"].clamp(0,1))
from _regen_util import build_regen_pipe, stable_regen
pipe=build_regen_pipe()
def crop(im,r): s=int(RES*r); o=(RES-s)//2; return im.crop((o,o,o+s,o+s)).resize((RES,RES))
def att(name,im,k):
    if name=="clean": return im
    if name=="jpeg50": b=io.BytesIO(); im.save(b,"JPEG",quality=50); b.seek(0); return Image.open(b).convert("RGB")
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise": x=np.asarray(im,np.float32)/255+np.random.RandomState(k).randn(RES,RES,3).astype(np.float32)*0.05; return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="bright": return ImageEnhance.Brightness(im).enhance(0.5)
    if name=="contrast": return ImageEnhance.Contrast(im).enhance(0.5)
    if name=="vaeB": return vae(im,VAEB)
    if name=="vaeC": return vae(im,VAEC)
    if name=="crop90": return crop(im,0.9)
    if name=="crop75": return crop(im,0.75)
    if name=="rot9": return im.rotate(9,resample=Image.BICUBIC)
    if name=="rot30": return im.rotate(30,resample=Image.BICUBIC)
    if name=="regen": return stable_regen(pipe,to(im),1234+k).resize((RES,RES))
    if name=="rinse": return stable_regen(pipe,stable_regen(pipe,to(im),1234+k).resize((RES,RES)),1235+k).resize((RES,RES))
ATTS=["clean","jpeg50","jpeg25","blur","noise","bright","contrast","vaeB","vaeC","crop90","crop75","rot9","rot30","regen","rinse"]
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
covers=[to(Image.open(f).convert("RGB")) for f in files]
M=build_methods(["trustmark_b","vine_r"],dev)
print(f"=== per-fragment full suite, n={N} ===",flush=True)
rows={}
for name,m in M.items():
    rng=np.random.RandomState(0); bits=rng.randint(0,2,m.n_bits).tolist(); gt=np.array(bits)
    acc={a:[] for a in ATTS}
    for k,cov in enumerate(covers):
        wm=to(m.embed(cov,bits))
        for a in ATTS:
            dec=np.array(m.decode(att(a,wm,k)))[:m.n_bits]; n=min(len(dec),len(gt))
            acc[a].append(float(np.mean(dec[:n]==gt[:n])))
    rows[name]={a:np.mean(acc[a]) for a in ATTS}; print(f"  {name} done",flush=True)
print("\n"+f"{'frag':12s} "+"".join(f"{a[:7]:>8s}" for a in ATTS))
for nm,acc in rows.items(): print(f"{nm:12s} "+"".join(f"{acc[a]:8.3f}" for a in ATTS))
print("FRAGSUITE_DONE")
