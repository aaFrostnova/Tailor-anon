"""Attack-induced quality (SSIM/PSNR) for the L1/L2/L3 attacks, SAME attack defs as measure_frag_suite.py.
Fills the '待测' quality column so each attack has a (bit-acc, SSIM) pair for the Performance-vs-Quality view.
Convention: SSIM/PSNR of attack(cover) vs clean cover (watermark near-invisible -> = attack's own quality cost)."""
import os, sys, io, glob, numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim_fn
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
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
def psnr(a,b):
    mse=np.mean((np.asarray(a,np.float32)-np.asarray(b,np.float32))**2); return 99.0 if mse<1e-9 else 10*np.log10(255**2/mse)
def ssim(a,b): return float(ssim_fn(np.asarray(a.convert("RGB")),np.asarray(b.convert("RGB")),channel_axis=2))
GEO={"crop90","crop75","rot9","rot30"}
ATTS=["jpeg50","jpeg25","blur","noise","bright","contrast","vaeB","vaeC","crop90","crop75","rot9","rot30","regen","rinse"]
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
covers=[to(Image.open(f).convert("RGB")) for f in files]
print(f"=== attack-induced quality, n={N} (SSIM/PSNR of attack(cover) vs cover) ===")
print(f"{'attack':9s} {'SSIM':>7s} {'PSNR':>7s}  {'level':>6s}  note")
res={}
for a in ATTS:
    ss=[]; ps=[]
    for k,c in enumerate(covers):
        atk=att(a,c,k); ss.append(ssim(atk,c)); ps.append(psnr(atk,c))
    sm=np.mean(ss); pm=np.mean(ps); res[a]=(sm,pm)
    lv="L1" if a in {"jpeg50","jpeg25","blur","noise","bright","contrast"} else ("L2" if a in GEO else "L3")
    note="geo-misalign (SSIM低=形变非画质)" if a in GEO else ("photometric (SSIM低但可逆)" if a in {"bright","contrast"} else "")
    print(f"{a:9s} {sm:7.3f} {pm:7.2f}  {lv:>6s}  {note}")
print("QUAL_DONE")
