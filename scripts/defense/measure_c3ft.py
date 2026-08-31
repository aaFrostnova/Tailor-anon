"""Evaluate C3 decoder-robustness finetune (VINE-R IP2P recipe, lpips=0) against the
n=100 same-res(256)/same-seed/PAIRED regen & rinse harness.
Models: C3 baseline, C3-FT-{500,1000,1500}, VINE-R (target).
Go/no-go: does FT lift rinse toward VINE-R's ~0.83 WITHOUT dropping single-regen / clean?
Each FT ckpt = C3 encoder (frozen) + finetuned decoder (CustomConvNeXt.pth)."""
import os,sys,glob,numpy as np,torch
from PIL import Image
from torchvision import transforms
VINE_REPO="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0,VINE_REPO); sys.path.insert(0,os.path.join(VINE_REPO,"vine","src"))
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
from _regen_util import build_regen_pipe, stable_regen
dev="cuda"; RES=256
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
FT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_C3_ft"
C3="%s/vine_official_C3/checkpoint-110000"%RT
t256=transforms.Compose([transforms.Resize((RES,RES),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()])
def load(spec):
    """spec: 'VINE-R' | ablation-ckpt-dir (has all 4 .pth) | ('C3enc+FTdec', ftdir) tuple."""
    if spec=="VINE-R":
        enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc"); dec=CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec")
        return enc.to(dev).eval(), dec.to(dev).eval()
    enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc")
    if isinstance(spec,tuple):   # C3 encoder + finetuned decoder from a FT checkpoint dir
        encdir,decdir=spec
    else:
        encdir=decdir=spec
    enc.unet.load_state_dict(torch.load(os.path.join(encdir,"UNet2DConditionModel.pth"),map_location="cpu"))
    enc.vae_a2b.load_state_dict(torch.load(os.path.join(encdir,"vae.pth"),map_location="cpu"))
    enc.sec_encoder.load_state_dict(torch.load(os.path.join(encdir,"ConditionAdaptor.pth"),map_location="cpu"))
    dec=CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec")
    dec.load_state_dict(torch.load(os.path.join(decdir,"CustomConvNeXt.pth"),map_location="cpu"))
    return enc.to(dev).eval(), dec.to(dev).eval()
def embed(enc,x,sec):
    with torch.no_grad(): w=enc(x*2-1,sec)
    return ((w+1)/2).clamp(0,1)
def ba(dec,x,sec):
    with torch.no_grad(): p=dec(x.clamp(0,1))
    return float(((p>0.5).float()==sec).float().mean())
def pil(t): return Image.fromarray((t[0].permute(1,2,0)*255).byte().cpu().numpy())
def tt(im): return transforms.ToTensor()(im.resize((RES,RES))).unsqueeze(0).to(dev)
pipe=build_regen_pipe()
# each FT ckpt: encoder = C3 (frozen), decoder = FT checkpoint's CustomConvNeXt.pth
MODELS=[("VINE-R (target)","VINE-R"),
        ("C3 baseline 110k",C3)]
for s in (500,1000,1500):
    d="%s/checkpoint-%d"%(FT,s)
    if os.path.exists(os.path.join(d,"CustomConvNeXt.pth")): MODELS.append(("C3-FT-%d"%s,(C3,d)))
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
imgs=[t256(Image.open(f).convert("RGB")).unsqueeze(0).to(dev) for f in files]
torch.manual_seed(0); secs=[torch.randint(0,2,(1,100)).float().to(dev) for _ in files]
res={}
for mn,spec in MODELS:
    enc,dec=load(spec); rg=[]; rn=[]; cl=[]
    for k,(img,sec) in enumerate(zip(imgs,secs)):
        wm=embed(enc,img,sec); cl.append(ba(dec,wm,sec))
        r1=tt(stable_regen(pipe,pil(wm),1234+k))
        r2=tt(stable_regen(pipe,pil(r1),5678+k))
        rg.append(ba(dec,r1,sec)); rn.append(ba(dec,r2,sec))
        if (k+1)%25==0: print(f"    {mn} [{k+1}/{N}]",flush=True)
    res[mn]=(float(np.mean(cl)),float(np.mean(rg)),float(np.mean(rn)),np.array(rg),np.array(rn))
    del enc,dec; torch.cuda.empty_cache(); print(f"  {mn}: clean {res[mn][0]:.3f}  regen {res[mn][1]:.3f}  rinse {res[mn][2]:.3f}",flush=True)
print(f"\n=== C3 decoder-FT (lpips=0, IP2P surrogate) — same-res/same-seed/paired (n={N}) ===")
print(f"{'model':20s} {'clean':>7s} {'regen':>7s} {'rinse':>7s}   {'regen 95%CI':>14s}")
for mn,_ in MODELS:
    c,rg,rn,rga,rna=res[mn]; ci=1.96*rga.std()/np.sqrt(N)
    print(f"{mn:20s} {c:7.3f} {rg:7.3f} {rn:7.3f}   ±{ci:.3f}")
# paired best-FT vs C3 baseline and vs VINE-R
best=max([m for m in res if m.startswith("C3-FT")], key=lambda m:res[m][2], default=None)
if best:
    dR=res[best][3]-res["C3 baseline 110k"][3]; dN=res[best][4]-res["C3 baseline 110k"][4]
    print(f"\nPAIRED {best} − C3 baseline:  regen Δ={dR.mean():+.3f} (±{1.96*dR.std()/np.sqrt(N):.3f})   rinse Δ={dN.mean():+.3f} (±{1.96*dN.std()/np.sqrt(N):.3f})")
    gR=res[best][4]-res["VINE-R (target)"][4]
    print(f"       {best} − VINE-R rinse gap: {gR.mean():+.3f}")
print("C3FT_DONE")
