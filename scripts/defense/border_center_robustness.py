"""Does C2 (no-GAN, CENTER-concentrated) have a DIFFERENT robustness profile than border-concentrated
VINE-R / C0?  Probe with border-erase vs center-erase (which directly test WHERE the recoverable mark lives),
plus standard geometry + regen. Hypothesis: C2(center) survives border-erase but dies on center-erase;
VINE-R/C0(border) are the opposite."""
import os,sys,glob,io,numpy as np,torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import kornia.geometry.transform as KT
VINE_REPO="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0,VINE_REPO); sys.path.insert(0,os.path.join(VINE_REPO,"vine","src"))
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev="cuda"; RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
def load_vine(ckpt=None):
    enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev).eval()
    dec=CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec").to(dev).eval()
    if ckpt:
        enc.unet.load_state_dict(torch.load(os.path.join(ckpt,"UNet2DConditionModel.pth"),map_location="cpu"))
        enc.vae_a2b.load_state_dict(torch.load(os.path.join(ckpt,"vae.pth"),map_location="cpu"))
        enc.sec_encoder.load_state_dict(torch.load(os.path.join(ckpt,"ConditionAdaptor.pth"),map_location="cpu"))
        dec.load_state_dict(torch.load(os.path.join(ckpt,"CustomConvNeXt.pth"),map_location="cpu")); enc.to(dev).eval(); dec.to(dev).eval()
    return enc,dec
t256=transforms.Compose([transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()])
def embed(enc,x,sec):
    with torch.no_grad(): w=enc(x*2-1,sec)
    return ((w+1)/2).clamp(0,1)
def ba(dec,x,sec):
    with torch.no_grad(): p=dec(x.clamp(0,1))
    return float(((p>0.5).float()==sec).float().mean())
H=256
def border_erase(x,frac):   # blank outer ring of width frac*H (keep CENTER)
    b=int(H*frac); y=x.clone(); y[:,:,:b,:]=0.5; y[:,:,-b:,:]=0.5; y[:,:,:,:b]=0.5; y[:,:,:,-b:]=0.5; return y
def center_erase(x,frac):   # blank central frac*H square (keep BORDER)
    c=int(H*frac); o=(H-c)//2; y=x.clone(); y[:,:,o:o+c,o:o+c]=0.5; return y
def crop_resize(x,r): c=int(H*r); return KT.resize(KT.center_crop(x,(c,c)),(H,H))
ATT={"clean":lambda x:x,
     "border_erase15":lambda x:border_erase(x,0.15),"border_erase25":lambda x:border_erase(x,0.25),
     "center_erase40":lambda x:center_erase(x,0.40),"center_erase50":lambda x:center_erase(x,0.50),
     "crop75_resize":lambda x:crop_resize(x,0.75)}
from _regen_util import build_regen_pipe, stable_regen
pipe=build_regen_pipe()
MODELS=[("VINE-R (border)",None),("C0 full 105k (border)",f"{RT}/vine_official_C0/checkpoint-105000"),
        ("C2 no-GAN 110k (CENTER)",f"{RT}/vine_official_C2/checkpoint-110000")]
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
imgs=[t256(Image.open(f).convert("RGB")).unsqueeze(0).to(dev) for f in files]
torch.manual_seed(0); secs=[torch.randint(0,2,(1,100)).float().to(dev) for _ in files]
print(f"=== border/center-targeted robustness, n={N} ===")
rows={}
for mn,ck in MODELS:
    enc,dec=load_vine(ck); acc={a:[] for a in list(ATT)+["regen"]}
    for k,(img,sec) in enumerate(zip(imgs,secs)):
        wm=embed(enc,img,sec)
        for a,fn in ATT.items(): acc[a].append(ba(dec,fn(wm),sec))
        pil=Image.fromarray((wm[0].permute(1,2,0)*255).byte().cpu().numpy())
        rg=transforms.ToTensor()(stable_regen(pipe,pil,1234+k).resize((256,256))).unsqueeze(0).to(dev)
        acc["regen"].append(ba(dec,rg,sec))
    rows[mn]={a:float(np.mean(acc[a])) for a in acc}; del enc,dec; torch.cuda.empty_cache(); print(f"  {mn} done",flush=True)
cols=list(ATT)+["regen"]
print(f"\n{'model':26s}"+"".join(f"{c[:13]:>14s}" for c in cols))
for mn in rows: print(f"{mn:26s}"+"".join(f"{rows[mn][c]:14.3f}" for c in cols))
print("\nkey: border_erase=blank OUTER ring (center-mark survives) · center_erase=blank CENTER (border-mark survives)")
print("BC_DONE")
