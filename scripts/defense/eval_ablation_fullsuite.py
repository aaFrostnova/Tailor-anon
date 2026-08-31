"""Full attack suite on ALL ablation VINE variants (VINE-R released / C0 full / C2 no-GAN / C3 no-LPIPS / C5 crop).
Attacks: L1 signal (jpeg50/jpeg25/blur/noise/bright/contrast/VAE-B/VAE-C) | L2 geometry (crop90/75/rot9/30) | L3 regen/rinse.
Reports bit-acc per (model,attack) + clean PSNR. 256x256 native VINE res. CtrlRegen+/UnMarker are separate cross-env (not here)."""
import os,sys,glob,io,numpy as np,torch
from PIL import Image, ImageEnhance
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
        dec.load_state_dict(torch.load(os.path.join(ckpt,"CustomConvNeXt.pth"),map_location="cpu"))
        enc.to(dev).eval(); dec.to(dev).eval()
    return enc,dec
t256=transforms.Compose([transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()])
def embed(enc,img01,sec):
    with torch.no_grad(): wm=enc(img01*2-1,sec)
    return ((wm+1)/2).clamp(0,1)
def dec_ba(dec,x01,sec):
    with torch.no_grad(): p=dec(x01)
    return float(((p>0.5).float()==sec).float().mean())
def psnr(a01,b01):
    mse=torch.mean((a01-b01)**2).item(); return 99.0 if mse<1e-12 else 10*np.log10(1/mse)
# ---- attack helpers (tensor[1,3,256,256] -> tensor) ----
def t2pil(x): return Image.fromarray((x[0].permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy())
def pil2t(im): return transforms.ToTensor()(im.convert("RGB")).unsqueeze(0).to(dev)
def a_jpeg(x,q):
    b=io.BytesIO(); t2pil(x).save(b,"JPEG",quality=q); b.seek(0); return pil2t(Image.open(b))
def a_blur(x,s):
    import torchvision.transforms.functional as TF; k=int(2*round(2*s)+1); return TF.gaussian_blur(x,k,[s,s])
def a_noise(x,sd): return (x+torch.randn_like(x)*sd).clamp(0,1)
def a_enh(x,kind,f): return pil2t((ImageEnhance.Brightness if kind=="b" else ImageEnhance.Contrast)(t2pil(x)).enhance(f))
def a_rot(x,d): return KT.rotate(x,torch.tensor([float(d)],device=x.device),mode="bilinear",padding_mode="reflection")
def a_crop(x,r): H=x.shape[-1]; c=int(H*r); return KT.resize(KT.center_crop(x,(c,c)),(H,H))
# VAE neural-compression attacks (compressai forward)
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
try:
    VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev)
    VAEC=cheng2020_anchor(quality=3,pretrained=True).eval().to(dev); VAE_OK=True
except Exception as e:
    print("VAE load failed:",e,flush=True); VAE_OK=False
def a_vae(x,model):
    with torch.no_grad(): return model(x)["x_hat"].clamp(0,1)
# regen
from _regen_util import build_regen_pipe, stable_regen
pipe=build_regen_pipe()
def a_regen(x,seed):
    return pil2t(stable_regen(pipe,t2pil(x),seed).resize((256,256)))
def a_rinse(x,seed):
    return a_regen(a_regen(x,seed),seed+1)
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
imgs=[t256(Image.open(f).convert("RGB")).unsqueeze(0).to(dev) for f in files]
torch.manual_seed(0); secs=[torch.randint(0,2,(1,100)).float().to(dev) for _ in files]
# attack list (name -> fn(x, idx))
ATT=[("clean",lambda x,k:x),
     ("jpeg50",lambda x,k:a_jpeg(x,50)),("jpeg25",lambda x,k:a_jpeg(x,25)),
     ("blur",lambda x,k:a_blur(x,2.0)),("noise",lambda x,k:a_noise(x,0.05)),
     ("bright",lambda x,k:a_enh(x,"b",0.5)),("contrast",lambda x,k:a_enh(x,"c",0.5))]
if VAE_OK: ATT+=[("vaeB",lambda x,k:a_vae(x,VAEB)),("vaeC",lambda x,k:a_vae(x,VAEC))]
ATT+=[("crop90",lambda x,k:a_crop(x,0.9)),("crop75",lambda x,k:a_crop(x,0.75)),
      ("rot9",lambda x,k:a_rot(x,9)),("rot30",lambda x,k:a_rot(x,30)),
      ("regen",lambda x,k:a_regen(x,1234+k)),("rinse",lambda x,k:a_rinse(x,1234+k))]
MODELS=[("VINE-R released",None),("C0 full 105k",f"{RT}/vine_official_C0/checkpoint-105000"),
        ("C2 no-GAN 110k",f"{RT}/vine_official_C2/checkpoint-110000"),
        ("C3 no-LPIPS 110k",f"{RT}/vine_official_C3/checkpoint-110000"),
        ("C5 crop 55k",f"{RT}/vine_official_C5/checkpoint-55000")]
print(f"=== full-suite ablation eval, n={N}, attacks={len(ATT)} ===",flush=True)
rows={}
for mname,ck in MODELS:
    enc,dec=load_vine(ck); acc={a:[] for a,_ in ATT}; ps=[]
    for k,(img,sec) in enumerate(zip(imgs,secs)):
        wm01=embed(enc,img,sec); ps.append(psnr(img,wm01))
        for a,fn in ATT: acc[a].append(dec_ba(dec,fn(wm01,k).clamp(0,1),sec))
    rows[mname]=(np.mean(ps),{a:np.mean(acc[a]) for a,_ in ATT})
    del enc,dec; torch.cuda.empty_cache(); print(f"  {mname:18s} done",flush=True)
atts=[a for a,_ in ATT]
print("\n"+"="*140)
print(f"{'model':18s} {'PSNR':>5s} "+"".join(f"{a[:7]:>8s}" for a in atts))
for m,(p,acc) in rows.items():
    print(f"{m:18s} {p:5.1f} "+"".join(f"{acc[a]:8.3f}" for a in atts))
print("FULLSUITE_DONE")
