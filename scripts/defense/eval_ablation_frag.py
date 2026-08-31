"""Evaluate the ABLATION-trained VINE variants as candidate fragments for our composite.
C5=crop-augmented (the one meant to fix VINE's crop gap) vs C0=full (our from-scratch repro) vs VINE-R released.
Measures clean/crop/rot/jpeg/regen bit-acc + PSNR, and SAVES watermarked image + residual heatmap.
Question: did crop-training buy crop robustness WITHOUT losing regen (VINE's whole value)?"""
import os,sys,glob,io,numpy as np,torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torchvision import transforms
import kornia.geometry.transform as KT
VINE_REPO="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0,VINE_REPO); sys.path.insert(0,os.path.join(VINE_REPO,"vine","src"))
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev="cuda"; RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
OUT="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/ablation_frag"
os.makedirs(OUT,exist_ok=True)
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
def a_rot(x,d): return KT.rotate(x,torch.tensor([float(d)],device=x.device),mode="bilinear",padding_mode="reflection")
def a_crop(x,r): H=x.shape[-1]; c=int(H*r); return KT.resize(KT.center_crop(x,(c,c)),(H,H))
def a_jpeg(x,q):
    im=Image.fromarray((x[0].permute(1,2,0)*255).byte().cpu().numpy()); b=io.BytesIO()
    im.save(b,"JPEG",quality=q); b.seek(0)
    return transforms.ToTensor()(Image.open(b).convert("RGB")).unsqueeze(0).to(x.device)
N=int(sys.argv[1]) if len(sys.argv)>1 else 16
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
imgs=[t256(Image.open(f).convert("RGB")).unsqueeze(0).to(dev) for f in files]
torch.manual_seed(0); secs=[torch.randint(0,2,(1,100)).float().to(dev) for _ in files]
ATT={"clean":lambda x:x,"crop90":lambda x:a_crop(x,0.9),"crop75":lambda x:a_crop(x,0.75),
     "rot9":lambda x:a_rot(x,9),"rot30":lambda x:a_rot(x,30),"jpeg50":lambda x:a_jpeg(x,50)}
from _regen_util import build_regen_pipe, stable_regen
pipe=build_regen_pipe()
MODELS=[("VINE-R released",None),("C0 full 105k",f"{RT}/vine_official_C0/checkpoint-105000"),
        ("C5 crop 55k",f"{RT}/vine_official_C5/checkpoint-55000")]
print(f"=== ablation VINE fragment eval, n={N} ===",flush=True)
rows={}
for mname,ck in MODELS:
    enc,dec=load_vine(ck)
    acc={a:[] for a in list(ATT)+["regen"]}; ps=[]
    for k,(img,sec) in enumerate(zip(imgs,secs)):
        wm01=embed(enc,img,sec); ps.append(psnr(img,wm01))
        for a,fn in ATT.items(): acc[a].append(dec_ba(dec,fn(wm01).clamp(0,1),sec))
        pil=Image.fromarray((wm01[0].permute(1,2,0)*255).byte().cpu().numpy())
        rt=transforms.ToTensor()(stable_regen(pipe,pil,1234+k).resize((256,256))).unsqueeze(0).to(dev)
        acc["regen"].append(dec_ba(dec,rt,sec))
        if k==0:  # save watermarked + residual for img0
            tag=mname.split()[0]+("_"+mname.split()[1] if len(mname.split())>1 else "")
            pil.save(f"{OUT}/wm_{tag}.png")
            res=np.abs(wm01[0].permute(1,2,0).cpu().numpy()-img[0].permute(1,2,0).cpu().numpy()).mean(2)
            plt.figure(figsize=(3,3)); plt.imshow(res,cmap="viridis"); plt.axis("off")
            plt.title(f"{mname}\nPSNR={ps[0]:.1f}dB",fontsize=8); plt.tight_layout()
            plt.savefig(f"{OUT}/resid_{tag}.png",dpi=110,bbox_inches="tight"); plt.close()
    rows[mname]=(np.mean(ps),{a:np.mean(acc[a]) for a in acc})
    del enc,dec; torch.cuda.empty_cache(); print(f"  {mname:18s} done",flush=True)
# table
atts=list(ATT)+["regen"]
print(f"\n{'model':18s} {'PSNR':>6s} "+"".join(f"{a:>8s}" for a in atts))
for m,(p,acc) in rows.items():
    print(f"{m:18s} {p:6.2f} "+"".join(f"{acc[a]:8.3f}" for a in atts))
# save cover img0 for reference
Image.fromarray((imgs[0][0].permute(1,2,0)*255).byte().cpu().numpy()).save(f"{OUT}/cover_img0.png")
print("ABL_EVAL_DONE")
