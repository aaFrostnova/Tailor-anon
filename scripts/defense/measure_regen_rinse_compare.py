"""Clean same-resolution (256), same-seed, PAIRED comparison of C0/C2/C3/VINE-B/VINE-R on regen & rinse, n=100.
All models: same images, same secrets, same regen seeds -> apples-to-apples. rinse reuses regen's 1st round."""
import os,sys,glob,numpy as np,torch
from PIL import Image
from torchvision import transforms
VINE_REPO="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0,VINE_REPO); sys.path.insert(0,os.path.join(VINE_REPO,"vine","src"))
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
from _regen_util import build_regen_pipe, stable_regen
dev="cuda"; RES=256; RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
t256=transforms.Compose([transforms.Resize((RES,RES),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()])
def load(spec):
    if spec in ("VINE-B","VINE-R"):
        enc=VINE_Turbo.from_pretrained(f"Shilin-LU/{spec}-Enc"); dec=CustomConvNeXt.from_pretrained(f"Shilin-LU/{spec}-Dec")
    else:
        enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc"); dec=CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec")
        enc.unet.load_state_dict(torch.load(os.path.join(spec,"UNet2DConditionModel.pth"),map_location="cpu"))
        enc.vae_a2b.load_state_dict(torch.load(os.path.join(spec,"vae.pth"),map_location="cpu"))
        enc.sec_encoder.load_state_dict(torch.load(os.path.join(spec,"ConditionAdaptor.pth"),map_location="cpu"))
        dec.load_state_dict(torch.load(os.path.join(spec,"CustomConvNeXt.pth"),map_location="cpu"))
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
MODELS=[("VINE-B",  "VINE-B"),("VINE-R","VINE-R"),
        ("C0 full 105k",f"{RT}/vine_official_C0/checkpoint-105000"),
        ("C2 no-GAN 110k",f"{RT}/vine_official_C2/checkpoint-110000"),
        ("C3 no-LPIPS 110k",f"{RT}/vine_official_C3/checkpoint-110000")]
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
imgs=[t256(Image.open(f).convert("RGB")).unsqueeze(0).to(dev) for f in files]
torch.manual_seed(0); secs=[torch.randint(0,2,(1,100)).float().to(dev) for _ in files]  # SAME across models
res={}
for mn,spec in MODELS:
    enc,dec=load(spec); rg=[]; rn=[]; cl=[]
    for k,(img,sec) in enumerate(zip(imgs,secs)):
        wm=embed(enc,img,sec); cl.append(ba(dec,wm,sec))
        r1=tt(stable_regen(pipe,pil(wm),1234+k))            # regen (1 round)  — SAME seed across models
        r2=tt(stable_regen(pipe,pil(r1),5678+k))            # rinse (2nd round)
        rg.append(ba(dec,r1,sec)); rn.append(ba(dec,r2,sec))
        if (k+1)%25==0: print(f"    {mn} [{k+1}/{N}]",flush=True)
    res[mn]=(float(np.mean(cl)),float(np.mean(rg)),float(np.mean(rn)),np.array(rg),np.array(rn))
    del enc,dec; torch.cuda.empty_cache(); print(f"  {mn}: clean {res[mn][0]:.3f}  regen {res[mn][1]:.3f}  rinse {res[mn][2]:.3f}",flush=True)
print(f"\n=== 同分辨率(256)/同种子/配对  regen & rinse  (n={N}) ===")
print(f"{'model':18s} {'clean':>7s} {'regen':>7s} {'rinse':>7s}   {'regen 95%CI':>16s}")
for mn,_ in MODELS:
    c,rg,rn,rga,rna=res[mn]; ci=1.96*rga.std()/np.sqrt(N)
    print(f"{mn:18s} {c:7.3f} {rg:7.3f} {rn:7.3f}   ±{ci:.3f}")
# paired C3 vs VINE-R (same seeds -> paired diff)
d_r=res['C3 no-LPIPS 110k'][3]-res['VINE-R'][3]; d_n=res['C3 no-LPIPS 110k'][4]-res['VINE-R'][4]
print(f"\nPAIRED C3 − VINE-R:  regen Δ={d_r.mean():+.3f} (95%CI ±{1.96*d_r.std()/np.sqrt(N):.3f})   rinse Δ={d_n.mean():+.3f} (±{1.96*d_n.std()/np.sqrt(N):.3f})")
print("CMP_DONE")
