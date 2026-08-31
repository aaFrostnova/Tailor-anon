"""regen (Zhao NeurIPS24) bit-acc for official C0/C5 (latest) + released VINE-B.
Embeds N covers x M secrets per model, runs stable_regen (SD2.1 VAE roundtrip noise_step=60),
decodes -> bit-acc. ALSO dumps 512px watermarked PNGs + secrets.npz for the separate ctrlregen pass."""
import sys, os, glob, numpy as np, torch
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts","defense"))
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
from _regen_util import build_regen_pipe, stable_regen
dev="cuda"; IMG=256; NCOV=6; NSEC=2
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
WORK="/scratch/workspace/mingzhel_umass_edu-ablator/regen_work"; os.makedirs(WORK,exist_ok=True)
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(RT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1])); return ds[-1],int(ds[-1].split("-")[-1])
c0d,c0s=latest("vine_official_C0"); c5d,c5s=latest("vine_official_C5")
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
idx=[3,7,12,20,30,45][:NCOV]
covers=[np.asarray(Image.open(NAT[k]).convert("RGB").resize((IMG,IMG)),np.float32)/255 for k in idx]
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
def load_enc(s): return (VINE_Turbo.from_pretrained(s.split(":",1)[1]) if str(s).startswith("pretrained:") else VINE_Turbo(ckpt_path=s,device=dev)).to(dev).eval()
def load_dec(spec):
    if str(spec).startswith("pretrained:"): return CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec").to(dev).eval()
    d=CustomConvNeXt(secret_size=100); d.load_state_dict(torch.load(os.path.join(spec,"CustomConvNeXt.pth"),map_location="cpu")); return d.to(dev).eval()
def decode(dec,img01):
    t=torch.from_numpy(np.ascontiguousarray(img01)).permute(2,0,1)[None].float().to(dev)
    with torch.no_grad(): return dec(t)
models=[("C0",c0d),("C5",c5d),("VINEB","pretrained:Shilin-LU/VINE-B-Enc")]
torch.manual_seed(0)
pipe=build_regen_pipe()
secrets={}
print(f"{'model':8s} {'clean':>8s} {'regen(Zhao)':>12s}",flush=True)
for mname,spec in models:
    enc=load_enc(spec); dec=load_dec(spec)
    mdir=os.path.join(WORK,mname); os.makedirs(mdir,exist_ok=True)
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    ca,ra=[],[]
    for ci,cov in enumerate(covers):
        for si in range(NSEC):
            sec=torch.randint(0,2,(1,100),device=dev).float()
            with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
            w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
            ca.append(((decode(dec,w01)>0.5).float()==sec).float().mean().item())
            # save 512 png + secret for ctrlregen
            key=f"{ci}_{si}"; Image.fromarray((w01*255).astype(np.uint8)).resize((512,512)).save(os.path.join(mdir,key+".png"))
            secrets[f"{mname}/{key}"]=sec.cpu().numpy().astype(np.int8)[0]
            # Zhao regen
            rg=stable_regen(pipe,Image.fromarray((w01*255).astype(np.uint8)),seed=1234+ci*10+si).resize((IMG,IMG))
            ra.append(((decode(dec,np.asarray(rg,np.float32)/255)>0.5).float()==sec).float().mean().item())
    print(f"{mname:8s} {np.mean(ca):8.3f} {np.mean(ra):12.3f}",flush=True)
    del enc,dec; torch.cuda.empty_cache()
np.savez(os.path.join(WORK,"secrets.npz"),**secrets)
print(f"saved 512 pngs + secrets to {WORK}\nREGEN_DONE",flush=True)
