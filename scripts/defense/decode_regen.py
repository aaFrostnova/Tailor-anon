"""Decode ctrlregen-attacked PNGs (per model) vs saved secrets.npz -> bit-acc. Arg: --tag 03/05/07."""
import sys, os, glob, argparse, numpy as np, torch
from PIL import Image
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev="cuda"; IMG=256
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"; WORK="/scratch/workspace/mingzhel_umass_edu-ablator/regen_work"
OUT="/scratch/workspace/mingzhel_umass_edu-ablator/regen_out"
ap=argparse.ArgumentParser(); ap.add_argument("--tags",default="03,05,07"); a=ap.parse_args()
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(RT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1])); return ds[-1]
def load_dec(spec):
    if spec=="VINEB": return CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec").to(dev).eval()
    d=CustomConvNeXt(secret_size=100); d.load_state_dict(torch.load(os.path.join(spec,"CustomConvNeXt.pth"),map_location="cpu")); return d.to(dev).eval()
sec=np.load(os.path.join(WORK,"secrets.npz")); specs={"C0":latest("vine_official_C0"),"C5":latest("vine_official_C5"),"VINEB":"VINEB"}
tags=[t for t in a.tags.split(",") if t]
print(f"{'model':8s} "+" ".join(f"ctrlreg_s{t:>3s}" for t in tags),flush=True)
for m,spec in specs.items():
    dec=load_dec(spec); row=[]
    for t in tags:
        d=os.path.join(OUT,f"{m}_s{t}"); accs=[]
        for f in sorted(glob.glob(os.path.join(d,"*.png"))):
            key=f"{m}/"+os.path.basename(f)[:-4]
            if key not in sec: continue
            img=np.asarray(Image.open(f).convert("RGB").resize((IMG,IMG)),np.float32)/255
            tt=torch.from_numpy(img).permute(2,0,1)[None].float().to(dev)
            with torch.no_grad(): bits=(dec(tt)>0.5).float().cpu().numpy()[0]
            accs.append((bits==sec[key]).mean())
        row.append(np.mean(accs) if accs else float("nan"))
    print(f"{m:8s} "+" ".join(f"{v:11.3f}" for v in row),flush=True)
    del dec; torch.cuda.empty_cache()
print("DECODE_REGEN_DONE",flush=True)
