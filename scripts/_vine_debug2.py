import os,sys,torch
VINE_REPO="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0,VINE_REPO); sys.path.insert(0,os.path.join(VINE_REPO,"vine","src"))
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
acc=Accelerator(mixed_precision="no"); dev=acc.device
enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev)
dec=CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec").to(dev)
for m in (enc.unet,enc.vae_a2b,enc.sec_encoder): m.requires_grad_(True)
dec.requires_grad_(True)
# instrument
_sec=enc.sec_encoder.forward
def sec_fwd(secret,x): o=_sec(secret,x); print(f"  sec_encoder: x={tuple(x.shape)} -> x_sec={tuple(o.shape)}",flush=True); return o
enc.sec_encoder.forward=sec_fwd
_ve=enc.vae_enc.forward
def ve_fwd(x,direction): o=_ve(x,direction); print(f"  vae_enc: in={tuple(x.shape)} -> latent={tuple(o.shape)}",flush=True); return o
enc.vae_enc.forward=ve_fwd
print("=== BEFORE prepare ===",flush=True)
img=Image.open("results/defense/ext_vtv100/img_00000.png").convert("RGB")
t=transforms.Compose([transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()])
x01=t(img).unsqueeze(0).to(dev); sec=torch.randint(0,2,(1,100)).float().to(dev)
try:
    with torch.no_grad(): wm=enc(x01*2-1,sec); print("  BEFORE prepare OK, wm",tuple(wm.shape),flush=True)
except Exception as e: print("  BEFORE prepare FAIL:",str(e)[:100],flush=True)
print("=== AFTER prepare (submodule pattern) ===",flush=True)
enc.unet,enc.vae_enc,enc.vae_dec,enc.sec_encoder,dec=acc.prepare(enc.unet,enc.vae_enc,enc.vae_dec,enc.sec_encoder,dec)
# re-instrument prepared wrappers
_sec2=enc.sec_encoder.forward
def sec_fwd2(secret,x): o=_sec2(secret,x); print(f"  [prep] sec_encoder: x={tuple(x.shape)} -> x_sec={tuple(o.shape)}",flush=True); return o
enc.sec_encoder.forward=sec_fwd2
try:
    with torch.no_grad(): wm=enc(x01*2-1,sec); print("  AFTER prepare OK, wm",tuple(wm.shape),flush=True)
except Exception as e: print("  AFTER prepare FAIL:",type(e).__name__,str(e)[:120],flush=True)
print("DEBUG2_DONE")
