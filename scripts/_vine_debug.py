import os,sys,numpy as np,torch
VINE_REPO="/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0,VINE_REPO); sys.path.insert(0,os.path.join(VINE_REPO,"vine","src"))
from PIL import Image
from torchvision import transforms
dev="cuda"
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev).eval()
dec=CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec").to(dev).eval()
img=Image.open("results/defense/ext_vtv100/img_00000.png").convert("RGB")
sec=torch.randint(0,2,(1,100)).float().to(dev)
for R in [256,512]:
    t=transforms.Compose([transforms.Resize((R,R),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()])
    x01=t(img).unsqueeze(0).to(dev)
    try:
        with torch.no_grad():
            wm=enc(x01*2-1,sec); wm01=((wm+1)/2).clamp(0,1); p=dec(wm01)
        ba=((p>0.5).float()==sec).float().mean().item()
        print(f"[R={R}] enc_out={tuple(wm.shape)} decode bit-acc={ba:.3f}",flush=True)
    except Exception as e:
        print(f"[R={R}] FAILED {type(e).__name__}: {str(e)[:140]}",flush=True)
print("DEBUG_DONE")
