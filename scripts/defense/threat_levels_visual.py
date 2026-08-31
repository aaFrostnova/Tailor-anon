"""Per-level visual montage for THREAT_MODEL_V2: the SAME watermarked image after a representative
attack at each level. SSIM computed from the actual shown image vs original; 'defend'=report bit-acc.
Reliable engines only (PIL / compressai / on-disk CtrlRegen+); no flaky diffusion proxy."""
import os, io, numpy as np, torch
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; D=256
orig=Image.open(f"{CF}/results/defense/ext_vtv1000/img_00000.png").convert("RGB").resize((D,D)); oa=np.asarray(orig)
def SS(x): return float(ssim_fn(oa, np.asarray(x.convert("RGB").resize((D,D))), channel_axis=2))
def jpeg(q): b=io.BytesIO(); orig.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
def noise(sd): x=np.asarray(orig,np.float32)/255+np.random.RandomState(0).randn(D,D,3).astype(np.float32)*sd; return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
def crop(r): s=int(D*r); o=(D-s)//2; return orig.crop((o,o,o+s,o+s)).resize((D,D))
from compressai.zoo import bmshj2018_hyperprior
VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().cuda()
def vae():
    t=torch.from_numpy(oa.astype(np.float32)/255).permute(2,0,1)[None].cuda()
    with torch.no_grad(): o=VAEB(t)["x_hat"].clamp(0,1)
    return Image.fromarray((o[0].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
def cr(tag): return Image.open(f"{CF}/results/defense/ext_vtv1000_ctrlregen_s{tag}/img_00000.png").convert("RGB").resize((D,D))
# (image, level, attack-name, defend-ba)  — SSIM computed live
P=[(orig,"Original","watermarked",None),
   (jpeg(25),"L1","jpeg25",0.97),(noise(0.05),"L1","noise",1.00),
   (crop(0.75),"L2","crop75*",0.96),(orig.rotate(30,resample=Image.BICUBIC),"L2","rotate30*",0.96),
   (vae(),"L3","VAE-B",0.87),(cr("03"),"L3","CtrlRegen s0.3 (mild)",0.82),
   (cr("05"),"L4","CtrlRegen+ s0.5",0.72),(cr("07"),"L4","CtrlRegen+ s0.7",0.64),(cr("09"),"L4","CtrlRegen+ s0.9",0.59)]
COL={"Original":"#333","L1":"#159e53","L2":"#0a78b8","L3":"#d1791f","L4":"#cc2222"}
fig,ax=plt.subplots(2,5,figsize=(15,7.4)); fig.subplots_adjust(hspace=0.42,wspace=0.06)
for k,(im,lv,nm,ba) in enumerate(P):
    a=ax[k//5,k%5]; a.imshow(np.asarray(im.convert("RGB"))); a.set_xticks([]); a.set_yticks([]); c=COL[lv]
    a.set_title(f"{lv} · {nm}" if lv!="Original" else "Original (watermarked)",fontsize=10.5,color=c,fontweight="bold")
    sub = "clean" if ba is None else f"SSIM {SS(im):.2f}  |  defend {ba:.2f}"
    a.set_xlabel(sub,fontsize=10,color=c)
    for s in a.spines.values(): s.set_edgecolor(c); s.set_linewidth(3)
fig.suptitle("What the image looks like after a representative attack at each level\n(same watermarked image · SSIM = actual shown image vs original · defend = post-attack bit-acc)",fontsize=12.5,y=0.99)
fig.text(0.5,0.015,"*geometric SSIM is a deformation artifact.  SSIM shown is for THIS single image (the report tables give n=1000/n=50 means, slightly lower).  L1–L3 stay visually usable.  L4 CtrlRegen+ RE-RENDERS the whole image: ControlNet keeps structure (this building stays recognizable) but every pixel is re-synthesized (SSIM ~0.5–0.6) and the mark is erased — for faces / fine textures the visible loss is larger.  UnMarker (not shown): SSIM 0.73, detection 1.00 (fully defended).",ha="center",fontsize=8.5,style="italic")
out=f"{CF}/results/defense/threat_levels_visual.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("SAVED",out)
