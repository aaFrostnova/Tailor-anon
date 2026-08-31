"""Measure Watermark-Anything (WAM, Meta) as an SMT candidate fragment: bit-acc across the
full attack suite (SIG/GEO/REG) + clean PSNR + embed/decode latency, SOLO, native 32-bit msg.
Mirrors the frag_suite / measure_c3ft methodology so the numbers drop into the SMT candidate set.
Prior (already-observed wbench_regen): WAM regen-dead (detection 1.0 -> 0.0). This quantifies the
full vector so the solver can decide if WAM is ever on the Pareto front."""
import os,sys,io,time,glob,numpy as np,torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF+"/scripts/defense")
from _regen_util import build_regen_pipe, stable_regen
dev="cuda"; RES=256
WAM_ROOT="/project/pi_shiqingma_umass_edu/mingzheli/watermark/wam/repo"
sys.path.insert(0,WAM_ROOT)
# ---- load WAM (chdir for its relative config paths) ----
from notebooks.inference_utils import load_model_from_checkpoint, default_transform, unnormalize_img
from watermark_anything.data.metrics import msg_predict_inference
import torch.nn.functional as F
_cwd=os.getcwd(); os.chdir(WAM_ROOT)
try:
    wam=load_model_from_checkpoint("checkpoints/params.json",
        "/project/pi_shiqingma_umass_edu/mingzheli/watermark/wam/models/checkpoint.pth").to(dev).eval()
finally: os.chdir(_cwd)
MSGBITS=32
torch.manual_seed(0); MSG=torch.randint(0,2,(1,MSGBITS)).float().to(dev)
def embed(pil):
    x=default_transform(pil).unsqueeze(0).to(dev)
    out=wam.embed(x,MSG); wm=unnormalize_img(out["imgs_w"]).clamp(0,1)[0]
    a=(wm.detach().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    p=Image.fromarray(a,"RGB")
    return p.resize(pil.size,Image.BILINEAR) if p.size!=pil.size else p
def decode_ba(pil):
    x=default_transform(pil).unsqueeze(0).to(dev)
    preds=wam.detect(x)["preds"]; mask=F.sigmoid(preds[:,0,:,:]); bits=preds[:,1:,:,:]
    pm=msg_predict_inference(bits,mask).cpu().float()
    return float((pm==MSG.cpu()).float().mean())
def psnr(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b.resize(a.shape[1::-1]) if b.size!=a.shape[1::-1] else b,np.float64)
    m=np.mean((a-b)**2); return 99. if m<1e-9 else 10*np.log10(255**2/m)
def _t(p): return TF.to_tensor(p).unsqueeze(0).to(dev)
def _p(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy())
VAEB=None
def get_vaeB():
    global VAEB
    if VAEB is None:
        from compressai.zoo import bmshj2018_hyperprior; VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev)
    return VAEB
pipe=None
def attack(name,im):
    global pipe
    if name=="clean": return im
    if name=="jpeg50": b=io.BytesIO(); im.save(b,"JPEG",quality=50); b.seek(0); return Image.open(b).convert("RGB")
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise":
        x=np.asarray(im,np.float32)/255+np.random.RandomState(0).randn(*np.asarray(im).shape).astype(np.float32)*0.05
        return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="crop90":
        W,H=im.size; c=int(min(W,H)*0.9); l=(W-c)//2; t=(H-c)//2; return im.crop((l,t,l+c,t+c)).resize((W,H),Image.BILINEAR)
    if name=="crop75":
        W,H=im.size; c=int(min(W,H)*0.75); l=(W-c)//2; t=(H-c)//2; return im.crop((l,t,l+c,t+c)).resize((W,H),Image.BILINEAR)
    if name=="rot9": return _p(TF.rotate(_t(im),9.0))
    if name=="rot30": return _p(TF.rotate(_t(im),30.0))
    if name=="vaeB":
        with torch.no_grad(): return _p(get_vaeB()(_t(im))["x_hat"].clamp(0,1))
    if name in ("regen","rinse"):
        if pipe is None: pipe=build_regen_pipe()
        r=stable_regen(pipe,im,1234)
        return stable_regen(pipe,r,1235) if name=="rinse" else r
ATTACKS=["clean","jpeg50","jpeg25","blur","noise","crop90","crop75","rot9","rot30","vaeB","regen","rinse"]
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
acc={a:[] for a in ATTACKS}; ps=[]; te=[]; td=[]
for i,fp in enumerate(files):
    orig=Image.open(fp).convert("RGB").resize((RES,RES))
    t0=time.time(); wm=embed(orig); te.append(time.time()-t0)
    ps.append(psnr(orig,wm))
    for a in ATTACKS:
        atk=attack(a,wm)
        t0=time.time(); ba=decode_ba(atk);
        if a=="clean": td.append(time.time()-t0)
        acc[a].append(ba)
    if (i+1)%10==0: print(f"  [{i+1}/{N}]",flush=True)
print(f"\n=== WAM (native {MSGBITS}-bit) solo, n={N} ===")
print(f"{'attack':10s} {'bit_acc':>8s}")
for a in ATTACKS: print(f"{a:10s} {np.mean(acc[a]):8.3f}")
print(f"\nclean PSNR {np.mean(ps):.2f} dB  ·  embed {np.mean(te)*1000:.0f} ms · decode {np.mean(td)*1000:.0f} ms")
import json
out={a:round(float(np.mean(acc[a])),4) for a in ATTACKS}
out.update(psnr=round(float(np.mean(ps)),2),embed_ms=round(float(np.mean(te)*1000),1),decode_ms=round(float(np.mean(td)*1000),1),n_bits=MSGBITS)
json.dump(out,open(CF+"/results/defense/wam_frag.json","w"),indent=2)
print("saved -> results/defense/wam_frag.json\nWAM_DONE")
