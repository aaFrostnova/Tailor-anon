"""Measure the REAL inputs for the SMT watermark optimizer (replaces estimated/calibrated values):
 (1) full-suite bit-acc for VINE / TrustMark / VideoSeal   (fills VideoSeal estimates)
 (2) clean PSNR per fragment                                (real quality cost)
 (3) embed_ms + decode_ms per fragment                      (real speed cost)
 (4) VINE alpha-sweep: PSNR + bit-acc vs strength alpha     (enables continuous-alpha in the solver)
Output -> results/defense/smt_inputs.json"""
import os,sys,io,glob,json,time,numpy as np,torch
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim_fn
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF+"/scripts"); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal"); sys.path.insert(0,CF)
from wbench.methods import build_methods
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
dev="cuda"; RES=256
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def pil2t(im): return torch.from_numpy(np.asarray(im.convert("RGB"),np.float32)/255).permute(2,0,1)[None].to(dev)
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev); VAEC=cheng2020_anchor(quality=3,pretrained=True).eval().to(dev)
def vae(im,m):
    with torch.no_grad(): o=m(pil2t(to(im)))["x_hat"].clamp(0,1)
    return Image.fromarray((o[0].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
def crop(im,r): s=int(RES*r); o=(RES-s)//2; return im.crop((o,o,o+s,o+s)).resize((RES,RES))
def att(name,im,k):
    if name=="clean": return im
    if name=="jpeg50": b=io.BytesIO(); im.save(b,"JPEG",quality=50); b.seek(0); return Image.open(b).convert("RGB")
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise": x=np.asarray(im,np.float32)/255+np.random.RandomState(k).randn(RES,RES,3).astype(np.float32)*0.05; return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="bright": return ImageEnhance.Brightness(im).enhance(0.5)
    if name=="contrast": return ImageEnhance.Contrast(im).enhance(0.5)
    if name=="vaeB": return vae(im,VAEB)
    if name=="vaeC": return vae(im,VAEC)
    if name=="crop90": return crop(im,0.9)
    if name=="crop75": return crop(im,0.75)
    if name=="rot9": return im.rotate(9,resample=Image.BICUBIC)
    if name=="rot30": return im.rotate(30,resample=Image.BICUBIC)
def psnr(a,b):
    mse=np.mean((np.asarray(a,np.float32)-np.asarray(b,np.float32))**2); return 99.0 if mse<1e-9 else 10*np.log10(255**2/mse)
ATTS=["clean","jpeg50","jpeg25","blur","noise","bright","contrast","vaeB","vaeC","crop90","crop75","rot9","rot30"]
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
covers=[to(Image.open(f).convert("RGB")) for f in files]
rng=np.random.RandomState(0); OUT={"n":N,"fragments":{},"alpha_sweep":{}}

# ---- VINE / TrustMark via build_methods (embed(pil,bits)->pil, decode(pil)->bits) ----
M=build_methods(["vine_r","trustmark_b"],dev)
def blend(cov,wm,a): c=np.asarray(to(cov),np.float64); w=np.asarray(to(wm),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def measure_method(name,m):
    bits=rng.randint(0,2,m.n_bits).tolist(); gt=np.array(bits)
    # timing (single img, warm)
    _=to(m.embed(covers[0],bits)); t0=time.perf_counter()
    for c in covers[:8]: to(m.embed(c,bits))
    emb_ms=(time.perf_counter()-t0)/8*1000
    wms=[to(m.embed(c,bits)) for c in covers]; ps=[psnr(c,w) for c,w in zip(covers,wms)]
    t0=time.perf_counter()
    for w in wms[:8]: m.decode(w)
    dec_ms=(time.perf_counter()-t0)/8*1000
    acc={}
    for a in ATTS:
        v=[]
        for k,w in enumerate(wms):
            d=np.array(m.decode(att(a,w,k)))[:m.n_bits]; n=min(len(d),len(gt)); v.append(float(np.mean(d[:n]==gt[:n])))
        acc[a]=round(float(np.mean(v)),3)
    OUT["fragments"][name]={"bit_acc":acc,"psnr":round(float(np.mean(ps)),2),"embed_ms":round(emb_ms,1),"decode_ms":round(dec_ms,1)}
    print(f"  {name}: PSNR {np.mean(ps):.2f}  embed {emb_ms:.0f}ms  decode {dec_ms:.0f}ms",flush=True)
    return wms,bits
vine_wms,vine_bits=measure_method("VINE",M["vine_r"])
measure_method("TrustMark",M["trustmark_b"])

# ---- VideoSeal via fragment class ----
sb=ShortenedBCH(); vs=VideoSealFragment(n_bits=sb.n,device=dev)
def vs_decode_bits(im): return (vs.raw_logits(to(im))>0).astype(int)
tgt=rng.randint(0,2,sb.n);
_=vs.embed_with_target(covers[0],tgt); t0=time.perf_counter()
for c in covers[:8]: vs.embed_with_target(c,tgt)
vs_emb=(time.perf_counter()-t0)/8*1000
vs_wms=[to(vs.embed_with_target(c,tgt)) for c in covers]; vs_ps=[psnr(c,w) for c,w in zip(covers,vs_wms)]
t0=time.perf_counter()
for w in vs_wms[:8]: vs.raw_logits(w)
vs_dec=(time.perf_counter()-t0)/8*1000
vs_acc={}
for a in ATTS:
    v=[float(np.mean(vs_decode_bits(att(a,w,k))[:sb.n]==tgt[:sb.n])) for k,w in enumerate(vs_wms)]; vs_acc[a]=round(float(np.mean(v)),3)
OUT["fragments"]["VideoSeal"]={"bit_acc":vs_acc,"psnr":round(float(np.mean(vs_ps)),2),"embed_ms":round(vs_emb,1),"decode_ms":round(vs_dec,1)}
print(f"  VideoSeal: PSNR {np.mean(vs_ps):.2f}  embed {vs_emb:.0f}ms  decode {vs_dec:.0f}ms",flush=True)

# ---- (4) VINE alpha-sweep: embed full, blend at alpha, PSNR + bit-acc on key attacks ----
KEY=["clean","jpeg25","crop75","noise"]
for al in [0.3,0.5,0.7,1.0]:
    rec={"psnr":[]}; accs={a:[] for a in KEY}
    for k,(c,w) in enumerate(zip(covers,vine_wms)):
        wa=blend(c,w,al); rec["psnr"].append(psnr(c,wa))
        for a in KEY:
            d=np.array(M["vine_r"].decode(att(a,wa,k)))[:len(vine_bits)]; accs[a].append(float(np.mean(d==np.array(vine_bits)[:len(d)])))
    OUT["alpha_sweep"][f"{al}"]={"psnr":round(float(np.mean(rec["psnr"])),2),**{a:round(float(np.mean(accs[a])),3) for a in KEY}}
    print(f"  VINE alpha={al}: PSNR {OUT['alpha_sweep'][str(al)]['psnr']}  " + " ".join(f"{a} {OUT['alpha_sweep'][str(al)][a]}" for a in KEY),flush=True)

json.dump(OUT,open(CF+"/results/defense/smt_inputs.json","w"),indent=1)
print("SAVED results/defense/smt_inputs.json\nMEASURE_DONE")
