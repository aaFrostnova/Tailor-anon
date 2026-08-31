"""Unified main-table runner. Backend-aware (fingerprint vs aiwm envs).
For each watermark: embed N UltraEdit images, apply attacks, decode -> bit-acc; + clean PSNR/SSIM.
Writes per-(wm,attack) rows to CSV. Run once per env; merge CSVs for the final table.
  fingerprint env: --backend fp   --wms dwtDct dwtDctSvd rivaGan trustmark vine_b
  aiwm env       : --backend aiwm --wms stega hidden
"""
import os, sys, io, glob, json, argparse, numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim_fn
ap=argparse.ArgumentParser()
ap.add_argument("--backend",required=True,choices=["fp","aiwm"])
ap.add_argument("--wms",nargs="+",required=True)
ap.add_argument("--n",type=int,default=20)
ap.add_argument("--attacks",nargs="+",default=["clean","jpeg50","jpeg25","blur","noise","bright","contrast","crop90","crop75"])
ap.add_argument("--out",required=True)
a=ap.parse_args()
SRC="/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source"
imgs=sorted(glob.glob(SRC+"/*.png"))[:a.n]
RES=512
def to512(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
# ---------- attacks (PIL, in-env cheap) ----------
def att(name, im):
    if name=="clean": return im
    if name=="jpeg50": b=io.BytesIO(); im.save(b,"JPEG",quality=50); b.seek(0); return Image.open(b).convert("RGB")
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise": x=np.asarray(im,np.float32)/255+np.random.randn(RES,RES,3).astype(np.float32)*0.05; return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="bright": return ImageEnhance.Brightness(im).enhance(0.5)
    if name=="contrast": return ImageEnhance.Contrast(im).enhance(0.5)
    # --- crop family: CenterCrop(ratio) + Resize back (RAVEN/WAVES CropLayer, geometric) ---
    if name=="crop90":  # RAVEN default: ratio 0.9 (crop 10%) -> mild geometric zoom
        s=int(RES*0.90); o=(RES-s)//2; return im.crop((o,o,o+s,o+s)).resize((RES,RES))
    if name=="crop75":  # harsher geometric probe: ratio 0.75 (crop 25%)
        s=int(RES*0.75); o=(RES-s)//2; return im.crop((o,o,o+s,o+s)).resize((RES,RES))
    raise ValueError(name)
def psnr(a_,b_):
    mse=np.mean((np.asarray(a_,np.float32)-np.asarray(b_,np.float32))**2); return 99.0 if mse<1e-9 else 10*np.log10(255**2/mse)
def ssim(a_,b_): return float(ssim_fn(np.asarray(a_.convert("L")),np.asarray(b_.convert("L"))))
# ---------- backend: watermark wrapper (embed(pil)->pil_wm, decode(pil)->bits, .bits) ----------
def make_backend(name):
    if a.backend=="fp":
        sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts")
        from wbench.methods import build_methods  # {name: method(embed(pil,bits)/decode(pil)/.n_bits)}
        m=build_methods([name],"cuda")[name]; nbits=m.n_bits
        rng=np.random.RandomState(0); bits=rng.randint(0,2,nbits).tolist()
        class W:
            def embed(self,pil): return to512(m.embed(to512(pil),bits))
            def decode(self,pil): return np.array(m.decode(to512(pil)))[:nbits]
            gt=np.array(bits)
        return W()
    else:  # aiwm
        import torch
        sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/external/ai-watermark/systems")
        import importlib; iw=importlib.import_module("watermarkers").init_watermarker
        wm=iw(name,batch_size=1,device="cuda")
        def t(pil): return torch.from_numpy(np.asarray(to512(pil),np.float32)/255).permute(2,0,1)[None].cuda()
        def p(ten): return Image.fromarray((ten.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
        class W:
            def embed(self,pil): return p(wm.encode(t(pil)))
            def decode(self,pil):
                raw=wm.decode_batch_raw(t(pil)); return (raw.detach().cpu().numpy()[0]>0.5).astype(int)
            gt=(wm.watermark.detach().cpu().numpy()[0]>0.5).astype(int) if wm.watermark is not None else None
        return W()
rows=[]
for wname in a.wms:
    W=make_backend(wname)
    per={at:[] for at in a.attacks}; ps=[]; ss=[]
    for ip in imgs:
        cov=to512(Image.open(ip).convert("RGB"))
        wm_im=W.embed(cov)
        ps.append(psnr(cov,wm_im)); ss.append(ssim(cov,wm_im))
        for at in a.attacks:
            dec=W.decode(att(at,wm_im))
            n=min(len(dec),len(W.gt)); per[at].append(float(np.mean(dec[:n]==W.gt[:n])))
    row={"wm":wname,"PSNR":round(float(np.mean(ps)),2),"SSIM":round(float(np.mean(ss)),3)}
    for at in a.attacks: row[at]=round(float(np.mean(per[at])),3)
    rows.append(row); print(json.dumps(row),flush=True)
os.makedirs(os.path.dirname(a.out),exist_ok=True)
json.dump(rows,open(a.out,"w"),indent=2)
print(f"SAVED {a.out}\nRUN_DONE",flush=True)
