"""W-Bench evaluation of the deployed 3-frag+head composite on the OFFICIAL W-Bench dataset.
For one task: embed VINE+TM+VideoSeal (alpha=0.70) on N official W-Bench source images, apply
that task's W-Bench attack, decode (head3 + best-path), report detection (composite_or) + quality.

Tasks:
  distortion  : conventional distortions (jpeg/blur/noise/crop/rotate) -> one row per distortion
  sto_regen   : stochastic regeneration (SD img2img, strength s)
  instruct    : global editing (InstructPix2Pix) with the dataset's per-image instruction
  local       : local editing (img2img-resynthesise the masked box, composite back) w/ dataset mask
"""
import os, sys, glob, json, io, argparse, csv
import numpy as np, torch
from PIL import Image, ImageFilter
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts")]: sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.fusion_head3 import load_head3
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
import lpips as lpips_mod
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0; ALPHA=0.70
sb=ShortenedBCH(); n=sb.n; tau=float(binom.ppf(0.99,n,0.5)+1)/n; TAU01=0.66
ap=argparse.ArgumentParser()
ap.add_argument("--task",required=True)
ap.add_argument("--wb_dir",required=True,help="W-Bench task image dir (sorted glob)")
ap.add_argument("--n",type=int,default=500)
ap.add_argument("--strength",type=float,default=0.6,help="regen/local strength")
ap.add_argument("--strength_list",default="0.2,0.3,0.4,0.5,0.6",help="for sto_sweep")
ap.add_argument("--caption_csv",default="")
ap.add_argument("--mask_dir",default="")
ap.add_argument("--out",default="")
a=ap.parse_args()

vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)
head=load_head3(os.path.join(REPO,"results/defense/frag3_head.pt"),dev)
lp=lpips_mod.LPIPS(net='alex').to(dev).eval()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def scale(c,w):
    C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64)
    return Image.fromarray(np.clip(C+ALPHA*(W-C),0,255).astype(np.uint8))
def embed(orig,iid):
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    x=scale(orig,to512(vine.embed_with_target(orig,apply_crypto(tx,pv,Mv))))
    x=scale(x,   to512(tm.embed_with_target(x,apply_crypto(tx,pt,Mt))))
    x=scale(x,   to512(vsf.embed_with_target(x,apply_crypto(tx,ps,Ms))))
    return x,tx.astype(np.uint8)
def lpips_d(x,y):
    ta=torch.tensor(np.asarray(x,np.float32)/127.5-1).permute(2,0,1)[None].to(dev)
    tb=torch.tensor(np.asarray(y,np.float32)/127.5-1).permute(2,0,1)[None].to(dev)
    with torch.no_grad(): return float(lp(ta,tb).item())
def decode(att,iid,tx):
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    av=np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob", n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
    at=np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
    as_=np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
    with torch.no_grad():
        H=head(torch.tensor(av[None],device=dev),torch.tensor(at[None],device=dev),torch.tensor(as_[None],device=dev)).cpu().numpy()[0]
    ver=bool(decode_and_verify(H,iid,codec=sb)["detected"]); zb=((H>0).astype(np.uint8)==tx).mean()>=tau
    best=any(bool(decode_and_verify(F,iid,codec=sb)["detected"]) for F in (av,at,as_))
    det=1.0 if (ver or zb or best) else 0.0
    return det, float(((H>0).astype(np.uint8)==tx).mean())

# attack model loaders (lazy)
_P={}
SD21_LOCAL="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
def sd_img2img():
    if "i2i" not in _P:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        pipe=StableDiffusionImg2ImgPipeline.from_pretrained(SD21_LOCAL,torch_dtype=torch.float16,safety_checker=None,local_files_only=True).to(dev)
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True); _P["i2i"]=pipe
    return _P["i2i"]
def ip2p():
    if "ip2p" not in _P:
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        pipe=StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",torch_dtype=torch.float16,safety_checker=None).to(dev)
        pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True); _P["ip2p"]=pipe
    return _P["ip2p"]
def att_sto_regen(im,seed):
    g=torch.Generator(dev).manual_seed(seed)
    return sd_img2img()(prompt="",image=im,strength=a.strength,num_inference_steps=50,guidance_scale=1.0,generator=g).images[0].resize((512,512))
def att_instruct(im,target_caption,seed):
    # W-Bench global editing: re-render toward the dataset's target_caption via SD img2img
    g=torch.Generator(dev).manual_seed(seed)
    return sd_img2img()(prompt=target_caption,image=im,strength=a.strength,num_inference_steps=50,guidance_scale=7.5,generator=g).images[0].resize((512,512))
def svd_pipe():
    if "svd" not in _P:
        from diffusers import StableVideoDiffusionPipeline
        pipe=StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",torch_dtype=torch.float16,variant="fp16",local_files_only=True).to(dev)
        pipe.set_progress_bar_config(disable=True); _P["svd"]=pipe
    return _P["svd"]
def att_svd(im,seed):
    # W-Bench image-to-video: SVD generates a video conditioned on the watermarked image.
    g=torch.Generator(dev).manual_seed(seed)
    cond=im.resize((1024,576))  # SVD-XT native
    frames=svd_pipe()(cond,num_frames=14,decode_chunk_size=4,motion_bucket_id=127,noise_aug_strength=0.02,generator=g).frames[0]
    return [f.convert("RGB").resize((512,512)) for f in frames]
def att_local(im,mask,seed):
    g=torch.Generator(dev).manual_seed(seed)
    regen=sd_img2img()(prompt="",image=im,strength=0.8,num_inference_steps=50,guidance_scale=1.0,generator=g).images[0].resize((512,512))
    m=(np.asarray(to512(mask).convert("L"),np.float32)/255.0)[...,None]
    out=m*np.asarray(regen,np.float32)+(1-m)*np.asarray(im,np.float32)
    return Image.fromarray(out.clip(0,255).astype(np.uint8))
DISTORT={"clean":lambda x:x,"jpeg25":lambda x:_jpeg(x,25),"jpeg10":lambda x:_jpeg(x,10),
 "blur3":lambda x:x.filter(ImageFilter.GaussianBlur(3)),"noise0.05":lambda x:_noise(x,0.05),
 "crop75":lambda x:_crop(x,0.75),"rot9":lambda x:x.rotate(9,resample=Image.BILINEAR)}
def _jpeg(x,q): b=io.BytesIO(); x.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
def _noise(x,s): a_=np.asarray(x,np.float64)+np.random.RandomState(0).normal(0,s*255,(512,512,3)); return Image.fromarray(np.clip(a_,0,255).astype(np.uint8))
def _crop(x,r): s=int(512*r); o=(512-s)//2; return x.crop((o,o,o+s,o+s)).resize((512,512))

imgs=sorted(glob.glob(os.path.join(a.wb_dir,"*.png")))[:a.n]
print(f"task={a.task} n={len(imgs)} wb_dir={a.wb_dir}",flush=True)
# captions / masks aligned by basename
caps={}  # idx -> target_caption (W-Bench INSTRUCT captions_new.csv)
if a.caption_csv and os.path.exists(a.caption_csv):
    with open(a.caption_csv) as f:
        for row in csv.DictReader(f):
            caps[str(row["idx"]).strip()]=row.get("target_caption","").strip()
def _idx_of(fp): return os.path.basename(fp).split("_")[0]
res={"task":a.task,"n":len(imgs)}
if a.task=="distortion":
    rows={k:{"det":[],"q":[]} for k in DISTORT}
    for j,fp in enumerate(imgs):
        iid=f"wb_{j:05d}"; orig=to512(Image.open(fp).convert("RGB")); wm,tx=embed(orig,iid)
        for k,fn in DISTORT.items():
            att=to512(fn(wm)); d,_=decode(att,iid,tx); rows[k]["det"].append(d)
            rows[k]["q"].append((float(_ssim(np.asarray(att),np.asarray(wm),channel_axis=2)),lpips_d(att,wm)))
        if (j+1)%50==0: print(f"  [{j+1}/{len(imgs)}]",flush=True)
    res["rows"]={k:{"det":float(np.mean(v["det"])),"ssim":float(np.mean([q[0] for q in v["q"]])),"lpips":float(np.mean([q[1] for q in v["q"]]))} for k,v in rows.items()}
    print("\n%-10s %6s %6s %6s"%("distort","det","ssim","lpips"))
    for k,v in res["rows"].items(): print("%-10s %6.3f %6.3f %6.3f"%(k,v["det"],v["ssim"],v["lpips"]))
elif a.task=="sto_sweep":
    strengths=[float(s) for s in a.strength_list.split(",")]
    rows={s:{"det":[],"q":[]} for s in strengths}
    pipe=sd_img2img()
    for j,fp in enumerate(imgs):
        iid=f"wb_{j:05d}"; orig=to512(Image.open(fp).convert("RGB")); wm,tx=embed(orig,iid)
        for s in strengths:
            g=torch.Generator(dev).manual_seed(7000+j)
            att=to512(pipe(prompt="",image=wm,strength=s,num_inference_steps=50,guidance_scale=1.0,generator=g).images[0])
            d,_=decode(att,iid,tx); rows[s]["det"].append(d)
            rows[s]["q"].append((float(_ssim(np.asarray(att),np.asarray(wm),channel_axis=2)),float(_psnr(np.asarray(wm),np.asarray(att),data_range=255)),lpips_d(att,wm)))
        if (j+1)%50==0: print(f"  [{j+1}/{len(imgs)}]",flush=True)
    res["sweep"]={str(s):{"det":float(np.mean(v["det"])),"ssim":float(np.mean([q[0] for q in v["q"]])),"psnr":float(np.mean([q[1] for q in v["q"]])),"lpips":float(np.mean([q[2] for q in v["q"]]))} for s,v in rows.items()}
    print("\n%6s %6s %6s %6s"%("s","det","ssim","psnr"))
    for s,v in res["sweep"].items(): print("%6s %6.3f %6.3f %6.1f"%(s,v["det"],v["ssim"],v["psnr"]))
elif a.task=="svd":
    framedet,anydet,ssim_,lpips_=[],[],[],[]
    for j,fp in enumerate(imgs):
        iid=f"wb_{j:05d}"; orig=to512(Image.open(fp).convert("RGB")); wm,tx=embed(orig,iid)
        frames=att_svd(wm,7000+j)
        dets=[decode(f,iid,tx)[0] for f in frames]
        framedet.append(float(np.mean(dets))); anydet.append(1.0 if max(dets)>0 else 0.0)
        mid=frames[len(frames)//2]; ssim_.append(float(_ssim(np.asarray(mid),np.asarray(wm),channel_axis=2))); lpips_.append(lpips_d(mid,wm))
        if (j+1)%20==0: print(f"  [{j+1}/{len(imgs)}] frame_det={np.mean(framedet):.3f}",flush=True)
    res.update(frame_det=float(np.mean(framedet)),any_frame_det=float(np.mean(anydet)),ssim=float(np.mean(ssim_)),lpips=float(np.mean(lpips_)),n_frames=14)
    print(f"\nsvd: mean-frame-det={res['frame_det']:.3f} any-frame-det={res['any_frame_det']:.3f} SSIM={res['ssim']:.3f} (n={len(imgs)})")
else:
    det,ssim_,psnr_,lpips_=[],[],[],[]
    for j,fp in enumerate(imgs):
        iid=f"wb_{j:05d}"; orig=to512(Image.open(fp).convert("RGB")); wm,tx=embed(orig,iid)
        if a.task=="sto_regen": att=att_sto_regen(wm,7000+j)
        elif a.task=="instruct":
            instr=caps.get(_idx_of(fp),"a high quality photo"); att=att_instruct(wm,instr,7000+j)
        elif a.task=="local":
            mp=os.path.join(a.mask_dir,os.path.basename(fp))
            mask=Image.open(mp) if os.path.exists(mp) else Image.new("L",(512,512),0); att=att_local(wm,mask,7000+j)
        else: raise SystemExit("bad task")
        att=to512(att); d,_=decode(att,iid,tx); det.append(d)
        ssim_.append(float(_ssim(np.asarray(att),np.asarray(wm),channel_axis=2))); psnr_.append(float(_psnr(np.asarray(wm),np.asarray(att),data_range=255))); lpips_.append(lpips_d(att,wm))
        if (j+1)%50==0: print(f"  [{j+1}/{len(imgs)}] det={np.mean(det):.3f}",flush=True)
    res.update(det=float(np.mean(det)),ssim=float(np.mean(ssim_)),psnr=float(np.mean(psnr_)),lpips=float(np.mean(lpips_)))
    print(f"\n{a.task}: detection={res['det']:.3f}  SSIM={res['ssim']:.3f}  PSNR={res['psnr']:.1f}  LPIPS={res['lpips']:.3f}  (n={len(imgs)})")
out=a.out or os.path.join(REPO,f"results/defense/wbench_{a.task}.json")
json.dump(res,open(out,"w"),indent=2); print("WBENCH_TASK_DONE",a.task)
