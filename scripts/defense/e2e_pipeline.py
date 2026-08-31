"""ONE image through the WHOLE pipeline, live, with per-step latency:
  embed composite -> (optional resync mark) -> attack -> (optional resync unwarp) -> decode -> dual-detect.
Usage: python e2e_pipeline.py --image <path|index> --attack rot30 [--resync]
Prints each step's wall-clock ms + result, so you SEE one image go end-to-end (not batch stats)."""
import os,sys,io,glob,time,argparse,numpy as np,torch
from PIL import Image, ImageEnhance, ImageFilter
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
KEY=b"v5_key_encoder_master"; dev="cuda"; RES=512; TAU=0.63
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def sr(c,w,a):
    c=np.asarray(to(c),np.float64); w=np.asarray(to(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def align1d(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]
def psnr(a,b): mse=np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2); return 99. if mse<1e-9 else 10*np.log10(255**2/mse)
class T:
    def __init__(s,label): s.l=label
    def __enter__(s): torch.cuda.synchronize(); s.t=time.perf_counter(); return s
    def __exit__(s,*a): torch.cuda.synchronize(); s.ms=(time.perf_counter()-s.t)*1000; print(f"    ⏱  {s.ms:7.1f} ms")
ap=argparse.ArgumentParser()
ap.add_argument("--image",default="1500"); ap.add_argument("--attack",default="rot30")
ap.add_argument("--resync",action="store_true"); a=ap.parse_args()
print("═"*64); print(f"  单管线端到端  ·  attack={a.attack}  ·  resync={a.resync}"); print("═"*64)
print("加载模型(一次性,不计入管线耗时)…",flush=True)
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
sync=None
if a.resync:
    from src.syncseal_frontend import load_sync; sync=load_sync()
VAEB=None
if a.attack=="vaeB":
    from compressai.zoo import bmshj2018_hyperprior; VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev)
pipe=None
if a.attack in ("regen","rinse"):
    from _regen_util import build_regen_pipe, stable_regen; pipe=build_regen_pipe()
import torchvision.transforms.functional as TF
def _t(pil): return TF.to_tensor(to(pil)).unsqueeze(0).to(dev)
def _p(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy())
def attack(name,im):
    if name=="clean": return im
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise": x=np.asarray(im,np.float32)/255+np.random.RandomState(0).randn(RES,RES,3).astype(np.float32)*0.05; return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="crop75": return _p(TF.resize(_t(im)[:,:,64:64+384,64:64+384],[RES,RES],antialias=True))
    if name=="rot9": return _p(TF.rotate(_t(im),9.0))
    if name=="rot30": return _p(TF.rotate(_t(im),30.0))
    if name=="vaeB":
        with torch.no_grad(): return _p(VAEB(_t(im))["x_hat"].clamp(0,1))
    if name in ("regen","rinse"):
        from _regen_util import stable_regen; r=stable_regen(pipe,to(im),1234)
        return stable_regen(pipe,to(r),1235) if name=="rinse" else r
def crypto_verify(hard,pay):
    # sb.decode returns (data_bits or None, n_err); crypto-verify = BCH decode succeeds AND payload matches
    data,nerr=sb.decode(np.asarray(hard,np.uint8))
    return int(data is not None and np.array_equal(data,pay))
# ---------- pick image ----------
src=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))
path=a.image if os.path.exists(a.image) else src[int(a.image)]
iid="demo_001"; pay=image_id_to_payload(iid,n_bits=sb.data_bits); tx=sb.encode(pay)
orig=to(Image.open(path).convert("RGB"))
pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
print(f"\n图: {os.path.basename(path)}   身份 ID(37bit payload): {''.join(map(str,pay[:37]))}")
print("\n[0] warmup(冷启动:CUDA 内核编译/autotune —— 第一张图必付,一次性)")
with T("warmup embed+decode"):
    _w=to(vine.embed_with_target(orig,apply_crypto(tx,pv_p,Mv))); _=vine.raw_probs(_w); _=tm.raw_logits(_w); _=vs.raw_logits(_w)
print("    ↑ 这是冷启动;下面各步是 warm(部署稳定态)延迟")
print("\n[1] 嵌入复合水印  VideoSeal@0.7 → TrustMark@0.7 → VINE@1.0")
with T("embed"):
    x1=sr(orig,to(vs.embed_with_target(orig,apply_crypto(tx,ps_p,Ms))),0.70)
    x2=sr(x1,to(tm.embed_with_target(x1,apply_crypto(tx,pt_p,Mt))),0.70)
    comp=sr(x2,to(vine.embed_with_target(x2,apply_crypto(tx,pv_p,Mv))),1.00)
print(f"    嵌入后 PSNR vs 原图: {psnr(orig,comp):.2f} dB  (水印近乎不可见)")
img=comp
if a.resync:
    print("\n[1b] 加 SyncSeal 同步标记(供解码端摆正几何)")
    with T("sync-embed"): img=_p(sync.embed(_t(img))["imgs_w"])
print(f"\n[2] 施加攻击: {a.attack}")
with T("attack"): atk=attack(a.attack,img)
dec_in=atk
if a.resync:
    print("\n[3] SyncSeal 检测 4 角 + 透视 unwarp(几何摆正)")
    with T("resync-unwarp"):
        d=sync.detect(_t(atk)); dec_in=_p(sync.unwarp(_t(atk),d["preds_pts"],(RES,RES)))
    print(f"    sync 存在分数: {float(d['preds'][0,0]):+.3f}")
print("\n[4] 解码三帧 → 去白化(align σ,M)→ 码字 LLR")
with T("decode 3 frags"):
    LV=align1d(np.clip(p2l(vine.raw_probs(dec_in)),-15,15),pv_p,Mv)
    LT=align1d(np.clip(tm.raw_logits(dec_in),-15,15),pt_p,Mt)
    LS=method_soft_to_codeword_llr(vs.raw_logits(dec_in),ps_p,Ms,kind="logit",n_codeword=sb.n)
ba=lambda L:((L>0).astype(int)==tx).mean()
print(f"    VINE      bit-acc {ba(LV):.3f}   {'✓ crypto-verify' if crypto_verify((LV>0).astype(int),pay) else '– (no id)'}")
print(f"    TrustMark bit-acc {ba(LT):.3f}   {'✓ crypto-verify' if crypto_verify((LT>0).astype(int),pay) else '– (no id)'}")
print(f"    VideoSeal bit-acc {ba(LS):.3f}   {'✓ crypto-verify' if crypto_verify((LS>0).astype(int),pay) else '– (no id)'}")
print("\n[5] 软融合(equal-MRC)+ 双重检测")
with T("fuse + detect"):
    LF=np.clip(LV,-15,15)+np.clip(LT,-15,15)+np.clip(LS,-15,15); baf=ba(LF)
    presence = baf>=TAU
    idv = crypto_verify((LF>0).astype(int),pay) or crypto_verify((LV>0).astype(int),pay) or crypto_verify((LT>0).astype(int),pay) or crypto_verify((LS>0).astype(int),pay)
print(f"    融合 bit-acc {baf:.3f}")
print(f"    presence 层: {baf:.3f} {'≥' if presence else '<'} τ={TAU}  → {'检出 ✓' if presence else '未检出 ✗'}")
print(f"    identity 层: crypto-verify (FPR 2⁻³⁷)  → {'身份验证通过 ✓' if idv else '未通过 ✗'}")
print("\n"+"═"*64)
verdict = presence or idv
print(f"  最终判定: {'✅ 水印检出' + (' + 身份已验证' if idv else ' (presence)') if verdict else '❌ 未检出（此攻击下丢失）'}")
print("═"*64)
