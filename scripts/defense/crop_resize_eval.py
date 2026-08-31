"""crop-ratio sweep: fidelity (PSNR/SSIM/LPIPS vs clean) + composite detection (VINE+TrustMark fused, dual).
Question: at light crops (crop90/95), is the attack >= regen fidelity (legal), and does TrustMark carry it?"""
import glob, os, sys, numpy as np
from PIL import Image
import torch, lpips
from scipy.stats import binom
from skimage.metrics import structural_similarity as ssim
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO)
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
RES=512; dev="cuda"; N=12
sb=ShortenedBCH(); tau=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits")}
frag={"vine":VineCryptoWrapper(master_key=b"v5_key_encoder_master",method_name="vine",n_bits=sb.n,device=dev),
      "trustmark":TrustMarkFragment(master_key=b"v5_key_encoder_master",method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)}
lp=lpips.LPIPS(net='alex').to(dev).eval()
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
def psnr(a,b):
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 10*np.log10(255.0**2/m)
def lpd(a,b):
    ta=torch.from_numpy(a/127.5-1).permute(2,0,1)[None].float().to(dev); tb=torch.from_numpy(b/127.5-1).permute(2,0,1)[None].float().to(dev)
    with torch.no_grad(): return float(lp(ta,tb).item())
def crop(pil,r):
    cw=int(round(RES*r)); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
ratios=[0.95,0.90,0.85,0.80,0.75,0.50]
files=sorted(glob.glob('/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg'))[4000:4000+N]
R={r:{"psnr":[],"ssim":[],"lpips":[],"vine":[],"tm":[],"fba":[],"or":[]} for r in ratios}
for i,fp in enumerate(files):
    C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"cr_{i:05d}"; Ca=arr(C)
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); img=C
    for nm in ["vine","trustmark"]:
        p,M=frag[nm].get_perm_M(iid); img=frag[nm].embed_with_target(img,apply_crypto(tx,p,M))
        if img.size!=(RES,RES): img=img.resize((RES,RES))
    for r in ratios:
        At=crop(img,r); Aa=arr(At)
        R[r]["psnr"].append(psnr(Aa,Ca)); R[r]["ssim"].append(ssim(Aa,Ca,channel_axis=2,data_range=255)); R[r]["lpips"].append(lpd(Aa,Ca))
        al={}; per={}
        for nm in ["vine","trustmark"]:
            kind,getter=SPEC[nm]; p,M=frag[nm].get_perm_M(iid); a=method_soft_to_codeword_llr(getattr(frag[nm],getter)(At),p,M,kind=kind,n_codeword=sb.n)
            al[nm]=a; per[nm]=float(np.mean(llr_to_bits(a)==tx))
        fused=fuse_llrs(al,weights=None,n_codeword=sb.n); fba=float(np.mean(llr_to_bits(fused)==tx)); fver=bool(decode_and_verify(fused,iid,codec=sb)["detected"])
        R[r]["vine"].append(per["vine"]); R[r]["tm"].append(per["trustmark"]); R[r]["fba"].append(fba)
        R[r]["or"].append(1.0 if (fba>=tau or fver) else 0.0)
    print(f"[{i+1}/{N}]",flush=True)
print(f"\n=== crop-ratio sweep (n={N}; regen anchor: PSNR>=20.1 SSIM>=0.54 LPIPS<=0.31) ===")
print(f"{'crop':>6}{'kept%':>7}{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'>=regen?':>9}{'VINE':>7}{'TM':>7}{'fused':>7}{'comp_or':>9}")
for r in ratios:
    p,s,l=np.mean(R[r]['psnr']),np.mean(R[r]['ssim']),np.mean(R[r]['lpips'])
    leg=(p>=20.1-0.5 and s>=0.54-0.02 and l<=0.31+0.02)
    print(f"{r:>6.2f}{int(r*100):>7}{p:>7.1f}{s:>7.3f}{l:>7.3f}{('LEGAL' if leg else 'excl'):>9}{np.mean(R[r]['vine']):>7.3f}{np.mean(R[r]['tm']):>7.3f}{np.mean(R[r]['fba']):>7.3f}{np.mean(R[r]['or']):>9.3f}")
print("CROP_SWEEP_DONE")
