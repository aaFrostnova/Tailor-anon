"""Per-image VARIANCE of each attack -> which cells the offline MEAN cannot represent.

The live-vs-offline decision is driven by variance, not cost:
  * ADVERSARIAL attacks (UnMarker) optimize per-image worst-case -> offline mean is not what a real
    attacker does to THIS image -> always live.
  * HIGH-VARIANCE attacks -> the specific image's outcome can sit either side of theta(f) even when the
    mean does not -> live when |mean - theta| < k*std.
This measures mean AND std across images for the live-measurable attacks so cells can be tagged.
"""
import sys, os, io, json
import numpy as np
from PIL import Image, ImageFilter
CF=os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC=os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF,f"{CF}/scripts",f"{CF}/scripts/defense"): sys.path.insert(0,p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from composite_external_eval import scale_resid
from live_measure import ATTACKS
import glob
n=int(sys.argv[1]) if len(sys.argv)>1 else 50
sb=ShortenedBCH(); NB=sb.n
FR={"VINE":VineCryptoWrapper(b"v5_key_encoder_master","vine",NB,"cuda",variant="R"),
    "TrustMark":TrustMarkFragment(b"v5_key_encoder_master","trustmark",NB,model_type="B",device="cuda"),
    "VideoSeal":VideoSealFragment(b"v5_key_encoder_master","videoseal",NB,device="cuda")}
def hard(fn,pil):
    v=np.asarray(FR[fn].raw_probs(pil) if fn=="VINE" else FR[fn].raw_logits(pil)).ravel()[:NB]
    return (v>(0.5 if fn=="VINE" else 0.0)).astype(np.uint8)
_c={}
def vae(p,w):
    import torch;from compressai.zoo import bmshj2018_hyperprior,cheng2020_anchor
    if w not in _c:_c[w]=((bmshj2018_hyperprior if w=="B" else cheng2020_anchor)(quality=3,pretrained=True)).eval().to("cuda")
    x=torch.from_numpy(np.asarray(p,np.float32).transpose(2,0,1)[None]/255.).to("cuda")
    import torch
    with torch.no_grad():y=_c[w](x)["x_hat"].clamp(0,1)
    return Image.fromarray((y[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
ATT={**{a:ATTACKS[a] for a in ATTACKS if a in ("clean","jpeg25","jpeg50","blur","noise","bright","contrast","crop90","crop75","crop50","rot9","rot30")},
     "vaeB":lambda p:vae(p,"B"),"vaeC":lambda p:vae(p,"C")}
covers=[Image.open(f).convert("RGB").resize((512,512)) for f in sorted(glob.glob(f"{SC}/pool/*/img/*.png"))[:n]]
out={}
for fn in FR:
    for a,af in ATT.items():
        per=[]
        for i,c in enumerate(covers):
            t=np.random.RandomState(i).randint(0,2,NB).astype(np.uint8)
            emb=FR[fn].embed_with_target(c,t)
            if emb.size!=(512,512):emb=emb.resize((512,512))
            per.append(float(np.mean(hard(fn,af(emb))==t)))
        per=np.array(per)
        out[f"{fn}/{a}"]={"mean":round(float(per.mean()),4),"std":round(float(per.std()),4),
                          "min":round(float(per.min()),4),"max":round(float(per.max()),4),"n":len(per)}
    print(f"  {fn}: "+" ".join(f"{a}(s{out[fn+'/'+a]['std']:.2f})" for a in list(ATT)[:6]),flush=True)
json.dump(out,open(f"{SC}/variance_profile.json","w"),indent=2)
# 高方差排行
hi=sorted(out.items(),key=lambda kv:-kv[1]["std"])[:12]
print("\n=== 图间方差最高的 12 个格子(std) ===")
for k,v in hi: print(f"  {k:22s} mean {v['mean']:.3f}  std {v['std']:.3f}  [{v['min']:.2f},{v['max']:.2f}]")
print("VARIANCE_DONE")
