"""SEARCH failure or INFORMATION failure? Decode the inner nested layers at the ORACLE view.

UnMarker preprocess = CenterCrop(460)+Resize(512) -> zoom 512/460 = 1.1130.
Layer K therefore sits at fraction K*1.1130 of the attacked frame (K=1.0 -> 1.113, cut off).
We decode VINE at the EXACT oracle scale and sweep finely around it, on three sets:
  A) full UnMarker (crop+scrub)   B) crop-0.9 only   C) pure-spectral only (no crop, layer stays at K)
If the oracle view decodes on (A) -> it is a SEARCH-GRID problem (fixable, finer grid).
If it does not                    -> the information is genuinely gone (search cannot help)."""
import os,sys,glob,json,numpy as np
from PIL import Image
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0,"scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.soft_fusion import method_soft_to_codeword_llr
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
sb=ShortenedBCH(); n=sb.n; tau=float(binom.ppf(0.99,n,0.5)+1)/n
SC="/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
V=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def view(pil,frac):
    if frac>=0.999: return to512(pil)
    s=int(round(512*frac)); o=(512-s)//2; return to512(pil).crop((o,o,o+s,o+s))
def dec(pil,iid,tx):
    p,M=V.get_perm_M(iid)
    rl=np.clip(method_soft_to_codeword_llr(V.raw_probs(pil),p,M,kind="prob",n_codeword=n),-CLAMP,CLAMP)
    ba=float(((rl>0).astype(np.uint8)==tx).mean())
    return ba, bool(decode_and_verify(rl,iid,codec=sb)["detected"]) or ba>=tau
ZOOM=512/460.0
GRID=[round(0.34+0.03*k,3) for k in range(23) if 0.34+0.03*k<=1.0001]
def nearest_grid(x): return min(GRID,key=lambda g:abs(g-x))
def run(setname, files_iids, oracle_fracs, fine=True):
    out={}
    for K,frac in oracle_fracs.items():
        rows_or=[]; rows_gr=[]; best_fine=[]
        for iid,path in files_iids:
            tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
            im=to512(Image.open(path).convert("RGB"))
            ba_o,v_o=dec(view(im,frac),iid,tx); rows_or.append((ba_o,v_o))
            g=nearest_grid(frac); ba_g,v_g=dec(view(im,g),iid,tx); rows_gr.append((ba_g,v_g))
            if fine:
                bb=ba_o
                for d in np.arange(-0.03,0.0301,0.005):
                    f=frac+d
                    if 0.3<f<=1.0:
                        b,_=dec(view(im,float(f)),iid,tx); bb=max(bb,b)
                best_fine.append(bb)
        o_ba=np.mean([r[0] for r in rows_or]); o_v=np.mean([1.0 if r[1] else 0.0 for r in rows_or])
        g_ba=np.mean([r[0] for r in rows_gr]); g_v=np.mean([1.0 if r[1] else 0.0 for r in rows_gr])
        out[f"K{K}"]={"oracle_frac":round(frac,4),"oracle_ba":round(float(o_ba),4),"oracle_verify":round(float(o_v),3),
                      "grid_frac":nearest_grid(frac),"grid_ba":round(float(g_ba),4),"grid_verify":round(float(g_v),3),
                      "best_ba_fine_sweep":round(float(np.mean(best_fine)),4) if fine else None}
        r=out[f"K{K}"]
        print(f"{setname:22} K={K}: oracle@{r['oracle_frac']:.4f} ba={r['oracle_ba']:.3f} verify={r['oracle_verify']:.2f} | "
              f"grid@{r['grid_frac']} ba={r['grid_ba']:.3f} verify={r['grid_verify']:.2f} | fine-sweep best ba={r['best_ba_fine_sweep']:.3f}",flush=True)
    return out
meta=json.load(open(f"{SC}/p3embed_Dum/meta.json"))["items"]
A=[(m["iid"],f"{SC}/p3um_D/{m['fname']}") for m in meta if os.path.exists(f"{SC}/p3um_D/{m['fname']}")][:15]
metaD=json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]
C=[(m["iid"],f"{SC}/p3um_nested_nocrop/{m['fname']}") for m in metaD if os.path.exists(f"{SC}/p3um_nested_nocrop/{m['fname']}")][:15]
# B) crop-0.9 only, generated from clean nested embeds
Bd=f"{SC}/crop09_nested"; os.makedirs(Bd,exist_ok=True); B=[]
for m in metaD[:15]:
    src=f"{SC}/p3embed_D/{m['fname']}"
    if not os.path.exists(src): continue
    im=to512(Image.open(src).convert("RGB")); s=460; o=(512-s)//2
    im.crop((o,o,o+s,o+s)).resize((512,512)).save(f"{Bd}/{m['fname']}"); B.append((m["iid"],f"{Bd}/{m['fname']}"))
res={}
res["B_crop09_only"]   = run("B) crop0.9 only",   B, {0.75:0.75*ZOOM, 0.5:0.5*ZOOM})
res["C_spectral_only"] = run("C) spectral only",  C, {1.0:1.0,        0.75:0.75, 0.5:0.5})
res["A_full_unmarker"] = run("A) FULL UnMarker",  A, {0.75:0.75*ZOOM, 0.5:0.5*ZOOM})
json.dump(res,open("results/defense/oracle_ring.json","w"),indent=2)
print("ORACLE_RING_DONE")
