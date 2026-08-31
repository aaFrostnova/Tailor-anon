"""Tiled-TM v2: proper 'any surviving tile detects' decoder.

Embed: TrustMark tiled g x g (full crypto codeword in every cell). VINE stays global.
Decode (composite detection = ANY of, all FPR-safe via 37-bit crypto exact match):
  (a) crypto-verify( VINE_global + best-window TM )        -> normal path (regen/signal-proc)
  (b) (VINE_global + best-window TM) zero-bit ba >= tau     -> single zero-bit test
  (c) crypto-verify( TM_window_k ) for ANY sliding window k -> crop/translation recovery
Slide step 32. Also measures FPR on unwatermarked images (verify-any over all windows).
"""
import os, sys, glob, io, json
import numpy as np
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0; G = 2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
STEP = 32
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
pipe = build_regen_pipe(); CELL = 512 // G
OFFSETS = list(range(0, 512 - CELL + 1, STEP))

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def psnr(a, b):
    e = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2); return 99.0 if e < 1e-9 else 10 * np.log10(255 * 255 / e)

def tiled_embed(base512, target):
    out = np.asarray(base512, np.float64).copy()
    for gy in range(G):
        for gx in range(G):
            cell = Image.fromarray(out[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL].astype(np.uint8))
            wm = tm.embed_with_target(cell, target); wm = wm.resize((CELL, CELL)) if wm.size != (CELL, CELL) else wm
            out[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL] = np.asarray(wm, np.float64)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

CANON = [(ox, oy) for oy in range(0, 512, CELL) for ox in range(0, 512, CELL)]  # fixed tile positions
def window_llrs(att512, perm, M):
    """LLR of every sliding window (for crypto-verify-any) + sum of the FIXED canonical
    tile windows (for the FPR-safe tau/fused path; no max-of-many selection)."""
    L = []
    for oy in OFFSETS:
        for ox in OFFSETS:
            win = att512.crop((ox, oy, ox + CELL, oy + CELL))
            L.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(win), perm, M, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
    L = np.array(L)
    canon = np.zeros(sb.n)
    for (ox, oy) in CANON:
        win = att512.crop((ox, oy, ox + CELL, oy + CELL))
        canon += np.clip(method_soft_to_codeword_llr(tm.raw_logits(win), perm, M, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
    return L, canon

# ---- attacks ----
def a_jpeg(im, q): b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def cresize(im, a): s=int(round(512*np.sqrt(a))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def corner(im, a): s=int(round(512*np.sqrt(a))); return im.crop((0,0,s,s)).resize((512,512))
def border(im, p): d=int(round(512*p)); reg=im.crop((d,d,512,512)); cv=Image.new("RGB",(512,512),(0,0,0)); cv.paste(reg,(0,0)); return cv
ATTACKS = [("clean",lambda im:im),("jpeg25",lambda im:a_jpeg(im,25)),("regen_x1",lambda im:to512(stable_regen(pipe,im,seed=11))),
           ("cresize0.5",lambda im:cresize(im,0.5)),("corner0.5",lambda im:corner(im,0.5)),
           ("border0.1",lambda im:border(im,0.1)),("border0.2",lambda im:border(im,0.2)),("border0.3",lambda im:border(im,0.3))]

imgs = (sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[180:180+N]
def det_new(av_global, Lwin, canon, iid, tx):
    fused = av_global + canon                                    # FIXED canonical tiles (no selection bias)
    if bool(decode_and_verify(fused, iid, codec=sb)["detected"]): return 1.0
    if ((fused>0).astype(np.uint8)==tx).mean() >= TAU: return 1.0
    for k in range(len(Lwin)):                                   # crop recovery: any-window crypto-verify (37-bit, FPR-safe)
        if bool(decode_and_verify(Lwin[k], iid, codec=sb)["detected"]): return 1.0
    return 0.0
def det_cur(av, at, iid, tx):
    fused = av + at
    if bool(decode_and_verify(fused, iid, codec=sb)["detected"]): return 1.0
    return 1.0 if ((fused>0).astype(np.uint8)==tx).mean() >= TAU else 0.0

rows = {nm: {"cur":[], "new":[]} for nm,_ in ATTACKS}; psG, psT = [], []
for j, fp in enumerate(imgs):
    iid = f"sw_{180+j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); tx = cw.astype(np.uint8)
    pv,Mv = vine.get_perm_M(iid); pt,Mt = tm.get_perm_M(iid); tgt = apply_crypto(cw, pt, Mt)
    v = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
    vt_cur = scale_resid(v, to512(tm.embed_with_target(v, tgt)), ALPHA)
    vt_new = scale_resid(v, tiled_embed(v, tgt), ALPHA)
    psG.append(psnr(orig, vt_cur)); psT.append(psnr(orig, vt_new))
    for nm, fn in ATTACKS:
        aC = to512(fn(vt_cur)); aN = to512(fn(vt_new))
        av_c = np.clip(method_soft_to_codeword_llr(vine.raw_probs(aC), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
        at_c = np.clip(method_soft_to_codeword_llr(tm.raw_logits(aC), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
        av_n = np.clip(method_soft_to_codeword_llr(vine.raw_probs(aN), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
        Lw, canon = window_llrs(aN, pt, Mt)
        rows[nm]["cur"].append(det_cur(av_c, at_c, iid, tx)); rows[nm]["new"].append(det_new(av_n, Lw, canon, iid, tx))
    if (j+1)%4==0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

# ---- FPR on unwatermarked (verify-any over all windows + global) ----
neg = (sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png"))) +
       sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[240:240+16]
fp_hits = 0; cur_fp = 0
for j, fp in enumerate(neg):
    iid = f"neg_{j:05d}"; im = to512(Image.open(fp).convert("RGB"))
    pv,Mv = vine.get_perm_M(iid); pt,Mt = tm.get_perm_M(iid); tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    av = np.clip(method_soft_to_codeword_llr(vine.raw_probs(im), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
    Lw, canon = window_llrs(im, pt, Mt)
    fp_hits += int(det_new(av, Lw, canon, iid, tx) > 0.5)
    at_n = np.clip(method_soft_to_codeword_llr(tm.raw_logits(im), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
    cur_fp += int(det_cur(av, at_n, iid, tx) > 0.5)

print(f"\nn={N} STEP={STEP} windows={len(OFFSETS)**2}  PSNR cur={np.mean(psG):.1f} tiled={np.mean(psT):.1f} (cost {np.mean(psG)-np.mean(psT):.1f}dB)")
print(f"FPR(unwm): CUR={cur_fp}/{len(neg)}  NEW(verify-any)={fp_hits}/{len(neg)}\n")
print(f"{'attack':11s} | {'CUR comp':>9s} | {'NEW(tiled v2)':>13s}")
print("-"*40)
out=[]
for nm,_ in ATTACKS:
    cur=float(np.mean(rows[nm]["cur"])); new=float(np.mean(rows[nm]["new"]))
    print(f"{nm:11s} | {cur:9.2f} | {new:13.2f}"); out.append({"attack":nm,"cur":cur,"new":new})
json.dump({"n":N,"step":STEP,"psnr_cur":float(np.mean(psG)),"psnr_tiled":float(np.mean(psT)),"fpr":fp_hits/len(neg),"rows":out},
          open(os.path.join(REPO,"results/defense/tiled_tm_v3.json"),"w"), indent=2)
print("TILEDV3_DONE")
