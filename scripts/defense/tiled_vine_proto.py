"""Test the user's idea: tile VINE too. Mirror the tiled-TM structure on the VINE
fragment and measure (a) whether tiling rescues VINE's crop robustness (global VINE dies
on every crop), (b) the embed-time cost (VINE = SD forward, ~10x TrustMark), (c) fidelity.

Compares VINE-ONLY detection: global-VINE (normal decode) vs tiled-VINE (2x2 cells, same
crypto codeword each, sliding-window crypto-verify-any). Attacks: clean + regen_x1
(regression guards) + border0.1/0.2/0.3 + cresize0.5 (crop families).
"""
import os, sys, glob, json, time
import numpy as np
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0; G = 2; STEP = 32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
pipe = build_regen_pipe(); CELL = 512 // G
OFFSETS = list(range(0, 512 - CELL + 1, STEP))

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def to(im, s): return im.resize((s, s)) if im.size != (s, s) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def psnr(a, b):
    e = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2); return 99.0 if e < 1e-9 else 10 * np.log10(255 * 255 / e)

def vine_tiled_embed(base512, target):
    out = np.asarray(base512, np.float64).copy()
    for gy in range(G):
        for gx in range(G):
            cell = Image.fromarray(out[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL].astype(np.uint8))
            wm = to(vine.embed_with_target(cell, target), CELL)
            out[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL] = np.asarray(wm, np.float64)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def vine_win_llr(att512, ox, oy, perm, M):
    win = att512.crop((ox, oy, ox + CELL, oy + CELL))
    return np.clip(method_soft_to_codeword_llr(vine.raw_probs(win), perm, M, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)

def vine_global_llr(att, perm, M):
    return np.clip(method_soft_to_codeword_llr(vine.raw_probs(to512(att)), perm, M, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)

def det_global(att, iid, perm, M, tx):
    llr = vine_global_llr(att, perm, M)
    if bool(decode_and_verify(llr, iid, codec=sb)["detected"]): return 1.0
    return 1.0 if ((llr > 0).astype(np.uint8) == tx).mean() >= TAU else 0.0
def det_tiled(att, iid, perm, M, tx):
    att = to512(att)
    canon = sum(vine_win_llr(att, ox, oy, perm, M) for oy in range(0, 512, CELL) for ox in range(0, 512, CELL))
    if bool(decode_and_verify(canon, iid, codec=sb)["detected"]): return 1.0
    if ((canon > 0).astype(np.uint8) == tx).mean() >= TAU: return 1.0
    for oy in OFFSETS:
        for ox in OFFSETS:
            if bool(decode_and_verify(vine_win_llr(att, ox, oy, perm, M), iid, codec=sb)["detected"]): return 1.0
    return 0.0

def cresize(im, a): s=int(round(512*np.sqrt(a))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def border(im, p): d=int(round(512*p)); reg=im.crop((d,d,512,512)); cv=Image.new("RGB",(512,512)); cv.paste(reg,(0,0)); return cv
ATTACKS = [("clean",lambda im:im),("regen_x1",lambda im:to512(stable_regen(pipe,im,seed=11))),
           ("cresize0.5",lambda im:cresize(im,0.5)),
           ("border0.1",lambda im:border(im,0.1)),("border0.2",lambda im:border(im,0.2)),("border0.3",lambda im:border(im,0.3))]

imgs = (sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[180:180+N]
rows = {nm:{"g":[], "t":[]} for nm,_ in ATTACKS}; psG, psT = [], []; tG, tT = 0.0, 0.0
for j, fp in enumerate(imgs):
    iid = f"sw_{180+j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); tx = cw.astype(np.uint8)
    pv, Mv = vine.get_perm_M(iid); tgt = apply_crypto(cw, pv, Mv)
    t0 = time.time(); vg = scale_resid(orig, to512(vine.embed_with_target(orig, tgt)), ALPHA); tG += time.time()-t0
    t0 = time.time(); vt = scale_resid(orig, vine_tiled_embed(orig, tgt), ALPHA); tT += time.time()-t0
    psG.append(psnr(orig, vg)); psT.append(psnr(orig, vt))
    for nm, fn in ATTACKS:
        aG = to512(fn(vg)); aT = to512(fn(vt))
        rows[nm]["g"].append(det_global(aG, iid, pv, Mv, tx)); rows[nm]["t"].append(det_tiled(aT, iid, pv, Mv, tx))
    if (j+1)%4==0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

print(f"\nn={N}  embed time/img: global-VINE={tG/N:.2f}s  tiled-VINE={tT/N:.2f}s ({tT/max(tG,1e-9):.1f}x)")
print(f"PSNR: global={np.mean(psG):.1f}  tiled={np.mean(psT):.1f} (cost {np.mean(psG)-np.mean(psT):.1f}dB)\n")
print(f"{'attack':11s} | {'VINE global':>11s} | {'VINE tiled':>10s}")
print("-"*40); out=[]
for nm,_ in ATTACKS:
    g=float(np.mean(rows[nm]["g"])); t=float(np.mean(rows[nm]["t"]))
    print(f"{nm:11s} | {g:11.2f} | {t:10.2f}"); out.append({"attack":nm,"global":g,"tiled":t})
json.dump({"n":N,"embed_s_global":tG/N,"embed_s_tiled":tT/N,"psnr_g":float(np.mean(psG)),"psnr_t":float(np.mean(psT)),"rows":out},
          open(os.path.join(REPO,"results/defense/tiled_vine_proto.json"),"w"), indent=2)
print("TILEDVINE_DONE")
