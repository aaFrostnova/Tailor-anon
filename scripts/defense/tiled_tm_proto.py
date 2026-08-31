"""Prototype: borrow the block-DWT 'spatial redundancy + self-sync' structure to close
the composite's crop gap. TrustMark is the ONLY crop-bearing fragment (VINE dies on all
crops), so we make IT redundant:

  TILED-TM embed : tile the image into a g x g grid (g=2 -> 256x256 cells); embed the
                   SAME crypto codeword via TrustMark into every cell. Any single intact
                   cell carries the full codeword.
  SLIDING decode : slide a (512/g)-window over the attacked image at a grid of offsets,
                   TrustMark-decode each window, undo-crypto, and take the BEST bit-acc /
                   crypto-verify-ANY (FPR stays ~Kx2^-37 from exact 37-bit match).

Compare composite detection: current (global VINE + global TM) vs proposed (global VINE +
TILED TM w/ sliding decode), on the crop families AND on clean/jpeg/regen (regression
guard) + the fidelity cost.
"""
import os, sys, glob, io, json
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, undo_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70; CLAMP = 15.0; G = 2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
pipe = build_regen_pipe()
CELL = 512 // G

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def psnr(a, b):
    e = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2); return 99.0 if e < 1e-9 else 10 * np.log10(255 * 255 / e)

def tiled_tm_embed(base512, target):
    """embed crypto target via TM into each g x g cell of base512 (PIL) -> PIL."""
    out = np.asarray(base512, np.float64).copy()
    for gy in range(G):
        for gx in range(G):
            cell = Image.fromarray(out[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL].astype(np.uint8))
            wm = tm.embed_with_target(cell, target)
            wm = wm.resize((CELL, CELL)) if wm.size != (CELL, CELL) else wm
            out[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL] = np.asarray(wm, np.float64)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

OFFSETS = list(range(0, 512 - CELL + 1, 64))   # sliding-window top-left offsets per axis
def tiled_tm_decode_llr(att512, perm, M):
    """slide a CELL window; return the aligned-codeword LLR of the window with max |evidence|."""
    best = None; best_score = -1
    for oy in OFFSETS:
        for ox in OFFSETS:
            win = att512.crop((ox, oy, ox + CELL, oy + CELL))
            llr = np.clip(method_soft_to_codeword_llr(tm.raw_logits(win), perm, M, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
            score = float(np.abs(llr).mean())     # confidence proxy; crypto-verify done by caller on best
            if score > best_score: best_score, best = score, llr
    return best

# ---- attacks ----
def a_jpeg(im, q): b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def cresize(im, a): s = int(round(512*np.sqrt(a))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def corner(im, a): s = int(round(512*np.sqrt(a))); return im.crop((0,0,s,s)).resize((512,512))
def border(im, p): d=int(round(512*p)); reg=im.crop((d,d,512,512)); cv=Image.new("RGB",(512,512),(0,0,0)); cv.paste(reg,(0,0)); return cv
ATTACKS = [("clean", lambda im: im), ("jpeg25", lambda im: a_jpeg(im,25)),
           ("regen_x1", lambda im: to512(stable_regen(pipe, im, seed=11))),
           ("cresize0.5", lambda im: cresize(im,0.5)), ("corner0.5", lambda im: corner(im,0.5)),
           ("border0.1", lambda im: border(im,0.1)), ("border0.2", lambda im: border(im,0.2)), ("border0.3", lambda im: border(im,0.3))]

imgs = (sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[180:180+N]
rec = {nm: {"cur_av":[], "cur_at":[], "new_at":[], "tx":[], "id":[]} for nm,_ in ATTACKS}
psG, psT = [], []
for j, fp in enumerate(imgs):
    iid = f"sw_{180+j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pv,Mv = vine.get_perm_M(iid); pt,Mt = tm.get_perm_M(iid); tgt = apply_crypto(cw, pt, Mt)
    v = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
    vt_global = scale_resid(v, to512(tm.embed_with_target(v, tgt)), ALPHA)       # current
    vt_tiled  = scale_resid(v, tiled_tm_embed(v, tgt), ALPHA)                     # proposed
    psG.append(psnr(orig, vt_global)); psT.append(psnr(orig, vt_tiled))
    for nm, fn in ATTACKS:
        aG = to512(fn(vt_global)); aT = to512(fn(vt_tiled)); r = rec[nm]
        r["cur_av"].append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(aG), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        r["cur_at"].append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(aG), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
        # proposed: VINE global (same) on tiled image + TILED sliding TM
        r["new_at"].append(tiled_tm_decode_llr(aT, pt, Mt))
        r.setdefault("new_av", []).append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(aT), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
        r["tx"].append(cw.astype(np.uint8)); r["id"].append(iid)
    if (j+1) % 5 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

def metr(F, tx, ids):
    det = []
    for i in range(len(F)):
        ver = bool(decode_and_verify(F[i], ids[i], codec=sb)["detected"])
        det.append(1.0 if (ver or ((F[i]>0).astype(np.uint8)==tx[i]).mean() >= TAU) else 0.0)
    return float(np.mean(det)), float(((F>0).astype(np.uint8)==tx).mean())

print(f"\nn={N}  PSNR global-TM={np.mean(psG):.1f}  tiled-TM={np.mean(psT):.1f} (cost {np.mean(psG)-np.mean(psT):.1f}dB)\n")
print(f"{'attack':11s} | {'CUR comp':>11s} | {'NEW comp(tiled)':>15s} | {'CUR TM':>9s} {'NEW TM':>9s}")
print("-"*64)
rows=[]
for nm,_ in ATTACKS:
    r = rec[nm]; cav=np.array(r["cur_av"]); cat=np.array(r["cur_at"]); nat=np.array(r["new_at"]); nav=np.array(r["new_av"]); tx=np.array(r["tx"]); ids=r["id"]
    cur = metr(cav+cat, tx, ids); new = metr(nav+nat, tx, ids); ct=metr(cat,tx,ids); nt=metr(nat,tx,ids)
    def c(x): return f"{x[0]:.2f}/{x[1]:.2f}"
    print(f"{nm:11s} | {c(cur):>11s} | {c(new):>15s} | {c(ct):>9s} {c(nt):>9s}")
    rows.append({"attack":nm,"cur":cur,"new":new,"cur_tm":ct,"new_tm":nt})
json.dump({"n":N,"psnr_global":float(np.mean(psG)),"psnr_tiled":float(np.mean(psT)),"rows":rows},
          open(os.path.join(REPO,"results/defense/tiled_tm_proto.json"),"w"), indent=2)
print("TILEDTM_DONE")
