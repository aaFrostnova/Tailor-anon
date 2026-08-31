"""FINAL (post-fix) pipeline benchmark, single local GPU, sequential.
Runs ONLY what the fixed pipeline still needs (skips crop-staircase/off-center already done at n=50 in
rerun_fixed_search.json). Stages:
  1 embed n=50 nested composite -> save to disk (reused by figures + clean-quality)
  2 clean baseline detection + PSNR/SSIM (n=50)
  3 rotation sweep n=50 through the full cascade (range 180)  [was only n=20 in wide_rot]
  4 decode existing CtrlRegen+ {s03,s07} with the fixed pipeline  [s05,s09 already in rerun_fixed_search.json]
  5 Q@P quality (PSNR/SSIM/LPIPS) for every regen/unmarker attacked set vs its matched clean embed
  6 figure data: (a) radial residual-energy profile of VINE vs TrustMark vs VideoSeal (spatial-signal compare)
                 (b) nested-VINE composite residual map (watermark shape)
Pipeline == nested VINE {1,.75,.5} full-strength + TrustMark@.70 + VideoSeal@.70 + SyncSeal;
decode = primary fused/bestpath -> geo_cascade (SyncSeal -> refine -> vine_scale@.005 -> blind rot +-180)."""
import os, sys, glob, json, argparse, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0, "scripts/defense")
import torch
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify

KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
FIG = f"{OUT}/final_figdata"; EMB = f"{SC}/final_embed_n50"
os.makedirs(FIG, exist_ok=True); os.makedirs(EMB, exist_ok=True)
free = torch.cuda.mem_get_info()[0] / 1e9
print(f"[gpu] visible device free={free:.1f} GB", flush=True)
assert free > 35, f"picked GPU only has {free:.1f} GB free — pick another card"

SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
FR = ["vine", "trustmark", "videoseal"]; sync = load_sync(dev=dev)
try:
    import lpips as _lp; LP = _lp.LPIPS(net="alex").to(dev).eval(); print("[lpips] alexnet ready", flush=True)
except Exception as e:
    LP = None; print(f"[lpips] unavailable ({e}) — PSNR/SSIM only", flush=True)

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scl(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def txc(iid, nm):
    p, M = F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def vine_nested(x, iid):
    x = to512(x)
    for K in (1.0, 0.75, 0.5):
        if K >= 0.999:
            x = to512(F["vine"].embed_with_target(x, txc(iid, "vine")))
        else:
            s = int(512 * K); o = (512 - s) // 2
            m = F["vine"].embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid, "vine"))
            out = x.copy(); out.paste(m.resize((s, s)), (o, o)); x = out
    return x
def embed(orig, iid):
    x = to512(orig)
    for nm in FR:
        x = vine_nested(x, iid) if nm == "vine" else scl(x, to512(F[nm].embed_with_target(x, txc(iid, nm))))
    return to512(sync_embed(sync, to512(x), dev))
def llr(nm, pil, iid):
    kind, g = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], g)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VS = np.arange(0.34, 1.0001, 0.005)
def vine_scale_search(pil, iid, tx):
    P = to512(pil)
    for f in VS:                                  # VERIFY-ONLY: zerobit ba>=tau repeated over 133 scales is
        if f >= 0.999: view = P                   # FPR-UNSAFE (union ~70%); crypto-verify union stays <=2^-30.
        else:
            s = int(round(512 * float(f))); o = (512 - s) // 2; view = P.crop((o, o, o + s, o + s))
        if cv(llr("vine", view, iid), iid): return True
    return False
def detect(att, iid, tx):
    al = {nm: llr(nm, att, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n)
    if cv(fused, iid) or ((fused > 0).astype(np.uint8) == tx).mean() >= tau: return 1.0
    if any(cv(al[nm], iid) for nm in FR): return 1.0
    if vine_scale_search(att, iid, tx): return 1.0
    rect, _ = sync_rectify(sync, att, dev)
    if any(cv(llr(nm, rect, iid), iid) for nm in FR): return 1.0
    best_d, best_ba = 0.0, -1.0
    for d in np.arange(-180, 180.01, 10.0):
        rl = llr("trustmark", rot(att, float(d)), iid)
        if cv(rl, iid): return 1.0
        ba = float(((rl > 0).astype(np.uint8) == tx).mean())
        if ba > best_ba: best_ba, best_d = ba, d
    if best_ba < 0.60: return 0.0
    for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
        r = rot(att, float(d))
        for nm in ("trustmark", "videoseal"):
            if cv(llr(nm, r, iid), iid): return 1.0
    return 0.0

def qual(ref_pil, att_pil):
    R = np.asarray(to512(ref_pil), np.float64); A = np.asarray(to512(att_pil), np.float64)
    p = psnr_fn(R, A, data_range=255); s = ssim_fn(R, A, channel_axis=2, data_range=255); lp = None
    if LP is not None:
        with torch.no_grad():
            def t(x): return (torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1).to(dev)
            lp = float(LP(t(np.asarray(to512(ref_pil))), t(np.asarray(to512(att_pil)))).item())
    return round(float(p), 2), round(float(s), 4), (round(lp, 4) if lp is not None else None)

report = {"pipeline": "nested-VINE{1,.75,.5}+TM.70+VideoSeal.70+SyncSeal; decode primary+geo_cascade(vine_scale .005, rot +-180)",
          "tau": tau, "fpr_bound": "2^-29.6 (union over 133 scale + 37 rot crypto-verify trials)"}
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:50]

# ---- STAGE 1+2: embed n=50 (save) + clean baseline ----
print("=== STAGE 1+2: embed n=50 + clean baseline ===", flush=True)
wms = []; clean_det = []; clean_p = []; clean_s = []; clean_l = []
for j, fp in enumerate(srcs):
    iid = f"final_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    wm = embed(orig, iid); wm.save(f"{EMB}/img_{j:05d}.png")
    wms.append((iid, wm, tx, orig))
    clean_det.append(detect(wm, iid, tx))
    p, s, l = qual(orig, wm); clean_p.append(p); clean_s.append(s)
    if l is not None: clean_l.append(l)
    if (j + 1) % 10 == 0: print(f"  embed+clean [{j+1}/50]", flush=True)
json.dump({"items": [{"i": j, "iid": f"final_{j:04d}"} for j in range(len(wms))], "tau": tau}, open(f"{EMB}/meta.json", "w"))
report["clean"] = {"detect": round(float(np.mean(clean_det)), 3), "psnr": round(float(np.mean(clean_p)), 2),
                   "ssim": round(float(np.mean(clean_s)), 4), "lpips": (round(float(np.mean(clean_l)), 4) if clean_l else None), "n": len(wms)}
print("clean:", report["clean"], flush=True)

# ---- STAGE 3: rotation sweep n=50 ----
print("=== STAGE 3: rotation sweep n=50 (full cascade, range 180) ===", flush=True)
ANG = [15, 30, 45, 60, 90, 120, 150, 180]
report["rotation"] = {}
for a in ANG:
    d = [detect(rot(wm, a), iid, tx) for iid, wm, tx, _ in wms]
    report["rotation"][a] = round(float(np.mean(d)), 3)
    print(f"  rot{a:3d}: {report['rotation'][a]:.3f}", flush=True)

# ---- STAGE 4: decode CtrlRegen+ s03,s07 (fixed pipeline) ----
print("=== STAGE 4: CtrlRegen+ s03,s07 decode (fixed pipeline) ===", flush=True)
def redecode(embed_dir, att_dir):
    meta = json.load(open(f"{embed_dir}/meta.json"))["items"]; d = []
    for m in meta:
        p = f"{att_dir}/{m['fname']}"
        if not os.path.exists(p): continue
        iid = m["iid"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        d.append(detect(to512(Image.open(p).convert("RGB")), iid, tx))
    return round(float(np.mean(d)), 3) if d else None, len(d)
report["ctrlregen_new"] = {}
for tag, s in [("s03", "s03"), ("s07", "s07")]:
    ad = f"{SC}/p3cr_D_{s}"
    if os.path.isdir(ad):
        v, cnt = redecode(f"{SC}/p3embed_D", ad); report["ctrlregen_new"][tag] = {"detect": v, "n": cnt}
        print(f"  CtrlRegen+ {tag}: {v} (n={cnt})", flush=True)

# ---- STAGE 5: Q@P quality for regen/unmarker sets ----
print("=== STAGE 5: Q@P quality (PSNR/SSIM/LPIPS) ===", flush=True)
def crop09(im):
    im = to512(im); s = 460; o = (512 - s) // 2
    return im.crop((o, o, o + s, o + s)).resize((512, 512))
report["quality"] = {}
# CtrlRegen+ : attacked vs matched clean embed (aligned)
metaD = {m["fname"]: m["iid"] for m in json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]}
for s in ("s03", "s05", "s07", "s09"):
    ad = f"{SC}/p3cr_D_{s}"
    if not os.path.isdir(ad): continue
    ps, ss, ls = [], [], []
    for fn in metaD:
        a = f"{ad}/{fn}"; e = f"{SC}/p3embed_D/{fn}"
        if not (os.path.exists(a) and os.path.exists(e)): continue
        p, sv, l = qual(Image.open(e).convert("RGB"), Image.open(a).convert("RGB")); ps.append(p); ss.append(sv)
        if l is not None: ls.append(l)
    report["quality"][f"ctrlregen_{s}"] = {"psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4),
                                            "lpips": (round(float(np.mean(ls)), 4) if ls else None), "n": len(ps)}
    print(f"  CtrlRegen+ {s} quality:", report["quality"][f"ctrlregen_{s}"], flush=True)
# UnMarker : attacked vs 0.9-cropped clean embed (aligned)
for tag, ad in [("default", f"{SC}/p3um_D"), ("strong", f"{SC}/p3um_strong")]:
    fs = sorted(glob.glob(f"{ad}/*.png"))
    ps, ss, ls = [], [], []
    for f in fs:
        e = f"{SC}/p3embed_Dum/{os.path.basename(f)}"
        if not os.path.exists(e): continue
        p, sv, l = qual(crop09(Image.open(e).convert("RGB")), Image.open(f).convert("RGB")); ps.append(p); ss.append(sv)
        if l is not None: ls.append(l)
    if ps: report["quality"][f"unmarker_{tag}"] = {"psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4),
                                                    "lpips": (round(float(np.mean(ls)), 4) if ls else None), "n": len(ps)}
    print(f"  UnMarker {tag} quality:", report["quality"].get(f"unmarker_{tag}"), flush=True)

# ---- STAGE 6: figure data ----
print("=== STAGE 6: figure data (spatial-signal compare + watermark shape) ===", flush=True)
def radial_profile(res):  # res: HxW residual magnitude; return mean by normalized radius bins
    H, W = res.shape; cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2); r = r / r.max()
    bins = np.linspace(0, 1, 33); idx = np.digitize(r.ravel(), bins) - 1
    prof = np.array([res.ravel()[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(32)])
    return (0.5 * (bins[:-1] + bins[1:])).tolist(), prof.tolist()
# single-fragment residual profiles (one representative image) + nested composite residual
base = to512(Image.open(srcs[0]).convert("RGB")); iid0 = "figref"
b = np.asarray(base, np.float64)
prof = {}
single = {"vine": to512(F["vine"].embed_with_target(base, txc(iid0, "vine"))),
          "trustmark": scl(base, to512(F["trustmark"].embed_with_target(base, txc(iid0, "trustmark")))),
          "videoseal": scl(base, to512(F["videoseal"].embed_with_target(base, txc(iid0, "videoseal"))))}
for nm, wm in single.items():
    res = np.abs(np.asarray(wm, np.float64) - b).mean(axis=2)
    x, y = radial_profile(res); prof[nm] = {"r": x, "energy": y, "total": float(res.sum())}
    np.save(f"{FIG}/resid_{nm}.npy", res.astype(np.float32))
# nested-VINE (rings) residual + full composite residual
nv = vine_nested(base, iid0); res_nv = np.abs(np.asarray(nv, np.float64) - b).mean(axis=2)
x, y = radial_profile(res_nv); prof["vine_nested"] = {"r": x, "energy": y, "total": float(res_nv.sum())}
np.save(f"{FIG}/resid_vine_nested.npy", res_nv.astype(np.float32))
comp = embed(base, iid0); res_comp = np.abs(np.asarray(comp, np.float64) - b).mean(axis=2)
np.save(f"{FIG}/resid_composite.npy", res_comp.astype(np.float32))
np.save(f"{FIG}/cover.npy", b.astype(np.float32))
report["radial_profiles"] = prof
json.dump(report, open(f"{OUT}/final_pipeline_bench.json", "w"), indent=2)
print("FINAL_PIPELINE_BENCH_DONE", flush=True)
