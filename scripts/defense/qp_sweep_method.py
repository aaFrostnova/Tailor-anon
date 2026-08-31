"""Per-method Q@P strength sweep on the EFFECTIVE (removable) attacks — rotate / crop-zoom / regen —
each over a strength grid, recording detection P and quality (PSNR/SSIM) at every strength so we can
plot P-vs-strength curves comparing methods. Ours uses the geo_cascade (SyncSeal rectify + crypto-verify)
on the geometric attacks; baselines use raw decode. Non-effective attacks (jpeg/blur/noise = Q@P inf)
are excluded per design."""
import os, sys, json, argparse, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.maskwm_wrapper import MaskWMWrapper
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
from _regen_util import build_regen_pipe, stable_regen
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"),
        "videoseal": ("logit", "raw_logits"), "maskwm": ("prob", "raw_scores")}
def build(nm):
    if nm == "vine": return VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
    if nm == "trustmark": return TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev)
    if nm == "videoseal": return VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)
    if nm == "maskwm": return MaskWMWrapper(ckpt_path="external/MaskWM/checkpoints/D_128bits.pth", master_key=KEY, method_name="maskwm", n_bits=n, device=dev)

ap = argparse.ArgumentParser()
ap.add_argument("--fragments", nargs="+", required=True)
ap.add_argument("--geo_cascade", action="store_true")
ap.add_argument("--src", default="/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()
frag = {nm: build(nm) for nm in a.fragments}
sync = load_sync(dev=dev) if a.geo_cascade else None
import glob
srcs = sorted(glob.glob(os.path.join(a.src, "*.png")))[:a.n]
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def embed(orig, iid):
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); x = orig
    for nm in a.fragments:
        p, M = frag[nm].get_perm_M(iid); x = scale(x, to512(frag[nm].embed_with_target(x, apply_crypto(tx, p, M))))
    if a.geo_cascade: x = sync_embed(sync, to512(x), dev)
    return to512(x), tx.astype(np.uint8)
def _cv(nm, pil, iid):
    kind, getter = SPEC[nm]; p, M = frag[nm].get_perm_M(iid)
    rl = np.clip(method_soft_to_codeword_llr(getattr(frag[nm], getter)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
    return rl
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    bw, bh = big.size; l = (bw - 512) // 2; return big.crop((l, l, l + 512, l + 512))
def detect(att, iid, tx, geo=False):
    al = {nm: _cv(nm, att, iid) for nm in a.fragments}
    fused = fuse_llrs(al, weights=None, n_codeword=n) if len(al) > 1 else next(iter(al.values()))
    det = bool(decode_and_verify(fused, iid, codec=sb)["detected"]) or (((fused > 0).astype(np.uint8) == tx).mean() >= tau) \
        or any(bool(decode_and_verify(v, iid, codec=sb)["detected"]) for v in al.values())
    if det: return 1.0
    if geo and a.geo_cascade:  # full geo_cascade: A SyncSeal rectify -> B local refine -> C blind angle search
        rect, _ = sync_rectify(sync, att, dev)
        for nm in a.fragments:                                              # A
            if bool(decode_and_verify(_cv(nm, rect, iid), iid, codec=sb)["detected"]): return 1.0
        if "trustmark" in a.fragments:                                      # B
            for d in (-3.0, 3.0, -6.0, 6.0):
                if bool(decode_and_verify(_cv("trustmark", rot(rect, d), iid), iid, codec=sb)["detected"]): return 1.0
        search = [nm for nm in ("trustmark", "videoseal") if nm in a.fragments]  # C blind coarse->fine, crypto-verify
        best_d, best_ba = 0.0, -1.0
        for d in np.arange(-30, 30.01, 10.0):
            for nm in search:
                if bool(decode_and_verify(_cv(nm, rot(att, float(d)), iid), iid, codec=sb)["detected"]): return 1.0
            if "trustmark" in a.fragments:
                ba = float(((_cv("trustmark", rot(att, float(d)), iid) > 0).astype(np.uint8) == tx).mean())
                if ba > best_ba: best_ba, best_d = ba, d
        if best_ba >= 0.60:
            for d in np.arange(best_d - 9, best_d + 9.01, 3.0):
                for nm in search:
                    if bool(decode_and_verify(_cv(nm, rot(att, float(d)), iid), iid, codec=sb)["detected"]): return 1.0
    return 0.0
def a_rot(x, deg): return x.rotate(deg, resample=Image.BICUBIC)  # black-fill (matches main eval; SyncSeal locates the true corners)
def a_crop(x, area):
    s = int(round(512 * np.sqrt(area))); o = (512 - s) // 2; return x.crop((o, o, o + s, o + s)).resize((512, 512))
GRID = {"rotate": [3, 9, 15, 20, 30, 45], "crop_zoom": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "regen": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
pipe = build_regen_pipe()
res = {"fragments": a.fragments, "n": len(srcs), "curves": {}}
for atk, grid in GRID.items():
    res["curves"][atk] = {"strength": grid, "det": [], "psnr": [], "ssim": []}
    print(f"== {a.fragments} / {atk} ==", flush=True)
    wms = []  # cache watermarked images
    for j, fp in enumerate(srcs):
        iid = f"qp_{j:05d}"; wms.append((iid, *embed(to512(Image.open(fp).convert("RGB")), iid)))
    for s in grid:
        dets, pss, sss = [], [], []
        for iid, wm, tx in wms:
            if atk == "rotate": att = a_rot(wm, s); geo = True
            elif atk == "crop_zoom": att = a_crop(wm, s); geo = True
            else: att = to512(stable_regen(pipe, wm, 7000 + hash(iid) % 1000, noise_step=int(s * 100))); geo = False
            dets.append(detect(att, iid, tx, geo=geo))
            aw, ww = np.asarray(att), np.asarray(wm); pss.append(_psnr(ww, aw, data_range=255)); sss.append(_ssim(ww, aw, channel_axis=2))
        res["curves"][atk]["det"].append(float(np.mean(dets))); res["curves"][atk]["psnr"].append(float(np.mean(pss))); res["curves"][atk]["ssim"].append(float(np.mean(sss)))
        print(f"  s={s}: det={np.mean(dets):.3f} psnr={np.mean(pss):.1f}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("QP_SWEEP_DONE", "_".join(a.fragments))
