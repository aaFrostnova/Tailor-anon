"""Method-parameterized W-Bench eval (baselines + composite on the SD-attack + distortion axes).
Same tasks/attacks as wbench_eval.py but embed/decode are configurable via --fragments so we can run
VINE / TrustMark / VideoSeal / MaskWM / composite consistently. (Geometry-resync cascade is NOT used
here — these axes are non-geometric; the geometry headline uses composite_external_eval --geo_cascade.)

Tasks: distortion | sto_sweep | sto_regen | instruct | local | svd."""
import os, sys, glob, json, io, argparse, csv
import numpy as np, torch
from PIL import Image, ImageFilter
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO, "scripts")]: sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.maskwm_wrapper import MaskWMWrapper
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.fusion_head3 import load_head3
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
import lpips as lpips_mod
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"),
        "videoseal": ("logit", "raw_logits"), "maskwm": ("prob", "raw_scores")}
def build_frag(name):
    if name == "vine":      return VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
    if name == "trustmark": return TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev)
    if name == "videoseal": return VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)
    if name == "maskwm":    return MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/D_128bits.pth"),
                                                 master_key=KEY, method_name="maskwm", n_bits=n, device=dev)
    raise SystemExit("unknown fragment " + name)

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--wb_dir", required=True)
ap.add_argument("--fragments", nargs="+", default=["vine", "trustmark", "videoseal"])
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--strength", type=float, default=0.6)
ap.add_argument("--strength_list", default="0.2,0.3,0.4,0.5,0.6")
ap.add_argument("--caption_csv", default=""); ap.add_argument("--mask_dir", default=""); ap.add_argument("--out", default="")
a = ap.parse_args()

frag = {name: build_frag(name) for name in a.fragments}
HEAD3 = a.fragments == ["vine", "trustmark", "videoseal"]
head = load_head3(os.path.join(REPO, "results/defense/frag3_head.pt"), dev) if HEAD3 else None
lp = lpips_mod.LPIPS(net="alex").to(dev).eval()
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def embed(orig, iid):
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    x = orig
    for name in a.fragments:
        p, M = frag[name].get_perm_M(iid)
        x = scale(x, to512(frag[name].embed_with_target(x, apply_crypto(tx, p, M))))
    return x, tx.astype(np.uint8)
def _llr(name, att, iid):
    kind, getter = SPEC[name]; p, M = frag[name].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(frag[name], getter)(att), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def decode(att, iid, tx):
    al = {name: _llr(name, att, iid) for name in a.fragments}
    if HEAD3:
        with torch.no_grad():
            fused = head(torch.tensor(al["vine"][None], device=dev), torch.tensor(al["trustmark"][None], device=dev),
                         torch.tensor(al["videoseal"][None], device=dev)).cpu().numpy()[0]
    else:
        fused = fuse_llrs(al, weights=None, n_codeword=n) if len(al) > 1 else next(iter(al.values()))
    ver = bool(decode_and_verify(fused, iid, codec=sb)["detected"])
    zb = ((fused > 0).astype(np.uint8) == tx).mean() >= tau
    best = any(bool(decode_and_verify(v, iid, codec=sb)["detected"]) for v in al.values())
    return (1.0 if (ver or zb or best) else 0.0), float(((fused > 0).astype(np.uint8) == tx).mean())

_P = {}; SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
def sd_img2img():
    if "i2i" not in _P:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD21, torch_dtype=torch.float16, safety_checker=None, local_files_only=True).to(dev)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True); _P["i2i"] = pipe
    return _P["i2i"]
def att_sto_regen(im, seed, s):
    g = torch.Generator(dev).manual_seed(seed)
    return sd_img2img()(prompt="", image=im, strength=s, num_inference_steps=50, guidance_scale=1.0, generator=g).images[0].resize((512, 512))
def att_instruct(im, cap, seed):
    g = torch.Generator(dev).manual_seed(seed)
    return sd_img2img()(prompt=cap, image=im, strength=a.strength, num_inference_steps=50, guidance_scale=7.5, generator=g).images[0].resize((512, 512))
def svd_pipe():
    if "svd" not in _P:
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16", local_files_only=False).to(dev)
        pipe.set_progress_bar_config(disable=True); _P["svd"] = pipe
    return _P["svd"]
def att_svd(im, seed):
    g = torch.Generator(dev).manual_seed(seed)
    frames = svd_pipe()(im.resize((1024, 576)), num_frames=14, decode_chunk_size=4, motion_bucket_id=127, noise_aug_strength=0.02, generator=g).frames[0]
    return [f.convert("RGB").resize((512, 512)) for f in frames]
def att_local(im, mask, seed):
    g = torch.Generator(dev).manual_seed(seed)
    regen = sd_img2img()(prompt="", image=im, strength=0.8, num_inference_steps=50, guidance_scale=1.0, generator=g).images[0].resize((512, 512))
    m = (np.asarray(to512(mask).convert("L"), np.float32) / 255.0)[..., None]
    return Image.fromarray((m * np.asarray(regen, np.float32) + (1 - m) * np.asarray(im, np.float32)).clip(0, 255).astype(np.uint8))
def _jpeg(x, q): b = io.BytesIO(); x.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def _noise(x, s): return Image.fromarray(np.clip(np.asarray(x, np.float64) + np.random.RandomState(0).normal(0, s*255, (512,512,3)), 0, 255).astype(np.uint8))
def _crop(x, r): s = int(512*r); o = (512-s)//2; return x.crop((o, o, o+s, o+s)).resize((512, 512))
DISTORT = {"clean": lambda x: x, "jpeg25": lambda x: _jpeg(x, 25), "jpeg10": lambda x: _jpeg(x, 10),
           "blur3": lambda x: x.filter(ImageFilter.GaussianBlur(3)), "noise0.05": lambda x: _noise(x, 0.05),
           "crop75": lambda x: _crop(x, 0.75), "rot9": lambda x: x.rotate(9, resample=Image.BILINEAR)}

imgs = sorted(glob.glob(os.path.join(a.wb_dir, "*.png")))[:a.n]
print(f"task={a.task} fragments={a.fragments} n={len(imgs)}", flush=True)
caps = {}
if a.caption_csv and os.path.exists(a.caption_csv):
    with open(a.caption_csv) as f:
        for row in csv.DictReader(f): caps[str(row["idx"]).strip()] = row.get("target_caption", "").strip()
res = {"task": a.task, "fragments": a.fragments, "n": len(imgs)}
if a.task == "distortion":
    rows = {k: [] for k in DISTORT}
    for j, fp in enumerate(imgs):
        iid = f"wb_{j:05d}"; wm, tx = embed(to512(Image.open(fp).convert("RGB")), iid)
        for k, fn in DISTORT.items(): rows[k].append(decode(to512(fn(wm)), iid, tx)[0])
        if (j+1) % 50 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)
    res["rows"] = {k: float(np.mean(v)) for k, v in rows.items()}
    for k, v in res["rows"].items(): print("%-10s %.3f" % (k, v))
elif a.task == "sto_sweep":
    strengths = [float(s) for s in a.strength_list.split(",")]; rows = {s: [] for s in strengths}
    for j, fp in enumerate(imgs):
        iid = f"wb_{j:05d}"; wm, tx = embed(to512(Image.open(fp).convert("RGB")), iid)
        for s in strengths: rows[s].append(decode(to512(att_sto_regen(wm, 7000+j, s)), iid, tx)[0])
        if (j+1) % 25 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)
    res["sweep"] = {str(s): float(np.mean(v)) for s, v in rows.items()}
    for s, v in res["sweep"].items(): print("s=%s det=%.3f" % (s, v))
elif a.task == "svd":
    fd, ad = [], []
    for j, fp in enumerate(imgs):
        iid = f"wb_{j:05d}"; wm, tx = embed(to512(Image.open(fp).convert("RGB")), iid)
        dets = [decode(f, iid, tx)[0] for f in att_svd(wm, 7000+j)]
        fd.append(float(np.mean(dets))); ad.append(1.0 if max(dets) > 0 else 0.0)
        if (j+1) % 20 == 0: print(f"  [{j+1}/{len(imgs)}] fd={np.mean(fd):.3f}", flush=True)
    res.update(frame_det=float(np.mean(fd)), any_frame_det=float(np.mean(ad)))
    print(f"svd frame_det={res['frame_det']:.3f} any={res['any_frame_det']:.3f}")
else:
    det = []
    for j, fp in enumerate(imgs):
        iid = f"wb_{j:05d}"; wm, tx = embed(to512(Image.open(fp).convert("RGB")), iid)
        if a.task == "sto_regen": att = att_sto_regen(wm, 7000+j, a.strength)
        elif a.task == "instruct": att = att_instruct(wm, caps.get(os.path.basename(fp).split("_")[0], "a high quality photo"), 7000+j)
        elif a.task == "local":
            mp = os.path.join(a.mask_dir, os.path.basename(fp)); mask = Image.open(mp) if os.path.exists(mp) else Image.new("L", (512, 512), 0)
            att = att_local(wm, mask, 7000+j)
        else: raise SystemExit("bad task")
        det.append(decode(to512(att), iid, tx)[0])
        if (j+1) % 50 == 0: print(f"  [{j+1}/{len(imgs)}] det={np.mean(det):.3f}", flush=True)
    res["det"] = float(np.mean(det)); print(f"{a.task}: detection={res['det']:.3f}")
json.dump(res, open(a.out or os.path.join(REPO, f"results/defense/wbm_{a.task}_{'_'.join(a.fragments)}.json"), "w"), indent=2)
print("WBM_DONE", a.task, "_".join(a.fragments))
