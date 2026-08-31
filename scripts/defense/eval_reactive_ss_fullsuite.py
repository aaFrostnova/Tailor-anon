"""Standalone full-attack-suite evaluation of the REACTIVE regen-orthogonal SS watermark.

Builds the spread-spectrum mark with carriers reactively projected into each image's regen-KEPT band,
then runs the full RAVEN 18-attack suite and reports per-attack bit-acc. Compares to VINE-R / TrustMark-B
(from results/defense/baselines_fullsuite.json). Key question: does optimizing for regen-survival
(low-mid band) keep it robust to the OTHER attacks, or break elsewhere (esp. geometric)?
"""
import argparse, glob, io, json, os, sys, tempfile
import numpy as np, torch
from PIL import Image
from scipy import ndimage

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.regen_ortho_projection import measure_survival_spectrum, build_kept_mask, project
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512
ATTACKS = ["clean", "jpeg", "blur", "noise", "bright", "contrast", "bm3d", "regen", "rinse2x", "rinse4x",
           "vae_b", "vae_c", "rs256", "hflip", "crop75", "crop50", "rot9", "crop_jpeg"]


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def to_pil(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def psnr(a, b):
    m = np.mean((a - b) ** 2); return float(99.0 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m))


def _ccr(img, frac):
    W, H = img.size; cw, ch = int(round(W * frac)), int(round(H * frac))
    l, t = (W - cw) // 2, (H - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)
def _rot(img, deg):
    W, H = img.size; a = np.array(img); pad = max(W, H) // 2
    refl = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))
def _rdu(img, mid):
    W, H = img.size; return img.resize((mid, mid), Image.BICUBIC).resize((W, H), Image.BICUBIC)
GEO = {"crop75": lambda im: _ccr(im, 0.75), "crop50": lambda im: _ccr(im, 0.50),
       "rot9": lambda im: _rot(im, 9.0), "rs256": lambda im: _rdu(im, 256),
       "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)}


def make_carriers(key, n_bits, H, W):
    cs = []; base = int.from_bytes(key[:4], "little")
    for i in range(n_bits):
        rs = np.random.RandomState((base + 1000 * i) % (2 ** 31))
        c = rs.randn(H, W).astype(np.float64); c -= c.mean(); c /= (np.linalg.norm(c) + 1e-9); cs.append(c)
    return cs


def embed_ss(x, eff, bits, target_psnr):
    H, W, _ = x.shape; d1 = np.zeros((H, W))
    for c, b in zip(eff, bits): d1 += (2 * int(b) - 1) * c
    d3 = np.repeat(d1[:, :, None], 3, axis=2)
    d3 *= (255.0 / 10 ** (target_psnr / 20.0)) / (np.sqrt(np.mean(d3 ** 2)) + 1e-12)
    return np.clip(x + d3, 0, 255)


def decode_ss(att, eff, hp=12.0):
    g = att.mean(axis=2)
    if hp: g = g - ndimage.gaussian_filter(g, hp)
    return np.array([1 if np.sum(g * c) > 0 else 0 for c in eff], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=10); ap.add_argument("--n_bits", type=int, default=64)
    ap.add_argument("--psnr", type=float, default=36.0); ap.add_argument("--n_probes", type=int, default=6)
    ap.add_argument("--out", default="results/defense/reactive_ss_fullsuite.json")
    args = ap.parse_args(); dev = "cuda"
    from regen_pipe import ReSDPipeline
    from wmattacker import (DiffWMAttacker, VAEWMAttacker, GaussianBlurAttacker,
                            GaussianNoiseAttacker, JPEGAttacker, BrightnessAttacker, ContrastAttacker, BM3DAttacker)
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    sig = {"jpeg": JPEGAttacker(quality=25), "blur": GaussianBlurAttacker(5, 1),
           "noise": GaussianNoiseAttacker(std=0.05), "bright": BrightnessAttacker(0.2),
           "contrast": ContrastAttacker(0.2), "bm3d": BM3DAttacker()}
    vae = {}
    for k, mn in [("vae_b", "bmshj2018-hyperprior"), ("vae_c", "cheng2020-anchor")]:
        try: vae[k] = VAEWMAttacker(mn, quality=3, metric="mse", device=dev)
        except Exception as e: print(f"[vae] skip {k}: {e}", flush=True)
    tmp = tempfile.mkdtemp()

    def regen_seeded(a, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        ip = os.path.join(tmp, "p.png"); op = os.path.join(tmp, "po.png")
        to_pil(a).save(ip); regen.attack([ip], [op]); return arr(Image.open(op).convert("RGB"))

    def attack_one(a, ip, op):
        if a == "clean": Image.open(ip).save(op); return
        if a in sig: sig[a].attack([ip], [op]); return
        if a in vae: vae[a].attack([ip], [op]); return
        if a == "regen": regen.attack([ip], [op]); return
        if a in ("rinse2x", "rinse4x"):
            k = 2 if a == "rinse2x" else 4; cur = ip
            for r in range(k):
                nxt = op.replace(".png", f"_r{r}.png"); regen.attack([cur], [nxt]); cur = nxt
            Image.open(cur).save(op); return
        if a in GEO:
            out = GEO[a](Image.open(ip).convert("RGB"))
            if out.size != (RES, RES): out = out.resize((RES, RES))
            out.save(op); return
        if a == "crop_jpeg":
            img = _ccr(Image.open(ip).convert("RGB"), 0.75)
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=25)
            Image.open(io.BytesIO(buf.getvalue())).convert("RGB").save(op); return
        raise ValueError(a)

    carriers = make_carriers(KEY, args.n_bits, RES, RES)
    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]
    rng = np.random.RandomState(0)
    R = {a: [] for a in ATTACKS}; psnrs = []; kept = []
    for i, fp in enumerate(files):
        x = arr(Image.open(fp).convert("RGB")); bits = rng.randint(0, 2, args.n_bits).astype(np.uint8)
        _, S = measure_survival_spectrum(x, regen_seeded, probe_seed=1234, n_probes=args.n_probes, probe_std=3.0)
        mask = build_kept_mask(S, RES, RES, mode="hard", thresh=0.5); kept.append(float(mask.mean()))
        eff = [project(np.repeat(c[:, :, None], 3, axis=2), mask)[:, :, 0] for c in carriers]
        eff = [c / (np.linalg.norm(c) + 1e-9) for c in eff]
        xw = embed_ss(x, eff, bits, args.psnr); psnrs.append(psnr(xw, x))
        ip = os.path.join(tmp, f"w_{i}.png"); to_pil(xw).save(ip)
        for a in ATTACKS:
            op = os.path.join(tmp, f"w_{i}_{a}.png")
            try:
                attack_one(a, ip, op); att = arr(Image.open(op).convert("RGB"))
                R[a].append(float(np.mean(decode_ss(att, eff) == bits)))
            except Exception as e:
                print(f"  [{a}] img{i} FAIL: {type(e).__name__}: {str(e)[:50]}", flush=True)
        print(f"  [{i+1}/{len(files)}] kept={kept[-1]:.3f} psnr={psnrs[-1]:.1f} "
              f"regen={np.mean(R['regen']):.2f} jpeg={np.mean(R['jpeg']):.2f} rot9={np.mean(R['rot9']):.2f}", flush=True)

    base = json.load(open(os.path.join(REPO, "results/defense/baselines_fullsuite.json")))["defenses"]
    summ = {a: float(np.mean(R[a])) for a in ATTACKS}
    json.dump({"reactive_ss": summ, "psnr": float(np.mean(psnrs)), "kept_frac": float(np.mean(kept)),
               "n_images": len(files), "n_bits": args.n_bits},
              open(os.path.join(REPO, args.out), "w"), indent=2)
    print(f"\n=== reactive regen-orthogonal SS — full suite (n={len(files)}, {args.n_bits} bits, "
          f"PSNR={np.mean(psnrs):.1f}, kept_frac={np.mean(kept):.3f}) ===")
    print(f"{'attack':<11}{'ReactiveSS':>11}{'VINE-R':>9}{'TrustMk-B':>11}")
    for a in ATTACKS:
        v = base.get('VINE-R', {}).get(a, {}).get('bit_acc', float('nan'))
        t = base.get('TrustMark-B', {}).get(a, {}).get('bit_acc', float('nan'))
        print(f"{a:<11}{summ[a]:>11.3f}{v:>9.3f}{t:>11.3f}")
    sig_at = ["jpeg", "blur", "noise", "bright", "contrast", "bm3d"]
    regen_at = ["regen", "rinse2x", "rinse4x", "vae_b", "vae_c"]
    geo_at = ["rs256", "hflip", "crop75", "crop50", "rot9", "crop_jpeg"]
    print(f"\nmean signal={np.mean([summ[a] for a in sig_at]):.3f}  "
          f"mean regen-family={np.mean([summ[a] for a in regen_at]):.3f}  "
          f"mean geometric={np.mean([summ[a] for a in geo_at]):.3f}")
    print("REACTIVE_SS_FULLSUITE_DONE")


if __name__ == "__main__":
    main()
