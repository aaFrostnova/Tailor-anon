"""Decisive PhaseMark study: DOMAIN (photo vs SD-generated) x VARIANT (APM/soft/alpha).

Settles the '>=35 dB' question. The paper reports ~34 dB on SD-GENERATED images measured
~against the VAE round-trip, using a SOFT variant. We measure, per corpus and per config:
  - PSNR vs TRUE CLEAN (residual mode = our honest deployed number)
  - PSNR vs VAE-RECON of the non-residual decode (= pure watermark perturbation = paper's ref)
  - PSNR vs clean of the non-residual decode (bounded by the VAE ceiling)
  - VAE round-trip ceiling for the corpus (PSNR(decode(encode(x)), x))
  - SSIM, LPIPS (residual)
  - PhaseMark-alone bit-acc under clean / regen(noise_step=60) / rinse2x
so we can (a) reproduce the paper on the right domain+reference and (b) show the photo verdict.
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.phasemark import PhaseMark

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"

CONFIGS = {
    "APM(gamma1,a1)":        dict(gamma=1.0),
    "soft(gamma0.6)":        dict(gamma=0.6),
    "soft(gamma0.4)":        dict(gamma=0.4),
    "soft(gamma0.6)+perc":   dict(gamma=0.6, perceptual=True),
    "alpha0.5":              dict(_alpha=0.5),
    "alpha0.5+gamma0.6":     dict(gamma=0.6, _alpha=0.5),
    "band[10,18]":           dict(band=(10.0, 18.0)),
}


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255; b = np.asarray(b, np.float64) / 255
    m = np.mean((a - b) ** 2)
    return 10 * np.log10(1 / m) if m > 1e-12 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=15)
    ap.add_argument("--n_bits", type=int, default=100)
    ap.add_argument("--output", default="results/defense/phasemark_domain_variant.json")
    args = ap.parse_args()
    dev = "cuda"

    import lpips as lpipsmod
    from pytorch_msssim import ssim as ssim_fn
    lp = lpipsmod.LPIPS(net="alex").to(dev).eval()
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    tmp = tempfile.mkdtemp()

    def to_t(pil):
        return torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)

    def regen_attack(pil, k):
        ip = os.path.join(tmp, "in.png"); pil.save(ip); cur = ip
        for r in range(k):
            nxt = os.path.join(tmp, f"r{r}.png"); regen.attack([cur], [nxt]); cur = nxt
        return Image.open(cur).convert("RGB").resize((512, 512))

    corpora = {
        "photo": sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[4000:4000 + args.n_images],
        "sd_gen": sorted(glob.glob(os.path.join(REPO, "results/sd21_corpus/*.png")))[:args.n_images],
    }
    rng = np.random.RandomState(1234)
    bitsets = [rng.randint(0, 2, args.n_bits).astype(np.uint8) for _ in range(args.n_images)]

    # cache one PhaseMark per distinct embed-config (gamma/perc/band); alpha is a call arg
    pm_cache = {}
    def get_pm(cfg):
        key = (cfg.get("gamma", 1.0), cfg.get("perceptual", False), tuple(cfg.get("band", (9.0, 19.0))))
        if key not in pm_cache:
            pm_cache[key] = PhaseMark("sd21", dev, gamma=key[0], perceptual=key[1], band=key[2])
        return pm_cache[key]

    out = {"n_images": args.n_images, "n_bits": args.n_bits, "corpora": {}}
    for corp, files in corpora.items():
        # VAE ceiling for this corpus (uses any pm's vae)
        base_pm = get_pm({})
        ceil = []
        origs = []
        for fp in files:
            o = Image.open(fp).convert("RGB").resize((512, 512)); origs.append(o)
            rec = base_pm.vae.decode(base_pm.vae.encode(o), size=(512, 512))
            ceil.append(psnr(o, rec))
        vae_ceiling = float(np.mean(ceil))
        out["corpora"][corp] = {"vae_ceiling_db": vae_ceiling, "configs": {}}
        print(f"\n### corpus={corp}  VAE round-trip ceiling = {vae_ceiling:.2f} dB (n={len(files)})", flush=True)
        print(f"{'config':<22}{'PSNRclean':>10}{'PSNRrecon':>10}{'rawPSNR':>9}{'SSIM':>7}{'LPIPS':>7}{'clean':>7}{'regen':>7}{'rinse':>7}")

        for name, cfg in CONFIGS.items():
            pm = get_pm(cfg); alpha = cfg.get("_alpha", 1.0)
            Pc, Pr, Praw, S, L = [], [], [], [], []
            bac, bar, bri = [], [], []
            for i, o in enumerate(origs):
                bits = bitsets[i]
                orig_lat = pm.vae.encode(o)
                orig_recon = pm.vae.decode(orig_lat, size=(512, 512))           # VAE recon (paper ref)
                wm_res = pm.embed(o, bits, residual=True, strength=alpha)        # deployed (vs clean)
                wm_raw = pm.embed(o, bits, residual=False, strength=alpha)       # raw decode
                Pc.append(psnr(o, wm_res))
                Pr.append(psnr(orig_recon, wm_raw))                             # pure watermark (paper)
                Praw.append(psnr(o, wm_raw))                                    # bounded by ceiling
                with torch.no_grad():
                    S.append(float(ssim_fn(to_t(o), to_t(wm_res), data_range=1.0)))
                    L.append(float(lp(to_t(o) * 2 - 1, to_t(wm_res) * 2 - 1).item()))
                bac.append(float(np.mean(pm.detect(wm_res, n_bits=args.n_bits) == bits)))
                bar.append(float(np.mean(pm.detect(regen_attack(wm_res, 1), n_bits=args.n_bits) == bits)))
                bri.append(float(np.mean(pm.detect(regen_attack(wm_res, 2), n_bits=args.n_bits) == bits)))
            rec = {"psnr_clean": float(np.mean(Pc)), "psnr_vs_recon": float(np.mean(Pr)),
                   "psnr_raw_vs_clean": float(np.mean(Praw)), "ssim": float(np.mean(S)), "lpips": float(np.mean(L)),
                   "ba_clean": float(np.mean(bac)), "ba_regen": float(np.mean(bar)), "ba_rinse2x": float(np.mean(bri))}
            out["corpora"][corp]["configs"][name] = rec
            print(f"{name:<22}{rec['psnr_clean']:>10.2f}{rec['psnr_vs_recon']:>10.2f}{rec['psnr_raw_vs_clean']:>9.2f}"
                  f"{rec['ssim']:>7.3f}{rec['lpips']:>7.3f}{rec['ba_clean']:>7.2f}{rec['ba_regen']:>7.2f}{rec['ba_rinse2x']:>7.2f}", flush=True)
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            json.dump(out, open(args.output, "w"), indent=2)

    print(f"\n[done] -> {args.output}\nPM_DOMAIN_DONE")


if __name__ == "__main__":
    main()
