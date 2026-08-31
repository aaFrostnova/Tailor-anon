"""PhaseMark quality<->regen-robustness Pareto: sweep the residual strength alpha.

Our PhaseMark sets 32*4 mid-band 2x2 latent-FFT blocks per channel to absolute +/-pi/2.
That rotation is the entire pixel-delta (residual mode already cancels the VAE roundtrip),
so PSNR is only 22.66 dB. The strength alpha linearly scales the latent perturbation:
  watermarked = clean + decode(orig_lat + alpha*dlat) - decode(orig_lat)
Lower alpha -> higher PSNR but weaker phase signal. KEY QUESTION: because diffusion regen
preserves the latent low/mid band, a SMALL phase nudge may still survive regen. This sweep
finds the alpha giving PSNR>=35 while PhaseMark-alone bit-acc under regen/rinse stays useful.

Per alpha (n images, random bits): PSNR/SSIM/LPIPS vs clean, and PhaseMark bit-acc under
clean / regen(noise_step=60) / rinse2x.
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.phasemark import PhaseMark

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255; b = np.asarray(b, np.float64) / 255
    m = np.mean((a - b) ** 2)
    return 10 * np.log10(1 / m) if m > 1e-12 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=15)
    ap.add_argument("--start_idx", type=int, default=4000)
    ap.add_argument("--strengths", nargs="+", type=float,
                    default=[1.0, 0.5, 0.35, 0.25, 0.18, 0.12])
    ap.add_argument("--n_bits", type=int, default=100)
    ap.add_argument("--output", default="results/defense/phasemark_strength_sweep.json")
    args = ap.parse_args()
    dev = "cuda"

    pm = PhaseMark("sd21", dev)
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

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[args.start_idx:args.start_idx + args.n_images]
    rng = np.random.RandomState(1234)
    bitsets = [rng.randint(0, 2, args.n_bits).astype(np.uint8) for _ in imgs]

    out = {"n_images": len(imgs), "n_bits": args.n_bits, "strengths": {}}
    print(f"[pm-sweep] {len(imgs)} imgs, strengths={args.strengths}", flush=True)

    def bitacc(pil, bits):
        rec = pm.detect(pil, n_bits=args.n_bits)
        return float(np.mean(rec == bits))

    for a in args.strengths:
        P, S, L = [], [], []
        ba_clean, ba_regen, ba_rinse = [], [], []
        for i, fp in enumerate(imgs):
            orig = Image.open(fp).convert("RGB").resize((512, 512))
            bits = bitsets[i]
            wm = pm.embed(orig, bits, residual=True, strength=a)
            if wm.size != (512, 512): wm = wm.resize((512, 512))
            P.append(psnr(orig, wm))
            with torch.no_grad():
                S.append(float(ssim_fn(to_t(orig), to_t(wm), data_range=1.0)))
                L.append(float(lp(to_t(orig) * 2 - 1, to_t(wm) * 2 - 1).item()))
            ba_clean.append(bitacc(wm, bits))
            ba_regen.append(bitacc(regen_attack(wm, 1), bits))
            ba_rinse.append(bitacc(regen_attack(wm, 2), bits))
        rec = {"psnr": float(np.mean(P)), "ssim": float(np.mean(S)), "lpips": float(np.mean(L)),
               "ba_clean": float(np.mean(ba_clean)), "ba_regen": float(np.mean(ba_regen)),
               "ba_rinse2x": float(np.mean(ba_rinse))}
        out["strengths"][f"{a:.2f}"] = rec
        print(f"  alpha={a:.2f}: PSNR={rec['psnr']:.2f} SSIM={rec['ssim']:.3f} LPIPS={rec['lpips']:.3f} "
              f"| ba clean={rec['ba_clean']:.2f} regen={rec['ba_regen']:.2f} rinse2x={rec['ba_rinse2x']:.2f}", flush=True)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)

    print("\n=== PhaseMark strength sweep (PSNR vs regen bit-acc) ===")
    print(f"{'alpha':>6}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'clean':>8}{'regen':>8}{'rinse2x':>9}")
    for a in args.strengths:
        r = out["strengths"][f"{a:.2f}"]
        print(f"{a:>6.2f}{r['psnr']:>8.2f}{r['ssim']:>8.3f}{r['lpips']:>8.3f}{r['ba_clean']:>8.2f}{r['ba_regen']:>8.2f}{r['ba_rinse2x']:>9.2f}")
    print(f"\n[done] -> {args.output}\nPM_SWEEP_DONE")


if __name__ == "__main__":
    main()
