"""RAVEN-faithful defense suite: defenses x RAVEN's EXACT attacks (Zhao WatermarkAttacker).

Unlike benchmark_defense_suite.py (which used our own harsher attacks), this uses the
canonical attack implementations RAVEN/WAVES inherit from XuandongZhao/WatermarkAttacker:
  Regen   = DiffWMAttacker(noise_step=60)         (gentle: noise to t=60/1000, denoise ~3 steps)
  Rinse2x/4x = 2 / 4 successive Regen passes
  VAE-B   = bmshj2018-hyperprior, quality 3       (compressai)
  VAE-C   = cheng2020-anchor, quality 3           (compressai)
  signal  = GaussianBlur / GaussianNoise / JPEG / Brightness / Contrast / BM3D (WAVES-style)
Calibration confirmed Regen matches RAVEN MS-COCO for DwtDct/DwtDctSvd/RivaGAN/VINE (4/5);
TrustMark diverges (its fine pixel residual dies under the SD-VAE roundtrip inside Regen).

Metric: bit-acc + detection @ per-method 1%-FPR binomial threshold. Prints our value next
to RAVEN's MS-COCO reference for each (defense, attack). Persistent JSON + log.
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np, torch
from PIL import Image
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "attack"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
# RAVEN MS-COCO Table-2 bit-acc reference (for the post-hoc bitstream defenses we run)
RAVEN_REF = {
    "DwtDct":   {"clean":0.863,"jpeg":0.516,"blur":0.677,"noise":0.859,"bright":0.572,"contrast":0.522,"bm3d":0.532,"regen":0.519,"rinse2x":0.508,"vae_b":0.523,"vae_c":0.521},
    "DwtDctSvd":{"clean":1.000,"jpeg":0.602,"blur":1.000,"noise":1.000,"bright":0.555,"contrast":0.473,"bm3d":0.784,"regen":0.644,"rinse2x":0.561,"vae_b":0.648,"vae_c":0.596},
    "RivaGAN":  {"clean":0.999,"jpeg":0.821,"blur":0.998,"noise":0.969,"bright":0.862,"contrast":0.986,"bm3d":0.934,"regen":0.608,"rinse2x":0.527,"vae_b":0.570,"vae_c":0.552},
    "TrustMark-Q":{"clean":1.000,"jpeg":0.992,"blur":1.000,"noise":0.997,"bright":0.962,"contrast":1.000,"bm3d":0.999,"regen":0.999,"rinse2x":0.999,"vae_b":0.982,"vae_c":0.987},
    "VINE-R":   {"clean":1.000,"jpeg":1.000,"blur":0.973,"noise":0.979,"bright":0.998,"contrast":0.998,"bm3d":1.000,"regen":0.881,"rinse2x":0.831,"vae_b":0.950,"vae_c":0.915},
}
ATTACKS = ["clean","jpeg","blur","noise","bright","contrast","bm3d","regen","rinse2x","rinse4x","vae_b","vae_c"]


def fpr1(n):
    return float(binom.ppf(0.99, n, 0.5) + 1) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--defenses", nargs="+",
                   default=["dwtDct","dwtDctSvd","rivaGan","trustmark","vine_r","maskwm","phasemark","dft_kred","quant_qim"])
    p.add_argument("--attacks", nargs="+", default=ATTACKS)
    p.add_argument("--output", default="results/defense/defense_faithful.json")
    args = p.parse_args()
    dev = "cuda"

    from regen_pipe import ReSDPipeline
    from wmattacker import (DiffWMAttacker, VAEWMAttacker, GaussianBlurAttacker,
                            GaussianNoiseAttacker, JPEGAttacker, BrightnessAttacker, ContrastAttacker, BM3DAttacker)
    from diffusers import DPMSolverMultistepScheduler
    from overwrite_matrix import build_all
    from scripts.defense.benchmark_composite_defense import GEO, _crop_then_jpeg

    print("[faithful] loading SD-2.1 ReSDPipeline + attackers", flush=True)
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=min(args.n_images, 4), noise_step=60)
    sig = {"jpeg": JPEGAttacker(quality=25), "blur": GaussianBlurAttacker(kernel_size=5, sigma=1),
           "noise": GaussianNoiseAttacker(std=0.05), "bright": BrightnessAttacker(brightness=0.2),
           "contrast": ContrastAttacker(contrast=0.2), "bm3d": BM3DAttacker()}
    vae = {}
    for k, mn in [("vae_b", "bmshj2018-hyperprior"), ("vae_c", "cheng2020-anchor")]:
        if k in args.attacks:
            try: vae[k] = VAEWMAttacker(mn, quality=3, metric="mse", device=dev)
            except Exception as e: print(f"  [vae] SKIP {k}: {e}", flush=True)

    methods = build_all(args.defenses, dev)
    names = [d for d in args.defenses if d in methods]
    tau = {d: fpr1(methods[d].n_bits) for d in names}
    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[args.start_idx:args.start_idx+args.n_images]
    tmp = tempfile.mkdtemp()
    print(f"[faithful] {len(imgs)} imgs, {len(names)} defenses, attacks={args.attacks}", flush=True)

    def run_attack(a, ins, outs):
        if a == "clean":
            for i, o in zip(ins, outs):
                Image.open(i).save(o)
            return
        if a in sig: sig[a].attack(ins, outs); return
        if a in vae: vae[a].attack(ins, outs); return
        if a == "regen": regen.attack(ins, outs); return
        if a in ("rinse2x", "rinse4x"):
            k = 2 if a == "rinse2x" else 4
            cur = ins
            for r in range(k):
                nxt = [o.replace(".png", f"_r{r}.png") for o in outs]
                regen.attack(cur, nxt); cur = nxt
            for c, o in zip(cur, outs): Image.open(c).save(o)
            return
        if a in GEO:
            for i, o in zip(ins, outs):
                g = GEO[a](Image.open(i).convert("RGB"))
                if g.size != (512, 512): g = g.resize((512, 512))
                g.save(o)
            return
        if a == "crop_jpeg":
            for i, o in zip(ins, outs): _crop_then_jpeg(i, o)
            return
        raise ValueError(a)

    out = {"n_images": len(imgs), "raven_ref": RAVEN_REF, "fpr1": {methods[d].name: tau[d] for d in names}, "defenses": {}}
    for d in names:
        m = methods[d]
        bitsL, ins = [], []
        for i, fp in enumerate(imgs):
            cover = Image.open(fp).convert("RGB").resize((512, 512))
            b = np.random.RandomState(7000 + i).randint(0, 2, m.n_bits).astype(np.uint8)
            wm = m.embed(cover, b)
            if wm.size != (512, 512): wm = wm.resize((512, 512))
            ip = os.path.join(tmp, f"{d}_{i}.png"); wm.save(ip); ins.append(ip); bitsL.append(b)
        rec = {}
        for a in args.attacks:
            outs = [os.path.join(tmp, f"{d}_{i}_{a}_o.png") for i in range(len(ins))]
            try:
                run_attack(a, ins, outs)
                accs = []
                for i in range(len(outs)):
                    att = Image.open(outs[i]).convert("RGB")
                    if att.size != (512, 512): att = att.resize((512, 512))
                    r = m.decode(att); nb = min(len(r), len(bitsL[i]))
                    accs.append(float(np.mean(r[:nb] == bitsL[i][:nb])))
                ba = float(np.mean(accs)); det = float(np.mean([a_ >= tau[d] for a_ in accs]))
            except Exception as e:
                print(f"  [{d}/{a}] FAIL: {type(e).__name__}: {str(e)[:60]}", flush=True); ba = det = float("nan")
            rec[a] = {"bit_acc": ba, "detect": det}
        out["defenses"][m.name] = rec
        ref = RAVEN_REF.get(m.name, {})
        row = " ".join(f"{a}={rec[a]['bit_acc']:.2f}" + (f"/R{ref[a]:.2f}" if a in ref else "") for a in args.attacks)
        print(f"  [{m.name}] {row}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n=== bit-acc (ours) vs RAVEN-MS-COCO [R] ===")
    for d in names:
        nm = methods[d].name; ref = RAVEN_REF.get(nm, {})
        print(f"\n{nm} (1%FPR tau={tau[d]:.2f}):")
        for a in args.attacks:
            r = f" RAVEN={ref[a]:.3f}" if a in ref else ""
            print(f"  {a:<9} ours_ba={out['defenses'][nm][a]['bit_acc']:.3f} detect={out['defenses'][nm][a]['detect']:.2f}{r}")
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
