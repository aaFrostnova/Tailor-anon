"""Compare TrustMark model variants (Q/P/B/C) under our faithful attack suite, to pick the
strongest OR-corroborator for the composite. Measures exactly what the composite uses:
TrustMarkMethod(variant).decode bit-acc + detect@0.75, per attack.
"""
import argparse, glob, os, sys, tempfile, json
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from wbench.methods import TrustMarkMethod
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
ATTACKS = ["clean", "jpeg", "noise", "regen", "rinse2x", "vae_b", "vae_c", "hflip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=15)
    ap.add_argument("--variants", nargs="+", default=["Q", "P", "B", "C"])
    ap.add_argument("--output", default="results/defense/trustmark_variants.json")
    args = ap.parse_args()
    dev = "cuda"

    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker, VAEWMAttacker, GaussianNoiseAttacker, JPEGAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    sig = {"jpeg": JPEGAttacker(quality=25), "noise": GaussianNoiseAttacker(std=0.05)}
    vae = {"vae_b": VAEWMAttacker("bmshj2018-hyperprior", quality=3, metric="mse", device=dev),
           "vae_c": VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=dev)}
    tmp = tempfile.mkdtemp()

    def attack_one(a, ip, op):
        if a == "clean": Image.open(ip).save(op); return
        if a in sig: sig[a].attack([ip], [op]); return
        if a in vae: vae[a].attack([ip], [op]); return
        if a == "regen": regen.attack([ip], [op]); return
        if a == "rinse2x":
            cur = ip
            for r in range(2):
                nxt = op.replace(".png", f"_r{r}.png"); regen.attack([cur], [nxt]); cur = nxt
            Image.open(cur).save(op); return
        if a == "hflip":
            Image.open(ip).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT).save(op); return
        raise ValueError(a)

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[4000:4000 + args.n_images]
    out = {"n_images": len(imgs), "variants": {}}
    print(f"[tm-variants] {len(imgs)} imgs, variants={args.variants}", flush=True)
    print(f"{'variant':<10}{'nbits':>6}" + "".join(f"{a[:7]:>8}" for a in ATTACKS))
    for v in args.variants:
        try:
            tm = TrustMarkMethod(v)
        except Exception as e:
            print(f"{v:<10} LOAD FAIL: {type(e).__name__}: {str(e)[:50]}", flush=True); continue
        nb = tm.n_bits
        # embed
        ins, secrets = [], []
        for i, fp in enumerate(imgs):
            cover = Image.open(fp).convert("RGB").resize((512, 512))
            b = np.random.RandomState(i).randint(0, 2, nb).astype(np.uint8)
            wm = tm.embed(cover, b)
            if wm.size != (512, 512): wm = wm.resize((512, 512))
            ip = os.path.join(tmp, f"{v}_{i}.png"); wm.save(ip); ins.append(ip); secrets.append(b)
        rec = {}
        for a in ATTACKS:
            accs = []
            for i in range(len(ins)):
                op = os.path.join(tmp, f"{v}_{i}_{a}.png")
                try:
                    attack_one(a, ins[i], op)
                    att = Image.open(op).convert("RGB")
                    if att.size != (512, 512): att = att.resize((512, 512))
                    r = tm.decode(att); m = min(len(r), len(secrets[i]))
                    accs.append(float(np.mean(r[:m] == secrets[i][:m])))
                except Exception:
                    accs.append(float("nan"))
            ba = float(np.nanmean(accs)); det = float(np.nanmean([1.0 if x >= 0.75 else 0.0 for x in accs]))
            rec[a] = {"bit_acc": ba, "detect": det}
        out["variants"][v] = {"n_bits": nb, "attacks": rec}
        print(f"{v:<10}{nb:>6}" + "".join(f"{rec[a]['bit_acc']:>8.3f}" for a in ATTACKS), flush=True)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== detect@0.75 rate ===")
    print(f"{'variant':<10}" + "".join(f"{a[:7]:>8}" for a in ATTACKS))
    for v, r in out["variants"].items():
        print(f"{v:<10}" + "".join(f"{r['attacks'][a]['detect']:>8.2f}" for a in ATTACKS))
    print(f"\n[done] -> {args.output}\nTM_VARIANTS_DONE")


if __name__ == "__main__":
    main()
