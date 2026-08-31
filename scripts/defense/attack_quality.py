"""Per-attack IMAGE QUALITY (attacked image vs clean original) for the 4-fused composite.

Quantifies the attacker's fidelity cost: an attack that kills the watermark but wrecks the
image is not useful. Measures PSNR/SSIM/LPIPS of the attacked watermarked image vs the pristine
cover. In-process attacks (signal/regen/VAE/geometric) are applied fresh; CtrlRegen+/UnMarker
are read from their saved attacked dirs. Reuses the already-embedded ext_4fused_coco set.
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from scripts.defense.benchmark_composite_defense import GEO, _crop_then_jpeg  # geometric helpers

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255; b = np.asarray(b, np.float64) / 255
    m = np.mean((a - b) ** 2)
    return 10 * np.log10(1 / m) if m > 1e-12 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=20)
    ap.add_argument("--embed_dir", default="results/defense/ext_4fused_coco")
    ap.add_argument("--output", default="results/defense/attack_quality.json")
    args = ap.parse_args()
    dev = "cuda"
    import lpips as lpipsmod
    from pytorch_msssim import ssim as ssim_fn
    lp = lpipsmod.LPIPS(net="alex").to(dev).eval()

    def to_t(pil):
        return torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)

    def quality(att, clean):
        a, c = to_t(att), to_t(clean)
        with torch.no_grad():
            return psnr(clean, att), float(ssim_fn(a, c, data_range=1.0)), float(lp(a * 2 - 1, c * 2 - 1).item())

    from regen_pipe import ReSDPipeline
    from wmattacker import (DiffWMAttacker, VAEWMAttacker, GaussianBlurAttacker,
                            GaussianNoiseAttacker, JPEGAttacker, BrightnessAttacker, ContrastAttacker, BM3DAttacker)
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    sig = {"jpeg": JPEGAttacker(quality=25), "blur": GaussianBlurAttacker(5, 1), "noise": GaussianNoiseAttacker(std=0.05),
           "bright": BrightnessAttacker(0.2), "contrast": ContrastAttacker(0.2), "bm3d": BM3DAttacker()}
    vae = {"vae_b": VAEWMAttacker("bmshj2018-hyperprior", quality=3, metric="mse", device=dev),
           "vae_c": VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=dev)}
    tmp = tempfile.mkdtemp()

    def attack_one(a, ip, op):
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
            if out.size != (512, 512): out = out.resize((512, 512))
            out.save(op); return
        if a == "crop_jpeg":
            _crop_then_jpeg(ip, op); return
        raise ValueError(a)

    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    inproc = ["jpeg", "blur", "noise", "bright", "contrast", "bm3d", "regen", "rinse2x", "rinse4x",
              "vae_b", "vae_c", "rs256", "hflip", "crop75", "crop50", "rot9", "crop_jpeg"]
    saved = {"ctrlregen_0.3": "ext_4fused_coco_ctrlregen_03", "ctrlregen_0.5": "ext_4fused_coco_ctrlregen_05",
             "ctrlregen_0.7": "ext_4fused_coco_ctrlregen_07", "unmarker": "ext_4fused_coco_unmarker"}

    acc = {a: {"psnr": [], "ssim": [], "lpips": []} for a in ["watermarked"] + inproc + list(saved)}
    for i in range(args.n_images):
        clean = Image.open(coco[4000 + i]).convert("RGB").resize((512, 512))
        wm_path = os.path.join(REPO, args.embed_dir, f"img_{i:05d}.png")
        wm = Image.open(wm_path).convert("RGB").resize((512, 512))
        for m, (P, S, L) in [("watermarked", quality(wm, clean))]:
            acc[m]["psnr"].append(P); acc[m]["ssim"].append(S); acc[m]["lpips"].append(L)
        ip = os.path.join(tmp, f"wm_{i}.png"); wm.save(ip)
        for a in inproc:
            op = os.path.join(tmp, f"wm_{i}_{a}.png")
            try:
                attack_one(a, ip, op); att = Image.open(op).convert("RGB").resize((512, 512))
                P, S, L = quality(att, clean)
                acc[a]["psnr"].append(P); acc[a]["ssim"].append(S); acc[a]["lpips"].append(L)
            except Exception as e:
                print(f"  [{a}] img{i} FAIL: {e}", flush=True)
        for name, d in saved.items():
            fp = os.path.join(REPO, "results/defense", d, f"img_{i:05d}.png")
            if os.path.exists(fp):
                att = Image.open(fp).convert("RGB").resize((512, 512)); P, S, L = quality(att, clean)
                acc[name]["psnr"].append(P); acc[name]["ssim"].append(S); acc[name]["lpips"].append(L)
        print(f"  [{i+1}/{args.n_images}]", flush=True)

    import json
    out = {a: {"psnr": float(np.mean(v["psnr"])), "ssim": float(np.mean(v["ssim"])), "lpips": float(np.mean(v["lpips"])), "n": len(v["psnr"])}
           for a, v in acc.items() if v["psnr"]}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n=== attacked-image quality vs clean original (n={args.n_images}) ===")
    print(f"{'attack':<14}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    for a in ["watermarked"] + inproc + list(saved):
        if a in out:
            r = out[a]; print(f"{a:<14}{r['psnr']:>8.2f}{r['ssim']:>8.3f}{r['lpips']:>8.3f}")
    print("ATTACK_QUALITY_DONE")


if __name__ == "__main__":
    main()
