"""Alignment-invariant attack quality: FID + KID of each attacked set vs a clean COCO reference.

PSNR/SSIM mislead for geometric/photometric/UnMarker attacks (spatial/global misalignment).
FID/KID compare DISTRIBUTIONS in Inception space -> alignment-invariant: they measure whether
attacked images still look like natural images. KID is unbiased and small-n robust (our attacked
sets are 50/10), so it is the primary metric; FID is reported alongside (biased high at small n,
use for relative comparison only).

Reference = 1000 held-out clean COCO (idx 6000-6999, distinct from all eval/FPR sets).
Attacked sets reuse the embedded ext_4fused_coco (50 watermarked images): in-process attacks are
applied fresh; CtrlRegen+/UnMarker are read from saved dirs.
"""
import argparse, copy, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from scripts.defense.benchmark_composite_defense import GEO, _crop_then_jpeg

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def to_u8(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB").resize((299, 299)), np.uint8)).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ref", type=int, default=1000)
    ap.add_argument("--n_atk", type=int, default=50)
    ap.add_argument("--embed_dir", default="results/defense/ext_4fused_coco")
    ap.add_argument("--output", default="results/defense/attack_fid_kid.json")
    args = ap.parse_args()
    dev = "cuda"
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))

    # --- reference: 1000 clean held-out COCO ---
    base_fid = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    base_kid = KernelInceptionDistance(feature=2048, subset_size=40, subsets=100, normalize=False).to(dev)
    print(f"[ref] loading {args.n_ref} clean COCO (idx 6000+)", flush=True)
    buf = []
    for i in range(args.n_ref):
        buf.append(to_u8(Image.open(coco[6000 + i])))
        if len(buf) == 50:
            b = torch.stack(buf).to(dev); base_fid.update(b, real=True); base_kid.update(b, real=True); buf = []
    if buf:
        b = torch.stack(buf).to(dev); base_fid.update(b, real=True); base_kid.update(b, real=True)

    # --- attackers (in-process) ---
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

    wm_paths = [os.path.join(REPO, args.embed_dir, f"img_{i:05d}.png") for i in range(args.n_atk)]
    wm_paths = [p for p in wm_paths if os.path.exists(p)]

    def score(pils):
        fid = copy.deepcopy(base_fid); kid = copy.deepcopy(base_kid)
        bb = []
        for p in pils:
            bb.append(to_u8(p))
            if len(bb) == 50:
                b = torch.stack(bb).to(dev); fid.update(b, real=False); kid.update(b, real=False); bb = []
        if bb:
            b = torch.stack(bb).to(dev); fid.update(b, real=False); kid.update(b, real=False)
        km, ks = kid.compute()
        return float(fid.compute().item()), float(km.item()), float(ks.item())

    out = {}
    # clean-held-out sanity floor (n_atk distinct clean imgs, idx 5500+)
    sets = {"clean_floor": [Image.open(coco[5500 + i]) for i in range(args.n_atk)],
            "watermarked": [Image.open(p) for p in wm_paths]}
    inproc = ["jpeg", "blur", "noise", "bright", "contrast", "bm3d", "regen", "rinse2x", "rinse4x",
              "vae_b", "vae_c", "rs256", "hflip", "crop75", "crop50", "rot9", "crop_jpeg"]
    saved = {"ctrlregen_0.3": "ext_4fused_coco_ctrlregen_03", "ctrlregen_0.5": "ext_4fused_coco_ctrlregen_05",
             "ctrlregen_0.7": "ext_4fused_coco_ctrlregen_07", "unmarker": "ext_4fused_coco_unmarker"}

    print(f"\n=== FID / KID vs {args.n_ref}-image clean reference ===")
    print(f"{'set':<14}{'n':>5}{'FID':>9}{'KID*1e3':>10}{'KID_std*1e3':>13}")

    def report(name, pils):
        if not pils: return
        f, km, ks = score(pils)
        out[name] = {"n": len(pils), "fid": f, "kid": km, "kid_std": ks}
        print(f"{name:<14}{len(pils):>5}{f:>9.1f}{km*1e3:>10.2f}{ks*1e3:>13.2f}", flush=True)
        import json; json.dump(out, open(args.output, "w"), indent=2)

    for name, pils in sets.items():
        report(name, pils)
    for a in inproc:
        atk = []
        for j, wp in enumerate(wm_paths):
            op = os.path.join(tmp, f"{a}_{j}.png")
            try:
                attack_one(a, wp, op); atk.append(Image.open(op).convert("RGB"))
            except Exception as e:
                print(f"  [{a}] {j} FAIL {e}", flush=True)
        report(a, atk)
    for name, d in saved.items():
        fps = sorted(glob.glob(os.path.join(REPO, "results/defense", d, "*.png")))
        report(name, [Image.open(p) for p in fps])

    print("FID_KID_DONE")


if __name__ == "__main__":
    main()
