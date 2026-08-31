"""Batch: measure the watermark POSITION trajectory (border/center energy ratio vs training step)
for the from-scratch ablation configs C0_full / C1_secretonly / C5_crop. For each checkpoint-N dir:
load VINE_Turbo(ckpt_path=dir) (unet+ConditionAdaptor+trained vae.pth) + CustomConvNeXt(dir/CustomConvNeXt.pth),
then measure (a) border/center residual ratio on gray+natural covers, (b) clean bit-acc. Writes a CSV
we then plot as position-vs-iteration. Re-runnable: measures whatever checkpoints currently exist.
Usage: python measure_abl_trajectory.py --root <vine_abl_scratch> --out <csv> [--configs C0_full ...]"""
import sys, os, glob, argparse, csv, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt

dev = "cuda"; IMG = 256
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--configs", nargs="+", default=["C0_full", "C1_secretonly", "C5_crop"])
a = ap.parse_args()

# border band (outer 10% frame) — identical definition to measure_vine_border.py
yy, xx = np.mgrid[0:IMG, 0:IMG]
Dist = np.minimum(np.minimum(yy, IMG - 1 - yy), np.minimum(xx, IMG - 1 - xx))
BORD = Dist < (IMG // 10)

NAT = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers = {"gray": np.full((IMG, IMG, 3), 0.5, np.float32),
          "natural": np.asarray(Image.open(NAT[0]).convert("RGB").resize((IMG, IMG)), np.float32) / 255}
nat_imgs = [np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255 for p in NAT[:8]]

def to_t(arr01): return (torch.from_numpy(arr01).permute(2, 0, 1)[None].float() * 2 - 1).to(dev)

def ckpt_step(d):
    try: return int(os.path.basename(d).split("-")[1])
    except Exception: return -1

rows = []
for cfg in a.configs:
    cdirs = sorted(glob.glob(os.path.join(a.root, cfg, "checkpoint-*")), key=ckpt_step)
    print(f"=== {cfg}: {len(cdirs)} checkpoints ===", flush=True)
    for cd in cdirs:
        step = ckpt_step(cd)
        need = ["UNet2DConditionModel.pth", "ConditionAdaptor.pth", "vae.pth", "CustomConvNeXt.pth"]
        if not all(os.path.exists(os.path.join(cd, f)) for f in need):
            print(f"  step {step}: incomplete, skip", flush=True); continue
        try:
            enc = VINE_Turbo(ckpt_path=cd, device=dev).to(dev).eval()
            dec = CustomConvNeXt(secret_size=100)
            dec.load_state_dict(torch.load(os.path.join(cd, "CustomConvNeXt.pth"), map_location="cpu"), strict=True)
            dec.to(dev).eval()
            try: T = int(enc.sched.config.num_train_timesteps) - 1
            except Exception: T = 999
            def embed(img01, secret):
                with torch.no_grad(): return enc(to_t(img01), secret=secret, timesteps=torch.tensor([T], device=dev).long())
            row = {"config": cfg, "step": step}
            # (a) position: border/center ratio on gray + natural
            for name, cov in covers.items():
                s = torch.randint(0, 2, (1, 100), device=dev).float()
                wm = embed(cov, s)
                res = (wm[0] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy() - cov
                r = np.abs(res).mean(2)
                bmean, cmean = float(r[BORD].mean()), float(r[~BORD].mean())
                row[f"ratio_{name}"] = bmean / max(cmean, 1e-9)
                row[f"border_{name}"] = bmean; row[f"center_{name}"] = cmean
            # (b) clean bit-acc (n=8 natural) + clean PSNR (natural[0])
            accs, psnrs = [], []
            for img in nat_imgs:
                s = torch.randint(0, 2, (1, 100), device=dev).float()
                wm = embed(img, s)
                wm01 = (wm[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                bits = None
                with torch.no_grad(): bits = dec(to_t(wm01) * 0.5 + 0.5)
                accs.append(float(((bits > 0.5).float() == s).float().mean().item()))
                mse = np.mean((wm01 - img) ** 2)
                psnrs.append(float(10 * np.log10(1.0 / max(mse, 1e-12))))
            row["bitacc_clean"] = float(np.mean(accs)); row["psnr_clean"] = float(np.mean(psnrs))
            rows.append(row)
            print(f"  step {step:5d}: ratio_gray={row['ratio_gray']:.2f} ratio_nat={row['ratio_natural']:.2f} "
                  f"bitacc={row['bitacc_clean']:.3f} psnr={row['psnr_clean']:.2f}", flush=True)
            del enc, dec; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  step {step}: ERROR {repr(e)[:200]}", flush=True)

if rows:
    keys = ["config", "step", "ratio_gray", "ratio_natural", "border_gray", "center_gray",
            "border_natural", "center_natural", "bitacc_clean", "psnr_clean"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print(f"WROTE {len(rows)} rows -> {a.out}", flush=True)
else:
    print("NO_ROWS (no complete checkpoints yet)", flush=True)
