"""Measure whether VINE's payload is at the BORDER + its crop robustness, for a given encoder/decoder
(pretrained VINE-B baseline OR a crop-fine-tuned checkpoint). Uses the agent-verified load recipe.
  --enc pretrained:Shilin-LU/VINE-B-Enc   or   <checkpoint-dir>   (VINE_Turbo(ckpt_path=...))
  --dec pretrained:Shilin-LU/VINE-B-Dec   or   <path/to/CustomConvNeXt.pth>
Metrics: (1) residual border/center energy ratio on gray+natural covers (payload location);
         (2) crop bit-acc: embed -> center-crop keep c -> resize 256 -> decode (crop robustness)."""
import sys, glob, argparse, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev = "cuda"; IMG = 256
ap = argparse.ArgumentParser(); ap.add_argument("--enc", required=True); ap.add_argument("--dec", required=True)
ap.add_argument("--tag", default="model"); a = ap.parse_args()
print(f"[load] enc={a.enc}  dec={a.dec}", flush=True)
enc = VINE_Turbo.from_pretrained(a.enc.split(":", 1)[1]) if a.enc.startswith("pretrained:") else VINE_Turbo(ckpt_path=a.enc, device=dev)
enc.to(dev).eval()
if a.dec.startswith("pretrained:"):
    dec = CustomConvNeXt.from_pretrained(a.dec.split(":", 1)[1])
else:
    dec = CustomConvNeXt(secret_size=100); dec.load_state_dict(torch.load(a.dec, map_location="cpu"), strict=True)
dec.to(dev).eval()
try: T = int(enc.sched.config.num_train_timesteps) - 1
except Exception: T = 999
print(f"[ok] models loaded, one-step timestep T={T}", flush=True)
yy, xx = np.mgrid[0:IMG, 0:IMG]; D = np.minimum(np.minimum(yy, IMG - 1 - yy), np.minimum(xx, IMG - 1 - xx))
BORD = D < (IMG // 10)   # outer 10% frame band
def to_t(arr01): return (torch.from_numpy(arr01).permute(2, 0, 1)[None].float() * 2 - 1).to(dev)
def embed(img01, secret):
    with torch.no_grad(): return enc(to_t(img01), secret=secret, timesteps=torch.tensor([T], device=dev).long())
def decode_bits(wm):
    with torch.no_grad(): return dec((wm * 0.5 + 0.5).clamp(0, 1))
def rand_secret(): return torch.randint(0, 2, (1, 100), device=dev).float()
covers = {"gray": np.full((IMG, IMG, 3), 0.5, np.float32),
          "natural": np.asarray(Image.open(sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[0]).convert("RGB").resize((IMG, IMG)), np.float32) / 255}
# (1) payload location: residual border/center energy ratio
print(f"=== [{a.tag}] payload location (residual border/center energy ratio) ===", flush=True)
for name, cov in covers.items():
    s = rand_secret(); wm = embed(cov, s)
    res = (wm[0] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy() - cov  # residual in [0,1]
    r = np.abs(res).mean(2)
    ratio = r[BORD].mean() / max(r[~BORD].mean(), 1e-9)
    print(f"  {name:8}: border/center energy ratio = {ratio:.2f}  (border {r[BORD].mean():.4f} vs center {r[~BORD].mean():.4f})", flush=True)
# (2) crop robustness: embed -> center-crop keep c -> resize 256 -> decode
print(f"=== [{a.tag}] crop robustness (bit-acc, n=8 natural) ===", flush=True)
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:8]
imgs = [np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255 for p in srcs]
def center_crop_resize(img01, c):
    s = int(IMG * c); o = (IMG - s) // 2
    return np.asarray(Image.fromarray((img01 * 255).astype(np.uint8)).crop((o, o, o + s, o + s)).resize((IMG, IMG)), np.float32) / 255
for c in [1.0, 0.9, 0.75, 0.5]:
    accs = []
    for img in imgs:
        s = rand_secret(); wm = embed(img, s)
        wm01 = (wm[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        att = center_crop_resize(wm01, c) if c < 0.999 else wm01
        bits = decode_bits(to_t(att))
        accs.append(float(((bits > 0.5).float() == s).float().mean().item()))
    print(f"  crop keep {int(c*100):3d}% : bit-acc = {np.mean(accs):.3f}", flush=True)
print(f"MEASURE_DONE [{a.tag}]", flush=True)
