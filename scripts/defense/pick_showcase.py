"""Rank the embedded watermarked images by fidelity and build a contact sheet of candidates
(clean | watermarked | 10x residual) so we can pick the nicest showcase cases for the report.
CPU-only (PIL/numpy)."""
import glob, os, sys
import numpy as np
from PIL import Image, ImageDraw

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
EMB = os.path.join(REPO, "results/defense/ext_4fused_coco")
OUT = os.path.join(REPO, "results/figures"); os.makedirs(OUT, exist_ok=True)


def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.mean((a - b) ** 2); return 10 * np.log10(255 ** 2 / m) if m > 1e-9 else 99.0


coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
rows = []
for i in range(50):
    wp = os.path.join(EMB, f"img_{i:05d}.png")
    if not os.path.exists(wp): continue
    clean = Image.open(coco[4000 + i]).convert("RGB").resize((512, 512))
    wm = Image.open(wp).convert("RGB").resize((512, 512))
    rows.append((psnr(clean, wm), i))
rows.sort(reverse=True)
print("rank by PSNR (idx within embedded set, COCO idx = 4000+idx):")
for r, i in rows: print(f"  img_{i:05d}  PSNR={r:.2f}")

# contact sheet of top 12
top = [i for _, i in rows[:12]]
TH = 200
sheet = Image.new("RGB", (TH * 3 + 40, TH * len(top) + 20), "white")
d = ImageDraw.Draw(sheet)
for row, i in enumerate(top):
    clean = Image.open(coco[4000 + i]).convert("RGB").resize((TH, TH))
    wm = Image.open(os.path.join(EMB, f"img_{i:05d}.png")).convert("RGB").resize((512, 512))
    cl512 = Image.open(coco[4000 + i]).convert("RGB").resize((512, 512))
    res = np.clip(np.abs(np.asarray(wm, np.int16) - np.asarray(cl512, np.int16)) * 10, 0, 255).astype(np.uint8)
    resim = Image.fromarray(res).resize((TH, TH))
    y = 10 + row * TH
    sheet.paste(clean, (10, y)); sheet.paste(wm.resize((TH, TH)), (10 + TH, y)); sheet.paste(resim, (10 + TH * 2, y))
    d.text((12, y + 2), f"img{i:05d}", fill="yellow")
sheet.save(os.path.join(OUT, "candidates_contact_sheet.png"))
print(f"\n[done] contact sheet -> {OUT}/candidates_contact_sheet.png (cols: clean | watermarked | 10x residual)")
