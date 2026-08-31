"""Compare VINE-B vs VINE-R watermark STRENGTH. Since finetune.py (VINE-R production) FREEZES the encoder and
trains only the decoder, the encoders should be IDENTICAL -> same watermark strength. Verify by (1) comparing
encoder & decoder weights, (2) embedding the same cover+secret with both encoders and comparing residual RMS /
PSNR / border concentration."""
import sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev = "cuda"; IMG = 256
print("[load] B/R encoders + decoders ...", flush=True)
encB = VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc").to(dev).eval()
encR = VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev).eval()
decB = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec").to(dev).eval()
decR = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec").to(dev).eval()
def weight_diff(m1, m2):
    s1, s2 = m1.state_dict(), m2.state_dict(); keys = [k for k in s1 if k in s2 and s1[k].shape == s2[k].shape]
    maxd = 0.0; ndiff = 0; tot = 0
    for k in keys:
        d = (s1[k].float() - s2[k].float()).abs().max().item(); maxd = max(maxd, d)
        if d > 1e-6: ndiff += 1
        tot += 1
    return maxd, ndiff, tot
mB, nB, tB = weight_diff(encB, encR); print(f"[weights] ENCODER B vs R: max|Δ|={mB:.2e}, params differing={nB}/{tB}", flush=True)
mD, nD, tD = weight_diff(decB, decR); print(f"[weights] DECODER B vs R: max|Δ|={mD:.2e}, params differing={nD}/{tD}", flush=True)
yy, xx = np.mgrid[0:IMG, 0:IMG]; D = np.minimum(np.minimum(yy, IMG-1-yy), np.minimum(xx, IMG-1-xx)); BORD = D < IMG//10
T = int(encB.sched.config.num_train_timesteps) - 1
def to_t(a01): return (torch.from_numpy(a01).permute(2,0,1)[None].float()*2-1).to(dev)
def embed(enc, img01, sec):
    with torch.no_grad(): wm = enc(to_t(img01), secret=sec, timesteps=torch.tensor([T], device=dev).long())
    return (wm[0]*0.5+0.5).permute(1,2,0).cpu().numpy()
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:5]
print("=== watermark strength (same cover+secret, B vs R encoder) ===", flush=True)
for tag, enc in [("VINE-B", encB), ("VINE-R", encR)]:
    rms, psnr, ratio = [], [], []
    for s in srcs:
        cov = np.asarray(Image.open(s).convert("RGB").resize((IMG,IMG)), np.float64)/255
        sec = torch.randint(0,2,(1,100),device=dev).float()
        wm = embed(enc, cov, sec); res = wm - cov
        rms.append(np.sqrt((res**2).mean())*255)
        psnr.append(10*np.log10(1.0/max((res**2).mean(),1e-12)))
        r = np.abs(res).mean(2); ratio.append(r[BORD].mean()/max(r[~BORD].mean(),1e-9))
    print(f"  {tag}: residual RMS={np.mean(rms):.3f}/255  PSNR={np.mean(psnr):.2f}dB  border/center={np.mean(ratio):.2f}", flush=True)
# also: does R-encoder's output decode with R-decoder better (they're paired)?
print("STRENGTH_DONE", flush=True)
