"""DECISIVE test for VideoSeal as a candidate fragment: it is a PIXEL-space neural
watermark (TrustMark/MaskWM sibling), so the deciding question is whether it survives
diffusion REGENERATION. If it dies (bit-acc -> 0.5) like every other pixel-domain mark,
it overlaps TrustMark's death mode and would be a redundant/negative fragment.

Pure VideoSeal: embed a fixed message, clean-decode (sanity ~1.0), then in-env multi-round
stable_regen x1/x2, decode bit-acc. No crypto/BCH needed for this go/no-go.
"""
import os, sys, glob
import numpy as np, torch
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
sys.path.insert(0, os.path.join(REPO, "external/videoseal"))
import videoseal
from _regen_util import build_regen_pipe, stable_regen

dev = "cuda"
model = videoseal.load("videoseal")        # image+video model, 256-bit
model = model.to(dev).eval()
nbits = model.nbits if hasattr(model, "nbits") else None
pipe = build_regen_pipe()

def to_t(pil): return torch.from_numpy(np.asarray(pil.convert("RGB").resize((512, 512)), np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)
def to_pil(t): return Image.fromarray((t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
def bits_of(det):
    p = det["preds"] if isinstance(det, dict) else det
    return (p[:, 1:] > 0).int()[0].cpu().numpy()          # drop detection bit, threshold message bits

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:188]
print(f"loaded videoseal; nbits={nbits}", flush=True)
clean, r1, r2, psn = [], [], [], []
for j, fp in enumerate(imgs):
    x = to_t(Image.open(fp).convert("RGB"))
    with torch.no_grad():
        out = model.embed(x, is_video=False) if "is_video" in model.embed.__code__.co_varnames else model.embed(x)
        msg = out["msgs"][0].cpu().numpy().astype(int)
        xw = out["imgs_w"]
        e = ((x - xw) ** 2).mean().item(); psn.append(99 if e < 1e-9 else 10 * np.log10(1.0 / e))
        clean.append(np.mean(bits_of(model.detect(xw, is_video=False)) == msg))
        pw = to_pil(xw)
        g1 = to_t(stable_regen(pipe, pw, seed=11)); r1.append(np.mean(bits_of(model.detect(g1, is_video=False)) == msg))
        g2 = to_t(stable_regen(pipe, to_pil(g1), seed=12)); r2.append(np.mean(bits_of(model.detect(g2, is_video=False)) == msg))
    print(f"  [{j+1}/{len(imgs)}] clean={clean[-1]:.3f} r1={r1[-1]:.3f} r2={r2[-1]:.3f}", flush=True)

print(f"\nVideoSeal  PSNR={np.mean(psn):.1f}dB")
print(f"  clean bit-acc   = {np.mean(clean):.3f}  (want ~1.0)")
print(f"  regen x1 bit-acc= {np.mean(r1):.3f}")
print(f"  regen x2 bit-acc= {np.mean(r2):.3f}")
print("verdict: " + ("SURVIVES regen -> worth full fragment integration" if np.mean(r1) > 0.62 else
                     "DIES under regen like TrustMark/MaskWM -> redundant/negative fragment"))
print("VSEAL_REGEN_DONE")
