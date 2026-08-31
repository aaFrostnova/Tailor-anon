"""Decode UnMarker no-crop strength sweep on TRUSTMARK wm: spectral attack effect + re-embed probe.

Mirror of unmarker_sweep_decode but for TrustMark (the high-freq pixel mark). Prediction: spectral UnMarker
(stage1 erases high-freq) KILLS TrustMark (opposite of VINE). Clean control reuses the VINE sweep's UnMarked
clean dirs (sweep_{tag}_cl) -- same COCO images, watermark-agnostic. Probe: iwr(wm) vs icr(clean) AUROC.
"""
import glob, os, sys
import numpy as np, torch
from PIL import Image
import lpips
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO)
from src.trustmark_fragment import TrustMarkFragment
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512
STR = [("s50", 50), ("s200", 200), ("s600", 600), ("s1000", 1000)]


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
def psnr(a, b):
    m = np.mean((a - b) ** 2); return float(99 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m))
def auroc(pos, neg):
    c = 0.0
    for a in pos:
        for b in neg: c += (a < b) + 0.5 * (a == b)
    return c / (len(pos) * len(neg) + 1e-9)


def main():
    tm = TrustMarkFragment(master_key=b"v5_key_encoder_master", method_name="trustmark", n_bits=100, model_type="B", device="cuda")
    lp = lpips.LPIPS(net="alex").to("cuda").eval()
    def lpd(a, b):
        ta = torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].float().cuda(); tb = torch.from_numpy(b / 127.5 - 1).permute(2, 0, 1)[None].float().cuda()
        with torch.no_grad(): return float(lp(ta, tb).item())
    def emb(a, iid):
        W = tm.embed(Image.fromarray(a.astype(np.uint8)), iid); return arr(W if W.size == (RES, RES) else W.resize((RES, RES)))
    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    print(f"\n=== UnMarker no-crop sweep on TrustMark-B (spectral attack) ===")
    print(f"{'iter':>6}{'PSNR':>8}{'LPIPS':>8}{'ba_w':>8}{'probeAUROC':>12}{'iwr':>7}{'icr':>7}")
    for tag, it in STR:
        wd = os.path.join(REPO, f"results/defense/tm_sweep_{tag}_wm"); cd = os.path.join(REPO, f"results/defense/sweep_{tag}_cl")
        if not (os.path.isdir(wd) and os.path.isdir(cd)): continue
        n = min(len(glob.glob(wd + "/*.png")), len(glob.glob(cd + "/*.png")))
        if n == 0: continue
        ps, lps, baw, iwr, icr = [], [], [], [], []
        for i in range(n):
            rid = f"unm_tm_{i:05d}"; wid = f"WRtm_{i:05d}"
            O = arr(Image.open(coco[4000 + i]).convert("RGB").resize((RES, RES)))
            Aw = arr(Image.open(os.path.join(wd, f"img_{i:05d}.png"))); Ac = arr(Image.open(os.path.join(cd, f"img_{i:05d}.png")))
            ps.append(psnr(Aw, O)); lps.append(lpd(Aw, O))
            baw.append(tm.detect(Image.fromarray(Aw.astype(np.uint8)), rid)["bit_accuracy"])
            iwr.append(rms(emb(Aw, rid) - Aw)); icr.append(rms(emb(Ac, rid) - Ac))
        print(f"{it:>6}{np.mean(ps):>8.1f}{np.mean(lps):>8.3f}{np.mean(baw):>8.3f}{auroc(iwr, icr):>12.3f}{np.mean(iwr):>7.2f}{np.mean(icr):>7.2f}")
    print("compare VINE no-crop sweep: ba_w stayed 0.99 (spectral immune). If TrustMark ba_w DROPS => spectral kills it.")
    print("TM_SWEEP_DECODE_DONE")


if __name__ == "__main__":
    main()
