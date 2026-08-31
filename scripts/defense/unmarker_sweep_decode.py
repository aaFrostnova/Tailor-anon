"""Decode the UnMarker strength sweep (no-crop): quality <-> de-watermarking (ba_w) <-> re-embed probe AUROC.

For each strength (stage1 iters), over the attacked watermarked & clean sets:
  quality: PSNR + LPIPS of attacked-wm vs ORIGINAL clean (aligned, no crop)
  de-watermarking: direct detect ba_w (low => attack succeeded)
  probe: round-1 re-embed AUROC (iwr<icr), + iww, surv
Answers (1) equal-fidelity: how much quality must UnMarker destroy to break detection; (2) does a high-fidelity
strength leave a probe sweet-spot (ba_w low but mark residual detectable).
"""
import glob, os, sys
import numpy as np, torch
from PIL import Image
import lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO)
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"; RES = 512
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
STRENGTHS = [("s50", 50), ("s200", 200), ("s600", 600), ("s1000", 1000)]


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
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device="cuda")
    lp = lpips.LPIPS(net="alex").to("cuda").eval()
    def lpd(a, b):
        ta = torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].float().cuda()
        tb = torch.from_numpy(b / 127.5 - 1).permute(2, 0, 1)[None].float().cuda()
        with torch.no_grad(): return float(lp(ta, tb).item())
    def emb(a, iid):
        W = vine.embed(Image.fromarray(a.astype(np.uint8)), iid); return arr(W if W.size == (RES, RES) else W.resize((RES, RES)))
    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))

    rows = []
    for tag, it in STRENGTHS:
        wd = os.path.join(REPO, f"results/defense/sweep_{tag}_wm"); cd = os.path.join(REPO, f"results/defense/sweep_{tag}_cl")
        if not (os.path.isdir(wd) and os.path.isdir(cd)): continue
        n = min(len(glob.glob(os.path.join(wd, "*.png"))), len(glob.glob(os.path.join(cd, "*.png"))))
        if n == 0: continue
        ps, lps, baw, iwr, icr, iww, surv = [], [], [], [], [], [], []
        for i in range(n):
            rid = f"unm_{i:05d}"; wid = f"WR_{i:05d}"
            O = arr(Image.open(coco[4000 + i]).convert("RGB").resize((RES, RES)))
            Aw = arr(Image.open(os.path.join(wd, f"img_{i:05d}.png"))); Ac = arr(Image.open(os.path.join(cd, f"img_{i:05d}.png")))
            ps.append(psnr(Aw, O)); lps.append(lpd(Aw, O))
            baw.append(vine.detect(Image.fromarray(Aw.astype(np.uint8)), rid)["bit_accuracy"])
            iwr.append(rms(emb(Aw, rid) - Aw)); iww.append(rms(emb(Aw, wid) - Aw)); icr.append(rms(emb(Ac, rid) - Ac))
            surv.append(rms(Aw - Ac))
        rows.append({"tag": tag, "iter": it, "n": n, "psnr": np.mean(ps), "lpips": np.mean(lps),
                     "ba_w": np.mean(baw), "auroc": auroc(iwr, icr), "iwr": np.mean(iwr), "icr": np.mean(icr),
                     "iww": np.mean(iww), "surv": rms(np.array(surv))})

    print(f"\n=== UnMarker STRENGTH SWEEP (no-crop, pure spectral) ===")
    print(f"{'iter':>6}{'PSNR':>8}{'LPIPS':>8}{'ba_w':>8}{'probeAUROC':>12}{'iwr':>7}{'icr':>7}{'surv':>7}")
    for r in rows:
        print(f"{r['iter']:>6}{r['psnr']:>8.1f}{r['lpips']:>8.3f}{r['ba_w']:>8.3f}{r['auroc']:>12.3f}{r['iwr']:>7.2f}{r['icr']:>7.2f}{r['surv']:>7.1f}")
    print("ba_w=0.5 => detection broken; PSNR<~28/LPIPS>0.1 => visible damage; AUROC>0.75 => probe works")

    its = [r["iter"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(its, [r["psnr"] for r in rows], "o-", color="#2ca02c", label="PSNR vs orig")
    ax2 = ax[0].twinx()
    ax2.plot(its, [r["ba_w"] for r in rows], "s-", color="#d62728", label="ba_w (detect)")
    ax2.plot(its, [r["auroc"] for r in rows], "^--", color="#1f77b4", label="probe AUROC")
    ax2.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax[0].set_xlabel("UnMarker stage1 iterations (strength)"); ax[0].set_ylabel("PSNR (dB)", color="#2ca02c")
    ax2.set_ylabel("bit-acc / AUROC"); ax2.set_ylim(0.3, 1.05); ax[0].set_xscale("log")
    ax[0].set_title("Strength: PSNR drops, ba_w drops, AUROC tracks ba_w")
    l1, la1 = ax[0].get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels(); ax2.legend(l1 + l2, la1 + la2, loc="center left", fontsize=8)
    # equal-fidelity
    ax[1].plot([r["psnr"] for r in rows], [r["ba_w"] for r in rows], "s-", color="#d62728", label="ba_w (detect)")
    ax[1].plot([r["psnr"] for r in rows], [r["auroc"] for r in rows], "^--", color="#1f77b4", label="probe AUROC")
    ax[1].axhline(0.5, ls=":", color="gray", lw=0.8)
    ax[1].set_xlabel("attacked-image PSNR vs orig (higher = more faithful)"); ax[1].set_ylabel("bit-acc / AUROC")
    ax[1].invert_xaxis(); ax[1].set_title("Equal-fidelity: at high PSNR does UnMarker break detection?"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "results/figures/fig_unmarker_strength_sweep.png"), dpi=130, bbox_inches="tight")
    print("[fig] fig_unmarker_strength_sweep.png\nSWEEP_DECODE_DONE")


if __name__ == "__main__":
    main()
