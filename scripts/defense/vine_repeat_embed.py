"""Repeatedly re-embed the SAME VINE payload (round r's input = round r-1's output watermarked image).

Per round r (1..R), with a FIXED image_id (same payload + same key) the whole time:
  W_r = VINE.embed(W_{r-1}, id)        (W_0 = clean)
  incremental residual = W_r - W_{r-1}  (what THIS round added)
  cumulative residual  = W_r - clean    (total drift from the original)
  bit-acc = VINE.detect(W_r, id)        (does it still decode the payload?)
Reports per-round incr_RMS, cum_RMS, PSNR(vs clean), bit-acc (mean over n_stats images),
and a visual fig_vine_repeat_embed.png for the hero images.

Question it answers: does same-payload re-embedding ACCUMULATE (cum_RMS ~ linear in r => additive template)
or SATURATE (encoder adds little once the mark is present => idempotent/content-adaptive)?
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO)
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
OUT = os.path.join(REPO, "results/figures/fig_vine_repeat_embed.png")
RES = 512


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
def psnr(a, b):
    m = np.mean((a - b) ** 2); return float(99.0 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n_stats", type=int, default=6)
    ap.add_argument("--heroes", nargs="+", type=int, default=[0, 5])
    args = ap.parse_args()
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device="cuda")
    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_stats]

    R = args.rounds
    incr = np.zeros((args.n_stats, R)); cum = np.zeros((args.n_stats, R))
    ps = np.zeros((args.n_stats, R)); ba = np.zeros((args.n_stats, R))
    hero = {}  # idx -> dict(clean, W[list], incrR[list], cumR[list], ba[list], cumrms[list])

    for idx, fp in enumerate(files):
        iid = f"rep_{idx:05d}"                      # FIXED id (same payload+key) for all rounds
        C0 = Image.open(fp).convert("RGB").resize((RES, RES)); C0a = arr(C0)
        cur = C0; prev_a = C0a
        if idx in args.heroes:
            hero[idx] = {"clean": C0a, "W": [], "incrR": [], "cumR": [], "ba": [], "cumrms": []}
        for r in range(R):
            W = vine.embed(cur, iid)
            if W.size != (RES, RES): W = W.resize((RES, RES))
            Wa = arr(W)
            incr[idx, r] = rms(Wa - prev_a); cum[idx, r] = rms(Wa - C0a)
            ps[idx, r] = psnr(Wa, C0a); ba[idx, r] = vine.detect(W, iid)["bit_accuracy"]
            if idx in args.heroes:
                hero[idx]["W"].append(Wa); hero[idx]["incrR"].append(Wa - prev_a)
                hero[idx]["cumR"].append(Wa - C0a); hero[idx]["ba"].append(ba[idx, r])
                hero[idx]["cumrms"].append(cum[idx, r])
            cur = W; prev_a = Wa
        print(f"  img{idx:05d}: ba={np.round(ba[idx],3).tolist()}  cum_rms={np.round(cum[idx],2).tolist()}", flush=True)

    print(f"\n=== VINE repeated same-payload embedding (R={R}, n={args.n_stats}, means) ===")
    print(f"{'round':<7}{'incr_RMS':>10}{'cum_RMS':>10}{'PSNR_vs_orig':>14}{'bit_acc':>10}")
    for r in range(R):
        print(f"{r+1:<7}{incr[:,r].mean():>10.2f}{cum[:,r].mean():>10.2f}{ps[:,r].mean():>14.2f}{ba[:,r].mean():>10.3f}")
    lin = cum[:, -1].mean() / cum[:, 0].mean()
    print(f"\ncum_RMS round{R}/round1 = {lin:.2f}  (={R:.1f} => linear accumulation; <{R} => sub-linear/saturating)")

    # ---- figure: per hero, 3 rows [W, incr x20, cum x10], cols = clean + R rounds ----
    AMP_I, AMP_C = 20, 10
    nrows = 3 * len(hero); fig, axes = plt.subplots(nrows, R + 1, figsize=(2.1 * (R + 1), 2.1 * nrows))
    blank = np.full((RES, RES, 3), 255, np.uint8)
    for h, (idx, d) in enumerate(hero.items()):
        rw, ri, rc = 3 * h, 3 * h + 1, 3 * h + 2
        axes[rw, 0].imshow(d["clean"].astype(np.uint8)); axes[ri, 0].imshow(blank); axes[rc, 0].imshow(blank)
        axes[rw, 0].set_ylabel(f"img{idx:05d}\nwatermarked", fontsize=8)
        axes[ri, 0].set_ylabel("incr resid x20", fontsize=8); axes[rc, 0].set_ylabel("cum resid x10", fontsize=8)
        axes[rw, 0].set_title("clean / round0", fontsize=8)
        for r in range(R):
            axes[rw, r + 1].imshow(np.clip(d["W"][r], 0, 255).astype(np.uint8))
            axes[ri, r + 1].imshow(np.clip(d["incrR"][r] * AMP_I + 128, 0, 255).astype(np.uint8))
            axes[rc, r + 1].imshow(np.clip(d["cumR"][r] * AMP_C + 128, 0, 255).astype(np.uint8))
            axes[rw, r + 1].set_title(f"r{r+1}  ba={d['ba'][r]:.2f}", fontsize=8)
            axes[rc, r + 1].set_title(f"cumRMS={d['cumrms'][r]:.1f}", fontsize=7)
        for ax in axes[rw], : pass
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("VINE: repeated same-payload re-embedding — incremental vs cumulative residual, and bit-acc per round",
                 fontsize=11, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"[fig] -> {OUT}\nVINE_REPEAT_DONE")


if __name__ == "__main__":
    main()
