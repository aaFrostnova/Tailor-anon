"""STABLE re-run of fig_vine_regen_residual + fig_vine_latent_residual (8-step regen, multi-hero).

Fixes the 3-step sampler-instability noise that made img0 look 'divergent'. Per hero:
  pixel fig:  clean | W | VINE residual (W-clean)x20 | regen(W) | surviving residual (regen(W)-regen(C))x20
  latent fig: clean | W | latent resid BEFORE ||zW-zC|| | latent resid AFTER ||zRw-zRc|| , with corr(before,after)
Paired same-seed regen so the surviving residual isolates the watermark.
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "defense"))
from src.vine_crypto_wrapper import VineCryptoWrapper
from _regen_util import build_regen_pipe, stable_regen
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
FIG = os.path.join(REPO, "results/figures"); RES = 512


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heroes", nargs="+", type=int, default=[0, 2, 5, 6])
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args(); dev = "cuda"
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=dev)
    pipe = build_regen_pipe(); vae = pipe.vae; sf = vae.config.scaling_factor
    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))

    def latent(pil):
        a = np.asarray(pil.convert("RGB").resize((RES, RES)), np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev, torch.float16) * 2 - 1
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * sf
        return z.float().cpu().numpy()[0]

    nH = len(args.heroes); AMP = 20
    figp, axp = plt.subplots(nH, 5, figsize=(15, 3.0 * nH))
    figl, axl = plt.subplots(nH, 4, figsize=(14, 3.4 * nH))
    pcols = ["clean", "watermarked W", "VINE residual x20", "regen(W) [8-step]", "surviving residual x20"]
    lcols = ["clean", "watermarked W", "latent resid BEFORE", "latent resid AFTER regen"]
    for hi, h in enumerate(args.heroes):
        iid = f"st_{h:05d}"
        C = Image.open(files[4000 + h]).convert("RGB").resize((RES, RES))
        W = vine.embed(C, iid)
        if W.size != (RES, RES): W = W.resize((RES, RES))
        baC = vine.detect(W, iid)["bit_accuracy"]
        Rc = stable_regen(pipe, C, args.seed, denoise_steps=args.steps)
        Rw = stable_regen(pipe, W, args.seed, denoise_steps=args.steps)
        baR = vine.detect(Rw, iid)["bit_accuracy"]
        Ca, Wa, Rca, Rwa = arr(C), arr(W), arr(Rc), arr(Rw)
        pix_b, pix_a = rms(Wa - Ca), rms(Rwa - Rca)
        # pixel row
        rb = np.clip((Wa - Ca) * AMP + 128, 0, 255).astype(np.uint8)
        ra = np.clip((Rwa - Rca) * AMP + 128, 0, 255).astype(np.uint8)
        for c, im in enumerate([Ca, Wa, rb, Rwa, ra]):
            axp[hi, c].imshow(np.clip(im, 0, 255).astype(np.uint8)); axp[hi, c].axis("off")
            if hi == 0: axp[hi, c].set_title(pcols[c], fontsize=10)
        axp[hi, 4].text(0.0, -0.08, f"img{h:05d}: bit-acc {baC:.2f}->{baR:.2f}  pix-RMS {pix_b:.1f}->{pix_a:.1f}",
                        transform=axp[hi, 4].transAxes, fontsize=9, va="top")
        # latent row
        zC, zW, zRc, zRw = latent(C), latent(W), latent(Rc), latent(Rw)
        lb, la = zW - zC, zRw - zRc
        mb, ma = np.sqrt((lb ** 2).sum(0)), np.sqrt((la ** 2).sum(0))
        cr = float(np.corrcoef(lb.ravel(), la.ravel())[0, 1]); vmax = max(mb.max(), ma.max())
        for c, im in enumerate([Ca, Wa, mb, ma]):
            ax = axl[hi, c]
            if c < 2: ax.imshow(np.clip(im, 0, 255).astype(np.uint8))
            else:
                hm = ax.imshow(im, cmap="magma", vmin=0, vmax=vmax); figl.colorbar(hm, ax=ax, fraction=0.046, pad=0.02)
            ax.axis("off")
            if hi == 0: ax.set_title(lcols[c], fontsize=10)
        axl[hi, 2].text(0.0, -0.10, f"img{h:05d}: bit-acc {baC:.2f}->{baR:.2f}  latRMS {rms(lb):.3f}->{rms(la):.3f}  "
                        f"corr(before,after)={cr:+.3f}", transform=axl[hi, 2].transAxes, fontsize=9, va="top")
        print(f"  img{h:05d}: bit-acc {baC:.2f}->{baR:.2f}  pixRMS {pix_b:.1f}->{pix_a:.1f}  latent_corr={cr:+.3f}", flush=True)

    figp.suptitle("VINE regen residual (STABLE 8-step regen) — pixel space", fontsize=12, y=0.997)
    figp.tight_layout(rect=[0, 0, 1, 0.99]); figp.savefig(os.path.join(FIG, "fig_vine_regen_residual_stable.png"), dpi=125, bbox_inches="tight")
    figl.suptitle("VINE regen residual (STABLE 8-step regen) — diffusion latent manifold", fontsize=12, y=0.997)
    figl.tight_layout(rect=[0, 0, 1, 0.99]); figl.savefig(os.path.join(FIG, "fig_vine_latent_residual_stable.png"), dpi=125, bbox_inches="tight")
    print("[figs] fig_vine_regen_residual_stable.png + fig_vine_latent_residual_stable.png\nSTABLE_RESID_DONE")


if __name__ == "__main__":
    main()
