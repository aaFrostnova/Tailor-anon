"""Visualize the VINE watermark residual IN THE DIFFUSION LATENT MANIFOLD, before vs after regen.

VINE puts its mark on a diffusion-VAE latent manifold; regen (encode->noise->denoise->decode) operates
on that SAME manifold, which is why the mark survives even though the pixel residual is re-synthesized.
We make that visible by encoding into the regen pipeline's own VAE (SD-2.1) latent space.

Per hero image, columns:
  clean | watermarked W | latent residual BEFORE (||z_W - z_C|| over 4 ch) | latent residual AFTER regen
The AFTER residual uses SAME-SEED regen of clean & watermarked so content divergence cancels and only the
watermark's latent footprint remains. Caption: VINE bit-acc clean->after-regen; latent-resid RMS before->after;
and corr(resid_before, resid_after) -- the key number: a high correlation means the watermark perturbation
PERSISTS in the latent manifold through regen (mechanistic reason bit-acc stays ~0.90).
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
OUT = os.path.join(REPO, "results/figures")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heroes", nargs="+", type=int, default=[0, 5])
    ap.add_argument("--seed", type=int, default=1234); args = ap.parse_args()
    dev = "cuda"
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=dev)
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    vae = pipe.vae; sf = vae.config.scaling_factor
    tmp = tempfile.mkdtemp()

    def regen_seeded(pil):
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        pil.save(ip); regen.attack([ip], [op])
        return Image.open(op).convert("RGB").resize((512, 512))

    def latent(pil):  # deterministic VAE-mean latent, (4,64,64)
        a = np.asarray(pil.convert("RGB").resize((512, 512)), np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev, torch.float16) * 2 - 1
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * sf
        return z.float().cpu().numpy()[0]

    def corr(a, b):
        a, b = a.ravel(), b.ravel()
        return float(np.corrcoef(a, b)[0, 1])

    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    nH = len(args.heroes)
    fig, axes = plt.subplots(nH, 4, figsize=(15, 3.8 * nH))
    if nH == 1: axes = axes[None, :]
    cols = ["clean", "watermarked", "latent resid BEFORE  ||z_W - z_C||",
            "latent resid AFTER regen  ||z_Rw - z_Rc||"]
    for hi, h in enumerate(args.heroes):
        iid = f"vl_{h:05d}"
        C = Image.open(coco[4000 + h]).convert("RGB").resize((512, 512))
        W = vine.embed(C, iid)
        if W.size != (512, 512): W = W.resize((512, 512))
        ba_clean = vine.detect(W, iid)["bit_accuracy"]
        zC, zW = latent(C), latent(W)
        Rc, Rw = regen_seeded(C), regen_seeded(W)
        ba_regen = vine.detect(Rw, iid)["bit_accuracy"]
        zRc, zRw = latent(Rc), latent(Rw)
        rb, ra = zW - zC, zRw - zRc                       # latent residuals (4,64,64)
        mb = np.sqrt((rb ** 2).sum(0)); ma = np.sqrt((ra ** 2).sum(0))   # per-pixel L2 over channels
        rms_b, rms_a = float(np.sqrt((rb ** 2).mean())), float(np.sqrt((ra ** 2).mean()))
        r_all = corr(rb, ra)
        r_ch = [corr(rb[c], ra[c]) for c in range(rb.shape[0])]
        vmax = max(mb.max(), ma.max())
        for c, (im, ttl) in enumerate(zip([C, W, mb, ma], cols)):
            ax = axes[hi, c]
            if c < 2:
                ax.imshow(im); ax.axis("off")
            else:
                hm = ax.imshow(im, cmap="magma", vmin=0, vmax=vmax)
                ax.axis("off"); fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.02)
            if hi == 0: ax.set_title(ttl, fontsize=11)
        axes[hi, 0].set_ylabel(f"img{h:05d}", fontsize=11)
        cap = (f"img{h:05d}:  VINE bit-acc clean={ba_clean:.2f} -> after regen={ba_regen:.2f}    "
               f"latent-resid RMS {rms_b:.3f} -> {rms_a:.3f}    "
               f"corr(before, after) = {r_all:+.3f}")
        axes[hi, 2].text(0.0, -0.08, cap, transform=axes[hi, 2].transAxes, fontsize=10, va="top")
        print(f"  img{h:05d}: ba {ba_clean:.2f}->{ba_regen:.2f}  latRMS {rms_b:.3f}->{rms_a:.3f}  "
              f"corr_all={r_all:+.3f}  corr_per_ch={[round(x,3) for x in r_ch]}", flush=True)
    fig.suptitle("VINE watermark in the diffusion latent manifold (SD-2.1 VAE) — the mark persists through regen "
                 "as a correlated, attenuated latent footprint", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT, "fig_vine_latent_residual.png"); fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[done] -> {out}\nVINE_LATENT_RESID_DONE")


if __name__ == "__main__":
    main()
