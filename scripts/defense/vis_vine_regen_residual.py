"""Visualize the VINE watermark residual before vs after one regen attack.

Per hero image, columns:
  clean | watermarked W | VINE residual (W-clean)x20 | regen(W) | surviving residual (regen(W)-regen(clean))x20
The last column isolates the WATERMARK's post-regen signature: regen clean & watermarked with the
SAME random seed so their difference cancels regen's content changes and leaves only the watermark.
Caption: VINE bit-acc clean->after-regen, and residual RMS before->after.
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
OUT = os.path.join(REPO, "results/figures"); AMP = 20


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/home/mingzhel_umass_edu/.conda/envs/fingerprint/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


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
    tmp = tempfile.mkdtemp()

    def regen_seeded(pil):
        torch.manual_seed(args.seed); np.random.seed(args.seed)  # same randomness for clean & wm
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        pil.save(ip); regen.attack([ip], [op])
        return Image.open(op).convert("RGB").resize((512, 512))

    def resid(a, b):
        return np.clip((np.asarray(a, np.int16) - np.asarray(b, np.int16)) * AMP + 128, 0, 255).astype(np.uint8)

    def rms(a, b):
        return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))

    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    TH = 300; F = _font(15)
    cols = ["clean", "watermarked", "VINE residual x20", "after regen", "surviving residual x20"]
    sheet = Image.new("RGB", (TH * 5 + 12, (TH + 46) * len(args.heroes) + 24), "white")
    d = ImageDraw.Draw(sheet)
    for c, t in enumerate(cols):
        d.text((4 + c * (TH + 2), 4), t, fill="black", font=F)
    for hi, h in enumerate(args.heroes):
        iid = f"vr_{h:05d}"
        C = Image.open(coco[4000 + h]).convert("RGB").resize((512, 512))
        W = vine.embed(C, iid)
        if W.size != (512, 512): W = W.resize((512, 512))
        ba_clean = vine.detect(W, iid)["bit_accuracy"]
        Rc = regen_seeded(C); Rw = regen_seeded(W)
        ba_regen = vine.detect(Rw, iid)["bit_accuracy"]
        rb = resid(W, C); ra = resid(Rw, Rc)
        rms_b, rms_a = rms(W, C), rms(Rw, Rc)
        y = 24 + hi * (TH + 46)
        for c, im in enumerate([C, W, Image.fromarray(rb), Rw, Image.fromarray(ra)]):
            sheet.paste(im.resize((TH, TH)), (2 + c * (TH + 2), y))
        d.text((4, y + TH + 2), f"img{h:05d}: VINE bit-acc clean={ba_clean:.2f} -> after regen={ba_regen:.2f}   "
                                f"residual RMS {rms_b:.2f} -> {rms_a:.2f}", fill="black", font=F)
        print(f"  img{h:05d}: ba {ba_clean:.2f}->{ba_regen:.2f}  rms {rms_b:.2f}->{rms_a:.2f}", flush=True)
    out = os.path.join(OUT, "fig_vine_regen_residual.png"); sheet.save(out)
    print(f"[done] -> {out}\nVINE_REGEN_RESID_DONE")


if __name__ == "__main__":
    main()
