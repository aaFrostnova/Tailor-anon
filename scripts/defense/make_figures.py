"""Showcase figures for the report:
  (1) fig_imperceptibility.png : diverse images, cols [clean | watermarked | 10x residual] + PSNR/SSIM
  (2) fig_robustness.png       : one hero image through many attacks, each labeled with the attacked
                                  image + our fused bit-acc + detect ✓/✗ (hero must be in the
                                  UnMarker set, idx 0-9).
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from scripts.defense.benchmark_composite_defense import GEO, _crop_then_jpeg

KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
EMB = os.path.join(REPO, "results/defense/ext_4fused_coco")
OUT = os.path.join(REPO, "results/figures"); os.makedirs(OUT, exist_ok=True)
SPEC = {"vine": ("prob", "raw_probs"), "dft": ("logit", "raw_logits"),
        "qim": ("logit", "raw_logits"), "trustmark": ("logit", "raw_logits")}


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/home/mingzhel_umass_edu/.conda/envs/fingerprint/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.mean((a - b) ** 2); return 10 * np.log10(255 ** 2 / m) if m > 1e-9 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imperc_idx", nargs="+", type=int, default=[9, 24, 21, 48])
    ap.add_argument("--hero", type=int, default=-1, help="-1 = auto-pick a representative hero (regen must detect)")
    ap.add_argument("--hero_candidates", nargs="+", type=int, default=[2, 5, 7, 3, 1, 4, 6, 8, 0, 9])
    args = ap.parse_args()
    dev = "cuda"
    sb = ShortenedBCH(); tau = float(__import__("scipy.stats", fromlist=["binom"]).binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    from pytorch_msssim import ssim as ssim_fn
    F1, F2 = _font(22), _font(18)

    # ---------- (1) imperceptibility ----------
    TH = 320
    rows = args.imperc_idx
    sheet = Image.new("RGB", (TH * 3 + 40, (TH + 34) * len(rows) + 10), "white")
    d = ImageDraw.Draw(sheet)
    d.text((14, 4), "clean", fill="black", font=F1); d.text((24 + TH, 4), "watermarked", fill="black", font=F1)
    d.text((34 + TH * 2, 4), "residual x10", fill="black", font=F1)
    for r, i in enumerate(rows):
        clean = Image.open(coco[4000 + i]).convert("RGB").resize((512, 512))
        wm = Image.open(os.path.join(EMB, f"img_{i:05d}.png")).convert("RGB").resize((512, 512))
        res = np.clip(np.abs(np.asarray(wm, np.int16) - np.asarray(clean, np.int16)) * 10, 0, 255).astype(np.uint8)
        a = torch.from_numpy(np.asarray(clean, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        b = torch.from_numpy(np.asarray(wm, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad(): ss = float(ssim_fn(a, b, data_range=1.0))
        y = 34 + r * (TH + 34)
        sheet.paste(clean.resize((TH, TH)), (10, y)); sheet.paste(wm.resize((TH, TH)), (20 + TH, y))
        sheet.paste(Image.fromarray(res).resize((TH, TH)), (30 + TH * 2, y))
        d.text((14, y + TH + 6), f"PSNR {psnr(clean, wm):.1f} dB   SSIM {ss:.3f}", fill="black", font=F2)
    sheet.save(os.path.join(OUT, "fig_imperceptibility.png"))
    print(f"[fig1] -> {OUT}/fig_imperceptibility.png", flush=True)

    # ---------- (2) robustness montage (hero through attacks) ----------
    frag = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
            "dft": DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev),
            "qim": QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev),
            "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)}
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker, VAEWMAttacker, JPEGAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    jp = JPEGAttacker(quality=25); vae_c = VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=dev)
    tmp = tempfile.mkdtemp()

    def setup_hero(h):
        iid = f"ext_{h:05d}"
        txx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        wm = Image.open(os.path.join(EMB, f"img_{h:05d}.png")).convert("RGB").resize((512, 512))
        p = os.path.join(tmp, "h.png"); wm.save(p)
        return iid, txx, wm, p

    def make(a, h, wmimg, ip):
        op = os.path.join(tmp, f"h_{a}.png")
        if a == "watermarked": return wmimg
        if a == "jpeg": jp.attack([ip], [op])
        elif a == "vae_c": vae_c.attack([ip], [op])
        elif a == "regen": regen.attack([ip], [op])
        elif a in GEO:
            o = GEO[a](wmimg); (o if o.size == (512, 512) else o.resize((512, 512))).save(op)
        elif a == "crop_jpeg": _crop_then_jpeg(ip, op)
        elif a == "ctrlregen_0.5":
            return Image.open(os.path.join(REPO, "results/defense/ext_4fused_coco_ctrlregen_05", f"img_{h:05d}.png")).convert("RGB").resize((512, 512))
        elif a == "unmarker":
            return Image.open(os.path.join(REPO, "results/defense/ext_4fused_coco_unmarker", f"img_{h:05d}.png")).convert("RGB").resize((512, 512))
        return Image.open(op).convert("RGB").resize((512, 512))

    def decode(att, image_id, tx):
        al = {}
        for name in ["vine", "dft", "qim", "trustmark"]:
            kind, getter = SPEC[name]; m = frag[name]; perm, M = m.get_perm_M(image_id)
            al[name] = method_soft_to_codeword_llr(getattr(m, getter)(att), perm, M, kind=kind, n_codeword=sb.n)
        fused = fuse_llrs(al, weights=None, n_codeword=sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        det = (fba >= tau) or bool(decode_and_verify(fused, image_id, codec=sb)["detected"])
        return fba, det

    # pick a representative hero: clean AND regen must detect (matches the 0.94 aggregate)
    cand = [args.hero] if args.hero >= 0 else args.hero_candidates
    hero, image_id, tx, wmimg, ip = None, None, None, None, None
    for h in cand:
        iid, txx, wm, p = setup_hero(h)
        dc = decode(make("watermarked", h, wm, p), iid, txx)[1]
        dr = decode(make("regen", h, wm, p), iid, txx)[1]
        print(f"  [hero-search] img{h:05d} clean_det={dc} regen_det={dr}", flush=True)
        if dc and dr:
            hero, image_id, tx, wmimg, ip = h, iid, txx, wm, p; break
    if hero is None:
        hero = cand[0]; image_id, tx, wmimg, ip = setup_hero(hero)

    attacks = ["watermarked", "jpeg", "regen", "vae_c", "rs256", "crop75", "hflip", "ctrlregen_0.5", "unmarker"]
    TH = 256; cols = len(attacks)
    mont = Image.new("RGB", (TH * cols + (cols + 1) * 6, TH + 60), "white")
    d = ImageDraw.Draw(mont)
    for c, a in enumerate(attacks):
        att = make(a, hero, wmimg, ip); fba, det = decode(att, image_id, tx)
        x = 6 + c * (TH + 6)
        mont.paste(att.resize((TH, TH)), (x, 30))
        d.text((x + 2, 6), a, fill="black", font=F2)
        col = (0, 140, 0) if det else (200, 0, 0)
        d.text((x + 2, TH + 34), f"ba {fba:.2f} {'OK' if det else 'X'}", fill=col, font=F2)
        print(f"  {a:<14} fused_ba={fba:.2f} detect={det}", flush=True)
    mont.save(os.path.join(OUT, "fig_robustness.png"))
    print(f"[fig2] hero=img_{hero:05d} -> {OUT}/fig_robustness.png\nFIGURES_DONE")


if __name__ == "__main__":
    main()
