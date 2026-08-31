"""Layer-by-layer embedding visualization for the report:
  Top row    : cumulative image after each fragment (clean -> +VINE -> +DFT -> +QIM -> +TrustMark),
               labelled with cumulative PSNR/SSIM vs clean.
  Bottom row : the MARGINAL residual each fragment adds (stage_i - stage_{i-1}) x AMP, labelled with
               the marginal PSNR (this fragment's own cost) -> shows where each watermark puts energy.
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.payload import image_id_to_payload
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
OUT = os.path.join(REPO, "results/figures"); os.makedirs(OUT, exist_ok=True)
AMP = 15


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
    ap.add_argument("--heroes", nargs="+", type=int, default=[5])
    args = ap.parse_args(); dev = "cuda"; sb = ShortenedBCH()
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
    dft = DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev)
    qim = QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev)
    tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
    from pytorch_msssim import ssim as SS
    def t(p): return torch.from_numpy(np.asarray(p.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    F1, F2 = _font(20), _font(16)
    coco = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    stages = ["clean", "+VINE", "+DFT-Kred", "+Quant-QIM", "+TrustMark-B"]

    TH = 300; nrow = len(args.heroes)
    sheet = Image.new("RGB", (TH * 5 + 12, (2 * TH + 56) * nrow + 6), "white")
    d = ImageDraw.Draw(sheet)
    for hi, hero in enumerate(args.heroes):
        iid = f"q_{hero:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        clean = Image.open(coco[4000 + hero]).convert("RGB").resize((512, 512))
        # cumulative images
        imgs = [clean]
        cur = clean
        pv, Mv = vine.get_perm_M(iid); cur = vine.embed_with_target(cur, apply_crypto(tx, pv, Mv)).resize((512, 512)); imgs.append(cur)
        cur = dft.embed(cur, iid, tx).resize((512, 512)); imgs.append(cur)
        cur = qim.embed(cur, iid, tx).resize((512, 512)); imgs.append(cur)
        pt, Mt = tm.get_perm_M(iid); cur = tm.embed_with_target(cur, apply_crypto(tx, pt, Mt)).resize((512, 512)); imgs.append(cur)

        y0 = hi * (2 * TH + 56)
        for c in range(5):
            x = 2 + c * (TH + 2)
            # top: cumulative image + cumulative PSNR/SSIM vs clean
            sheet.paste(imgs[c].resize((TH, TH)), (x, y0 + 24))
            if c == 0:
                d.text((x + 2, y0 + 2), "clean (original)", fill="black", font=F2)
            else:
                with torch.no_grad(): ss = float(SS(t(clean), t(imgs[c]), data_range=1.0))
                d.text((x + 2, y0 + 2), f"{stages[c]}  {psnr(clean, imgs[c]):.1f}dB ssim{ss:.3f}", fill="black", font=F2)
            # bottom: marginal residual (this fragment) x AMP + marginal PSNR
            yb = y0 + 24 + TH + 8
            if c == 0:
                blank = Image.new("RGB", (TH, TH), (245, 245, 245)); sheet.paste(blank, (x, yb))
                d.text((x + 6, yb + TH // 2), "(no watermark)", fill="gray", font=F2)
            else:
                res = (np.asarray(imgs[c], np.int32) - np.asarray(imgs[c - 1], np.int32)) * AMP + 128
                resim = Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)).resize((TH, TH))
                sheet.paste(resim, (x, yb))
                mp = psnr(imgs[c], imgs[c - 1])
                d.text((x + 2, yb + TH + 2), f"{stages[c][1:]} Δ  ({mp:.1f}dB)", fill="black", font=F2)
        print(f"  hero img{hero:05d} done", flush=True)

    out = os.path.join(OUT, "fig_layered.png"); sheet.save(out)
    print(f"[layered] x{AMP} residual -> {out}\nLAYERED_DONE")


if __name__ == "__main__":
    main()
