"""Evaluate a trained T_lat checkpoint on 20 test images.

Reports PSNR + BCH detection + TPR@1%FPR across clean and three SD regen
strengths. Compares to the random-T_lat baseline using a matched master key.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reuse prototype's embed/verify
sys.path.insert(0, str(REPO / "scripts"))
from test_latent_prototype import (   # noqa: E402
    embed_latent, verify_latent, verify_latent_learned, verify_latent_ensemble,
    make_T_lat, regen_attack, load_vae, DEVICE,
)
from src.payload import BCHCodec  # noqa: E402
from src.latent_decoder import LatentDecoder, LatentDecoderXR  # noqa: E402


def tpr_threshold(n_bits: int = 127, fpr: float = 0.01) -> float:
    from scipy.stats import binom
    for k in range(n_bits, n_bits // 2, -1):
        if binom.sf(k - 1, n_bits, 0.5) > fpr:
            return (k + 1) / n_bits
    return 0.5


def evaluate(T_lat, master_key: bytes, image_files, codec, alphas, strengths, thr, label: str,
             pixel_jnd: bool = False, decoder=None, ensemble_weight: float = None):
    """Sweep (alpha, strength); return table of metrics.

    If `decoder` is given and `ensemble_weight` is None: use learned CNN only.
    If `decoder` is given and `ensemble_weight` is a float in [0, 1]:
        use ensemble of matched filter and CNN with this weight on the CNN.
    Otherwise: use matched filter.
    """
    if decoder is not None and ensemble_weight is not None:
        detector_label = f"ensemble (decoder w={ensemble_weight:.2f})"
    elif decoder is not None:
        detector_label = "learned CNN"
    else:
        detector_label = "matched filter"
    print(f"\n========== {label} (pixel_jnd={pixel_jnd}, detector={detector_label}) ==========")
    for alpha in alphas:
        rows = []
        for img_idx, fp in enumerate(image_files):
            pil = Image.open(fp).convert("RGB").resize((256, 256), Image.BILINEAR)
            image_id = f"eval_{img_idx:04d}"

            pil_w, info = embed_latent(pil, master_key, image_id, T_lat, alpha=alpha,
                                       codec=codec, pixel_jnd=pixel_jnd)

            def _verify(susp_pil):
                if decoder is not None and ensemble_weight is not None:
                    return verify_latent_ensemble(
                        pil, susp_pil, master_key, image_id,
                        T_lat, decoder, codec, decoder_weight=ensemble_weight,
                    )
                if decoder is not None:
                    return verify_latent_learned(pil, susp_pil, master_key, image_id,
                                                  T_lat, decoder, codec)
                return verify_latent(pil, susp_pil, master_key, image_id, T_lat, codec)

            res_clean = _verify(pil_w)
            regen = {}
            for s in strengths:
                pil_regen = regen_attack(pil_w, strength=s)
                regen[s] = _verify(pil_regen)

            rows.append({"psnr": info["psnr"], "clean": res_clean, "regen": regen})

        psnr = float(np.mean([r["psnr"] for r in rows]))
        clean_acc = np.mean([r["clean"]["raw_bit_acc"] for r in rows])
        clean_bch = np.mean([float(r["clean"]["detected"]) for r in rows])
        clean_tpr = np.mean([float(r["clean"]["raw_bit_acc"] >= thr) for r in rows])
        msg = (f"  α={alpha:5.2f} PSNR={psnr:5.2f}dB | "
               f"clean: BCH={clean_bch*100:3.0f}% TPR={clean_tpr*100:3.0f}% acc={clean_acc:.3f}")
        for s in strengths:
            acc = np.mean([r["regen"][s]["raw_bit_acc"] for r in rows])
            bch = np.mean([float(r["regen"][s]["detected"]) for r in rows])
            tpr = np.mean([float(r["regen"][s]["raw_bit_acc"] >= thr) for r in rows])
            msg += f" | s={s}: BCH={bch*100:3.0f}% TPR={tpr*100:3.0f}% acc={acc:.3f}"
        print(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="T_lat .npz checkpoint")
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--master_key", default="latent_train_master_key_v1",
                   help="Must match training master_key for trained T_lat eval")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    p.add_argument("--strengths", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    p.add_argument("--also_random_baseline", action="store_true",
                   help="Also eval the random T_lat baseline for comparison")
    p.add_argument("--pixel_jnd", action="store_true",
                   help="Apply pixel-domain JND mask after VAE decode (free PSNR boost)")
    p.add_argument("--decoder_ckpt", default=None,
                   help="Path to a trained LatentDecoder .pt. If set, "
                        "swap matched filter for the learned CNN.")
    p.add_argument("--ensemble_weights", type=float, nargs="+", default=None,
                   help="If set with --decoder_ckpt, evaluate ensemble of "
                        "matched filter + CNN with these CNN-weight values "
                        "(e.g. 0.2 0.5 0.8). Use 0 for matched-filter only, "
                        "1 for CNN only.")
    args = p.parse_args()

    codec = BCHCodec()
    thr = tpr_threshold(codec.n, fpr=0.01)
    master_key = args.master_key.encode("utf-8")

    image_files = sorted((REPO / "images").glob("*.png"))[: args.n_images]
    print(f"[setup] {len(image_files)} test images, TPR threshold={thr:.4f}")

    # Load trained T_lat
    d = np.load(args.ckpt, allow_pickle=True)
    T_lat_trained = d["T_lat"]
    rms = float(np.sqrt(np.mean(T_lat_trained ** 2)))
    print(f"[setup] trained T_lat shape={T_lat_trained.shape} rms={rms:.4f}")
    print(f"[setup] master_key={args.master_key}")

    # Optional learned decoder
    decoder = None
    if args.decoder_ckpt:
        # Build region masks (same as embed/training) to construct the decoder.
        # The training-time version returns an ndarray (n_bits, C, H, W);
        # eval-time uses the prototype list version, so convert.
        from train_T_lat import make_latent_region_masks as _ml_rmasks
        region_masks_np = _ml_rmasks(codec.n, (4, 32, 32))
        region_masks_t = torch.from_numpy(region_masks_np).to(DEVICE)

        ckpt = torch.load(args.decoder_ckpt, map_location=DEVICE, weights_only=False)
        in_ch_residual = int(ckpt.get("in_ch_residual", 4))
        feat_ch = int(ckpt.get("feat_ch", 64))
        n_bits = int(ckpt.get("n_bits", codec.n))
        arch = ckpt.get("arch", "base")
        xr_layers = int(ckpt.get("xr_layers", 2))
        if arch == "xr":
            decoder = LatentDecoderXR(
                in_ch_residual=in_ch_residual, n_bits=n_bits, feat_ch=feat_ch,
                region_masks=region_masks_t, n_layers=xr_layers, n_heads=4,
            ).to(DEVICE)
        else:
            decoder = LatentDecoder(
                in_ch_residual=in_ch_residual, n_bits=n_bits, feat_ch=feat_ch,
                region_masks=region_masks_t,
            ).to(DEVICE)
        decoder.load_state_dict(ckpt["state_dict"])
        decoder.eval()
        n_params = sum(p.numel() for p in decoder.parameters())
        print(f"[setup] learned decoder loaded: arch={arch} {n_params:,} params, "
              f"in_ch_residual={in_ch_residual}, feat_ch={feat_ch}, n_bits={n_bits}")

    if args.ensemble_weights is not None and decoder is not None:
        for w in args.ensemble_weights:
            evaluate(T_lat_trained, master_key, image_files, codec,
                     args.alphas, args.strengths, thr, label="trained T_lat",
                     pixel_jnd=args.pixel_jnd, decoder=decoder, ensemble_weight=w)
    else:
        evaluate(T_lat_trained, master_key, image_files, codec,
                 args.alphas, args.strengths, thr, label="trained T_lat",
                 pixel_jnd=args.pixel_jnd, decoder=decoder)

    if args.also_random_baseline:
        # Random baseline with same master key
        T_lat_rand = make_T_lat(master_key)
        evaluate(T_lat_rand, master_key, image_files, codec,
                 [0.05, 0.08, 0.12], args.strengths, thr,
                 label="random T_lat (baseline)",
                 pixel_jnd=args.pixel_jnd, decoder=decoder)


if __name__ == "__main__":
    main()
