"""High-fidelity FLUX regeneration attack vs the post-hoc fused tier (VINE+DFT+QIM).

Tests the previously-missing 16-channel / flow-matching regenerator: FLUX VAE
round-trip (autoencoder bottleneck) and FLUX img2img (flow-matching re-synthesis).
Reports content PSNR (fidelity) and the fused-tier / VINE detection under each,
alongside SD-1.5 regen as the 4-channel baseline.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.fused_detector import FusedDetector
from src.soft_fusion import fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from benchmark_fused import apply_attack, psnr, adaptive_weights

DEVICE = "cuda"
IMG_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "flux_vae", "flux_i2i_20", "flux_i2i_30", "regen_10_sd15", "regen_30_sd15"])
    p.add_argument("--output", default="results/fused/flux_regen.json")
    args = p.parse_args()

    fd = FusedDetector(device=DEVICE, use_dft=True, work_res=args.resolution)
    files = sorted([f for f in Path(IMG_DIR).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[4000:4000 + args.n_images]
    agg = {a: {"psnr": [], "fused": [], "vine": []} for a in args.attacks}

    for i, fp in enumerate(files):
        image_id = f"flux_{i:05d}"
        pil = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        wm = fd.embed(pil, image_id)
        for atk in args.attacks:
            att = apply_attack(atk, wm)
            agg[atk]["psnr"].append(psnr(wm, att))
            aligned, _ = fd.aligned_llrs(att, image_id)
            fused = fuse_llrs(aligned, weights=adaptive_weights(aligned), n_codeword=fd.sb.n)
            agg[atk]["fused"].append(float(decode_and_verify(fused, image_id, codec=fd.sb, p=8)["detected"]))
            # VINE-alone detection (its own shortened-BCH decode)
            hard = llr_to_bits(aligned["vine"]); data, ne = fd.sb.decode(hard)
            exp = image_id_to_payload(image_id, fd.sb.data_bits)
            agg[atk]["vine"].append(float(data is not None and np.array_equal(data, exp)))
        print(f"  [{i+1}/{len(files)}]", flush=True)

    summary = {"n_images": len(files), "resolution": args.resolution, "attacks": {}}
    for atk in args.attacks:
        d = agg[atk]
        summary["attacks"][atk] = {"psnr": float(np.mean(d["psnr"])),
                                   "fused_detect": float(np.mean(d["fused"])),
                                   "vine_detect": float(np.mean(d["vine"]))}
    Path(REPO / args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(REPO / args.output, "w"), indent=2)
    print(f"\n{'attack':<16}{'PSNR(dB)':>10}{'fused':>8}{'vine':>7}")
    print("-" * 44)
    for atk in args.attacks:
        s = summary["attacks"][atk]
        print(f"{atk:<16}{s['psnr']:>10.1f}{s['fused_detect']:>8.2f}{s['vine_detect']:>7.2f}", flush=True)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
