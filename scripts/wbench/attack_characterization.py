"""Component B: characterize how each attack damages the watermark signal.

For each attack and each bit-method, records fine-grained damage that aggregate
bit-accuracy hides:
  - per-logical-bit survival (which of the 100 codeword bits flip);
  - per-carrier survival for the location-aware methods, binned by DFT radial
    frequency (DFT-Kred) and by spatial block (Quant-QIM);
  - residual-spectrum energy of (attacked - clean) green channel by radius;
  - per-method soft-margin (|LLR|) distribution, clean vs attacked.

Outputs results/attack_char/{attack}.json plus heatmap arrays in {attack}.npz,
and derives:
  - weights.json   : per-(method, attack) inverse-variance-style fusion weights
                     (oracle, attack-known) plus a pooled attack-agnostic set;
  - erasure_calib.json : clean per-method |LLR| stats to set erasure thresholds.

These feed Component A (weights) and Component C (erasure map / tamper-loc).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.fused_detector import FusedDetector
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import apply_crypto
from src.soft_fusion import llr_to_bits
from benchmark_fused import apply_attack, psnr  # reuse attack suite


def carrier_eff_decode(mag, delta):
    """Per-carrier effective-bit decode via the cos-QIM feature (0 on bit-0 lattice)."""
    cosv = np.cos(2.0 * np.pi * mag / delta)
    return (cosv < 0).astype(np.uint8)


def embedded_eff(method, image_id, tx):
    """The effective bits actually written per carrier-bit: secret XOR bit_flip."""
    perm, M = method.get_perm_M(image_id)
    secret = apply_crypto(np.asarray(tx, np.uint8), perm, M).astype(np.uint8)  # (n_bits,)
    return (secret ^ method.bit_flip.astype(np.uint8))  # (n_bits,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=30)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "jpeg_75", "jpeg_50", "blur_1.5", "blur_2.5", "noise_50",
        "crop_90", "resize_110", "regen_10_sd15", "regen_20_sd15",
    ])
    p.add_argument("--out_dir", default="results/attack_char")
    args = p.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = FusedDetector(device="cuda")
    dft, qim = fd.methods["dft_kred"], fd.methods["quant_qim"]
    radii = dft.carrier_radii().reshape(-1)               # (n_bits*KM,)
    rad_bins = np.linspace(radii.min(), radii.max() + 1e-6, 9)  # 8 bands
    rad_idx = np.clip(np.digitize(radii, rad_bins) - 1, 0, len(rad_bins) - 2)
    block_coords = qim.carrier_block_coords().reshape(-1, 2)    # (n_bits*npos, 2)

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[char] {len(files)} imgs, {len(args.attacks)} attacks", flush=True)

    # Pre-embed all images once (clean wm + tx + green channel cache).
    embedded = []
    for i, fp in enumerate(files):
        image_id = f"char_{i:05d}"
        tx = fd.codeword(image_id)
        wm = fd.embed(Image.open(fp).convert("RGB"), image_id)
        green = np.asarray(wm, np.float32)[:, :, 1] / 255.0
        embedded.append((image_id, tx, wm, green))
    print("[char] embedding done", flush=True)

    results = {}
    for atk in args.attacks:
        bit_surv = {m: [] for m in fd.methods}        # per-logical-bit survival
        abs_llr = {m: [] for m in fd.methods}         # per-bit |LLR|
        dft_rad_surv = np.zeros((len(rad_bins) - 1, 2))   # [sum, count]
        qim_block = np.zeros((qim.n_by, qim.n_bx, 2))     # [sum, count]
        res_energy = []                                # radial residual profile
        for image_id, tx, wm, green0 in embedded:
            att = apply_attack(atk, wm)
            aligned, _ = fd.aligned_llrs(att, image_id)
            for m in fd.methods:
                hard = llr_to_bits(aligned[m])
                bit_surv[m].append((hard == tx).astype(np.float32))
                abs_llr[m].append(np.abs(aligned[m]))
            # DFT per-carrier survival by radius
            mag = dft.carrier_magnitudes(att).reshape(-1)
            eff_hat = carrier_eff_decode(mag, dft.decode_delta())
            eff_emb = np.repeat(embedded_eff(dft, image_id, tx), dft.K * dft.M)
            surv = (eff_hat == eff_emb).astype(np.float32)
            for b in range(len(rad_bins) - 1):
                sel = rad_idx == b
                dft_rad_surv[b, 0] += surv[sel].sum(); dft_rad_surv[b, 1] += sel.sum()
            # QIM per-carrier survival by block
            mu = qim.carrier_means(att).reshape(-1)
            eff_hat_q = carrier_eff_decode(mu, qim.decode_delta())
            eff_emb_q = np.repeat(embedded_eff(qim, image_id, tx), qim.n_pos)
            surv_q = (eff_hat_q == eff_emb_q).astype(np.float32)
            for (by, bx), s in zip(block_coords, surv_q):
                qim_block[by, bx, 0] += s; qim_block[by, bx, 1] += 1
            # residual spectrum (green channel) by radius
            ga = np.asarray(att.convert("RGB").resize(wm.size), np.float32)[:, :, 1] / 255.0
            resid = ga - green0
            R = np.abs(np.fft.fftshift(np.fft.fft2(resid)))
            H, W = R.shape; cy, cx = H // 2, W // 2
            yy, xx = np.ogrid[:H, :W]
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
            prof = np.bincount(rr.ravel(), R.ravel()) / (np.bincount(rr.ravel()) + 1e-9)
            res_energy.append(prof[:min(64, len(prof))])

        bit_surv_mean = {m: np.mean(bit_surv[m], axis=0).tolist() for m in fd.methods}
        mean_abs_llr = {m: float(np.mean(abs_llr[m])) for m in fd.methods}
        var_llr = {m: float(np.var(np.concatenate(abs_llr[m]))) for m in fd.methods}
        n_collapsed = {m: float(np.mean([(np.abs(x) < 0.5).sum() for x in abs_llr[m]]))
                       for m in fd.methods}
        dft_rs = (dft_rad_surv[:, 0] / (dft_rad_surv[:, 1] + 1e-9)).tolist()
        qim_map = (qim_block[:, :, 0] / (qim_block[:, :, 1] + 1e-9))
        minlen = min(len(e) for e in res_energy)
        res_prof = np.mean([e[:minlen] for e in res_energy], axis=0)

        results[atk] = {
            "logical_bit_survival": bit_surv_mean,
            "mean_abs_llr": mean_abs_llr,
            "var_abs_llr": var_llr,
            "n_collapsed": n_collapsed,
            "dft_radial_survival": dft_rs,
            "residual_energy_by_radius": res_prof.tolist(),
        }
        np.savez(out_dir / f"{atk}.npz", qim_block_survival=qim_map,
                 dft_radial_survival=np.array(dft_rs),
                 residual_energy=res_prof)
        with open(out_dir / f"{atk}.json", "w") as f:
            json.dump(results[atk], f, indent=2)
        ov = {m: round(float(np.mean(bit_surv_mean[m])), 3) for m in fd.methods}
        print(f"  [{atk}] bit-survival {ov}  |LLR| "
              f"{ {m: round(mean_abs_llr[m],2) for m in fd.methods} }", flush=True)

    # ----- derive fusion weights (inverse-variance-style reliability) -----
    # Reliability r_m = mean correct-direction LLR = mean over bits of llr*(2*tx-1).
    # We approximate per-attack using mean_abs_llr / sqrt(var) as a robust proxy.
    weights = {"oracle": {}, "pooled": {}}
    for atk in args.attacks:
        r = results[atk]
        w = {m: r["mean_abs_llr"][m] / (np.sqrt(r["var_abs_llr"][m]) + 1e-6)
             for m in fd.methods}
        s = sum(w.values()) + 1e-9
        weights["oracle"][atk] = {m: w[m] / s for m in fd.methods}
    pooled = {m: float(np.mean([weights["oracle"][a][m] for a in args.attacks]))
              for m in fd.methods}
    ps = sum(pooled.values()) + 1e-9
    weights["pooled"] = {m: pooled[m] / ps for m in fd.methods}
    with open(out_dir / "weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    # ----- erasure calibration from clean margins -----
    if "clean" in results:
        calib = {m: {"mean_abs_llr_clean": results["clean"]["mean_abs_llr"][m]}
                 for m in fd.methods}
        with open(out_dir / "erasure_calib.json", "w") as f:
            json.dump(calib, f, indent=2)

    print(f"[done] -> {out_dir} (weights.json pooled={weights['pooled']})", flush=True)


if __name__ == "__main__":
    main()
