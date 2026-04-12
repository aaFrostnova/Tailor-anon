#!/usr/bin/env python3
"""
Test 1: Model-agnostic robustness test (multi-domain, parallelized).

Fingerprint images with multi-domain fragments → apply transformations →
check which fragments survive. Reports per-fragment and aggregate detection.

Usage:
    python scripts/test_robustness.py --input_dir ./images --output_dir ./results/robustness
    python scripts/test_robustness.py --input_dir ./images --output_dir ./results/robustness --num_fragments 8 --workers 8
    python scripts/test_robustness.py --input_dir ./images --output_dir ./results/robustness --single_domain --strategies pixel dct dwt-dct
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.keygen import generate_master_key, derive_all_subkeys, image_id_from_path
from src.fragment import generate_fragment
from src.embed import embed_multi_domain, extract_multi_domain, embed, extract
from src.verify import verify_multi_domain, verify_fingerprint, fisher_combine_pvalues, pearson_correlation
from src.fragment_config import get_default_configs
from src.metrics import psnr, ssim
from src.pipeline import load_image, load_config
from src.transforms import TRANSFORM_SUITE

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Model-agnostic robustness test")
    parser.add_argument("--input_dir", required=True, help="Directory of input images")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--num_fragments", type=int, nargs="+", default=[8])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[4/255, 8/255])
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--transforms", nargs="+", default=None,
                        help="Specific transforms to test (default: all)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    # Legacy single-domain mode
    parser.add_argument("--single_domain", action="store_true")
    parser.add_argument("--strategies", nargs="+", default=["pixel", "dct", "dwt-dct"])
    return parser.parse_args()


# --- Alignment for crop (size mismatch) ---

def align_cropped(image_full, cropped_fp):
    """Find where cropped_fp comes from in image_full using phase correlation."""
    _, th, tw = cropped_fp.shape
    _, oh, ow = image_full.shape

    if th >= oh and tw >= ow:
        return 0, 0  # no crop

    # Phase correlation
    padded = np.zeros_like(image_full)
    padded[:, :th, :tw] = cropped_fp
    best_pos = (0, 0)
    best_val = -1
    for c in range(image_full.shape[0]):
        f_orig = np.fft.fft2(image_full[c])
        f_pad = np.fft.fft2(padded[c])
        cross = f_orig * np.conj(f_pad)
        cross /= np.maximum(np.abs(cross), 1e-10)
        corr_map = np.fft.ifft2(cross).real
        valid = corr_map[:oh - th + 1, :ow - tw + 1]
        peak = np.unravel_index(np.argmax(valid), valid.shape)
        if valid[peak] > best_val:
            best_val = valid[peak]
            best_pos = peak

    # Fine refinement ±3 pixels
    ct, cl = best_pos
    best_mse = float("inf")
    for dt in range(max(0, ct - 3), min(oh - th + 1, ct + 4)):
        for dl in range(max(0, cl - 3), min(ow - tw + 1, cl + 4)):
            mse = np.mean((cropped_fp - image_full[:, dt:dt + th, dl:dl + tw]) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_pos = (dt, dl)

    return best_pos


# --- Log-polar alignment for crop+resize ---

def estimate_scale_logpolar(image_full, suspect):
    """Estimate scale factor between two same-size images using log-polar DFT.

    When an image is cropped and resized back, the content is scaled.
    In log-polar coordinates, scaling becomes translation, detectable
    via phase correlation.

    Returns:
        Estimated scale factor (>1 means suspect is zoomed in / cropped+resized).
    """
    from scipy.ndimage import map_coordinates

    C, H, W = image_full.shape
    # Use grayscale average for alignment
    gray_orig = np.mean(image_full, axis=0)
    gray_susp = np.mean(suspect, axis=0)

    # Step 1: DFT magnitude (translation-invariant)
    mag_orig = np.abs(np.fft.fftshift(np.fft.fft2(gray_orig)))
    mag_susp = np.abs(np.fft.fftshift(np.fft.fft2(gray_susp)))

    # Apply high-pass filter to suppress DC
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    hp = 1.0 - np.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * (min(H, W) * 0.05)**2))
    mag_orig = mag_orig * hp
    mag_susp = mag_susp * hp

    # Step 2: Convert to log-polar coordinates
    max_r = min(H, W) // 2
    num_angles = 360
    num_radii = 200
    log_base = np.exp(np.log(max_r) / num_radii)

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    radii = log_base ** np.arange(num_radii)

    # Build log-polar sample grid
    angle_grid, radius_grid = np.meshgrid(angles, radii)
    y_coords = cy + radius_grid * np.sin(angle_grid)
    x_coords = cx + radius_grid * np.cos(angle_grid)

    lp_orig = map_coordinates(mag_orig, [y_coords, x_coords], order=1, mode='constant')
    lp_susp = map_coordinates(mag_susp, [y_coords, x_coords], order=1, mode='constant')

    # Step 3: Phase correlation on log-polar images
    f1 = np.fft.fft2(lp_orig)
    f2 = np.fft.fft2(lp_susp)
    cross = f1 * np.conj(f2)
    cross /= np.maximum(np.abs(cross), 1e-10)
    corr = np.fft.ifft2(cross).real

    # Peak in radii dimension = log(scale)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift_r = peak[0]
    if shift_r > num_radii // 2:
        shift_r -= num_radii

    scale = log_base ** shift_r
    return scale


def align_crop_resize(image_full, suspect, fragments, configs, weights, K, alpha, master_key, image_id, epsilon):
    """Try to align a crop+resized image using log-polar scale estimation.

    1. Estimate scale via log-polar
    2. Resize suspect to estimated crop size
    3. Use phase correlation to find offset
    4. Extract and verify on the aligned region
    """
    from PIL import Image as PILImage
    from src.transforms import _to_pil, _from_pil

    _, H, W = image_full.shape

    # Step 1: Estimate scale
    scale = estimate_scale_logpolar(image_full, suspect)

    # Only try if scale is reasonable (crop ratios 0.5-0.99)
    if scale < 1.01 or scale > 2.5:
        return None  # Not a crop+resize, or can't estimate

    # Step 2: Resize suspect back to estimated crop size
    crop_h = int(H / scale)
    crop_w = int(W / scale)
    if crop_h < 16 or crop_w < 16:
        return None

    suspect_pil = _to_pil(suspect)
    suspect_resized = _from_pil(suspect_pil.resize((crop_w, crop_h), PILImage.BILINEAR))

    # Step 3: Phase correlation to find offset
    top, left = align_cropped(image_full, suspect_resized)
    _, th, tw = suspect_resized.shape
    orig_crop = image_full[:, top:top + th, left:left + tw]

    # Step 4: Extract and verify
    from src.embed import extract_multi_domain
    residuals = extract_multi_domain(orig_crop, suspect_resized, configs)
    corrs_k, pvals_k = [], []
    for k in range(K):
        frag_crop = fragments[k][:, top:top + th, left:left + tw] * weights[k]
        # Resize fragment to match
        if frag_crop.shape != residuals[k].shape:
            # For strategies returning different shapes, use pixel correlation
            r_val, p_val = pearson_correlation(
                residuals[k].flatten()[:1000],
                frag_crop.flatten()[:1000]
            )
        else:
            r_val, p_val = pearson_correlation(residuals[k], frag_crop)
        corrs_k.append(r_val)
        pvals_k.append(p_val)

    chi2, combined_p = fisher_combine_pvalues(pvals_k)
    return {
        "detected": combined_p < alpha,
        "combined_p_value": combined_p,
        "mean_correlation": float(np.mean(corrs_k)),
        "fragment_survived": [p < alpha for p in pvals_k],
        "estimated_scale": scale,
        "post_psnr": psnr(orig_crop, suspect_resized),
        "post_ssim": ssim(orig_crop, suspect_resized),
    }


# --- Worker function for parallel execution ---

def run_one_transform(args_tuple):
    """Worker: run one (K, eps, transform) combination across all images.

    Receives pre-computed fingerprinted images and fragments to avoid
    redundant computation.
    """
    (t_name, t_fn_name, K, eps, alpha, master_key_hex, configs,
     images_data, image_ids, fingerprinted_data, fragments_data, weights) = args_tuple

    master_key = bytes.fromhex(master_key_hex)

    # Reconstruct transform function from name
    from src.transforms import TRANSFORM_SUITE as _TS
    if t_name == "none":
        t_fn = lambda x: x
    else:
        t_fn = _TS[t_name]

    detected_count = 0
    correlations = []
    fp_psnrs, fp_ssims, post_psnrs, post_ssims = [], [], [], []
    frag_survive_counts = [0] * K
    n = len(images_data)

    for idx in range(n):
        try:
            image = images_data[idx]
            image_id = image_ids[idx]
            fingerprinted = fingerprinted_data[idx]
            fragments = fragments_data[idx]

            # Apply transform
            if t_name == "none":
                transformed_fp = fingerprinted
            else:
                transformed_fp = t_fn(fingerprinted)

            # Handle size mismatch (crop without resize)
            if transformed_fp.shape != image.shape:
                top, left = align_cropped(image, transformed_fp)
                _, th, tw = transformed_fp.shape
                orig_crop = image[:, top:top + th, left:left + tw]

                residuals = extract_multi_domain(orig_crop, transformed_fp, configs)
                corrs_k, pvals_k = [], []
                for k in range(K):
                    frag_crop = fragments[k][:, top:top + th, left:left + tw] * weights[k]
                    r_val, p_val = pearson_correlation(residuals[k], frag_crop)
                    corrs_k.append(r_val)
                    pvals_k.append(p_val)

                chi2, combined_p = fisher_combine_pvalues(pvals_k)
                detected = combined_p < alpha
                mean_corr = float(np.mean(corrs_k))
                fragment_survived = [p < alpha for p in pvals_k]

                post_p = psnr(orig_crop, transformed_fp)
                post_s = ssim(orig_crop, transformed_fp)
            else:
                result = verify_multi_domain(
                    image, transformed_fp, master_key, image_id,
                    configs=configs, epsilon=eps, weights=weights, alpha=alpha,
                )
                detected = result["detected"]
                mean_corr = result["mean_correlation"]
                per_frag = result["per_fragment"]
                fragment_survived = [per_frag["p_values"][k] < alpha for k in range(K)]

                post_p = psnr(image, transformed_fp)
                post_s = ssim(image, transformed_fp)

                # If normal verification fails, try log-polar alignment
                # (handles crop+resize where size is the same but content is scaled)
                if not detected:
                    try:
                        lp_result = align_crop_resize(
                            image, transformed_fp, fragments, configs,
                            weights, K, alpha, master_key, image_id, eps,
                        )
                        if lp_result and lp_result["detected"]:
                            detected = True
                            mean_corr = lp_result["mean_correlation"]
                            fragment_survived = lp_result["fragment_survived"]
                            post_p = lp_result["post_psnr"]
                            post_s = lp_result["post_ssim"]
                    except Exception:
                        pass  # log-polar failed, keep original result

            if detected:
                detected_count += 1
            correlations.append(mean_corr)
            fp_psnrs.append(psnr(image, fingerprinted))
            fp_ssims.append(ssim(image, fingerprinted))
            post_psnrs.append(post_p)
            post_ssims.append(post_s)
            for k, surv in enumerate(fragment_survived):
                if surv:
                    frag_survive_counts[k] += 1

        except Exception as e:
            print(f"  ERROR: K={K}/{t_name}/img{idx}: {e}")

    det_rate = detected_count / n if n > 0 else 0
    frag_survive_rates = [c / n for c in frag_survive_counts] if n > 0 else [0] * K

    frag_summary = []
    for k in range(K):
        c = configs[k]
        frag_summary.append({
            "index": k,
            "strategy": c["strategy"],
            "region": c.get("region", "full"),
            "freq_range": c.get("freq_range"),
            "level": c.get("level"),
            "survival_rate": frag_survive_rates[k],
        })

    return {
        "mode": "multi-domain",
        "num_fragments": K,
        "epsilon": eps, "epsilon_255": eps * 255,
        "transform": t_name,
        "num_images": n, "detected": detected_count,
        "detection_rate": det_rate,
        "mean_correlation": float(np.mean(correlations)) if correlations else 0,
        "fp_psnr_mean": float(np.mean(fp_psnrs)) if fp_psnrs else 0,
        "fp_ssim_mean": float(np.mean(fp_ssims)) if fp_ssims else 0,
        "post_psnr_mean": float(np.mean(post_psnrs)) if post_psnrs else 0,
        "post_ssim_mean": float(np.mean(post_ssims)) if post_ssims else 0,
        "per_fragment_survival": frag_summary,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.workers or min(cpu_count(), 16)

    # Load images
    input_dir = Path(args.input_dir)
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])[:args.max_images]

    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    images = []
    image_ids = []
    for f in image_files:
        images.append(load_image(str(f)))
        image_ids.append(image_id_from_path(str(f)))

    print(f"Loaded {len(images)} images")
    print(f"Workers: {num_workers}")

    # Select transforms
    transforms = {"none": lambda x: x}
    if args.transforms:
        for t in args.transforms:
            if t in TRANSFORM_SUITE:
                transforms[t] = TRANSFORM_SUITE[t]
    else:
        transforms.update(TRANSFORM_SUITE)

    master_key = generate_master_key(32)
    master_key_hex = master_key.hex()

    all_results = []
    start_time = time.time()

    if args.single_domain:
        # Single-domain mode (sequential, legacy)
        print(f"Mode: single-domain (sequential)")
        from src.fragment import generate_all_fragments
        for strategy in args.strategies:
            embed_kwargs = {}
            if strategy == "dct":
                embed_kwargs = {"block_size": 8, "freq_range": (10, 40)}
            elif strategy == "dwt-dct":
                embed_kwargs = {"wavelet": "haar", "level": 2, "block_size": 8, "freq_range": (10, 40)}

            for K in args.num_fragments:
                for eps in args.epsilons:
                    for t_name in transforms:
                        t_fn = transforms[t_name]
                        detected_count = 0
                        correlations, fp_psnrs, fp_ssims, post_psnrs, post_ssims = [], [], [], [], []

                        for img, img_id in zip(images, image_ids):
                            try:
                                subkeys = derive_all_subkeys(master_key, img_id, K)
                                _, aggregated = generate_all_fragments(subkeys, img.shape, eps)
                                fingerprinted = embed(img, aggregated, strategy=strategy, **embed_kwargs)

                                transformed_fp = fingerprinted if t_name == "none" else t_fn(fingerprinted)
                                residual = extract(img, transformed_fp, strategy=strategy, **embed_kwargs)
                                result = verify_fingerprint(residual, master_key, img_id,
                                                          num_fragments=K, epsilon=eps, alpha=args.alpha)
                                if result["detected"]:
                                    detected_count += 1
                                correlations.append(result["mean_correlation"])
                                fp_psnrs.append(psnr(img, fingerprinted))
                                fp_ssims.append(ssim(img, fingerprinted))
                                post_psnrs.append(psnr(img, transformed_fp))
                                post_ssims.append(ssim(img, transformed_fp))
                            except Exception as e:
                                print(f"  ERROR: {strategy}/{K}/{t_name}: {e}")

                        n = len(images)
                        det_rate = detected_count / n if n > 0 else 0
                        entry = {
                            "mode": "single-domain", "strategy": strategy,
                            "num_fragments": K, "epsilon": eps, "epsilon_255": eps * 255,
                            "transform": t_name, "num_images": n, "detected": detected_count,
                            "detection_rate": det_rate,
                            "mean_correlation": float(np.mean(correlations)) if correlations else 0,
                            "fp_psnr_mean": float(np.mean(fp_psnrs)) if fp_psnrs else 0,
                            "fp_ssim_mean": float(np.mean(fp_ssims)) if fp_ssims else 0,
                            "post_psnr_mean": float(np.mean(post_psnrs)) if post_psnrs else 0,
                            "post_ssim_mean": float(np.mean(post_ssims)) if post_ssims else 0,
                        }
                        all_results.append(entry)
                        status = "PASS" if det_rate >= 0.9 else ("WEAK" if det_rate > 0 else "FAIL")
                        print(f"[{status}] {strategy:8s} K={K} eps={eps*255:.0f}/255 | "
                              f"{t_name:30s} | det={detected_count}/{n} ({det_rate*100:.0f}%)")
    else:
        # Multi-domain mode (parallel by transform)
        print(f"Mode: multi-domain (parallel)")
        for K in args.num_fragments:
            configs = get_default_configs(K)
            weights = [1.0 / K] * K
            strat_counts = {}
            for c in configs:
                s = c["strategy"]
                strat_counts[s] = strat_counts.get(s, 0) + 1
            layout_str = ", ".join(f"{v}×{k}" for k, v in strat_counts.items())
            print(f"\nK={K}: {layout_str}")

            for eps in args.epsilons:
                # Pre-compute fingerprinted images + fragments (once per eps)
                print(f"  Embedding eps={eps*255:.0f}/255 ...", end=" ", flush=True)
                fingerprinted_list = []
                fragments_list = []
                for img, img_id in zip(images, image_ids):
                    subkeys = derive_all_subkeys(master_key, img_id, K)
                    frags = [generate_fragment(subkeys[k], img.shape, eps, fragment_index=k)
                             for k in range(K)]
                    fp = embed_multi_domain(img, frags, configs, weights)
                    fingerprinted_list.append(fp)
                    fragments_list.append(frags)
                print("done")

                # Build task list: one task per transform
                tasks = []
                for t_name in transforms:
                    tasks.append((
                        t_name, t_name, K, eps, args.alpha, master_key_hex, configs,
                        images, image_ids, fingerprinted_list, fragments_list, weights,
                    ))

                # Run in parallel
                print(f"  Running {len(tasks)} transforms × {len(images)} images "
                      f"with {num_workers} workers ...")
                with Pool(num_workers) as pool:
                    results = pool.map(run_one_transform, tasks)

                for entry in results:
                    all_results.append(entry)
                    det_rate = entry["detection_rate"]
                    status = "PASS" if det_rate >= 0.9 else ("WEAK" if det_rate > 0 else "FAIL")
                    frag_survive_rates = [fs["survival_rate"] for fs in entry["per_fragment_survival"]]
                    surv_str = "".join("●" if r >= 0.5 else "○" for r in frag_survive_rates)
                    print(f"  [{status}] K={K} eps={eps*255:.0f}/255 | "
                          f"{entry['transform']:30s} | "
                          f"det={entry['detected']}/{entry['num_images']} "
                          f"({det_rate*100:.0f}%) | "
                          f"mean_r={entry['mean_correlation']:.4f} | "
                          f"frags [{surv_str}]")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save full results
    results_path = output_dir / "robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print per-fragment survival table for multi-domain
    if not args.single_domain:
        print(f"\n{'='*100}")
        print(f"PER-FRAGMENT SURVIVAL RATES")
        print(f"{'='*100}")

        for K in args.num_fragments:
            configs = get_default_configs(K)
            print(f"\nK={K} fragment layout:")
            for k, c in enumerate(configs):
                region = c.get("region", "full")
                freq = c.get("freq_range", "-")
                level = c.get("level", "-")
                print(f"  [{k}] {c['strategy']:8s} region={region:8s} freq={str(freq):12s} level={str(level)}")

            print(f"\n{'Transform':32s} ", end="")
            for k in range(K):
                print(f" F{k}", end="")
            print(f"  {'Agg.Det':>8s}")
            print("-" * (38 + K * 4 + 10))

            k_results = [r for r in all_results
                         if r.get("mode") == "multi-domain" and r["num_fragments"] == K]
            for r in k_results:
                print(f"{r['transform']:32s} ", end="")
                for fs in r["per_fragment_survival"]:
                    rate = fs["survival_rate"]
                    symbol = " ✓" if rate >= 0.9 else (" ~" if rate >= 0.5 else " ✗")
                    print(f"{symbol:>3s}", end="")
                print(f"  {r['detection_rate']*100:>7.0f}%")

    # Save summary
    summary_path = output_dir / "robustness_summary.txt"
    with open(summary_path, "w") as f:
        for r in all_results:
            mode = r.get("mode", "unknown")
            strategy = r.get("strategy", "multi")
            det_str = f"{r['detection_rate']*100:.0f}%"
            f.write(f"{mode:12s} {strategy:8s} K={r['num_fragments']} eps={r['epsilon_255']:.1f}/255 "
                    f"{r['transform']:32s} det={det_str:>5s} "
                    f"mean_r={r['mean_correlation']:.4f} "
                    f"FP_PSNR={r['fp_psnr_mean']:.1f}dB\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
