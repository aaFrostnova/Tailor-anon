#!/usr/bin/env python3
"""
Visualize robustness test results.

Usage:
    python scripts/plot_results.py --results_dir ./results/robustness_k8_100img --output_dir ./figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_results(results_dir):
    with open(os.path.join(results_dir, "robustness_results.json")) as f:
        return json.load(f)


def get_fragment_names(result):
    """Extract short fragment names from per_fragment_survival."""
    names = []
    for fs in result["per_fragment_survival"]:
        s = fs["strategy"]
        freq = fs.get("freq_range")
        level = fs.get("level")
        region = fs.get("region", "full")

        if s == "pixel":
            names.append(f"pixel\n({region})")
        elif s == "dct":
            tag = "low" if freq and freq[0] < 5 else ("mid" if freq and freq[0] < 20 else "high")
            names.append(f"DCT\n({tag})")
        elif s == "dwt-dct":
            tag = "low" if freq and freq[0] < 5 else ("mid" if freq and freq[0] < 20 else "high")
            names.append(f"DWT-DCT\nL{level} ({tag})")
        elif s == "quantization":
            bits = fs.get("bits")
            if bits is None:
                # Fallback: read from config
                from src.fragment_config import get_default_configs
                cfgs = get_default_configs(len(result["per_fragment_survival"]))
                bits = cfgs[fs["index"]].get("bits", "?")
            names.append(f"Quant\n(bits={bits})")
        elif s == "warping":
            strength = fs.get("strength", "?")
            names.append(f"Warp\n(s={strength})")
        elif s == "color-filter":
            intensity = fs.get("intensity", "?")
            names.append(f"Color\n(i={intensity})")
        else:
            names.append(s)
    return names


def short_transform_name(name):
    """Shorten transform names for display."""
    replacements = {
        "center_crop_resize_": "crop+rsz ",
        "random_crop_resize_": "rnd_crop+rsz ",
        "center_crop_": "center_crop ",
        "random_crop_": "random_crop ",
        "jpeg_q60+center_crop_80%": "JPEG60+crop80",
        "jpeg_q40+random_crop_resize_60%": "JPEG40+crop_rsz60",
        "jpeg_q": "JPEG Q=",
        "resize_": "resize ",
        "noise_": "noise σ=",
        "blur_": "blur r=",
        "bright_+20%": "bright +20%",
        "bright_-20%": "bright -20%",
        "contrast_+20%": "contrast +20%",
        "contrast_-20%": "contrast -20%",
    }
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
            break
    return name


def split_by_eps(data):
    """Split results by epsilon value."""
    eps_groups = {}
    for r in data:
        eps = r["epsilon_255"]
        if eps not in eps_groups:
            eps_groups[eps] = []
        eps_groups[eps].append(r)
    return eps_groups


# ============================================================
# Plot 1: Heatmap — Fragment survival rate matrix
# ============================================================

def plot_heatmap(data, output_dir, eps_val=None):
    """Heatmap: rows=transforms, cols=fragments, color=survival rate."""
    if eps_val:
        subset = [r for r in data if r["epsilon_255"] == eps_val]
        title_suffix = f" (ε={eps_val:.0f}/255)"
    else:
        subset = data
        title_suffix = ""

    if not subset or "per_fragment_survival" not in subset[0]:
        return

    transforms = [r["transform"] for r in subset]
    K = len(subset[0]["per_fragment_survival"])
    frag_names = get_fragment_names(subset[0])

    matrix = np.zeros((len(transforms), K))
    for i, r in enumerate(subset):
        for j, fs in enumerate(r["per_fragment_survival"]):
            matrix[i, j] = fs["survival_rate"]

    fig, ax = plt.subplots(figsize=(10, max(6, len(transforms) * 0.45)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(K))
    ax.set_xticklabels(frag_names, fontsize=8)
    ax.set_yticks(range(len(transforms)))
    ax.set_yticklabels([short_transform_name(t) for t in transforms], fontsize=9)

    # Add text annotations
    for i in range(len(transforms)):
        for j in range(K):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.0%}", ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Survival Rate", shrink=0.8)
    ax.set_title(f"Per-Fragment Survival Rate Heatmap{title_suffix}", fontsize=12)
    ax.set_xlabel("Fragment")
    ax.set_ylabel("Transform")

    plt.tight_layout()
    tag = f"_eps{eps_val:.0f}" if eps_val else ""
    plt.savefig(os.path.join(output_dir, f"heatmap{tag}.png"), dpi=150)
    plt.close()
    print(f"  Saved heatmap{tag}.png")


# ============================================================
# Plot 2: Bar chart — Detection rate per transform (ε comparison)
# ============================================================

def plot_detection_bar(data, output_dir):
    """Grouped bar chart: detection rate per transform, ε=4 vs ε=8."""
    eps_groups = split_by_eps(data)
    eps_values = sorted(eps_groups.keys())

    if len(eps_values) < 2:
        # Single epsilon
        eps_values = eps_values[:1]

    transforms = [r["transform"] for r in eps_groups[eps_values[0]]]
    short_names = [short_transform_name(t) for t in transforms]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(transforms))
    width = 0.35 if len(eps_values) >= 2 else 0.5
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for idx, eps in enumerate(eps_values):
        rates = [r["detection_rate"] * 100 for r in eps_groups[eps]]
        offset = (idx - (len(eps_values) - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width * 0.9,
                      label=f"ε={eps:.0f}/255", color=colors[idx], alpha=0.85)
        # Add value labels on bars below 100%
        for bar, rate in zip(bars, rates):
            if rate < 100:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel("Detection Rate (%)", fontsize=11)
    ax.set_title("Detection Rate by Transform and Perturbation Budget", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 108)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.3, label='90% threshold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_rate_bar.png"), dpi=150)
    plt.close()
    print("  Saved detection_rate_bar.png")


# ============================================================
# Plot 3: Radar chart — Per-fragment robustness profile
# ============================================================

def plot_radar(data, output_dir, eps_val=None):
    """Radar chart: each fragment's survival across attack types."""
    if eps_val:
        subset = [r for r in data if r["epsilon_255"] == eps_val]
    else:
        subset = data

    if not subset or "per_fragment_survival" not in subset[0]:
        return

    transforms = [r["transform"] for r in subset if r["transform"] != "none"]
    K = len(subset[0]["per_fragment_survival"])
    frag_names = get_fragment_names(subset[0])

    # Build survival matrix (fragments × transforms)
    matrix = np.zeros((K, len(transforms)))
    for j, r in enumerate([r for r in subset if r["transform"] != "none"]):
        for i, fs in enumerate(r["per_fragment_survival"]):
            matrix[i, j] = fs["survival_rate"]

    angles = np.linspace(0, 2 * np.pi, len(transforms), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.Dark2(np.linspace(0, 1, K))
    for i in range(K):
        values = matrix[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, markersize=4,
                label=frag_names[i].replace('\n', ' '), color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([short_transform_name(t) for t in transforms], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    eps_tag = f" (ε={eps_val:.0f}/255)" if eps_val else ""
    ax.set_title(f"Per-Fragment Robustness Profile{eps_tag}", fontsize=13, pad=20)

    plt.tight_layout()
    tag = f"_eps{eps_val:.0f}" if eps_val else ""
    plt.savefig(os.path.join(output_dir, f"radar{tag}.png"), dpi=150)
    plt.close()
    print(f"  Saved radar{tag}.png")


# ============================================================
# Plot 4: Stacked bar — Which fragments contribute to detection
# ============================================================

def plot_stacked_bar(data, output_dir, eps_val=None):
    """Stacked bar: for each transform, show which fragments survived."""
    if eps_val:
        subset = [r for r in data if r["epsilon_255"] == eps_val]
    else:
        subset = data

    if not subset or "per_fragment_survival" not in subset[0]:
        return

    transforms = [r["transform"] for r in subset]
    K = len(subset[0]["per_fragment_survival"])
    frag_names = get_fragment_names(subset[0])

    matrix = np.zeros((len(transforms), K))
    for i, r in enumerate(subset):
        for j, fs in enumerate(r["per_fragment_survival"]):
            matrix[i, j] = fs["survival_rate"]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(transforms))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    bottom = np.zeros(len(transforms))

    for j in range(K):
        ax.bar(x, matrix[:, j], bottom=bottom, label=frag_names[j].replace('\n', ' '),
               color=colors[j], alpha=0.85, width=0.7)
        bottom += matrix[:, j]

    ax.set_ylabel("Cumulative Survival (sum of per-fragment rates)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([short_transform_name(t) for t in transforms],
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.axhline(y=K * 0.5, color='red', linestyle='--', alpha=0.3)

    eps_tag = f" (ε={eps_val:.0f}/255)" if eps_val else ""
    ax.set_title(f"Fragment Contribution to Detection{eps_tag}", fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    tag = f"_eps{eps_val:.0f}" if eps_val else ""
    plt.savefig(os.path.join(output_dir, f"stacked_bar{tag}.png"), dpi=150)
    plt.close()
    print(f"  Saved stacked_bar{tag}.png")


# ============================================================
# Plot 5: Original vs fingerprinted comparison
# ============================================================

def plot_image_comparison(output_dir, image_dir=None):
    """Side-by-side: original, fingerprinted, residual (magnified)."""
    from src.pipeline import FingerprintPipeline, load_image, save_image
    from src.keygen import generate_master_key
    from src.metrics import psnr, ssim

    if image_dir is None:
        image_dir = os.path.join(os.path.dirname(__file__), "..", "images")

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:1]

    if not image_files:
        print("  No images found for comparison plot")
        return

    key = generate_master_key()
    pipeline = FingerprintPipeline(master_key=key, multi_domain=True)

    for fname in image_files:
        image = load_image(os.path.join(image_dir, fname))
        fp, _ = pipeline.fingerprint_image(image, "demo")

        residual = fp - image
        p = psnr(image, fp)
        s = ssim(image, fp)

        # Convert CHW -> HWC for display
        img_hwc = np.clip(image.transpose(1, 2, 0), 0, 1)
        fp_hwc = np.clip(fp.transpose(1, 2, 0), 0, 1)

        # Magnify residual for visibility
        res_hwc = residual.transpose(1, 2, 0)
        res_mag = np.clip(res_hwc * 10 + 0.5, 0, 1)  # 10x magnification

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_hwc)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(fp_hwc)
        axes[1].set_title(f"Fingerprinted\nPSNR={p:.1f}dB, SSIM={s:.4f}", fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(res_mag)
        axes[2].set_title("Residual (10x magnified)", fontsize=12)
        axes[2].axis('off')

        plt.suptitle(f"Fingerprint Imperceptibility — {fname}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "image_comparison.png"), dpi=150)
        plt.close()
        print("  Saved image_comparison.png")


# ============================================================
# Plot 6: Post-PSNR vs Detection Rate scatter
# ============================================================

def plot_quality_vs_detection(data, output_dir):
    """Scatter plot: post-transform PSNR vs detection rate."""
    eps_groups = split_by_eps(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'4.0': '#2196F3', '8.0': '#FF9800'}
    markers = {'4.0': 'o', '8.0': 's'}

    for eps, results in eps_groups.items():
        psnrs = [r["post_psnr_mean"] for r in results]
        det_rates = [r["detection_rate"] * 100 for r in results]
        labels = [r["transform"] for r in results]

        key = f"{eps:.1f}"
        ax.scatter(psnrs, det_rates, c=colors.get(key, 'gray'),
                  marker=markers.get(key, 'o'), s=80, alpha=0.8,
                  label=f"ε={eps:.0f}/255", edgecolors='black', linewidths=0.5)

        # Label points below 100%
        for x, y, label in zip(psnrs, det_rates, labels):
            if y < 100:
                ax.annotate(short_transform_name(label), (x, y),
                          textcoords="offset points", xytext=(5, -10),
                          fontsize=6, alpha=0.8)

    ax.set_xlabel("Post-Transform PSNR (dB)", fontsize=11)
    ax.set_ylabel("Detection Rate (%)", fontsize=11)
    ax.set_title("Image Quality After Attack vs Fingerprint Detection Rate", fontsize=13)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.3, label='90% threshold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(60, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_vs_detection.png"), dpi=150)
    plt.close()
    print("  Saved quality_vs_detection.png")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Plot robustness results")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_dir", default=None,
                        help="Image directory for comparison plot")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.results_dir)
    eps_groups = split_by_eps(data)
    eps_values = sorted(eps_groups.keys())

    print(f"Loaded {len(data)} results, eps values: {eps_values}")
    print(f"Generating figures in {output_dir}/\n")

    # Plot 1: Heatmaps (one per epsilon)
    print("Plot 1: Heatmap")
    for eps in eps_values:
        plot_heatmap(data, str(output_dir), eps_val=eps)

    # Plot 2: Detection rate bar chart
    print("Plot 2: Detection rate bar chart")
    plot_detection_bar(data, str(output_dir))

    # Plot 3: Radar charts (one per epsilon)
    print("Plot 3: Radar chart")
    for eps in eps_values:
        plot_radar(data, str(output_dir), eps_val=eps)

    # Plot 4: Stacked bar (one per epsilon)
    print("Plot 4: Stacked bar")
    for eps in eps_values:
        plot_stacked_bar(data, str(output_dir), eps_val=eps)

    # Plot 5: Image comparison
    print("Plot 5: Image comparison")
    image_dir = args.image_dir or os.path.join(os.path.dirname(__file__), "..", "images")
    plot_image_comparison(str(output_dir), image_dir)

    # Plot 6: Quality vs detection scatter
    print("Plot 6: Quality vs detection scatter")
    plot_quality_vs_detection(data, str(output_dir))

    print(f"\nDone! All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
