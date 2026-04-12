#!/usr/bin/env python3
"""
Test 2: Memorization test — train DDPM on fingerprinted data, check if
generated images carry the fingerprint signal.

Usage:
    python scripts/test_memorization.py --output_dir ./results/memorization
    python scripts/test_memorization.py --output_dir ./results/memorization --epochs 50 --fp_ratios 1.0 0.5 0.1
    python scripts/test_memorization.py --output_dir ./results/memorization --strategy dwt-dct --num_fragments 8

Trains DDPM on CIFAR-10 with various fingerprinting ratios and measures
fingerprint signal in generated images.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.keygen import generate_master_key, derive_all_subkeys
from src.fragment import generate_all_fragments
from src.embed import embed
from src.verify import verify_fingerprint, pearson_correlation
from src.ddpm import SimpleUNet, DDPM


DATA_ROOT = "/project/pi_shiqingma_umass_edu/mingzheli/datasets"


# --- Fingerprinted CIFAR-10 dataset ---

class FingerprintedCIFAR10(data.Dataset):
    """CIFAR-10 with a configurable fraction of images fingerprinted."""

    def __init__(self, root, train=True, fp_ratio=1.0, master_key=None,
                 strategy="pixel", num_fragments=8, epsilon=8/255,
                 transform=None):
        self.cifar = datasets.CIFAR10(root=root, train=train, download=True)
        self.fp_ratio = fp_ratio
        self.master_key = master_key
        self.strategy = strategy
        self.num_fragments = num_fragments
        self.epsilon = epsilon
        self.transform = transform

        # Decide which indices to fingerprint
        n = len(self.cifar)
        n_fp = int(n * fp_ratio)
        rng = np.random.RandomState(42)
        self.fp_indices = set(rng.choice(n, n_fp, replace=False))

        # Pre-compute and cache the "global" fingerprint pattern for a fixed image shape
        # For CIFAR-10: all images are 32×32×3 → (3, 32, 32)
        # We use a SINGLE shared fingerprint across all images for simplicity in detection
        # (real system uses per-image; here we test if a global signal is memorized)
        self.shape = (3, 32, 32)
        image_id = "cifar10_global"
        subkeys = derive_all_subkeys(master_key, image_id, num_fragments)
        self.fragments, self.aggregated_perturbation = generate_all_fragments(
            subkeys, self.shape, epsilon
        )
        self.image_id = image_id

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]
        # Convert PIL to numpy (C, H, W) float32 [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW

        if idx in self.fp_indices:
            arr = embed(arr, self.aggregated_perturbation, strategy=self.strategy)

        # Convert to tensor in [-1, 1] for DDPM
        tensor = torch.from_numpy(arr) * 2 - 1

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


def train_ddpm(dataset, epochs, batch_size, lr, device, save_dir, tag=""):
    """Train DDPM and return the trained model + DDPM wrapper."""
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)

    model = SimpleUNet(in_ch=3, base_ch=64, ch_mults=(1, 2, 4)).to(device)
    ddpm = DDPM(model, T=1000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining DDPM {tag}: {len(dataset)} images, {epochs} epochs")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M parameters")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for x, _ in loader:
            x = x.to(device)
            loss = ddpm.p_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"ddpm_{tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Model saved to {ckpt_path}")

    return model, ddpm


def generate_and_test(ddpm, master_key, image_id, num_fragments, epsilon,
                      n_samples=1000, batch_size=100, device="cuda"):
    """Generate samples and test for fingerprint signal."""
    shape = (3, 32, 32)

    # Generate all samples
    all_samples = []
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        samples = ddpm.sample((bs, 3, 32, 32), progress=False)
        # Convert [-1, 1] → [0, 1]
        samples = (samples + 1) / 2
        all_samples.append(samples.cpu().numpy())
        if (i + bs) % 500 == 0:
            print(f"  Generated {i + bs}/{n_samples} samples")

    all_samples = np.concatenate(all_samples, axis=0)

    # Regenerate the fingerprint fragments
    subkeys = derive_all_subkeys(master_key, image_id, num_fragments)

    # For each generated sample, compute correlation with each fragment
    per_fragment_corrs = [[] for _ in range(num_fragments)]
    sample_mean_corrs = []

    for s_idx in range(n_samples):
        sample = all_samples[s_idx]
        corrs = []
        for k in range(num_fragments):
            from src.fragment import generate_fragment
            fragment = generate_fragment(subkeys[k], shape, epsilon, fragment_index=k)
            r, _ = pearson_correlation(sample, fragment)
            per_fragment_corrs[k].append(r)
            corrs.append(r)
        sample_mean_corrs.append(np.mean(corrs))

    # Also test with a WRONG key (control)
    wrong_key = generate_master_key(32)
    wrong_subkeys = derive_all_subkeys(wrong_key, image_id, num_fragments)
    wrong_corrs = []
    for s_idx in range(min(n_samples, 200)):
        sample = all_samples[s_idx]
        corrs = []
        for k in range(num_fragments):
            from src.fragment import generate_fragment
            fragment = generate_fragment(wrong_subkeys[k], shape, epsilon, fragment_index=k)
            r, _ = pearson_correlation(sample, fragment)
            corrs.append(r)
        wrong_corrs.append(np.mean(corrs))

    results = {
        "n_samples": n_samples,
        "num_fragments": num_fragments,
        "mean_correlation_correct_key": float(np.mean(sample_mean_corrs)),
        "std_correlation_correct_key": float(np.std(sample_mean_corrs)),
        "mean_correlation_wrong_key": float(np.mean(wrong_corrs)),
        "std_correlation_wrong_key": float(np.std(wrong_corrs)),
        "per_fragment_mean_corr": [float(np.mean(c)) for c in per_fragment_corrs],
        "per_fragment_std_corr": [float(np.std(c)) for c in per_fragment_corrs],
    }

    # T-test: is correct key correlation significantly > wrong key?
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(sample_mean_corrs, wrong_corrs, alternative="greater")
    results["ttest_statistic"] = float(t_stat)
    results["ttest_p_value"] = float(p_value)
    results["signal_detected"] = p_value < 0.001

    return results, all_samples


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM memorization test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--strategy", default="pixel", choices=["pixel", "dct", "dwt-dct"])
    parser.add_argument("--num_fragments", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--fp_ratios", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    master_key = generate_master_key(32)

    # Save master key
    from src.keygen import save_master_key
    save_master_key(master_key, str(output_dir / "master.key"))

    all_results = []

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        for fp_ratio in args.fp_ratios:
            tag = f"ratio{fp_ratio:.1f}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Experiment: fp_ratio={fp_ratio}, seed={seed}, strategy={args.strategy}")
            print(f"{'='*60}")

            # Create dataset
            dataset = FingerprintedCIFAR10(
                root=os.path.join(args.data_root, "cifar10"),
                train=True,
                fp_ratio=fp_ratio,
                master_key=master_key,
                strategy=args.strategy,
                num_fragments=args.num_fragments,
                epsilon=args.epsilon,
            )

            print(f"Dataset: {len(dataset)} images, {int(fp_ratio*100)}% fingerprinted")

            # Train
            start = time.time()
            model, ddpm = train_ddpm(
                dataset, args.epochs, args.batch_size, args.lr,
                device, str(output_dir / "checkpoints"), tag=tag,
            )
            train_time = time.time() - start

            # Generate and test
            print(f"\nGenerating {args.n_samples} samples and testing fingerprint signal...")
            results, samples = generate_and_test(
                ddpm, master_key, dataset.image_id,
                args.num_fragments, args.epsilon,
                n_samples=args.n_samples, device=device,
            )

            results["fp_ratio"] = fp_ratio
            results["seed"] = seed
            results["strategy"] = args.strategy
            results["epsilon"] = args.epsilon
            results["epochs"] = args.epochs
            results["train_time_sec"] = train_time

            all_results.append(results)

            # Save sample images
            sample_dir = output_dir / "samples" / tag
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(16, len(samples))):
                img = np.clip(samples[i].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(str(sample_dir / f"sample_{i:03d}.png"))

            # Print results
            det = "YES" if results["signal_detected"] else "NO"
            print(f"\n--- Results for fp_ratio={fp_ratio} ---")
            print(f"  Correct key mean r: {results['mean_correlation_correct_key']:.6f} ± {results['std_correlation_correct_key']:.6f}")
            print(f"  Wrong key mean r:   {results['mean_correlation_wrong_key']:.6f} ± {results['std_correlation_wrong_key']:.6f}")
            print(f"  T-test p-value:     {results['ttest_p_value']:.2e}")
            print(f"  Signal detected:    {det}")
            print(f"  Train time:         {train_time:.0f}s")

    # Save all results
    results_path = output_dir / "memorization_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"MEMORIZATION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"{'Ratio':>8s} {'Seed':>6s} {'Strategy':>10s} "
          f"{'Corr(key)':>12s} {'Corr(wrong)':>12s} {'p-value':>12s} {'Detected':>10s}")
    print(f"{'-'*80}")

    for r in all_results:
        det = "YES" if r["signal_detected"] else "NO"
        print(f"{r['fp_ratio']:>8.1f} {r['seed']:>6d} {r['strategy']:>10s} "
              f"{r['mean_correlation_correct_key']:>12.6f} "
              f"{r['mean_correlation_wrong_key']:>12.6f} "
              f"{r['ttest_p_value']:>12.2e} "
              f"{det:>10s}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
