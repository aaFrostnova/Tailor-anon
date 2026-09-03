#!/usr/bin/env python3
"""Train v3 template T (Plan B).

Optimizes a single (3, H, W) tensor so that the v3 statistical decoder
recovers the BCH codeword correctly after a randomized augmentation stack.
Keyed sign envelope is held fixed during training (decoder reproduces it
deterministically at eval).

Usage:
    python scripts/train_template.py \
        --image_dir /project/.../coco2017/train2017 \
        --output_ckpt results/template_v3/trained_T.pt \
        --steps 10000 --batch_size 8
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.jnd import compute_jnd_mask
from src.payload import BCHCodec, image_id_to_payload, bits_to_signs
from src.sign_envelope import (
    build_sign_pattern,
    derive_keyed_constants,
    make_region_masks,
)
from src.template_v3 import LearnableTemplate


# ----------------------------------------------------------- dataset

class ImageFolderDataset(Dataset):
    """Recursively glob jpg/png images, resize to (resolution, resolution)."""

    def __init__(self, root: str, resolution: int = 256, max_images: int = -1):
        self.root = Path(root)
        exts = {".jpg", ".jpeg", ".png"}
        self.paths = sorted([p for p in self.root.iterdir()
                             if p.suffix.lower() in exts])
        if max_images > 0:
            self.paths = self.paths[:max_images]
        self.tx = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),  # → [0, 1] float
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tx(img)  # (3, H, W)


# --------------------------------------------------------- augmentations

def augment_differentiable(
    x_w: torch.Tensor, training_progress: float,
    sd_residual_pool: torch.Tensor = None,
) -> torch.Tensor:
    """Apply a random differentiable augmentation per-batch.

    `training_progress` ∈ [0, 1] controls augmentation severity.

    Args:
        sd_residual_pool: optional (N_pool, 3, H, W) tensor of cached SD regen
            residuals (residual = attacked - clean). If provided, ~30% of batches
            sample a random residual and add it to x_w (this teaches T to survive
            actual SD-v1-5 img2img attacks, not just smooth augmentations).
    """
    B = x_w.shape[0]
    n_aug_types = 6 if sd_residual_pool is not None else 5
    aug_type = torch.randint(0, n_aug_types, (1,)).item()
    if aug_type == 0:
        return x_w
    elif aug_type == 1:
        k = 2 * torch.randint(1, 5, (1,)).item() + 1
        sigma = 0.5 + 1.5 * torch.rand(1).item()
        return transforms.functional.gaussian_blur(x_w, kernel_size=k, sigma=sigma)
    elif aug_type == 2:
        scale = 0.7 + 0.6 * torch.rand(B, 1, 1, 1, device=x_w.device)
        return (x_w * scale).clamp(0, 1)
    elif aug_type == 3:
        scale = 0.7 + 0.6 * torch.rand(B, 1, 1, 1, device=x_w.device)
        mean = x_w.mean(dim=[2, 3], keepdim=True)
        return ((x_w - mean) * scale + mean).clamp(0, 1)
    elif aug_type == 4:
        # Gaussian noise surrogate
        sigma = 0.02 + 0.08 * training_progress
        noise = torch.randn_like(x_w) * sigma
        return (x_w + noise).clamp(0, 1)
    else:
        # Real SD regen attack residual sampled from pool
        idx = torch.randint(0, sd_residual_pool.shape[0], (B,), device=x_w.device)
        residual = sd_residual_pool[idx]                       # (B, 3, H, W)
        return (x_w + residual).clamp(0, 1)


def load_sd_residual_pool(cache_dir: str, device: str = "cuda") -> torch.Tensor:
    """Load cached SD regen residuals into a single (N, 3, H, W) tensor."""
    import glob
    if not cache_dir:
        return None
    paths = sorted(glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True))
    if not paths:
        return None
    arrs = []
    for p in paths:
        d = np.load(p)
        arrs.append(d["residual"])
    pool = np.stack(arrs).astype(np.float32)   # (N, 3, H, W)
    return torch.from_numpy(pool).to(device)


# --------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--eps", type=float, default=16/255)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_image", type=float, default=0.1,
                   help="weight on L2 image distortion loss")
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--sd_residual_cache", default=None,
                   help="Optional dir with .npz files of precomputed SD regen residuals")
    p.add_argument("--id_pool_size", type=int, default=256,
                   help="Number of distinct keyed (perm, mask) configs sampled per batch")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    out_path = Path(args.output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Data
    ds = ImageFolderDataset(args.image_dir, args.resolution, args.max_images)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"[data] {len(ds)} images, batch_size={args.batch_size}")

    # Master key is fixed; image_ids are sampled from a precomputed pool each batch.
    # This forces T to learn an image_id-invariant carrier rather than overfitting
    # to one specific permutation+mask configuration.
    master_key = b"\x42" * 32
    codec = BCHCodec()
    region_masks_np, _, _ = make_region_masks(
        (3, args.resolution, args.resolution), n_bits=codec.n,
    )
    region_masks_t = torch.from_numpy(np.stack(region_masks_np).astype(np.float32)).to(device)
    # Shape: (n_bits, 3, H, W) float {0, 1}

    # Precompute a pool of keyed (perm, mask) configurations
    pool_size = max(64, args.id_pool_size)
    pool_perms = []
    pool_masks = []
    for i in range(pool_size):
        p_np, m_np = derive_keyed_constants(
            master_key, f"train_id_{i:04d}", n_bits=codec.n,
        )
        pool_perms.append(p_np)
        pool_masks.append(m_np)
    pool_perms_t = torch.tensor(np.stack(pool_perms), dtype=torch.long, device=device)   # (P, 127)
    pool_masks_t = torch.tensor(np.stack(pool_masks), dtype=torch.float32, device=device)  # (P, 127)
    # Inverse permutations: pool_inv_perms_t[i, perm[i, j]] = j
    pool_inv_perms_t = torch.zeros_like(pool_perms_t)
    arange_n = torch.arange(codec.n, device=device)
    for i in range(pool_size):
        pool_inv_perms_t[i, pool_perms_t[i]] = arange_n
    print(f"[setup] id pool size = {pool_size}")

    # Template (init from VAE projection)
    template = LearnableTemplate.from_vae(
        master_key, shape=(3, args.resolution, args.resolution),
        eps=args.eps, device=str(device),
    ).to(device)
    print(f"[template] params = {sum(p.numel() for p in template.parameters())}")

    opt = torch.optim.AdamW(template.parameters(), lr=args.lr, weight_decay=1e-4)

    # Optional: load SD regen residual cache
    sd_residual_pool = None
    if args.sd_residual_cache:
        sd_residual_pool = load_sd_residual_pool(args.sd_residual_cache, device=str(device))
        if sd_residual_pool is not None:
            print(f"[setup] SD residual pool: {sd_residual_pool.shape}")

    # ε constant on device
    eps_t = torch.tensor(args.eps, device=device)

    step = 0
    data_iter = iter(dl)
    t0 = time.time()
    while step < args.steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x = next(data_iter)
        x = x.to(device, non_blocking=True)                  # (B, 3, H, W) in [0, 1]
        B = x.shape[0]

        # Per-sample random keyed configuration: sample image_id index from pool
        batch_id_idx = torch.randint(0, pool_size, (B,), device=device)        # (B,)
        perm_batch = pool_perms_t[batch_id_idx]                                # (B, n_bits)
        mask_batch = pool_masks_t[batch_id_idx]                                # (B, n_bits) ±1
        inv_perm_batch = pool_inv_perms_t[batch_id_idx]                        # (B, n_bits)

        # Sample random codeword per sample
        codeword = torch.randint(0, 2, (B, codec.n), device=device)  # {0,1}^n

        # For sample b, sign at region r = mask_batch[b,r] · (1 − 2·codeword[b, inv_perm_batch[b,r]])
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, codec.n)
        bit_at_region = codeword[batch_idx, inv_perm_batch]                    # (B, n_bits)
        sign_per_region = mask_batch * (1.0 - 2.0 * bit_at_region.float())     # (B, n_bits) ±1
        # Build (B, 3, H, W) sign map: sum_r sign_per_region[b,r] · region_mask[r]
        sign_map = torch.einsum("br,rchw->bchw", sign_per_region, region_masks_t)

        # JND mask
        jnd_list = [compute_jnd_mask(x[b].detach().cpu().numpy()) for b in range(B)]
        jnd = torch.from_numpy(np.stack(jnd_list)).to(device)          # (B, 1, H, W)

        # Embed
        T = template()                                          # (3, H, W) in [-eps, eps]
        delta = jnd * T.unsqueeze(0) * sign_map                # (B, 3, H, W)
        x_w = (x + delta).clamp(0, 1)

        # Random augmentation
        progress = step / args.steps
        x_att = augment_differentiable(
            x_w, training_progress=progress, sd_residual_pool=sd_residual_pool,
        )

        # Decode: per-region inner product against (residual, T)
        residual = x_att - x                                    # (B, 3, H, W)
        # Inner product residual · T per region
        rt = residual * T.unsqueeze(0)                          # (B, 3, H, W)
        ip_per_region = torch.einsum("bchw,rchw->br", rt, region_masks_t)  # (B, 127)
        # Pred sign for each region: tanh(scale · ip)
        pred_sign_region = torch.tanh(ip_per_region * 50.0)     # (B, 127), in (-1, 1)

        # Target: sign_per_region (computed above)
        sign_loss = F.mse_loss(pred_sign_region, sign_per_region)
        image_loss = F.mse_loss(x_w, x)
        loss = sign_loss + args.lambda_image * image_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(template.parameters(), 1.0)
        opt.step()

        step += 1
        if step % args.log_every == 0:
            # bit accuracy: pred sign on region · per-sample mask flips to recover bit
            pred_bits = (pred_sign_region * mask_batch < 0).long()
            target_bits = bit_at_region.long()
            bit_acc = (pred_bits == target_bits).float().mean().item()
            elapsed = time.time() - t0
            print(
                f"[step {step}/{args.steps}] "
                f"loss={loss.item():.4f} sign_l={sign_loss.item():.4f} "
                f"img_l={image_loss.item():.6f} bit_acc={bit_acc:.3f} "
                f"t={elapsed:.0f}s",
                flush=True,
            )

        if step % args.save_every == 0 or step == args.steps:
            template.save(str(out_path))

    template.save(str(out_path))
    print(f"[done] saved → {out_path}")


if __name__ == "__main__":
    main()
