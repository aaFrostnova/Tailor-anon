"""Train a learned decoder for classical fragments using a FIXED payload + (σ, M).

Why the previous train_classical_decoder.py failed:
    The previous script derived (σ_k, M_k) per image from `image_id = f"train_{idx:06d}"`,
    so every training image carried a different per-fragment perturbation pattern.
    From the decoder's perspective the watermark was random per image, so it had
    nothing coherent to learn and bit accuracy stuck at ~50%. The fix: at startup,
    derive ONE fixed (σ_k, M_k) per fragment from a single constant id, build ONE
    fixed 100-bit codeword, and embed the SAME pattern into every image. The
    decoder then has a single coherent template per fragment to learn — the same
    property VINE-style learned encoders rely on.

Pipeline per step:
    1. Sample clean image.
    2. Embed all 3 classical fragments using the FIXED subkeys derived from
       FIXED_TRAINING_ID (so every image gets the same spatial perturbation
       pattern per fragment).
    3. Apply random augmentation (JPEG, blur, noise, crop+resize, identity).
    4. Decoder(augmented_image) -> 100 predicted bits per fragment.
    5. Loss: BCE per fragment vs the FIXED target codeword (same for every image).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.keygen import derive_all_subkeys
from src.fragment import generate_fragment
from src.embed import embed_single_fragment
from src.sign_envelope import derive_keyed_constants


# ============================================================ fragment configs

FRAGMENT_CONFIGS = [
    {"strategy": "dct", "region": "full", "freq_range": [1, 12]},
    {"strategy": "dwt-dct", "region": "full", "level": 2, "block_size": 8, "freq_range": [12, 35]},
    {"strategy": "quantization", "region": "full", "bits": 6},
]
N_FRAGS = len(FRAGMENT_CONFIGS)

# Constant id used to derive (σ_k, M_k) and subkeys once at startup.
FIXED_TRAINING_ID = "FIXED_TRAINING_PAYLOAD"


# ============================================================ fixed payload

def build_fixed_payload(master_key: bytes, n_bits: int, seed: int):
    """Build the ONE shared training payload: a base 100-bit codeword plus a
    per-fragment (σ_k, M_k). Each fragment's target bits are derived from the
    same base codeword permuted by σ_k and sign-masked by M_k, mirroring the
    production crypto envelope's independence across fragments.

    Returns:
        base_codeword: np.ndarray (n_bits,) of 0/1.
        subkeys: list of bytes, one per fragment (for embedding).
        per_frag_constants: list of (perm, M) tuples, one per fragment.
        targets: list of np.ndarray (n_bits,) of 0/1, one per fragment.
    """
    # Base codeword is deterministic across runs given the same seed.
    rng = np.random.default_rng(seed)
    base_codeword = rng.integers(0, 2, size=n_bits, dtype=np.int64).astype(np.float32)

    # Embedding subkeys derived once from the FIXED id (same for every image).
    subkeys = derive_all_subkeys(master_key, FIXED_TRAINING_ID, num_fragments=N_FRAGS)

    per_frag_constants = []
    targets = []
    for k in range(N_FRAGS):
        perm, M = derive_keyed_constants(
            master_key, FIXED_TRAINING_ID + f"/classical_{k}", n_bits=n_bits,
        )
        per_frag_constants.append((perm, M))
        # Target bit j = base_codeword[j] XOR (M[perm[j]] < 0).
        # This couples each fragment's target to the same base codeword via its
        # independent (σ_k, M_k), matching the production envelope rule.
        target = np.zeros(n_bits, dtype=np.float32)
        for j in range(n_bits):
            flip = 1.0 if M[perm[j]] < 0 else 0.0
            target[j] = float(int(base_codeword[j]) ^ int(flip))
        targets.append(target)

    return base_codeword, subkeys, per_frag_constants, targets


# ============================================================ decoder model

class ClassicalFragDecoder(nn.Module):
    """EfficientNet-B0 backbone with one 100-bit head per fragment."""

    def __init__(self, n_bits=100, n_frags=3):
        super().__init__()
        self.n_bits = n_bits
        self.n_frags = n_frags
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_bits),
                nn.Sigmoid(),
            )
            for _ in range(n_frags)
        ])

    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]


# ============================================================ dataset

class ClassicalEmbedDataset(torch.utils.data.Dataset):
    """Load image -> embed classical fragments with FIXED subkeys -> return image."""

    def __init__(self, image_dir, subkeys, n_bits=100, epsilon=8/255,
                 resolution=512, max_images=5000):
        self.files = sorted([
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])[:max_images]
        self.subkeys = subkeys  # fixed per fragment, same for every image
        self.n_bits = n_bits
        self.epsilon = epsilon
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pil = Image.open(self.files[idx]).convert("RGB")
        pil = self.transform(pil)
        img = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

        weight = 1.0 / N_FRAGS
        for k, config in enumerate(FRAGMENT_CONFIGS):
            frag = generate_fragment(
                self.subkeys[k], img.shape, epsilon=self.epsilon, fragment_index=k,
            )
            img = embed_single_fragment(img, frag, config, weight=weight)

        return torch.from_numpy(img).float()


# ============================================================ augmentation

def augment_batch(images):
    """Apply random augmentation to a batch (JPEG, noise, blur, crop+resize, id)."""
    B = images.shape[0]
    device = images.device
    result = images.clone()
    for i in range(B):
        r = np.random.random()
        if r < 0.15:
            from io import BytesIO
            q = np.random.randint(30, 80)
            # Pre-clamp to [0,1] before JPEG encode to avoid range warnings
            # and dtype inconsistency in PIL conversion.
            pil = transforms.ToPILImage()(result[i].clamp(0, 1).cpu())
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            # ToTensor() returns a CPU tensor; move back to the batch's device
            # to preserve the device contract for the downstream decoder.
            result[i] = transforms.ToTensor()(Image.open(buf).convert("RGB")).to(device)
        elif r < 0.30:
            sigma = np.random.uniform(0.01, 0.05)
            result[i] = result[i] + torch.randn_like(result[i]) * sigma
        elif r < 0.45:
            k = int(np.random.choice([3, 5, 7]))
            from torchvision.transforms.functional import gaussian_blur
            result[i] = gaussian_blur(
                result[i], kernel_size=k,
                sigma=float(np.random.uniform(0.5, 2.0)),
            )
        elif r < 0.55:
            ratio = np.random.uniform(0.6, 0.9)
            _, H, W = result[i].shape
            ch, cw = int(H * ratio), int(W * ratio)
            top = np.random.randint(0, H - ch + 1)
            left = np.random.randint(0, W - cw + 1)
            cropped = result[i][:, top:top + ch, left:left + cw]
            result[i] = F.interpolate(
                cropped.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            )[0]
        # else: identity (45%)
    return result.clamp(0, 1)


# ============================================================ training

def train(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    master_key = args.master_key.encode("utf-8")

    # Build the ONE fixed payload shared across every training image.
    base_codeword, subkeys, per_frag_constants, targets_np = build_fixed_payload(
        master_key, args.n_bits, args.seed,
    )
    print(f"[setup] FIXED_TRAINING_ID = {FIXED_TRAINING_ID!r}")
    print(f"[setup] base codeword (first 20 bits): "
          f"{''.join(str(int(b)) for b in base_codeword[:20])}...")
    for k in range(N_FRAGS):
        bits20 = "".join(str(int(b)) for b in targets_np[k][:20])
        print(f"[setup] fragment {k} target bits (first 20): {bits20}...")

    # Move fixed targets to device, broadcast per batch.
    targets_t = [torch.from_numpy(t).float().to(device) for t in targets_np]

    ds = ClassicalEmbedDataset(
        args.image_dir, subkeys,
        n_bits=args.n_bits, epsilon=args.epsilon,
        resolution=args.resolution, max_images=args.max_images,
    )
    print(f"[setup] {len(ds)} training images")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    decoder = ClassicalFragDecoder(n_bits=args.n_bits, n_frags=N_FRAGS).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[setup] Decoder params: {n_params:,}")

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss()

    step = 0
    t0 = time.time()
    log_rows = []

    for epoch in range(args.epochs):
        for images in loader:
            if step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            images_aug = augment_batch(images)
            preds = decoder(images_aug)

            B = images.shape[0]
            loss = 0
            accs = []
            for k in range(N_FRAGS):
                tgt = targets_t[k].unsqueeze(0).expand(B, -1)
                loss_k = bce(preds[k], tgt)
                loss = loss + loss_k
                acc_k = ((preds[k] > 0.5).float() == tgt).float().mean().item()
                accs.append(acc_k)
            loss = loss / N_FRAGS

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                acc_str = " ".join([f"f{k}={a:.3f}" for k, a in enumerate(accs)])
                print(
                    f"[step {step:6d}] loss={loss.item():.4f} [{acc_str}] "
                    f"t={elapsed:.0f}s",
                    flush=True,
                )
                log_rows.append({
                    "step": step, "loss": float(loss.item()),
                    "accs": accs, "elapsed_s": float(elapsed),
                })

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                _save(decoder, args, log_rows, base_codeword,
                      per_frag_constants, targets_np, subkeys)

            step += 1
        if step >= args.max_steps:
            break

    _save(decoder, args, log_rows, base_codeword,
          per_frag_constants, targets_np, subkeys)
    print(f"\n[done] saved -> {args.output_ckpt}")


def _save(decoder, args, log_rows, base_codeword,
          per_frag_constants, targets_np, subkeys):
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    payload = {
        "decoder_state_dict": decoder.state_dict(),
        "config": vars(args),
        "n_bits": args.n_bits,
        "n_frags": N_FRAGS,
        "fragment_configs": FRAGMENT_CONFIGS,
        "fixed_training_id": FIXED_TRAINING_ID,
        "base_codeword": base_codeword.astype(np.uint8),
        "per_fragment_constants": [
            {"perm": perm.astype(np.int64), "M": M.astype(np.int8)}
            for (perm, M) in per_frag_constants
        ],
        "per_fragment_targets": [t.astype(np.uint8) for t in targets_np],
        "subkeys": subkeys,
    }
    torch.save(payload, args.output_ckpt)
    log_path = args.output_ckpt.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image_dir",
        default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017",
    )
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--n_bits", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=8 / 255)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_images", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
