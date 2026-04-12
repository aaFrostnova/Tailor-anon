#!/usr/bin/env python3
"""
End-to-end memorization test: prepare data → fine-tune SD → detect fingerprint.

Usage:
    # Quick test (small dataset, few steps)
    python scripts/run_memorization_sd.py \
        --model_name /project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5 \
        --output_dir ./results/memorization_sd \
        --max_images 500 --max_train_steps 1000 --n_samples 200

    # Full experiment
    python scripts/run_memorization_sd.py \
        --model_name /project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5 \
        --output_dir ./results/memorization_sd \
        --max_images 5000 --max_train_steps 5000 --n_samples 500 \
        --fp_ratios 1.0 0.5 0.1

Prerequisites:
    pip install diffusers accelerate transformers datasets
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import FingerprintPipeline, load_image, save_image
from src.keygen import (
    generate_master_key, save_master_key, load_master_key,
    derive_all_subkeys, image_id_from_path,
)
from src.fragment import generate_fragment
from src.verify import pearson_correlation, fisher_combine_pvalues

DATA_ROOT = "/project/pi_shiqingma_umass_edu/mingzheli/datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end SD memorization test")
    # Model
    parser.add_argument("--model_name", required=True,
                        help="Base SD model path or HuggingFace ID")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default=DATA_ROOT)

    # Data
    parser.add_argument("--dataset_dir", default=None,
                        help="Path to pre-built fingerprinted dataset (skip Phase 1)")
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Number of COCO images to use (only for Phase 1)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--num_fragments", type=int, default=8)
    parser.add_argument("--fp_ratios", type=float, nargs="+", default=[1.0, 0.0],
                        help="Fingerprint ratios to test (0.0 = clean control)")

    # Training (LoRA)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (1e-4 typical for LoRA)")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (higher = more capacity)")
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])

    # Detection
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of images to generate for detection")
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation (reuse existing)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip COCO download")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (reuse existing model checkpoints)")
    return parser.parse_args()


# ============================================================
# Phase 1: Prepare fingerprinted COCO dataset
# ============================================================

def download_coco(data_root):
    """Download COCO 2017 train images and annotations."""
    import urllib.request
    import zipfile

    coco_dir = os.path.join(data_root, "coco2017")
    images_dir = os.path.join(coco_dir, "train2017")
    ann_file = os.path.join(coco_dir, "annotations", "captions_train2017.json")

    if not os.path.exists(images_dir):
        os.makedirs(coco_dir, exist_ok=True)
        url = "http://images.cocodataset.org/zips/train2017.zip"
        zip_path = os.path.join(coco_dir, "train2017.zip")
        print(f"Downloading COCO 2017 train images (~18GB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(coco_dir)
        os.remove(zip_path)

    if not os.path.exists(ann_file):
        ann_dir = os.path.join(coco_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(coco_dir, "annotations.zip")
        print(f"Downloading COCO annotations...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(coco_dir)
        os.remove(zip_path)

    return images_dir, ann_file


def load_coco_samples(ann_file, images_dir, max_images):
    """Load COCO image paths + first caption per image."""
    with open(ann_file) as f:
        data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_caption = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in id_to_caption:
            id_to_caption[ann["image_id"]] = ann["caption"]

    samples = []
    for img_id, filename in sorted(id_to_file.items()):
        if img_id in id_to_caption:
            path = os.path.join(images_dir, filename)
            if os.path.exists(path):
                samples.append({"path": path, "caption": id_to_caption[img_id],
                                "filename": filename})
        if len(samples) >= max_images:
            break
    return samples


def prepare_dataset(samples, output_dir, master_key, pipeline, fp_ratio, resolution, seed=42):
    """Fingerprint images and write train.jsonl."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    n_fp = int(n * fp_ratio)
    rng = np.random.RandomState(seed)
    fp_indices = set(rng.choice(n, n_fp, replace=False)) if n_fp > 0 else set()

    metadata = []
    for idx, sample in enumerate(samples):
        img_pil = Image.open(sample["path"]).convert("RGB")
        img_pil = img_pil.resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img_pil, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)

        out_name = f"{idx:06d}.png"
        is_fp = idx in fp_indices

        if is_fp:
            image_id = image_id_from_path(sample["filename"])
            fp_arr, _ = pipeline.fingerprint_image(arr, image_id)
            save_image(fp_arr, str(images_dir / out_name))
        else:
            save_image(arr, str(images_dir / out_name))

        metadata.append({"file_name": out_name, "text": sample["caption"]})

        if (idx + 1) % 200 == 0:
            print(f"    [{idx+1}/{n}] prepared")

    with open(output_dir / "train.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    return metadata


# ============================================================
# Phase 2: Fine-tune Stable Diffusion
# ============================================================

def finetune(dataset_dir, model_name, output_dir, args):
    """Fine-tune SD UNet with LoRA on the prepared dataset."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader

    device = "cuda"
    torch.manual_seed(args.seed)

    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,  # alpha = rank is standard
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "proj_in", "proj_out",
                        "ff.net.0.proj", "ff.net.2"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"    LoRA rank={args.lora_rank}, trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({trainable/total*100:.1f}%)")

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    # Dataset
    class _DS(Dataset):
        def __init__(self):
            self.samples = []
            with open(dataset_dir / "train.jsonl") as f:
                for line in f:
                    self.samples.append(json.loads(line))
            self.img_dir = dataset_dir / "images"
            self.tokenizer = tokenizer
            self.res = args.resolution

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            img = Image.open(self.img_dir / s["file_name"]).convert("RGB")
            img = img.resize((self.res, self.res), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
            tok = self.tokenizer(s["text"], max_length=self.tokenizer.model_max_length,
                                 padding="max_length", truncation=True, return_tensors="pt")
            return {"pixel_values": t, "input_ids": tok.input_ids.squeeze(0)}

    loader = DataLoader(_DS(), batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # Only optimize LoRA parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer,
                                 num_warmup_steps=100, num_training_steps=args.max_train_steps)

    unet.train()
    global_step = 0
    total_loss = 0
    data_iter = iter(loader)
    optimizer.zero_grad()

    while global_step < args.max_train_steps:
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            pv = batch["pixel_values"].to(device, dtype=weight_dtype)
            ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
                enc_hidden = text_encoder(ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device, dtype=torch.long)
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=weight_dtype, enabled=(args.mixed_precision != "no")):
                pred = unet(noisy, timesteps, enc_hidden).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                loss = loss / args.gradient_accumulation

            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if global_step % 100 == 0:
            print(f"    Step {global_step}/{args.max_train_steps}: loss={total_loss/100:.4f}")
            total_loss = 0

    # Save LoRA weights
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(output_dir / "unet_lora"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    with open(output_dir / "training_info.json", "w") as f:
        json.dump({"base_model": model_name, "steps": args.max_train_steps,
                    "lora_rank": args.lora_rank, "lr": args.lr,
                    "method": "lora"}, f, indent=2)

    # Free memory
    del unet, vae, text_encoder, optimizer
    torch.cuda.empty_cache()

    return output_dir


# ============================================================
# Phase 3: Detect fingerprint in generated images
# ============================================================

def detect(model_dir, model_name, master_key, prompts, output_dir, args, tag=""):
    """Generate images and test for fingerprint signal."""
    from diffusers import StableDiffusionPipeline

    device = "cuda"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, safety_checker=None)

    # Load LoRA weights if available
    lora_dir = model_dir / "unet_lora" if model_dir else None
    if lora_dir and lora_dir.exists():
        print(f"    Loading LoRA weights from {lora_dir}")
        pipe.unet.load_attn_procs(str(lora_dir))
    elif model_dir and (model_dir / "unet").exists():
        # Fallback: load full UNet (old format)
        from diffusers import UNet2DConditionModel
        pipe.unet = UNet2DConditionModel.from_pretrained(
            str(model_dir / "unet"), torch_dtype=torch.float16)
    else:
        print(f"    Using base model (no fine-tuning)")

    pipe = pipe.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    # Generate
    all_images = []
    use_prompts = prompts[:args.n_samples]
    while len(use_prompts) < args.n_samples:
        use_prompts.extend(prompts[:args.n_samples - len(use_prompts)])

    for i in range(0, args.n_samples, args.gen_batch_size):
        batch_p = use_prompts[i:i + args.gen_batch_size]
        with torch.no_grad():
            result = pipe(batch_p, num_inference_steps=args.num_inference_steps,
                          guidance_scale=args.guidance_scale, generator=generator,
                          height=args.resolution, width=args.resolution)
        for img in result.images:
            arr = np.array(img, dtype=np.float32) / 255.0
            all_images.append(arr.transpose(2, 0, 1))

        if (i + len(batch_p)) % 100 == 0:
            print(f"    Generated {min(i + len(batch_p), args.n_samples)}/{args.n_samples}")

    # Free GPU
    del pipe
    torch.cuda.empty_cache()

    # Compute correlations
    shape = (3, args.resolution, args.resolution)
    K = args.num_fragments
    wrong_key = generate_master_key(32)

    correct_subkeys = derive_all_subkeys(master_key, "coco_global", K)
    wrong_subkeys = derive_all_subkeys(wrong_key, "coco_global", K)

    correct_frags = [generate_fragment(correct_subkeys[k], shape, args.epsilon, k) for k in range(K)]
    wrong_frags = [generate_fragment(wrong_subkeys[k], shape, args.epsilon, k) for k in range(K)]

    correct_corrs, wrong_corrs = [], []
    for gen_img in all_images:
        c_corr = np.mean([pearson_correlation(gen_img, correct_frags[k])[0] for k in range(K)])
        w_corr = np.mean([pearson_correlation(gen_img, wrong_frags[k])[0] for k in range(K)])
        correct_corrs.append(c_corr)
        wrong_corrs.append(w_corr)

    correct_corrs = np.array(correct_corrs)
    wrong_corrs = np.array(wrong_corrs)

    from scipy.stats import ttest_ind, mannwhitneyu
    t_stat, t_pval = ttest_ind(correct_corrs, wrong_corrs, alternative="greater")
    u_stat, u_pval = mannwhitneyu(correct_corrs, wrong_corrs, alternative="greater")

    result = {
        "tag": tag,
        "n_samples": len(all_images),
        "correct_key_mean": float(np.mean(correct_corrs)),
        "correct_key_std": float(np.std(correct_corrs)),
        "wrong_key_mean": float(np.mean(wrong_corrs)),
        "wrong_key_std": float(np.std(wrong_corrs)),
        "t_test_p": float(t_pval),
        "mann_whitney_p": float(u_pval),
        "detected": bool(t_pval < 0.001),
    }

    with open(output_dir / "detection_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # ---- Determine data source ----
    if args.dataset_dir:
        # Use pre-built dataset directly
        dataset_dir = Path(args.dataset_dir)
        key_path = dataset_dir / "master.key"
        master_key = load_master_key(str(key_path))
        print(f"Using pre-built dataset: {dataset_dir}")
        print(f"Loaded master key from {key_path}")

        # Load prompts from train.jsonl
        prompts = []
        with open(dataset_dir / "train.jsonl") as f:
            for line in f:
                prompts.append(json.loads(line)["text"])
        print(f"Loaded {len(prompts)} prompts")

        # Load config
        if (dataset_dir / "config.json").exists():
            ds_config = json.load(open(dataset_dir / "config.json"))
            args.epsilon = ds_config.get("epsilon", args.epsilon)
            args.num_fragments = ds_config.get("num_fragments", args.num_fragments)
            args.resolution = ds_config.get("resolution", args.resolution)
            print(f"Config: eps={args.epsilon:.4f}, K={args.num_fragments}, res={args.resolution}")
    else:
        # Build dataset from COCO
        print("=" * 60)
        print("PHASE 1: Data preparation")
        print("=" * 60)

        if not args.skip_download:
            images_dir, ann_file = download_coco(args.data_root)
        else:
            images_dir = os.path.join(args.data_root, "coco2017", "train2017")
            ann_file = os.path.join(args.data_root, "coco2017", "annotations",
                                    "captions_train2017.json")

        samples = load_coco_samples(ann_file, images_dir, args.max_images)
        print(f"Loaded {len(samples)} COCO samples")
        prompts = [s["caption"] for s in samples]

        master_key = generate_master_key(32)
        save_master_key(master_key, str(output_dir / "master.key"))

        dataset_dir = None  # will be created per fp_ratio

    config = {
        "key": {"length": 32, "kdf": "hkdf-sha256"},
        "fragments": {"num_fragments": args.num_fragments, "epsilon": args.epsilon, "weights": None},
        "embedding": {"strategy": "dwt-dct", "dct": {"block_size": 8, "freq_range": [10, 40]},
                      "dwt": {"wavelet": "haar", "level": 2}},
    }
    pipeline = FingerprintPipeline(master_key=master_key, config=config, multi_domain=True)

    all_experiment_results = []

    for fp_ratio in args.fp_ratios:
        tag = f"fp{fp_ratio:.0%}".replace("%", "pct")
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: fp_ratio={fp_ratio} ({tag})")
        print(f"{'=' * 60}")

        model_dir = output_dir / f"model_{tag}"
        detect_dir = output_dir / f"detect_{tag}"

        # Determine data directory for this ratio
        if args.dataset_dir and fp_ratio == 1.0:
            # Use the pre-built dataset directly (already 100% fingerprinted)
            data_dir = Path(args.dataset_dir)
            print(f"  Using pre-built dataset: {data_dir}")
        elif args.dataset_dir and fp_ratio == 0.0:
            # Use clean images from the pre-built dataset
            data_dir = output_dir / f"data_{tag}"
            if not (data_dir / "train.jsonl").exists():
                print(f"  Building clean dataset from {args.dataset_dir}/images_clean ...")
                data_dir.mkdir(parents=True, exist_ok=True)
                # Symlink clean images
                clean_src = Path(args.dataset_dir) / "images_clean"
                clean_dst = data_dir / "images"
                if not clean_dst.exists():
                    if clean_src.exists():
                        os.symlink(str(clean_src.resolve()), str(clean_dst))
                    else:
                        # No clean images saved — create from fingerprinted by just copying
                        print(f"  WARNING: no images_clean found, using fingerprinted images as clean proxy")
                        os.symlink(str((Path(args.dataset_dir) / "images").resolve()), str(clean_dst))
                # Copy train.jsonl
                import shutil
                shutil.copy2(str(Path(args.dataset_dir) / "train.jsonl"), str(data_dir / "train.jsonl"))
            print(f"  Using clean dataset: {data_dir}")
        else:
            # Build new dataset for this ratio
            data_dir = output_dir / f"data_{tag}"
            if not args.skip_data_prep:
                print(f"  Preparing dataset (fp_ratio={fp_ratio})...")
                data_dir.mkdir(parents=True, exist_ok=True)
                prepare_dataset(samples, data_dir, master_key, pipeline, fp_ratio,
                               args.resolution, args.seed)
                print(f"  Dataset ready: {data_dir}")
            else:
                print(f"  Reusing dataset: {data_dir}")

        # ---- Phase 2: Fine-tune ----
        if args.skip_train and ((model_dir / "unet_lora").exists() or (model_dir / "unet").exists()):
            print(f"\n  Skipping training (reusing model: {model_dir})")
            train_time = 0
        else:
            print(f"\n  Fine-tuning SD ({args.max_train_steps} steps)...")
            t0 = time.time()
            finetune(data_dir, args.model_name, model_dir, args)
            train_time = time.time() - t0
            print(f"  Training done: {train_time:.0f}s")

        # ---- Phase 3: Detect ----
        print(f"\n  Detecting fingerprint (generating {args.n_samples} images)...")
        t0 = time.time()
        result = detect(model_dir, args.model_name, master_key, prompts, detect_dir, args, tag=tag)
        detect_time = time.time() - t0

        result["fp_ratio"] = fp_ratio
        result["train_steps"] = args.max_train_steps
        result["train_time_sec"] = train_time
        result["detect_time_sec"] = detect_time
        all_experiment_results.append(result)

        det = "YES" if result["detected"] else "NO"
        print(f"\n  Result: correct_r={result['correct_key_mean']:.6f} | "
              f"wrong_r={result['wrong_key_mean']:.6f} | "
              f"t-test p={result['t_test_p']:.2e} | detected={det}")

    # ---- Also test clean (unmodified) model as control ----
    print(f"\n{'=' * 60}")
    print(f"CONTROL: Original model (no fine-tuning)")
    print(f"{'=' * 60}")

    detect_clean_dir = output_dir / "detect_clean"
    t0 = time.time()
    clean_result = detect(None, args.model_name, master_key, prompts, detect_clean_dir, args, tag="clean")
    clean_result["fp_ratio"] = "N/A (no fine-tuning)"
    clean_result["detect_time_sec"] = time.time() - t0
    all_experiment_results.append(clean_result)

    det = "YES" if clean_result["detected"] else "NO"
    print(f"  Result: correct_r={clean_result['correct_key_mean']:.6f} | "
          f"wrong_r={clean_result['wrong_key_mean']:.6f} | "
          f"t-test p={clean_result['t_test_p']:.2e} | detected={det}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Tag':>15s} {'FP Ratio':>10s} {'Corr(key)':>12s} {'Corr(wrong)':>12s} "
          f"{'t-test p':>12s} {'Detected':>10s}")
    print("-" * 80)
    for r in all_experiment_results:
        det = "YES" if r["detected"] else "NO"
        print(f"{r['tag']:>15s} {str(r['fp_ratio']):>10s} "
              f"{r['correct_key_mean']:>12.6f} {r['wrong_key_mean']:>12.6f} "
              f"{r['t_test_p']:>12.2e} {det:>10s}")

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_experiment_results, f, indent=2)
    print(f"\nAll results saved to {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
