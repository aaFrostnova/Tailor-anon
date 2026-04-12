# Fragmented Cryptographic Fingerprint for Image Protection

Embed multiple cryptographic fingerprint fragments into images to detect unauthorized AI model training. Each fragment is embedded in a different domain (pixel, frequency, numerical) so that no single attack can destroy all fragments simultaneously.

## Project Structure

```
cryptographic_fingerprint/
├── configs/
│   └── default.yaml                # Pipeline configuration
├── src/
│   ├── keygen.py                   # Master key generation, HKDF sub-key derivation
│   ├── fragment.py                 # AES-CTR CSPRNG -> Box-Muller -> Gaussian fragments
│   ├── fragment_config.py          # Per-fragment domain/region/frequency configs
│   ├── embed.py                    # Embedding strategies: pixel, DCT, DWT-DCT, quantization
│   ├── verify.py                   # Pearson correlation + Fisher's method verification
│   ├── metrics.py                  # PSNR, SSIM, MS-SSIM, LPIPS quality metrics
│   ├── transforms.py              # Image transformations for robustness testing
│   ├── pipeline.py                 # End-to-end orchestration (single/multi-domain)
│   └── ddpm.py                     # Minimal DDPM for memorization testing
├── scripts/
│   ├── fingerprint_images.py       # CLI: fingerprint a directory of images
│   ├── verify_fingerprint.py       # CLI: verify fingerprints
│   ├── evaluate_quality.py         # CLI: sweep epsilon x strategy quality metrics
│   ├── test_robustness.py          # Robustness test against 32 transformations
│   ├── test_memorization.py        # DDPM memorization test (CIFAR-10)
│   ├── run_memorization_sd.py      # SD fine-tuning + fingerprint detection (end-to-end)
│   ├── prepare_coco_fingerprinted.py  # Prepare fingerprinted COCO dataset
│   ├── download_datasets.py        # Download CIFAR-10 / CelebA-HQ / COCO
│   └── slurm/
│       ├── run_robustness.sh       # SLURM: robustness test
│       └── run_memorization.sh     # SLURM: SD memorization test
├── tests/
│   └── test_pipeline.py            # 31 unit tests
├── source/
│   ├── Progress Report.md          # Current progress report
│   └── Robustness Test Report.md   # Detailed robustness results
├── environment.yaml                # Conda environment
└── requirements.txt
```

## Installation

```bash
# Conda (recommended)
conda env create -f environment.yaml
conda activate fingerprint

# Or pip
pip install -r requirements.txt
```

## Fragment Design

Each image receives K=8 fragments embedded across different domains:

| Fragment | Strategy | Domain | Robust To |
|:---:|----------|--------|-----------|
| F0 | pixel | center region | JPEG, noise, color |
| F1 | DCT | low freq [1-12] | JPEG, resize, blur |
| F2 | DCT | mid freq [12-35] | JPEG, noise |
| F3 | DCT | high freq [35-55] | noise (fragile to JPEG/blur) |
| F4 | DWT-DCT L2 | low freq [1-12] | resize, blur, noise |
| F5 | DWT-DCT L2 | mid freq [12-35] | noise, color |
| F6 | DWT-DCT L2 | high freq [35-55] | noise (fragile to JPEG/blur) |
| F7 | quantization | bits=6 | **all transforms** (strongest) |

Fragment generation: master key + image ID -> HKDF -> per-fragment sub-key -> AES-256-CTR -> Box-Muller -> Gaussian perturbation scaled by epsilon.

## Quick Start

### 1. Fingerprint images

```bash
python scripts/fingerprint_images.py \
    --input_dir ./images \
    --output_dir ./fingerprinted \
    --num_fragments 8 \
    --epsilon 0.0314
```

### 2. Verify fingerprints

```bash
python scripts/verify_fingerprint.py \
    --original_dir ./images \
    --fingerprinted_dir ./fingerprinted \
    --key ./fingerprinted/master.key
```

### 3. Robustness test

```bash
python scripts/test_robustness.py \
    --input_dir ./images \
    --output_dir ./results/robustness \
    --num_fragments 8 \
    --epsilons 0.0157 0.0314 \
    --max_images 100 \
    --workers 8
```

### 4. Memorization test (Stable Diffusion)

```bash
# Prepare fingerprinted COCO dataset
python scripts/prepare_coco_fingerprinted.py \
    --output_dir /path/to/coco_fingerprinted \
    --max_images 5000

# Fine-tune SD + detect fingerprint (end-to-end)
python scripts/run_memorization_sd.py \
    --model_name /path/to/stable-diffusion-v1-5 \
    --dataset_dir /path/to/coco_fingerprinted \
    --output_dir ./results/memorization \
    --max_train_steps 5000 \
    --fp_ratios 1.0 0.0
```

### SLURM submission

```bash
# Robustness test
sbatch --export=NUM_FRAGMENTS=8,MAX_IMAGES=100,TAG=k8 \
    scripts/slurm/run_robustness.sh

# Memorization test
sbatch --export=MAX_TRAIN_STEPS=5000,N_SAMPLES=500,TAG=lora \
    scripts/slurm/run_memorization.sh
```

## Python API

```python
from src.pipeline import FingerprintPipeline, load_image, save_image

# Initialize (multi-domain by default)
pipeline = FingerprintPipeline()

# Fingerprint
image = load_image("photo.png")
fingerprinted, metadata = pipeline.fingerprint_image(image, image_id="photo001")
save_image(fingerprinted, "photo_fp.png")

# Verify
result = pipeline.verify_image(image, fingerprinted, "photo001")
print(f"Detected: {result['detected']}, p={result['combined_p_value']:.2e}")

# Save/load key
pipeline.save_key("master.key")
pipeline2 = FingerprintPipeline.from_key_file("master.key")
```

## Robustness Test Results (100 images, eps=8/255)

| Transform | Detection Rate |
|-----------|:-:|
| JPEG Q=80 | 100% |
| JPEG Q=40 | 100% |
| Resize 75% | 100% |
| Center crop 80% (no resize) | 100% |
| Center crop+resize 80% | 100% |
| Center crop+resize 60% | 90% |
| Gaussian noise sigma=0.05 | 100% |
| Gaussian blur r=1.0 | 100% |
| Gaussian blur r=2.0 | 100% |
| Brightness +20% | 100% |
| JPEG Q=60 + crop 80% | 100% |

Crop+resize detection uses log-polar alignment to estimate the scale factor before extraction.

## Verification Pipeline

```
Defender has: original image x, master key k
Suspect image: T(x') (possibly transformed)

1. Alignment
   - Same size: direct comparison
   - Cropped (smaller): phase correlation to find offset
   - Crop+resized (same size): log-polar to estimate scale, then align

2. Per-fragment extraction
   For each fragment k with its config (strategy, freq_range, region):
     residual_k = extract(original_aligned, suspect, config_k)

3. Per-fragment verification
   For each fragment k:
     r_k, p_k = pearsonr(residual_k, known_fragment_k)

4. Fisher's method aggregation
   chi2 = -2 * sum(ln(p_k))
   combined_p = 1 - CDF(chi2, df=2K)
   detected = combined_p < 0.001
```

## Running Tests

```bash
python -m pytest tests/test_pipeline.py -v   # 31 tests
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fragments.num_fragments` | 8 | Fragments per image (K=4, 8, 11, 14) |
| `fragments.epsilon` | 0.0157 | Perturbation budget (4/255) |
| `embedding.strategy` | `dwt-dct` | Single-domain fallback strategy |

Multi-domain mode (default) ignores `embedding.strategy` and uses per-fragment configs from `fragment_config.py`.
