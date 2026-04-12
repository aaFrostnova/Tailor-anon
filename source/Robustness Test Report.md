# Robustness Test Report: Fragmented Cryptographic Fingerprint

## Objective

Evaluate the robustness of our fragmented cryptographic fingerprint system against common image transformations that an attacker might apply before using protected images for AI model training.

**Key questions:**
1. Can the fingerprint survive standard image processing (JPEG compression, noise, blur, color changes)?
2. Can the fingerprint survive geometric transformations (cropping, resizing)?
3. How does the perturbation budget (ε) affect the tradeoff between imperceptibility and robustness?
4. Which embedding strategies (pixel/DCT/DWT-DCT) are most resilient to which attacks?

## Experimental Setup

- **Dataset:** 100 images from Lexica (512×704, diverse content)
- **Fragments:** K=8 per image, multi-domain embedding
- **Perturbation budgets:** ε = 4/255 (~0.0157) and ε = 8/255 (~0.0314)
- **Detection threshold:** Fisher's method combined p-value < 0.001
- **Detection mode:** Image-level (requires original image), with phase correlation alignment for cropped images

### Fragment Layout (K=8)

| Fragment | Strategy | Domain | Robust To |
|:---:|----------|--------|-----------|
| F0 | pixel | q1 (top-left) | JPEG, noise |
| F1 | pixel | q2 (top-right) | JPEG, noise |
| F2 | pixel | q3 (bottom-left) | JPEG, noise |
| F3 | pixel | q4 (bottom-right) | JPEG, noise |
| F4 | pixel | center | JPEG, noise |
| F5 | DCT | low freq [1-12] | JPEG, resize |
| F6 | DCT | mid freq [12-35] | mild attacks |
| F7 | DWT-DCT L2 | low freq [1-12] | resize, blur |

## Results

### Overall Detection Rate and Post-Transform Image Quality (100 images)

The table shows detection rate alongside the image quality **after** the transformation is applied. Post-PSNR and Post-SSIM measure how much the attack itself distorts the image (comparing transformed image against the original). Lower post-PSNR means the attack is more destructive to the image content.

#### ε = 4/255 (FP PSNR = 51.6 dB)

| Transform | Det. Rate | Mean r | Post-PSNR | Post-SSIM |
|-----------|:---------:|:------:|:---------:|:---------:|
| **No transform** | 100% | 0.5421 | 51.6 dB | 0.9965 |
| | | | | |
| **JPEG compression** | | | | |
| Q=95 | 100% | 0.0458 | 45.7 dB | 0.9926 |
| Q=80 | 100% | 0.0167 | 41.3 dB | 0.9849 |
| Q=60 | 100% | 0.0054 | 34.3 dB | 0.9232 |
| Q=40 | 100% | 0.0088 | 32.6 dB | 0.8952 |
| | | | | |
| **Resize (down + back)** | | | | |
| 75% | 100% | 0.0242 | 33.4 dB | 0.9363 |
| 50% | 98% | 0.0063 | 30.1 dB | 0.8695 |
| | | | | |
| **Center crop (no resize)** | | | | |
| 80% area | 100% | 0.2760 | 51.5 dB | 0.9965 |
| 60% area | 100% | 0.2719 | 51.2 dB | 0.9964 |
| 40% area | 100% | 0.2594 | 50.8 dB | 0.9962 |
| | | | | |
| **Random crop (no resize)** | | | | |
| 80% area | 100% | 0.2683 | 51.5 dB | 0.9965 |
| 60% area | 100% | 0.2453 | 51.2 dB | 0.9964 |
| 40% area | 100% | 0.2150 | 51.0 dB | 0.9963 |
| | | | | |
| **Crop + resize back** | | | | |
| Center 80% + resize | 0% | 0.0000 | 15.2 dB | 0.4202 |
| Center 60% + resize | 0% | 0.0000 | 12.9 dB | 0.3691 |
| Center 40% + resize | 0% | 0.0000 | 11.5 dB | 0.3401 |
| Random 80% + resize | 0% | -0.0001 | 14.4 dB | 0.4073 |
| Random 60% + resize | 0% | 0.0000 | 12.3 dB | 0.3578 |
| Random 40% + resize | 0% | 0.0000 | 11.2 dB | 0.3374 |
| | | | | |
| **Gaussian noise** | | | | |
| σ=0.01 | 100% | 0.1435 | 39.7 dB | 0.9483 |
| σ=0.05 | 100% | 0.0295 | 26.2 dB | 0.5349 |
| σ=0.10 | 100% | 0.0147 | 20.4 dB | 0.3001 |
| | | | | |
| **Gaussian blur** | | | | |
| r=0.5 | 100% | 0.1420 | 39.6 dB | 0.9849 |
| r=1.0 | 100% | 0.0224 | 31.7 dB | 0.9100 |
| r=2.0 | **46%** | 0.0022 | 27.6 dB | 0.7877 |
| | | | | |
| **Color adjustment** | | | | |
| Brightness +20% | 100% | 0.0577 | 21.8 dB | 0.9614 |
| Brightness -20% | 100% | 0.0410 | 20.4 dB | 0.9563 |
| Contrast +20% | 100% | 0.0577 | 27.7 dB | 0.9209 |
| Contrast -20% | 100% | 0.0411 | 26.7 dB | 0.9549 |
| | | | | |
| **Compound attacks** | | | | |
| JPEG Q=60 + center crop 80% | 100% | 0.0043 | 33.8 dB | 0.9182 |
| JPEG Q=40 + random crop 60% | **87%** | 0.0026 | 32.2 dB | 0.8938 |
| JPEG Q=60 + crop resize 80% | 0% | 0.0000 | 15.2 dB | 0.4202 |
| JPEG Q=40 + crop resize 60% | 0% | 0.0000 | 12.2 dB | 0.3573 |

#### ε = 8/255 (FP PSNR = 45.6 dB)

| Transform | Det. Rate | Mean r | Post-PSNR | Post-SSIM |
|-----------|:---------:|:------:|:---------:|:---------:|
| **No transform** | 100% | 0.5419 | 45.6 dB | 0.9861 |
| | | | | |
| **JPEG compression** | | | | |
| Q=95 | 100% | 0.0928 | 44.5 dB | 0.9891 |
| Q=80 | 100% | 0.0346 | 41.1 dB | 0.9835 |
| Q=60 | 100% | 0.0144 | 34.2 dB | 0.9220 |
| Q=40 | 100% | 0.0153 | 32.5 dB | 0.8948 |
| | | | | |
| **Resize (down + back)** | | | | |
| 75% | 100% | 0.0457 | 33.4 dB | 0.9343 |
| 50% | 100% | 0.0126 | 30.1 dB | 0.8687 |
| | | | | |
| **Center crop (no resize)** | | | | |
| 80% area | 100% | 0.2758 | 45.5 dB | 0.9862 |
| 60% area | 100% | 0.2718 | 45.2 dB | 0.9860 |
| 40% area | 100% | 0.2593 | 44.8 dB | 0.9850 |
| | | | | |
| **Random crop (no resize)** | | | | |
| 80% area | 100% | 0.2682 | 45.5 dB | 0.9862 |
| 60% area | 100% | 0.2452 | 45.2 dB | 0.9858 |
| 40% area | 100% | 0.2149 | 45.0 dB | 0.9855 |
| | | | | |
| **Crop + resize back** | | | | |
| Center 80% + resize | 0% | 0.0000 | 15.2 dB | 0.4172 |
| Center 60% + resize | 0% | 0.0000 | 12.9 dB | 0.3664 |
| Center 40% + resize | 0% | 0.0000 | 11.5 dB | 0.3376 |
| Random 80% + resize | 0% | -0.0001 | 14.4 dB | 0.4045 |
| Random 60% + resize | 0% | 0.0000 | 12.3 dB | 0.3553 |
| Random 40% + resize | 0% | 0.0000 | 11.2 dB | 0.3350 |
| | | | | |
| **Gaussian noise** | | | | |
| σ=0.01 | 100% | 0.2603 | 39.0 dB | 0.9394 |
| σ=0.05 | 100% | 0.0588 | 26.1 dB | 0.5333 |
| σ=0.10 | 100% | 0.0293 | 20.4 dB | 0.2997 |
| | | | | |
| **Gaussian blur** | | | | |
| r=0.5 | 100% | 0.2403 | 38.9 dB | 0.9757 |
| r=1.0 | 100% | 0.0358 | 31.6 dB | 0.9076 |
| r=2.0 | **91%** | 0.0041 | 27.6 dB | 0.7872 |
| | | | | |
| **Color adjustment** | | | | |
| Brightness +20% | 100% | 0.1088 | 21.8 dB | 0.9490 |
| Brightness -20% | 100% | 0.0781 | 20.4 dB | 0.9497 |
| Contrast +20% | 100% | 0.1082 | 27.6 dB | 0.9095 |
| Contrast -20% | 100% | 0.0782 | 26.6 dB | 0.9481 |
| | | | | |
| **Compound attacks** | | | | |
| JPEG Q=60 + center crop 80% | 100% | 0.0087 | 33.8 dB | 0.9173 |
| JPEG Q=40 + random crop 60% | 100% | 0.0052 | 32.2 dB | 0.8931 |
| JPEG Q=60 + crop resize 80% | 0% | 0.0000 | 15.2 dB | 0.4197 |
| JPEG Q=40 + crop resize 60% | 0% | 0.0000 | 12.2 dB | 0.3570 |

### Imperceptibility (Fingerprint Only, Before Any Attack)

| ε | FP PSNR | FP SSIM | Proposal Target |
|---|---------|---------|:-:|
| 4/255 | **51.6 dB** | 0.9965 | > 40 dB, > 0.95 ✅ |
| 8/255 | **45.6 dB** | 0.9861 | > 40 dB, > 0.95 ✅ |

Both ε values exceed the proposal's quality targets by a wide margin. The fingerprint is imperceptible to human viewers.

### Observations on Post-Transform Quality

- **Crop without resize** preserves near-original quality (Post-PSNR > 50 dB) — the attack barely changes the image, and fingerprint is perfectly detectable.
- **Crop + resize** destroys image quality severely (Post-PSNR = 11-15 dB, SSIM = 0.33-0.42) — the interpolation creates massive artifacts. Fingerprint detection fails, but the image itself is heavily degraded.
- **Heavy noise (σ=0.10)** and **strong blur (r=2.0)** reduce image quality to Post-PSNR < 28 dB — these are destructive attacks that degrade the image's usability for training.
- **JPEG Q=40** drops quality to 32.6 dB but fingerprint still survives at 100% — the fingerprint is more resilient than the image content itself.

### Per-Fragment Survival Analysis

Under challenging attacks, different fragments show distinct robustness profiles:

**Gaussian blur r=2.0 (ε=8/255, 91% overall detection):**

| Fragment | Strategy | Survival |
|:---:|----------|:---:|
| F0-F4 | pixel (spatial regions) | 15-31% |
| F5 | DCT low freq | **68%** |
| F6 | DCT mid freq | 0% |
| F7 | DWT-DCT L2 low freq | **73%** |

**Insight:** Low-frequency fragments (F5, F7) survive blur much better than pixel or mid-frequency fragments. This validates the multi-domain design — even when pixel fragments are destroyed, frequency-domain fragments carry the detection.

**JPEG Q=40 + random crop 60% (ε=4/255, 87% detection):**

| Fragment | Strategy | Survival |
|:---:|----------|:---:|
| F0-F4 | pixel (spatial regions) | 14-22% |
| F5 | DCT low freq | **59%** |
| F6 | DCT mid freq | 7% |
| F7 | DWT-DCT L2 low freq | 0% |

**Insight:** Under compound attack, DCT low-frequency fragment is the last survivor. The per-fragment independence ensures that even with most fragments destroyed, Fisher's method can still aggregate enough evidence.

**Resize 50% (ε=4/255, 98% detection):**

| Fragment | Strategy | Survival |
|:---:|----------|:---:|
| F0-F4 | pixel (spatial regions) | 40-56% |
| F5 | DCT low freq | **95%** |
| F6 | DCT mid freq | 20% |
| F7 | DWT-DCT L2 low freq | **87%** |

**Insight:** Low-frequency fragments are highly resilient to resize. DCT low freq achieves 95% survival even at 50% downscale.

## Key Findings

### 1. Strong robustness for non-geometric attacks
JPEG compression (all quality levels), Gaussian noise (all intensities), brightness/contrast changes, and mild blur achieve **100% detection rate** at both ε values. The multi-domain fragmentation ensures reliable detection under these common transformations.

### 2. Crop robustness depends on resize behavior
- **Crop without resize: 100% detection** — Phase correlation alignment precisely locates the cropped region in the original image, enabling perfect fragment extraction. Works for both center and random crop, even at 40% area retention.
- **Crop with resize: 0% detection** — Bilinear interpolation during resize destroys the pixel-level alignment that all current extraction methods depend on. This is a fundamental limitation of image-level detection.

### 3. ε tradeoff is favorable
Increasing ε from 4/255 to 8/255 improves detection in challenging scenarios:
- Blur r=2.0: 46% → 91%
- JPEG Q=40 + random crop: 87% → 100%
- Resize 50%: 98% → 100%

While maintaining excellent imperceptibility (PSNR still > 45 dB, well above the 40 dB target).

### 4. Multi-domain design validated
The per-fragment survival analysis confirms that different strategies survive different attacks:
- **Pixel fragments:** Best for JPEG, noise, color changes
- **DCT low-freq fragments:** Best for resize and blur
- **DWT-DCT L2 low-freq fragments:** Best for resize and moderate blur
- **Mid/high-freq fragments:** Destroyed by strong blur and JPEG

This justifies the multi-domain design over single-domain approaches.

### 5. False positive rate = 0%
Tested with 10 wrong keys: all returned p > 0.001, zero false detections. The cryptographic key binding ensures no false attribution.

## Limitations

1. **Crop + resize is undetectable** at the image level. All extraction methods (pixel, DCT, DWT-DCT) require pixel-aligned comparison between original and suspect image. Bilinear interpolation after crop destroys this alignment irreversibly.

2. **Strong blur (r ≥ 2.0) degrades detection** at low ε. This is expected — heavy blur removes high-frequency information where much of the fingerprint energy resides.

3. **Image-level detection requires the original image.** Without the original, there is no reference to compute the residual against.

## Next Steps

### Immediate: Memorization Test (Step 2.2)
The image-level robustness test validates that our fingerprints survive common transformations. The critical next question is: **do these fingerprints survive model training?**

- Fine-tune Stable Diffusion v1.5 on 5,000 fingerprinted COCO images
- Generate images from the fine-tuned model
- Test whether generated images carry detectable fingerprint signal
- Compare fingerprint ratios: 100%, 50%, 10% of training data
- Slurm job prepared: `scripts/slurm/run_memorization.sh`

### Short-term: Model-Level Detection (Component 2)
Develop detection methods that do not require the original image:
- **White-box:** Correlate model weights/activations with known fingerprint patterns
- **Black-box:** Generate images, extract residuals via denoising (BM3D/DnCNN), test correlation
- This is the **fundamental solution** to crop+resize — detection bypasses image alignment entirely

### Medium-term: Inherent Backdoor Alignment (Step 2.1b)
Current fingerprints are random Gaussian perturbations. The proposal calls for aligning them with naturally learnable features:
- Train a proxy model, extract feature directions from intermediate layers
- Project fragments onto the learnable subspace
- This should increase memorization rate during model training

### Long-term: Full Evaluation
- Multi-model testing: SD v1.5, SDXL, SD 3.5, FLUX.1
- Baseline comparisons: WOUAF, PALADIN, Radioactive Data
- Robustness against model-level attacks: fine-tuning, pruning, distillation
