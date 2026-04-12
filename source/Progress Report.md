# Progress Report: Invisible Fragmented Cryptographic Fingerprints

## Project Overview

**Goal:** Embed fragmented cryptographic fingerprints into images to detect unauthorized AI model training. Only the holder of the master secret key can verify whether a model was trained on protected data.

**Paper:** *Invisible Fragmented Cryptographic Fingerprints for Protecting Images from Unauthorized AI Training* (Ma & Li, UMass Amherst)

---

## Current Status: Step 2.1 — Basic Fingerprint Generation Pipeline ✅

### What Has Been Built

A modular Python pipeline implementing the core fingerprint generation and embedding described in Step 2 of the proposal.

#### Architecture

```
Master Key (k, 256-bit) + Image ID
        ↓
    HKDF-SHA256 → K independent sub-keys
        ↓
    AES-256-CTR CSPRNG → uniform byte stream
        ↓
    Box-Muller transform → Gaussian samples
        ↓
    Scale by ε → fragment δᵢ
        ↓
    Aggregate: x' = x + Σ(αᵢ · δᵢ)  [pixel / DCT / DWT-DCT]
        ↓
    Fingerprinted image x'
```

#### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| Key derivation | `src/keygen.py` | Master key generation, HKDF-SHA256 per-image per-fragment sub-key derivation |
| Fragment generation | `src/fragment.py` | AES-256-CTR → Box-Muller → Gaussian perturbation, deterministic and reproducible |
| Embedding | `src/embed.py` | Three strategies: pixel-space, DCT mid-frequency, DWT-DCT hybrid |
| Verification | `src/verify.py` | Per-fragment Pearson correlation + Fisher's method p-value combination |
| Quality metrics | `src/metrics.py` | PSNR, SSIM, MS-SSIM, LPIPS |
| Pipeline | `src/pipeline.py` | End-to-end orchestration (fingerprint + verify + file I/O) |

#### CLI Scripts

| Script | Purpose |
|--------|---------|
| `scripts/fingerprint_images.py` | Batch fingerprint a directory of images |
| `scripts/verify_fingerprint.py` | Verify fingerprint presence (single or batch) |
| `scripts/evaluate_quality.py` | Sweep ε × strategy for quality metrics table |
| `scripts/download_datasets.py` | Download CIFAR-10 / CelebA-HQ / COCO / WikiArt |

#### Test Suite

25 unit tests covering all modules — all passing:

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestKeyGen` | 6 | Key randomness, determinism, per-image/fragment uniqueness, save/load |
| `TestFragmentGeneration` | 6 | Shape, determinism, Gaussian distribution, ε scaling, fragment independence |
| `TestEmbedding` | 5 | Pixel/DCT/DWT-DCT shape & range, pixel roundtrip recovery |
| `TestVerification` | 4 | Correct key detects, wrong key fails, wrong image_id fails, Fisher combine |
| `TestMetrics` | 2 | Identical image metrics, quality above threshold |
| `TestPipeline` | 2 | In-memory and file I/O end-to-end |

### Preliminary Results

Single image demo (DWT-DCT strategy, ε = 4/255, K = 8 fragments):

| Metric | Value | Target |
|--------|-------|--------|
| PSNR | 60.43 dB | > 40 dB ✅ |
| SSIM | 0.9997 | > 0.98 ✅ |
| L2 perturbation norm | 4.92 | — |
| L∞ perturbation norm | 0.027 | — |
| Verification p-value | ≈ 0 | < 0.001 ✅ |
| Mean Pearson r | 0.054 | > 0 ✅ |
| Fisher χ² statistic | 11052.41 | — |

**Observation:** At ε = 4/255 with DWT-DCT embedding, the fingerprint is completely imperceptible (PSNR > 60dB) and deterministically verifiable with the correct key. Wrong key or wrong image ID both fail detection, confirming cryptographic security properties.

### Security Properties Verified

- ✅ **Determinism:** Same key + image ID → identical fingerprint (reproducible)
- ✅ **Per-image uniqueness:** Different images receive different fingerprint compositions
- ✅ **Fragment independence:** Cross-fragment Pearson |r| < 0.1
- ✅ **Cryptographic unlinkability:** Wrong key → detection fails (p ≫ 0.001)
- ✅ **Image ID binding:** Wrong image ID → detection fails

---

## What's Missing vs. The Full Proposal

### Step 2: Core Fingerprint Generation (partially done)

| Sub-task | Status | Notes |
|----------|--------|-------|
| Basic fragment generation pipeline | ✅ Done | AES-CTR + Box-Muller + 3 embedding strategies |
| Multi-dataset quality sweep (ε × strategy) | 🔲 TODO | Need to run `evaluate_quality.py` on CIFAR-10, CelebA-HQ |
| **Inherent backdoor alignment** | 🔲 TODO | Align δᵢ with naturally learnable features via proxy model |
| **Memorization-optimized perturbations** | 🔲 TODO | max_δᵢ E[gᵢ(M)] optimization loop |
| **TensorCrypt GPU acceleration** | 🔲 TODO | Port AES/ChaCha20 to neural network computation graph |
| Iterate on fragment design | 🔲 TODO | Refine based on memorization experiments |

### Step 3: Detection Framework (not started)

| Sub-task | Status | Notes |
|----------|--------|-------|
| White-box detection | 🔲 TODO | Correlate model weights/activations with fingerprints |
| Black-box detection | 🔲 TODO | Query model, extract residuals, test correlation |
| Latent optimization | 🔲 TODO | Optimize latent codes to elicit fingerprint responses |
| Empirical null distribution | 🔲 TODO | Train ~20 clean models for calibration |
| Statistical hypothesis testing (full) | 🔲 TODO | Wilcoxon, trimmed means, empirical Bayes |

### Step 4: Evaluation (not started)

| Sub-task | Status | Notes |
|----------|--------|-------|
| Baseline comparisons (WOUAF, PALADIN, Radioactive Data, etc.) | 🔲 TODO | — |
| Robustness attacks (JPEG, cropping, blur, fine-tuning, pruning) | 🔲 TODO | — |
| Ablation studies (K, ε, strategy, aggregation) | 🔲 TODO | — |

---

## Roadmap

### Phase 1: Fingerprint Quality Validation (current → 1-2 weeks)

**Goal:** Establish the operating point (best ε × strategy) across real datasets.

- [ ] Download and prepare datasets (CIFAR-10, CelebA-HQ, COCO subset)
- [ ] Run `evaluate_quality.py` on each dataset with full ε sweep: {2/255, 4/255, 6/255, 8/255, 12/255, 16/255}
- [ ] Compare pixel vs. DCT vs. DWT-DCT on quality metrics table
- [ ] Add robustness pre-checks: JPEG compression survival, crop survival at image level (before any model training)
- [ ] **Deliverable:** Quality metrics table, operating point selection document

### Phase 2: Memorization Proof-of-Concept (weeks 2-4)

**Goal:** Confirm fingerprints survive model training and appear in generated outputs.

- [ ] Train unconditional DDPM on CIFAR-10 (fingerprinted vs. clean)
- [ ] Fine-tune Stable Diffusion v1.5 on CelebA-HQ subset (fingerprinted vs. clean)
- [ ] Generate 10K samples from each trained model
- [ ] Compute Pearson correlation between generated-image residuals and known fingerprints
- [ ] Fingerprinting ratio sweep: 100%, 50%, 10% of training data
- [ ] **Deliverable:** Correlation signal confirmation (does training memorize the fingerprint?)

### Phase 3: Inherent Backdoor Alignment (weeks 4-6)

**Goal:** Make fingerprints "model-friendly" so they are memorized more strongly.

- [ ] Train proxy model on clean data, extract feature directions from intermediate layers
- [ ] Project cryptographic fragments onto learnable feature subspace
- [ ] Compare memorization: random δᵢ vs. aligned δᵢ
- [ ] Implement memorization optimization: max_δᵢ E[gᵢ(M)] with perceptual constraint
- [ ] **Deliverable:** Improved fragment design with higher correlation in generated outputs

### Phase 4: Detection Framework (weeks 6-9)

**Goal:** Build the full statistical detection pipeline.

- [ ] White-box: weight/activation correlation with fingerprint directions
- [ ] Black-box: generate → denoise → extract residual → test correlation
- [ ] Latent optimization: optimize z to maximize fingerprint presence in G(z)
- [ ] Build empirical null distribution (~20 clean models)
- [ ] Implement full hypothesis testing: Fisher's method + Wilcoxon + trimmed means
- [ ] **Deliverable:** Detection pipeline with calibrated thresholds, AUROC > 0.95

### Phase 5: Robustness & Baselines (weeks 9-12)

**Goal:** Benchmark against attacks and prior work.

- [ ] Robustness matrix: JPEG Q∈{95,80,60,40} × crop {80%,60%,40%} × noise × blur × fine-tuning × pruning
- [ ] Baseline comparisons: WOUAF, PALADIN, Radioactive Data, DiffusionShield, Glaze/Nightshade
- [ ] Scale to SDXL, SD 3.5, FLUX.1
- [ ] Ablation: K ∈ {4, 8, 16}, ε sweep, aggregation rules
- [ ] **Deliverable:** Full evaluation tables for paper

### Phase 6: GPU Acceleration & Paper (weeks 12+)

- [ ] TensorCrypt integration for large-scale fingerprinting (if needed for final experiments)
- [ ] Draft paper with all results

---

## Environment

- **Conda env:** `fingerprint` (Python 3.12, PyTorch 2.x, CUDA 12.1)
- **Data root:** `/project/pi_shiqingma_umass_edu/mingzheli/datasets/`
- **Codebase:** `/home/mingzhel_umass_edu/cryptographic_fingerprint/`

---

## Key Design Decisions

1. **DWT-DCT as default strategy** — Literature suggests >93% fingerprint survival under JPEG Q=20 + 25% cropping, compared to ~60% for pixel-space.
2. **K = 8 fragments** — Balances redundancy (robustness to partial removal) with computational cost.
3. **ε = 4/255 default** — PSNR > 60dB confirms strong imperceptibility; may increase to 8/255 if memorization is weak.
4. **Fisher's method for p-value combination** — Standard approach for combining independent test statistics; will add Wilcoxon and trimmed means later.
5. **HKDF-SHA256 for key derivation** — Industry standard, deterministic, supports per-image per-fragment uniqueness.
