# Warm-up Phase 

## Part 1: Environment Setup


## 1a. Candidate Generative Models

The four candidate models span two architectural families — U-Net-based latent diffusion (SD v1.5, SDXL) and Diffusion Transformer / rectified flow (SD 3.5, FLUX.1) — covering the full spectrum of models an unauthorized trainer is likely to use. Together they vary in backbone (U-Net vs. DiT/MMDiT), training objective (ε-prediction vs. velocity/flow matching), conditioning mechanism (cross-attention vs. bidirectional MMDiT attention), and scale (860 M to 12 B parameters).

### (1) Stable Diffusion v1.5

| Attribute | Detail |
|-----------|--------|
| **Developer** | Stability AI / CompVis / Runway (2022) |
| **Architecture** | U-Net denoiser + CLIP ViT-L/14 text encoder + KL-regularized VAE |
| **Training objective** | ε-prediction (DDPM) |
| **Resolution** | 512×512 |
| **Parameters** | 860 M (U-Net) |
| **Latent space** | 4-channel, 64×64 (8× downsampling) |
| **HuggingFace ID** | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| **License** | CreativeML OpenRAIL-M |

SD v1.5 remains the most widely fine-tuned open-weight text-to-image model, with thousands of community checkpoints and LoRA adapters on platforms like Civitai. It is the single most likely model to be used for unauthorized fine-tuning on protected data, making it the essential first evaluation target. Its relatively small size also allows fast experimental iteration.



### (2) Stable Diffusion XL (SDXL 1.0)

| Attribute | Detail |
|-----------|--------|
| **Developer** | Stability AI (Jul 2023) |
| **Architecture** | Larger U-Net + dual text encoder (CLIP ViT-L/14 + OpenCLIP ViT-bigG/14) + SDXL-VAE + optional refiner |
| **Training objective** | ε-prediction (DDPM) |
| **Resolution** | 1024×1024 |
| **Parameters** | ~3.5 B (U-Net base + refiner) |
| **Latent space** | 4-channel, 128×128 |
| **HuggingFace ID** | `stabilityai/stable-diffusion-xl-base-1.0` |
| **License** | CreativeML OpenRAIL-M |

SDXL represents the high-capacity end of the U-Net family. Its dual text encoder, higher resolution, and 4× larger parameter count test whether fingerprint signals survive dilution in a substantially bigger model. The optional two-stage pipeline (base + refiner) also tests fingerprint persistence through cascaded generation.



### (3) Stable Diffusion 3.5 (SD 3.5)

| Attribute | Detail |
|-----------|--------|
| **Developer** | Stability AI (Oct 2024) |
| **Architecture** | Multimodal Diffusion Transformer (MMDiT / MMDiT-X) + 3 text encoders (CLIP-L, OpenCLIP-bigG, T5-XXL) + VAE; QK-normalization for training stability |
| **Training objective** | Rectified flow |
| **Resolution** | 1024×1024 (1 MP) |
| **Parameters** | 8.1 B (Large) / 2.5 B (Medium) |
| **Key innovation** | Bidirectional text-image attention — text and image tokens jointly attend to each other inside each MMDiT block, unlike U-Net cross-attention where text conditions the image unidirectionally |
| **License** | Stability AI Community License (free for research and commercial use < $1 M revenue) |

SD 3.5 marks the architectural shift from U-Net to DiT. The MMDiT's bidirectional attention may process fingerprint perturbations differently than U-Net cross-attention, making it essential for validating architecture-agnostic detection. Three variants are available:

| Variant | Params | Steps | VRAM (excl. text enc.) | HuggingFace ID |
|---------|--------|-------|------------------------|----------------|
| **SD 3.5 Large** | 8.1 B | 28–50 | ~18 GB | `stabilityai/stable-diffusion-3.5-large` |
| **SD 3.5 Large Turbo** | 8.1 B | 4 | ~18 GB | `stabilityai/stable-diffusion-3.5-large-turbo` |
| **SD 3.5 Medium** | 2.5 B | 28–50 | ~9.9 GB | `stabilityai/stable-diffusion-3.5-medium` |



### (4) FLUX.1 (Black Forest Labs)

| Attribute | Detail |
|-----------|--------|
| **Developer** | Black Forest Labs (Aug 2024) — founded by the original creators of Stable Diffusion (Rombach, Blattmann, Esser) |
| **Architecture** | Hybrid Diffusion Transformer: **Double-Stream blocks** (separate image/text paths with cross-attention) → **Single-Stream blocks** (merged sequence with full self-attention); dual text encoders (CLIP-L + T5-XXL); rotary positional embeddings (RoPE) |
| **Training objective** | Rectified flow — predicts velocity vectors along straight-line trajectories from noise to data, rather than ε-prediction |
| **Resolution** | 1024×1024 (supports variable aspect ratios) |
| **Parameters** | **12 B** |
| **Key innovation** | Flow matching replaces DDPM-style noise prediction; double→single stream transition enables separate-then-joint processing of modalities |

FLUX.1 is the largest open-weight text-to-image model and uses a fundamentally different training objective (velocity prediction under rectified flow). Fingerprints designed for ε-prediction may behave differently under flow matching, making FLUX.1 the most challenging and informative test case. Three variants are available:

| Variant | Steps | License | HuggingFace ID |
|---------|-------|---------|----------------|
| **FLUX.1 [dev]** | 50 | Non-commercial (open weights) | `black-forest-labs/FLUX.1-dev` |
| **FLUX.1 [schnell]** | 1–4 | **Apache 2.0** (fully open) | `black-forest-labs/FLUX.1-schnell` |
| **FLUX.1 [pro]** | Variable | API-only (commercial) | API at `api.bfl.ml` |




### Architecture Comparison

| | SD v1.5 | SDXL | SD 3.5 | FLUX.1 |
|---|---------|------|--------|--------|
| **Backbone** | U-Net | U-Net (larger) | MMDiT / MMDiT-X | Hybrid DiT (double+single stream) |
| **Training objective** | ε-prediction | ε-prediction | Rectified flow | Rectified flow (velocity) |
| **Text encoders** | CLIP ViT-L/14 | CLIP-L + OpenCLIP-bigG | CLIP-L + OpenCLIP-bigG + T5-XXL | CLIP-L + T5-XXL |
| **Conditioning** | Cross-attention | Cross-attention | Bidirectional MMDiT attention | Double→Single stream attention |
| **Parameters** | 860 M | ~3.5 B | 2.5–8.1 B | 12 B |
| **Resolution** | 512×512 | 1024×1024 | 1024×1024 | 1024×1024 |
| **Positional encoding** | Learned | Learned | Learned + crop augmentation | RoPE |

The four models cover the two major axes of variation that affect fingerprint behavior: (1) U-Net vs. DiT backbone — different feature extraction and gradient flow patterns may alter how perturbation signals are memorized during training; (2) ε-prediction vs. velocity/flow matching — different loss objectives mean the model optimizes different functions of the fingerprinted input, potentially amplifying or attenuating fingerprint signals differently

---

## 1b. Datasets

### 1b.1 Dataset Selection Matrix

| Dataset | Size | Resolution | Content Type | Experimental Purpose | Access |
|---------|------|------------|--------------|----------------------|--------|
| **CIFAR-10** | 60 K | 32×32 | General objects | Fast prototyping; fingerprint embedding feasibility | `torchvision.datasets.CIFAR10(download=True)` |
| **CelebA-HQ** | 30 K | 1024×1024 | Faces | Medium-scale training; high-resolution fingerprint testing | `datasets.load_dataset("mattymchen/celeba-hq")` |
| **COCO 2017** | 118 K train | Variable | Multi-object scenes | Text-to-image conditional generation fingerprint detection | `datasets.load_dataset("HuggingFaceM4/COCO")` |
| **ImageNet-1K** | 1.28 M | Variable | 1000-class objects | Large-scale experiments; fingerprint dilution ratio testing | `datasets.load_dataset("ILSVRC/imagenet-1k")` (registration required) |
| **WikiArt** | ~80 K | Variable | Artworks | Fingerprint robustness in style-transfer scenarios | `datasets.load_dataset("huggan/wikiart")` |
| **LAION-Aesthetics** | ~600 M URLs | Variable | Web images | Fingerprint survivability in very-large-scale training | `img2dataset` download pipeline |

```bash
pip install datasets webdataset img2dataset Pillow torchvision
```

### 1b.2 Dataset Usage Strategy

- **Fingerprint injection ratio experiments:** Set up four injection ratios—100 %, 50 %, 10 %, 1 %—on the same dataset.
- **Dataset scale gradient:** CIFAR-10 (60 K) → CelebA-HQ (30 K @ high resolution) → COCO (118 K) → ImageNet (1.28 M).
- **Control group:** For every experiment, maintain one model trained on a completely clean dataset as the H₀ null baseline.

---

## 1c. Evaluation Metrics

### 1c.1 Fingerprint Imperceptibility Metrics

Measure whether image quality is preserved after fingerprint embedding.

| Metric | Target | Meaning | Library & Usage |
|--------|--------|---------|-----------------|
| **PSNR** | > 40 dB | Peak signal-to-noise ratio; > 38 dB is "excellent" | `skimage.metrics.peak_signal_noise_ratio(ref, test, data_range=255)` |
| **SSIM** | > 0.95 | Structural similarity; 1.0 = identical | `pytorch_msssim.ssim(img1, img2, data_range=1.0)` (GPU-accelerated, differentiable) |
| **MS-SSIM** | > 0.95 | Multi-scale structural similarity; more robust than SSIM | `pytorch_msssim.ms_ssim(img1, img2, data_range=1.0)` |
| **LPIPS** | < 0.10 | Learned perceptual similarity; lower is better | `lpips.LPIPS(net='alex').cuda()` — input Nx3xHxW, range [-1, 1] |
| **FID (clean vs. fingerprinted)** | < 1.0 | Distributional distance between fingerprinted and original sets | `clean_fid.fid.compute_fid("path/clean", "path/fp")` |

```bash
pip install pytorch-msssim lpips==0.1.4 clean-fid scikit-image piq
```

### 1c.2 Detection Efficacy Metrics

Measure whether the fingerprint signal can be detected in a trained model.

| Metric | Target | Application |
|--------|--------|-------------|
| **Pearson correlation** (generated-image residuals vs. fingerprint pattern) | Significantly above the null distribution | Core metric for both white-box and black-box detection |
| **Cosine similarity** (model weights/activations vs. fingerprint direction) | > 3σ deviation from null mean | White-box detection (analogous to Radioactive Data) |
| **Spectral coherence** | Significant DFT cross-correlation | Frequency-domain black-box detection |
| **AUROC** (end-to-end detection pipeline) | > 0.95 | Overall detection performance |
| **TPR @ FPR = 1 %** | > 0.90 | Strict operational standard for real-world deployment |
| **p-value** (hypothesis test) | < 0.001 | Statistical significance claim |

### 1c.3 Model Utility Metrics

Ensure that fingerprint embedding does not degrade model training quality.

| Metric | Target | Library |
|--------|--------|---------|
| **FID** (generation quality) | Gap between fingerprinted-model FID and clean-model FID < 5 % | `clean-fid` or `pytorch-fid` |
| **Training loss curves** | Fingerprinted-model convergence curve approximately overlaps with clean model | `wandb` / `tensorboard` |
| **IS (Inception Score)** | Reference comparison | `piq.inception_score()` |

### 1c.4 Robustness Evaluation Matrix

Detection success rate for each attack × each embedding strategy:

| Attack Type | Parameter Range | Expected Threat Level |
|-------------|-----------------|----------------------|
| JPEG compression | Q ∈ {95, 80, 60, 40} | **High** (Q ≤ 60 is destructive for pixel-domain embedding) |
| Cropping | Retain {80 %, 60 %, 40 %} area | **Medium** (fragmented design can resist) |
| Gaussian noise | σ ∈ {0.01, 0.05, 0.1} | **Medium** |
| Resize + interpolation | Scale to 50 %, 75 % then restore | **Medium** |
| Color jitter | Brightness / contrast ± 0.2 | **Low** |
| Gaussian blur | σ ∈ {0.5, 1.0, 2.0} | **High** (destructive for high-frequency embedding) |
| Fine-tuning | Extra clean data, 10–50 % epochs | **Medium–High** |
| Model pruning | 30–70 % weight removal | **Medium** |



---

## 1d. Cryptographic Methods & Acceleration

### 1d.1 Cryptographic Primitive Selection

The core of fingerprint generation is deterministically deriving a pseudo-random perturbation sequence matching the image dimensions from a secret key k. The following three stream ciphers can all serve as CSPRNGs (Cryptographically Secure Pseudo-Random Number Generators):

| Primitive | Type | Key Length | Characteristics | Recommended Use |
|-----------|------|-----------|-----------------|-----------------|
| **AES-256-CTR** | Block cipher (CTR mode) | 256 bit | Industry standard; broad hardware acceleration (AES-NI) | Default choice; best compatibility |
| **ChaCha20** | Stream cipher | 256 bit + 96-bit nonce | High software efficiency; no hardware acceleration needed | Preferred for GPU scenarios |
| **Salsa20** | Stream cipher | 256 bit + 64-bit nonce | Predecessor of ChaCha20; slightly lower security margin | Fallback option |

```bash
pip install pycryptodome==3.23.0
# hashlib, secrets, os.urandom are Python built-ins
```

### 1d.2 Fingerprint Generation Pipeline

```
          Master Key (k)
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
Fragment 1  Fragment 2  ... Fragment m
    │          │          │
    ▼          ▼          ▼
AES-CTR     AES-CTR     AES-CTR
(nonce=1)   (nonce=2)   (nonce=m)
    │          │          │
    ▼          ▼          ▼
PRNG stream PRNG stream PRNG stream
    │          │          │
    ▼          ▼          ▼
Box-Muller → Gaussian → Scale to ε budget
    │          │          │
    ▼          ▼          ▼
  δ₁         δ₂       ... δₘ
    │          │          │
    └──────────┼──────────┘
               ▼
   x' = x + Σ αᵢ·δᵢ  (fingerprinted image)
```



### 1d.3 Three Embedding Domain Strategies

| Strategy | Domain | JPEG Resistance | Crop Resistance | Blur Resistance | Complexity |
|----------|--------|-----------------|-----------------|-----------------|------------|
| **Pixel-domain addition** | Spatial | ★★☆ | ★★★ | ★☆☆ | Low |
| **DCT mid-frequency embedding** | Frequency (8×8 block DCT) | ★★★ | ★★☆ | ★★☆ | Medium |
| **DWT-DCT hybrid** | Wavelet + frequency | ★★★ | ★★★ | ★★★ | High |

The **DWT-DCT hybrid strategy** is recommended as the primary approach: embed fingerprints in the DCT coefficients of the level-2 DWT LL subband for optimal overall robustness.

### 1d.4 GPU Acceleration: TensorCrypt

A key technical challenge of the project is **efficient fingerprint generation at scale**. Traditional CPU cryptographic operations are a bottleneck; TensorCrypt provides the solution.

**TensorCrypt** (Jin, Ma & Lin, NDSS 2025) translates cipher algorithms into neural network computation graphs for batch encryption on GPU:

| Cipher | Speedup over Existing GPU Implementations | Speedup over CPU |
|--------|-------------------------------------------|------------------|
| AES-256  | **4.09×** | Higher (CPU-dependent) |
| ChaCha20 | **5.44×** | Higher |
| Salsa20  | **5.06×** | Higher |

```bash
git clone https://github.com/OSUSecLab/TensorCrypt.git
cd TensorCrypt && pip install -r requirements.txt && cd ..
```

**Repository structure:**

```
TensorCrypt/
├── models/       # Pretrained checkpoints for each cipher
├── inference/    # Encryption/decryption inference scripts
├── src/          # Neural network DSL implementations
└── requirements.txt
```

**Important notes:**
- TensorCrypt is built on **TensorFlow**, while the main project uses PyTorch.
- Two integration paths:
  1. **Dual-framework coexistence:** Use TensorFlow only for cryptographic stream generation and PyTorch for training, bridging via NumPy arrays.
  2. **PyTorch port:** Translate TensorCrypt's computation graph into PyTorch operations (recommended long-term).

```bash
# If choosing the dual-framework path
pip install tensorflow[and-cuda]  # Must be CUDA-compatible with the PyTorch installation
```

### 1d.5 Scaled Production Pipeline Design

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Master Key  │────▶│  TensorCrypt │────▶│  Fingerprint │
│  + Image IDs │     │  (GPU batch  │     │  Perturbations│
│              │     │   AES/ChaCha)│     │  (GPU tensors)│
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Raw Images  │────▶│  Data Loader │────▶│  Fingerprinted│
│  (Dataset)   │     │  (PyTorch)   │     │  Images       │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                         ┌──────▼───────┐
                                         │  Diffusion   │
                                         │  Model Train │
                                         └──────────────┘
```

**Throughput estimates:**
- PyCryptodome on CPU: ~50 K images/min @ 256×256×3
- TensorCrypt on a single A100 GPU: **~250 K images/min** (5× speedup)
- For ImageNet-1K (1.28 M images, 8 fragments/image): CPU ~3.4 h → GPU **~41 min**

### 1d.6 Security Properties

| Property | Guarantee |
|----------|-----------|
| **Cryptographic unlinkability** | Without key k, fingerprints across different images cannot be linked |
| **Per-image uniqueness** | image_id participates in key derivation; each image's fingerprint combination is unique |
| **Fragment independence** | Each fragment uses an independent sub-key; an attacker cannot infer other fragments from one |
| **Verifiability** | Possessing k allows regeneration of all fingerprints for detection and verification |
| **Clustering resistance** | Without k, clustering analysis of fingerprint patterns is infeasible even with many fingerprinted images |

---

## Part 2: Literature review

This review covers **37 papers** across twelve categories, each selected for direct relevance to the cryptographic fingerprinting proposal.

### 2a. Classical image watermarking — foundations and limitations

**Cox et al., "Secure Spread Spectrum Watermarking for Multimedia" (IEEE TIP, 1997).** The seminal spread-spectrum watermarking paper embeds an i.i.d. Gaussian random vector into the most perceptually significant DCT components. Robust to JPEG, filtering, D/A conversion, and geometric transforms. Establishes the core principle the proposal exploits — imperceptible frequency-domain embedding — but was designed for direct extraction from media, not survival through neural network training.

**Barni et al., "Improved Wavelet-Based Watermarking Through Pixel-Wise Masking" (IEEE TIP, 2001).** Presents a DWT-based watermarking algorithm with pixel-wise perceptual masking that adapts to texture and luminance in all subbands. Achieves blind detection via correlation without the original image. Represents the best of classical wavelet-domain watermarking but shares the fundamental limitation: robustness is designed for signal processing attacks, not the nonlinear transformations of deep learning training pipelines.

**Cox, Miller & Bloom, *Digital Watermarking* (Morgan Kaufmann, 2002).** The comprehensive textbook formalizes watermark embedding as a communication problem and provides frameworks for evaluating robustness, capacity, and imperceptibility trade-offs. Essential background for understanding why classical methods fail under AI training — they assume a signal processing threat model, not gradient-based statistical learning.

### 2b. Data attribution and radioactive data — tracing through training

**Sablayrolles et al., "Radioactive Data: Tracing Through Training" (ICML 2020).** The closest precursor to the proposal. Makes imperceptible changes to training samples by aligning feature representations with a specific direction in feature space. Any model trained on radioactive data bears a detectable mark via statistical testing — detection at **p < 10⁻⁴** when only **1% of training data** is radioactive. The proposal extends this by using multiple fragmented cryptographic fingerprints instead of a single global marking direction, potentially offering greater capacity and resistance to removal.

**Wenger et al., "Data Isotopes for Data Provenance in DNNs" (PoPETs 2024).** Introduces "data isotopes" — special data points with out-of-distribution spurious features that create detectable correlations in any trained DNN. Unlike radioactive data, requires only **black-box query access** and no knowledge of the training process. Achieves discrimination among hundreds of different marks. Differs from the proposal in using visible features rather than cryptographic perturbations.

**Li et al., "Black-Box Dataset Ownership Verification via Backdoor Watermarking" (IEEE TIFS, 2023).** Inserts watermarking samples into datasets so any trained model learns a hidden association detectable via API queries. Introduces both targeted and untargeted variants with statistical testing. Shares the goal of dataset protection but relies on backdoor behavior (intentional misclassification), which the proposed cryptographic approach avoids.

### 2c. Unlearnable examples — making data unexploitable

**Huang et al., "Unlearnable Examples: Making Personal Data Unexploitable" (ICLR 2021, Spotlight).** Introduces error-minimizing noise — imperceptible perturbations that reduce training loss to near zero, causing models trained on them to achieve near-random test accuracy. Works in sample-wise and class-wise forms on CIFAR-10 and face recognition. Represents the "denial" approach to data protection (make data useless) versus the proposal's "detection" approach (allow training but embed traceable fingerprints). Known to be **fragile to adversarial training and strong augmentations**.

**Fu et al., "Robust Unlearnable Examples: Protecting Data Against Adversarial Learning" (ICLR 2022).** Addresses unlearnable examples' vulnerability to adversarial training by introducing robust error-minimizing noise that directly reduces adversarial training loss. Highlights the arms race in data protection — the proposal sidesteps this entirely by not trying to prevent learning but ensuring learning itself embeds detectable signatures.

**Fowl et al., "Adversarial Examples Make Strong Poisons" (NeurIPS 2021).** Demonstrates that targeted adversarial perturbations (TAP) can serve as potent data poisons for availability attacks. Effective at scale, connecting adversarial robustness to data poisoning. While TAP degrades model utility, the proposal preserves utility while embedding fingerprints — a more practical approach for content creators.

### 2d. Backdoor watermarking for dataset ownership

**Adi et al., "Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring" (USENIX Security 2018).** Pioneers using backdoor attacks as DNN watermarks. Creates a trigger set of abstract images with random labels, trains models to classify both normal and trigger inputs, and uses cryptographic commitments for ownership verification. Establishes the conceptual bridge between backdoors and watermarking — while Adi et al. watermark the *model*, the proposal watermarks the *data*.

**Tang et al., "Did You Train on My Dataset? Towards Public Dataset Protection with Clean-Label Backdoor Watermarking" (KDD Explorations, 2023).** Proposes clean-label backdoor watermarking where perturbed samples retain consistent labels, making them stealthy and resistant to anomaly detection. Adding **just 1% watermarking samples** injects a traceable function detectable via black-box API queries. Works across text, image, and audio. The proposal shares the clean-label property but embeds multiple fragmented cryptographic signatures rather than single trigger patterns.

**Li et al., "Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection" (NeurIPS 2023).** Explores untargeted backdoor watermarking where model behavior on triggers is non-deterministic rather than targeting a specific class. Reduces the security risks of traditional targeted backdoors. Aligns with the proposal's goal of producing statistically detectable but functionally benign signatures.

### 2e. Anti-neuron watermarking — the closest prior paradigm

**Zou et al., "Anti-Neuron Watermarking: Protecting Personal Data Against Unauthorized Neural Networks" (ECCV 2022).** Introduces the Personal Image Protection (PIP) problem and proposes specialized **linear color transformations** (LCT) that imprint watermark signatures into a model's neurons when trained on watermarked images. A third-party verifier detects unauthorized use by inferring the signature via loss analysis. This is **the most directly related prior work** — both embed imperceptible signatures that transfer to trained models. However, Zou et al. use a single global linear color transform, while the proposal uses multiple fragmented cryptographic perturbations, potentially offering greater capacity, robustness, and cryptographic verifiability.

**Shan et al., "Fawkes: Protecting Privacy Against Unauthorized Facial Recognition Models" (USENIX Security 2020).** Adds imperceptible pixel-level "cloaks" that shift image representations away from the person's true identity, causing trained models to learn incorrect features. While Fawkes targets privacy protection by disrupting learning, it shares the core mechanism of imperceptible perturbations affecting downstream training.

### 2f. Inherent backdoors — why fingerprints can persist

**Tao et al., "Exploring Inherent Backdoors in Deep Learning Models" (ACSAC 2024).** Demonstrates that backdoor vulnerabilities exist in normally trained, clean models — without any poisoning or modification. A systematic study on **54 pre-trained models** found **315 inherent backdoors** across all categories. This fundamentally supports the proposal's feasibility: if neural networks naturally develop backdoor-like structures, they can certainly learn carefully designed cryptographic fingerprint patterns.

**Wenger et al., "Finding Naturally Occurring Physical Backdoors in Image Datasets" (NeurIPS 2022, Datasets Track).** Shows that popular datasets like ImageNet naturally contain co-located objects that function as backdoor triggers without any digital manipulation. Models trained on these curated subsets achieve equivalent backdoor success rates to poisoned datasets. Demonstrates that DNNs inherently memorize spurious correlations — the fundamental property the proposal exploits.

**Liu et al., "Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks" (ECCV 2020).** Uses natural glass reflections as backdoor triggers that appear natural and resist detection. Achieves high success while evading defenses because triggers resemble natural phenomena. Reinforces that natural-looking perturbations persist through training — directly supporting the viability of imperceptible cryptographic fingerprints.

### 2g. Origin attribution of AI-generated images

**Wang et al., "Where Did I Come From? Origin Attribution of AI-Generated Images" (NeurIPS 2023).** Develops an alteration-free, model-agnostic attribution method that determines whether an image was generated by a specific model via reverse-engineering (inverting the model's input for a given image). Works on pre-trained models without impairing generation quality. Directly relevant as it addresses image provenance from the generative model side, complementing the proposal's training-data-side approach.

**Yu et al., "Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data" (ICCV 2021).** Embeds artificial fingerprints into training data of generative models so generated images carry traceable signatures back to specific training sets. A conceptual precursor to the proposed approach — both embed patterns in training data that persist through generation.

### 2h. Neural cryptography

**Jin et al., "TensorCrypt: Repurposing Neural Networks for Efficient Cryptographic Computation" (NDSS 2025).** Converts AES, ChaCha20, and Salsa20 into neural network computation graphs via a program translation framework. Achieves up to **5.44× speedup** over GPU crypto implementations with preserved security. Directly relevant as it demonstrates that neural architectures can faithfully implement cryptographic operations, supporting integration of crypto primitives within training pipelines.

**Gilad-Bachrach et al., "CryptoNets: Applying Neural Networks to Encrypted Data" (ICML 2016).** Demonstrates neural network inference on homomorphically encrypted data, achieving 99% accuracy on MNIST at 58,982 predictions/hour. Establishes the foundational concept of combining cryptographic computation with neural inference.

**Ng & Chow, "SoK: Cryptographic Neural-Network Computation" (IEEE S&P 2023).** Systematizes 53 privacy-preserving neural network papers (2016–2022), covering homomorphic encryption and secure computation approaches. Provides essential context for the intersection of cryptography and neural computation.

### 2i. Diffusion model watermarking — state of the art

**Wen et al., "Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust" (NeurIPS 2023).** Embeds watermarks as structured patterns in the **Fourier space of the initial noise vector** rather than in the image. Detection inverts the diffusion process to retrieve the noise vector. Fourier-space patterns are invariant to convolutions, crops, and rotations. Demonstrates that perturbation patterns embedded in the generative process persist through sampling — analogous to fingerprints persisting through training.

**Fernandez et al., "The Stable Signature: Rooting Watermarks in Latent Diffusion Models" (ICCV 2023).** Fine-tunes the latent decoder conditioned on a binary signature so all generated images contain an invisible watermark. Recovers signatures with **>90% accuracy** even after cropping to 10% of content, at false positive rates below **10⁻⁶**. Embeds watermarks within the model architecture itself, parallel to the proposal's data-level approach.

**Cui et al., "DiffusionShield: A Watermark for Copyright Protection against Generative Diffusion Models" (NeurIPS 2023).** Encodes ownership information into imperceptible blockwise watermarks that generative diffusion models easily learn and reproduce. Uses pattern uniformity via repeated basic patches. **Most directly relevant** to the proposal as it validates the core idea: watermarks in training images are learned and reproduced by generative models.

**Yang et al., "Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models" (CVPR 2024).** A training-free, plug-and-play method that maps watermarks to latent representations following a standard Gaussian distribution, making them indistinguishable from non-watermarked representations. Provides a **theoretical proof of performance-lossless watermarking** and outperforms Tree-Ring and Stable Signature in robustness.

### 2j. Membership inference attacks on generative models

**Duan et al., "Are Diffusion Models Vulnerable to Membership Inference Attacks?" (ICML 2023).** Proposes SecMI, exploiting DDIM inversion's deterministic reverse process. Introduces the "t-error" metric and achieves **81–89% accuracy** on CIFAR-10 and Stable Diffusion. Provides techniques the proposal could leverage for fingerprint detection.

**Kong et al., "An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization" (ICLR 2024).** Achieves competitive MIA performance with only **two model queries** (6× more efficient than prior SOTA). Demonstrates that highly efficient inference is possible, supporting the feasibility of lightweight fingerprint detection.

**Tang et al., "Membership Inference Attacks on Diffusion Models via Quantile Regression" (arXiv 2023).** Uses quantile regression to predict reconstruction loss distributions on non-training examples, enabling custom per-example thresholds with substantially lower computational cost than shadow-model approaches.

### 2k. Robust watermarking against transformations

**Zhu et al., "HiDDeN: Hiding Data With Deep Networks" (ECCV 2018).** Introduces the first end-to-end trainable encoder-noise-decoder framework for data hiding, jointly training an encoder, differentiable noise layers (Gaussian blur, JPEG, cropping), and a decoder. This foundational architecture underpins most subsequent deep watermarking methods.

**Tancik et al., "StegaStamp: Invisible Hyperlinks in Physical Photographs" (CVPR 2020).** Encodes 56-bit hyperlink bitstrings into photographs that survive the **physical print-and-photograph pipeline** — printing, lighting variation, perspective changes, and camera capture. If deep watermarks survive physical-domain distortions, fingerprints can plausibly survive AI training.

**Lukas et al., "SoK: How Robust is Deep Neural Network Image Classification Watermarking?" (IEEE S&P 2022).** Systematically evaluates DNN watermarking robustness against fine-tuning, pruning, model extraction, and input preprocessing. Finds many schemes claiming robustness fail under comprehensive evaluation. Essential reading for designing rigorous robustness evaluations of the proposed fingerprints.

### 2l. Adversarial perturbation-based protection tools

**Shan et al., "Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models" (USENIX Security 2023).** Applies barely perceptible style cloaks that mislead generative models attempting style mimicry, achieving **>92% disruption**. The proposal extends this concept — embedding cryptographic fingerprints for detectability rather than style-disrupting perturbations.

**Shan et al., "Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models" (IEEE S&P 2024).** Demonstrates that as few as **50–100 optimized poison samples** can completely corrupt a prompt's output in SDXL, with effects bleeding to semantically related concepts. Proves that carefully crafted training-time perturbations have powerful effects on generative models — the proposal's fingerprints exploit this same vulnerability for detection.

**Salman et al., "Raising the Cost of Malicious AI-Powered Image Editing" [PhotoGuard] (ICML 2023).** Proposes encoder attacks and diffusion attacks to immunize images against manipulation by latent diffusion models. Demonstrates that imperceptible perturbations can fundamentally alter how diffusion models process specific images, supporting the premise that fingerprint perturbations persist through and affect model behavior.

**Liang & Wu, "Mist: Towards Improved Adversarial Examples for Diffusion Models" (arXiv 2023).** Combines texture-aware and semantics-aware losses (targeting CLIP features) to protect images from diffusion-model mimicry. Claims robustness against purification attacks including DiffPure. Represents the current state of the art in adversarial image protection, complementary to the proposal's cryptographic approach.

---

## Part 3: Preliminary validation plan

### 3.1 Fingerprint embedding feasibility

The first experiments verify that PRNG-derived perturbations are imperceptible at sufficient strength for memorization.

**Perturbation generation.** Use AES-256 in CTR mode to produce a deterministic pseudo-random byte stream, convert to Gaussian-distributed values via Box-Muller transform, and scale to a desired epsilon budget. For fragmentation, generate **K = 4–8 independent fragments** per image, each with its own AES key/nonce pair, targeting different spatial or frequency regions.

**Target quality thresholds:**

| Metric | Target | Rationale |
|--------|--------|-----------|
| PSNR | > 40 dB | Standard imperceptibility; >38 dB is "excellent" in watermarking literature |
| SSIM | > 0.98 | Near-identical structural similarity |
| LPIPS (AlexNet) | < 0.06 | Well below 0.5 threshold; typical high-quality watermarking achieves 0.05–0.15 |

**Epsilon sweep experiment.** Test ε ∈ {2/255, 4/255, 6/255, 8/255, 12/255, 16/255} on 1,000 images from each target dataset. For each epsilon, compute PSNR, SSIM, LPIPS (both AlexNet and VGG backends), and perceptual hash distance. Plot all metrics versus epsilon to identify the **optimal operating point** — expected to be **ε = 4/255 to 8/255**, yielding PSNR ~40–46 dB.

**Three embedding strategies to compare:**

- **Pixel-space only** — direct addition of scaled PRNG noise; simplest but least robust to JPEG
- **DCT mid-frequency** — embed in 8×8 block DCT mid-frequency coefficients; designed for JPEG resilience
- **DWT-DCT hybrid** — embed in level-2 DWT LL subband's DCT coefficients; best overall robustness per watermarking literature (>93% bit accuracy under JPEG Q=20 and 25% cropping)

Additional validation beyond metrics should include FID between clean and fingerprinted datasets (target < 1.0), perceptual hash distance (target Hamming distance = 0 for >95% of images), and optionally a small 2AFC user study with 20 participants targeting discrimination accuracy ≤ 55%.

### 3.2 Memorization tests on small-scale diffusion models

**Setup.** Train unconditional DDPM on CIFAR-10 (32×32) or CelebA-HQ 64×64 subset using the standard Ho et al. hyperparameters: U-Net with attention at 16×16, Adam optimizer (lr = 2×10⁻⁴), batch size 128, T = 1000 linear-schedule diffusion steps, EMA decay 0.9999, simple MSE ε-prediction loss. For preliminary experiments, reduce to **200K–400K iterations** with **5,000–10,000 images** and batch size 64.

**Fingerprinting ratio sweep.** Train four configurations systematically, each with 3 random seeds:

| Config | Total images | Fingerprinted | Ratio |
|--------|-------------|---------------|-------|
| A | 10,000 | 10,000 | 100% |
| B | 10,000 | 5,000 | 50% |
| C | 10,000 | 1,000 | 10% |
| D (control) | 10,000 | 0 | 0% |

**Measuring memorization.** Generate N = 10,000 samples from each trained model. For each sample, extract the high-frequency residual (subtract a Gaussian-smoothed version with σ = 1.0), then compute Pearson correlation between the residual and the known fingerprint pattern. The test statistic is the **mean correlation across all generated samples**. Track training loss (every 1K steps), FID (every 50K steps), mean fingerprint correlation (every 50K steps), and cosine similarity of noise predictions at t = 1 (every 50K steps) throughout training.

### 3.3 Detection feasibility — white-box and black-box

**White-box correlation (model weights accessible).** Adapt the radioactive data detection framework: extract U-Net bottleneck features for the fingerprint pattern, compute projection of model weight vectors onto the fingerprint direction, and test if this projection is significantly larger than expected under H₀. Additionally, pass 1,000 Gaussian noise samples through the first denoising steps, record intermediate activations, and compute Pearson correlation with the fingerprint's frequency-domain representation.

**Black-box detection (only generated samples available).** Generate N = 5,000–10,000 samples, compute residuals via generic denoising (BM3D or DnCNN), and calculate the mean Pearson correlation with the known fingerprint. For advanced detection, use **matched-filter detection** in the frequency domain: cross-correlate sample DFTs with fingerprint DFTs and aggregate scores across fragments. Three complementary metrics — Pearson correlation, cosine similarity, and spectral coherence — should be computed for each sample.

**Sample size requirements (power analysis).** For a one-sample t-test on mean correlation at α = 0.01 and 90% power: effect size d = 0.3 (moderate) requires **~156 samples**, d = 0.1 (small) requires **~1,398 samples**, d = 0.5 (large) requires **~57 samples**. Recommendation: generate at least 1,000 samples; 5,000–10,000 for robust significance. The effect size should be estimated empirically from the 100% fingerprinted model first.

### 3.4 Baseline comparisons

# Baseline Methods for Comparison

## Primary Baselines

The two primary baselines are **WOUAF** and **PALADIN**, both addressing neural fingerprinting for text-to-image diffusion models — the same problem setting as our proposal. They are complemented by four secondary baselines from adjacent paradigms (data tracing, training denial, output watermarking) to provide broader context.

| # | Method | Paper | Venue | Core Mechanism | Fingerprint Target | Detection Mode | Stealth | Robustness | Multi-signal | Open Source | Key Limitation (Our Advantage) |
|---|--------|-------|-------|----------------|-------------------|----------------|---------|------------|-------------|-------------|-------------------------------|
| **B1** | **WOUAF** | Patel et al., "WOUAF: Weight Modulation for User Attribution and Fingerprinting in Text-to-Image Diffusion Models" | CVPR 2024 | Modulates generator weights per user via a learned fingerprint; each user receives a uniquely fine-tuned model whose outputs carry an identifiable signature extractable by a trained decoder | Model output (user attribution) | Black-box: trained decoder extracts fingerprint bits from generated images | High (near-imperceptible; FID degradation < 1) | Moderate — robust to JPEG, blur, noise; degrades under heavy cropping and strong augmentation | Single fingerprint per user-specific model; no fragmentation | Yes (`github.com/maitreya-patel/WOUAF`) | Single-model-per-user paradigm; does not address unauthorized training data usage; fingerprint lives in model weights, not in training data; no cryptographic derivation; attribution accuracy < 100% |
| **B2** | **PALADIN** | Murthy & Tripathi, "PALADIN: Robust Neural Fingerprinting for Text-to-Image Diffusion Models" | arXiv 2025 (2506.03170) | Built atop WOUAF; adds cyclic error-correcting codes (from coding theory) to achieve 100% user-attribution accuracy; improves decoder architecture and loss function for better quality–accuracy trade-off | Model output (user attribution) | Black-box: extract fingerprint bits from generated images; cyclic code enables bit-error recovery | High (improved over WOUAF; FID comparable) | High — error-correcting codes recover fingerprint bits even after post-processing distortions | Single fingerprint per user with ECC redundancy; no multi-fragment design | Not yet (built on WOUAF codebase) | Same paradigm limitation as WOUAF — fingerprints model output, not training data; requires model-provider cooperation (can't protect data from unauthorized scraping); no cryptographic key derivation |
| **B3** | **Radioactive Data** | Sablayrolles et al. | ICML 2020 | Aligns feature representations with a single random direction u in feature space via imperceptible perturbation | Training data → model weights | White-box: cosine similarity of trained model's weight direction with marking vector u | High (~42 dB) | Partial — validated on classifiers; vulnerable to strong augmentation | Single global direction — single point of failure | Yes (Facebook Research) | Single marking direction; validated only on classifiers, not generative models; no cryptographic derivation |
| **B4** | **Clean-Label Backdoor WM** | Tang et al. | KDD Explorations 2023 | Adds imperceptible trigger perturbations to 1–5% of dataset; model learns hidden trigger-output association | Training data → model behavior | Black-box: query with trigger pattern, observe output shift; Wilcoxon signed-rank test | High (~40 dB, clean-label) | Moderate — vulnerable to fine-tuning on additional clean data | Single trigger pattern — single point of failure | Yes | Single trigger; relies on backdoor misclassification behavior; not validated on diffusion models |
| **B5** | **DiffusionShield** | Cui et al. | NeurIPS 2023 | Encodes ownership info as imperceptible blockwise watermarks with pattern uniformity; diffusion models learn and reproduce them in outputs | Training data → generated images | Black-box: trained decoder extracts blockwise watermark from generated images | High (blockwise invisible) | Moderate — block structure adds robustness, but fixed pattern is identifiable | Fixed block pattern; limited capacity per image | Yes (`github.com/Yingqiancui/DiffusionShield`) | Fixed uniform pattern per user (not cryptographically derived); no multi-fragment resilience; evaluated primarily on DDPM, limited to smaller-scale diffusion models |
| **B6** | **Glaze + Nightshade** | Shan et al. | USENIX Sec 2023 / IEEE S&P 2024 | Adversarial perturbations that either disrupt style mimicry (Glaze) or corrupt specific prompt-concept associations (Nightshade) | Training data → model corruption | N/A (goal is disruption, not detection) | Medium–High | Declining — bypassed by LightShed purification (USENIX Sec 2025), noisy upscaling, and other data cleaning methods | N/A | Yes (binary tools) | No detection capability; "deny" paradigm rather than "detect"; actively degrades model quality; purification attacks increasingly effective |

---

### 3.5 Statistical hypothesis testing framework

**Formulation.** H₀: the model was NOT trained on fingerprinted data. H₁: the model WAS trained on fingerprinted data. Test statistic T = mean Pearson correlation between generated image residuals and known fingerprint pattern across N samples.

**Null distribution estimation.** Train **M = 20 clean DDPM models** with different random seeds, generate 5,000 samples from each, compute T for each, and fit the null distribution T_null ~ N(μ₀, σ₀²) (verify normality via Shapiro-Wilk test; use empirical quantiles or bootstrap if non-normal). With M = 20, the minimum resolvable empirical p-value is ~0.048; parametric fitting enables finer resolution. Resource estimate: **~370 GPU-hours** on A100 for 20 CIFAR-10 DDPM models at 200K steps each, fully parallelizable.

**Multi-fragment aggregation.** For K fragments, compute a K-dimensional vector of correlations and use **Fisher's method** to combine p-values: χ² = −2 Σ ln(pₖ) ~ χ²(2K) under H₀. This provides substantially more power than any single fragment alone.

**Significance thresholds.** Single-model test: α = 0.001 (z ≈ 3.1). Multiple model tests: Bonferroni correction α_adj = α/m. Alternative: Benjamini-Hochberg at FDR = 0.05.

**Recommended workflow:** (1) Train one 100%-fingerprinted model and estimate effect size d from 5,000 samples. (2) Use observed d to compute required N for 90% power at α = 0.001. (3) Adjust all subsequent experiments accordingly.

### 3.6 Robustness spot-checks

Apply each transformation to fingerprinted images **before training**, then measure whether the fingerprint signal persists in the trained model:

| Transformation | Parameters | Expected impact |
|---------------|------------|-----------------|
| JPEG compression | Q = {95, 80, 60, 40} | HIGH on pixel-space; DWT survives Q ≥ 60 |
| Cropping | {80%, 60%, 40%} area | MODERATE; DWT survives ≥ 60% area |
| Gaussian noise | σ = {0.01, 0.05, 0.1} | σ = 0.01 minimal; σ = 0.1 destructive |
| Resize | Scale to {50%, 75%} then back | MODERATE; interpolation blurs high frequencies |
| Color jitter | Brightness/Contrast ±0.2 | LOW; luminance fingerprints survive |
| Gaussian blur | σ = {0.5, 1.0, 2.0} | HIGH on high-frequency; LOW on mid/low |

For each transformation × embedding strategy (pixel, DCT, DWT-DCT): apply transformation to all 10,000 fingerprinted images, train DDPM on the transformed dataset (200K steps), generate 5,000 samples, run the detection test, and record p-value and power. The **fragmented approach provides defense-in-depth** — even if some fragments are destroyed by a transformation, remaining fragments still carry signal. Based on prior literature, the most threatening transformations in order are: JPEG ≤ Q60, Gaussian noise σ ≥ 0.05, heavy cropping < 60%, and aggressive resize with interpolation.

### 3.7 Ten-week experiment timeline

| Phase | Weeks | GPU-hours | Deliverable |
|-------|-------|-----------|-------------|
| Embedding feasibility + epsilon sweep | 1 | ~10 | Quality metrics table, operating-point selection |
| Memorization proof-of-concept (100% FP) | 2–3 | ~100 | Correlation signal confirmation |
| Ratio experiments (50%, 10%) | 3–4 | ~200 | Detection threshold vs. contamination ratio |
| Null distribution (20 clean models) | 4–6 | ~370 | Calibrated p-values, power estimates |
| Baseline comparisons | 6–8 | ~300 | Comparative detection accuracy tables |
| Robustness spot-checks | 8–10 | ~500 | Full robustness matrix |
| **Total** | **~10 weeks** | **~1,480** | **Complete preliminary validation report** |

## Conclusion

Three insights emerge from this warm-up plan that should guide the project's early decisions. First, the **DWT-DCT hybrid embedding strategy** deserves priority over pixel-space perturbation — the watermarking literature consistently shows mid-frequency wavelet coefficients survive JPEG, cropping, and resize far better than high-frequency pixel noise, and these are precisely the transformations images encounter in AI training pipelines. Second, the **fragmentation strategy is the proposal's key differentiator** from Radioactive Data and anti-neuron watermarking, and early experiments should quantify exactly how many fragments (K = 4–8) provide meaningful redundancy gains versus diminishing returns from splitting a finite perturbation budget. Third, the **null distribution estimation** (20 clean models, ~370 GPU-hours) is the most resource-intensive preliminary step, but it is non-negotiable for credible statistical claims — the team should begin training clean models in parallel from day one while conducting the faster embedding feasibility experiments. The literature review reveals that no existing method combines cryptographic verifiability, imperceptibility, multi-fragment robustness, and diffusion-model compatibility — this gap is exactly where the proposal's contribution lies, and the validation plan above will determine within ten weeks whether the core hypothesis holds.