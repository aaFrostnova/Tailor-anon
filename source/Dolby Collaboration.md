## Invisible Fragmented Cryptographic Fingerprints for Protecting Images from Unauthorized AI Training


### To-Do List

#### Step 1: Warm-up and initial setup

- [ ]  **Warm-up: environment setup, literature review, and preliminary validation**
  Set up the experimental environment, prepare the datasets and candidate generative models, review the most relevant watermarking and data attribution baselines, and conduct preliminary feasibility checks for fingerprint embedding and detection.

{%preview https://hackmd.io/@KjCKDZkaQmOkcksG4OndfA/rkvJ1FvjWe %}
    
#### Step 2: Core fingerprint generation pipeline

- [ ]  **Implement fragmented cryptographic fingerprint generation**
  Build a pipeline that derives multiple fingerprint fragments *δ₁, …, δₘ* from a master secret key and embeds them into each image.
- [ ]  **Design imperceptible yet learnable perturbations**
  Optimize the fingerprint perturbations so that they remain visually subtle while still being memorized by unauthorized generative model training.
- [ ]  **Accelerate large-scale fingerprint embedding**
  Develop an efficient generation pipeline for cryptographic fingerprints, potentially using neural-cryptography-inspired or GPU-friendly acceleration.
- [ ]  **Iterate on fingerprint generation and embedding design**🔁
Refine the fragment construction, perturbation strength, and embedding strategy based on perceptual quality, memorization behavior, and computational efficiency.

#### Step 3: Detection framework development

- [ ]  **Develop the white-box detection component**
  Design probes that detect correlations between embedded fingerprints and model weights, activations, or internal representations.
- [ ]  **Develop the black-box detection component**
  Build a query-based detection method, such as adaptive prompting or latent optimization, to reveal fingerprint responses without access to model internals.
- [ ]  **Construct a statistical hypothesis testing framework**
  Aggregate per-fingerprint response scores into a global test statistic and calibrate detection thresholds using an empirical null distribution.
- [ ]  **Iterate on the detection pipeline**🔁
Refine the probing strategy, score aggregation, and threshold calibration based on early detection performance and observed failure cases.

#### Step 4: Evaluation and refinement

- [ ]  **Run baseline comparisons and ablation studies**
  Compare against prior watermarking and data attribution methods, and ablate key factors such as fragment number, perturbation type, and score aggregation rule.
- [ ]  **Benchmark against post-processing and adaptation attacks**
  Evaluate robustness under resizing, JPEG compression, augmentation, retraining, pruning, or other attacker-side countermeasures.
- [ ]  **Iterate on method refinement through evaluation**🔁
  Use robustness results, baseline comparisons, and attack benchmarks to refine both the fingerprint generation module and the detection framework, and repeat the evaluation as needed.
 
#### Step 5: Paper writing
- [ ]  **Draft the paper**
  Organize the motivation, method, and evaluation results into a complete manuscript, including figures, tables, ablation summaries, and discussion of limitations.

### Results

