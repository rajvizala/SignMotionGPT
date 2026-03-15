# Agent Experiments: ASL OOD Generalization for `VQ-VAE + LLM`

This folder contains a research-oriented experiment package to improve out-of-domain (OOD) generalization for ASL motion generation while keeping your fixed pipeline:

- Motion tokenizer: `VQ-VAE` (unchanged)
- Sequence model: autoregressive `LLM` over motion tokens (unchanged)
- Data: **same datasets only** (`ASL Citizen` word-level + `How2Sign` sentence-level)

The package focuses on **training strategy**, **regularization**, **domain-aware validation**, and **continual-learning stabilization**.

---

## 1) Problem framing

You already observed:

- Strong in-domain training metrics after long training
- Poor unseen test performance on `How2Sign`

This is typical of overfitting and distribution shift in sequence generation. The plan here is to optimize for:

1. Better robustness to unseen sentence constructions
2. Better retention of useful word-level priors during sentence-level fine-tuning
3. More trustworthy model selection for real generalization (not only training loss)

---

## 2) Research hypothesis

If we preserve the current architecture but change optimization and training protocol, then:

- Flat-minima optimization (`SAM`, `SWA`) should improve OOD behavior.
- Consistency regularization (`R-Drop`) should reduce brittle token distributions.
- Continual-learning regularization (`EWC`) plus replay should preserve word-level compositional priors.
- Group-aware validation (pseudo-domains) should improve checkpoint selection for true generalization.

---

## 3) What is implemented in code

Code lives under `agent_experiments/experiments/`.

### 3.1 Core trainer

`train_sentence_generalization.py`:

- Trains sentence-level model with optional:
  - `R-Drop` consistency regularization
  - `SAM` optimizer wrapper
  - `SWA` checkpoint averaging at end of training
  - `EWC` penalty (computed from a compact word-level replay subset)
  - Replay sampling from word-level examples during sentence training
  - Label smoothing
  - Group-aware validation by pseudo-domain buckets

### 3.2 Training utilities

- `losses.py`
  - Label-smoothed CE
  - Symmetric KL for R-Drop
  - EWC state estimation and penalty
- `optim.py`
  - Lightweight SAM optimizer wrapper
  - SWA model averaging helper
- `domains.py`
  - Pseudo-domain construction from available metadata and sequence statistics
- `config_schema.py`
  - Typed config with defaults and validation

### 3.3 Reproducible configs

`agent_experiments/configs/` includes experiment presets:

- `baseline_repro.toml`
- `exp_a_rdrop_sam.toml`
- `exp_b_rdrop_sam_swa.toml`
- `exp_c_add_replay_ewc.toml`
- `exp_d_full_recipe.toml`

---

## 4) Why these techniques are chosen

### A) R-Drop

Two stochastic forward passes with dropout are encouraged to match (bidirectional KL).

- Why relevant: lowers train/infer discrepancy and reduces unstable token-level logits.
- Risk: over-regularization if KL weight too high.

### B) SAM and SWA

- `SAM`: seeks flatter neighborhoods (better robustness under shift).
- `SWA`: averages late checkpoints to land in wider minima.

These are architecture-preserving and usually low implementation risk.

### C) Replay + EWC from word-level stage

Sentence-level fine-tuning can forget compositional priors learned at word level. Replay and EWC are added to stabilize this transfer without changing model topology.

### D) Pseudo-domain validation

Generalization should not be measured only by random validation split. We evaluate per-group slices (length buckets, signer buckets if available, lexical rarity proxy) and choose checkpoints that do not collapse in hard groups.

---

## 5) Recommended experiment order

Run sequentially and keep all logs.

1. **Baseline reproduction**
   - `baseline_repro.toml`
2. **Flatness + consistency**
   - `exp_a_rdrop_sam.toml`
3. **Add SWA**
   - `exp_b_rdrop_sam_swa.toml`
4. **Add retention controls**
   - `exp_c_add_replay_ewc.toml`
5. **Full recipe**
   - `exp_d_full_recipe.toml`

Important: do not compare single runs only; use at least 3 seeds.

---

## 6) Metrics and interpretation guide

Primary:

- DTW-JPE / DTW-PA-JPE (or your current encoder-style metrics)
- FID/diversity/multimodality (if available in your pipeline)
- Group-wise worst-case score (most important for OOD)

When reading results:

- If mean improves but worst-group degrades, reject that recipe.
- If train loss improves but unseen metrics worsen, this is overfitting, not progress.
- Prefer recipes with smaller seed variance.

Use `RESULTS_TEMPLATE.md` for structured logging.

---

## 7) Run commands

From repo root:

- `python -m agent_experiments.experiments.train_sentence_generalization --config agent_experiments/configs/baseline_repro.toml`
- `python -m agent_experiments.experiments.train_sentence_generalization --config agent_experiments/configs/exp_d_full_recipe.toml`

You can override any value from CLI, for example:

- `--set training.epochs=40`
- `--set regularization.rdrop_alpha=2.0`

---

## 8) What could make this paper-worthy

A realistic publication angle is:

1. Show that architecture-constant interventions (`VQ-VAE + LLM` unchanged) significantly improve unseen sentence generalization.
2. Demonstrate that standard random validation would pick worse checkpoints than pseudo-domain validation.
3. Show that replay + EWC preserves word-level priors while improving sentence-level OOD quality.

If gains are modest but consistent across seeds and groups, that is still valuable as an engineering-science contribution.

---

## 9) Citations (techniques used)

1. How2Sign dataset:
   - Duarte et al., CVPR 2021, "How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language"
   - https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html

2. ASL Citizen:
   - Desai et al., NeurIPS Datasets and Benchmarks 2023, "ASL Citizen"
   - https://openreview.net/forum?id=zbEYTg2F1U

3. VQ-VAE:
   - van den Oord et al., NeurIPS 2017, "Neural Discrete Representation Learning"
   - https://papers.nips.cc/paper/7210-neural-discrete-representation-learning

4. MotionGPT:
   - Jiang et al., 2023, "MotionGPT: Human Motion as a Foreign Language"
   - https://hf.co/papers/2306.14795

5. R-Drop:
   - Liang et al., NeurIPS 2021, "R-Drop: Regularized Dropout for Neural Networks"
   - https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html

6. SAM:
   - Foret et al., 2020, "Sharpness-Aware Minimization for Efficiently Improving Generalization"
   - https://hf.co/papers/2010.01412

7. SWA:
   - Izmailov et al., UAI 2018, "Averaging Weights Leads to Wider Optima and Better Generalization"
   - https://hf.co/papers/1803.05407

8. EWC:
   - Kirkpatrick et al., 2016, "Overcoming catastrophic forgetting in neural networks"
   - https://hf.co/papers/1612.00796

9. Domain generalization benchmark perspective:
   - Gulrajani and Lopez-Paz, ICLR 2021, "In Search of Lost Domain Generalization"
   - https://github.com/facebookresearch/DomainBed

10. Sign language production architecture context:
   - Saunders et al., 2020, "Progressive Transformers for End-to-End Sign Language Production"
   - https://hf.co/papers/2004.14874
