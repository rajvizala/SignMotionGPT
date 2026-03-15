# Agent Experiments: Towards Generalised ASL Sign Language Generation

## Problem Statement

The current SignMotionGPT pipeline (VQ-VAE + Qwen3-0.6B LLM) achieves good training
metrics on How2Sign sentence-level data after extensive training (1000 VQ-VAE epochs +
40 LLM epochs), but **fails on unseen test data**. The model memorises training
sequences rather than learning generalisable text-to-motion mappings.

**Goal**: Achieve out-of-domain generalisation -- generate accurate sign motions for
sentences that are *similar to but not identical to* training data.

**Constraint**: Keep the VQ-VAE + LLM pipeline structure. Keep the same data (ASL
Citizen word-level + How2Sign sentence-level). All changes must be architectural
improvements, training technique modifications, or augmentation strategies.

---

## Diagnosis: Why the Current Pipeline Fails to Generalise

Based on analysis of the codebase and literature, the root causes are:

1. **Codebook bottleneck**: A flat 512-code VQ-VAE must simultaneously represent body
   pose, hand articulation, facial expression, and signer identity in a single code.
   This entangles features that should be independent, making novel combinations
   impossible.

2. **Sequence memorisation**: The LLM sees ~2K unique sentences x 4 templates for 40
   epochs. Standard cross-entropy loss rewards exact token reproduction, not semantic
   understanding. The model overfits to exact token patterns.

3. **No compositional mechanism**: The word-level knowledge (83K+ samples) is used only
   for VQ-VAE pre-training. At sentence level, the model cannot decompose sentences
   into known word-level signs.

4. **Insufficient regularisation**: No dropout, no data augmentation, no smoothing
   beyond curriculum learning. The model capacity (0.6B params) vastly exceeds the
   effective training set size.

---

## Experiment Overview

| # | Name | Target | Key Technique | Paper References |
|---|------|--------|---------------|-----------------|
| 1 | Residual VQ-VAE (RQ-VAE) | VQ-VAE | Multi-level residual quantisation | MOGO [1], SoundStream [2], MoSa [3] |
| 2 | Part-Aware VQ-VAE | VQ-VAE | Per-body-part codebooks + fusion | MotionGPT-2 [4], SOKE [5] |
| 3 | Semantic Regularisation | LLM | Contrastive + label smoothing + temporal | SignCLIP [6], MoCLIP [7] |
| 4 | Motion Augmentation | LLM | Token-level data augmentation | EnsAug [8], XmDA [9] |
| 5 | Retrieval-Augmented Gen. | LLM | Word-level sign dictionary as context | SOKE [5], RAG [10] |

---

## Experiment 1: Residual Vector Quantisation (RQ-VAE)

### Location
`exp1_residual_vq/`

### Motivation
The baseline VQ-VAE uses a single codebook of 512 vectors. With 182 input dimensions
and complex motion patterns, this codebook either (a) collapses (many codes unused) or
(b) over-specialises to training data.

### Architecture
Replace the single `QuantizeEMAReset` with a multi-level residual quantiser:

```
Input z_enc -> [Level 0: quantise(z_enc)] -> residual_0
            -> [Level 1: quantise(residual_0)] -> residual_1
            -> [Level 2: quantise(residual_1)] -> residual_2
            -> [Level 3: quantise(residual_2)] -> residual_3
Final: z_q = sum(quantised_0, quantised_1, quantised_2, quantised_3)
```

Each level has its own codebook of K codes. Level 0 captures coarse motion structure;
subsequent levels capture progressively finer detail.

### Key Innovations
- **Level dropout**: During training, randomly skip higher RQ levels (p=0.1). This
  forces level 0 to be self-sufficient, improving the coarse representation quality.
  Inspired by SoundStream.
- **EMA codebook updates with dead-code reset**: Per-level EMA (mu=0.99) with
  automatic replacement of unused codes.
- **Per-level perplexity tracking**: Monitor codebook utilisation at each level to
  detect collapse early.

### Expected Impact
- Effective codebook capacity: 512^4 = 68 billion combinations vs 512
- Better reconstruction of unseen motions (codes can combine in new ways)
- The LLM can learn coarse-to-fine generation patterns

### How to Run
```bash
python -m agent_experiments.exp1_residual_vq.train \
    --data-dir /path/to/npz_word_data \
    --output-dir ./agent_experiments/outputs/exp1 \
    --num-rq-levels 4 \
    --epochs 300 \
    --batch-size 32
```

### What to Measure
1. **Reconstruction loss** (train + val) -- should be lower than baseline
2. **Per-level perplexity** -- all levels should have perplexity > 100 (no collapse)
3. **Codebook utilisation** -- % of codes used per level per epoch
4. **Downstream LLM performance** -- after retraining LLM on RQ-VAE tokens

### References
- [1] MOGO: arXiv:2506.05952, 2025
- [2] SoundStream: IEEE/ACM Trans. Audio Speech Lang. Process., 2022
- [3] MoSa: arXiv:2511.01200, 2024

---

## Experiment 2: Part-Aware VQ-VAE

### Location
`exp2_part_aware_vq/`

### Motivation
Sign language meaning is carried primarily by hands (90 dimensions out of 182). The
current VQ-VAE treats all 182 dimensions equally in a single codebook, entangling hand
articulation with body pose and facial expression.

### Architecture
Split the 182-dim SMPL-X feature into four anatomical groups:
- **Body** (63-dim): body_pose
- **Left Hand** (45-dim): lhand_pose
- **Right Hand** (45-dim): rhand_pose
- **Face+Meta** (29-dim): shape(10) + jaw(3) + expression(10) + root(3) + cam(3)

Each group has its own encoder, codebook, and decoder. A **Cross-Part Fusion**
transformer (2 layers, 4 heads) lets parts interact before decoding.

### Key Innovations
- **Independent hand codebooks**: Hands get dedicated 256-code, 128-dim codebooks with
  higher loss weight (12x vs 10x for body). This preserves fine finger articulation.
- **Cross-part attention fusion**: Unlike fully independent part models (which lose
  coordination), our fusion layer enables the decoder to use cross-part context while
  keeping codebooks separate.
- **Compositional expressivity**: With 256 body codes x 256 left-hand codes x 256
  right-hand codes x 256 face codes = 4.3 billion combinations.

### Expected Impact
- Novel body+hand combinations become possible (key for generalisation)
- Hand reconstruction quality improves significantly
- The LLM can use multi-head decoding (predict 4 part tokens per timestep)

### How to Run
```bash
python -m agent_experiments.exp2_part_aware_vq.train \
    --data-dir /path/to/npz_word_data \
    --output-dir ./agent_experiments/outputs/exp2 \
    --code-num 256 --code-dim 128 \
    --epochs 300 --batch-size 32
```

### What to Measure
1. **Per-part reconstruction loss** -- especially hands vs body
2. **Per-part codebook perplexity** -- all codebooks should be well-utilised
3. **Overall reconstruction quality** vs single-codebook baseline
4. **Compositional test**: encode a body pose from sample A with hands from sample B

### References
- [4] MotionGPT-2: arXiv:2410.21747, 2024
- [5] SOKE: ICCV 2025, "Signs as Tokens"

---

## Experiment 3: Semantic-Aware Regularisation for LLM Training

### Location
`exp3_semantic_regularization/`

### Motivation
The current LLM training uses standard cross-entropy loss, which treats all wrong
tokens as equally bad and all right tokens as the only acceptable output. This leads to
memorisation rather than semantic understanding.

### Architecture
Three complementary regularisation techniques added to the LLM training loop:

**(A) Contrastive Motion-Text Alignment Loss** (inspired by SignCLIP [6] and MoCLIP [7])
- Extract text hidden states and motion hidden states from the LLM
- Project both into a shared 256-dim space
- InfoNCE loss: matched (text, motion) pairs should be close, mismatched pairs far

**(B) Codebook-Aware Label Smoothing** (novel contribution)
- For motion token targets, replace hard one-hot with a soft distribution
- Soft target for code i: p(j) ~ exp(-dist(codebook[i], codebook[j]) / tau)
- Blend: target = (1-alpha) * one_hot + alpha * soft_target
- This teaches the model that generating a "nearby" code is much better than a random one

**(C) Temporal Consistency Regularisation**
- KL divergence between predicted distributions at adjacent timesteps
- Prevents oscillation between unrelated codes in generated sequences

### Key Innovations
- Contrastive loss within the LLM's own hidden space (no external encoder needed)
- Codebook-distance-aware label smoothing is a novel technique specific to VQ-based
  generation that bridges the gap between NLU label smoothing and motion generation
- Combination of all three techniques provides multi-level regularisation

### Expected Impact
- Similar text prompts will produce similar motion outputs (semantic alignment)
- The model tolerates "close" codes, reducing brittleness on unseen data
- Smoother, more coherent generated sequences

### How to Run
```bash
python -m agent_experiments.exp3_semantic_regularization.train \
    --dataset-path /path/to/sentence_dataset.json \
    --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
    --output-dir ./agent_experiments/outputs/exp3 \
    --lambda-contrastive 0.1 \
    --lambda-temporal 0.01 \
    --label-smooth-alpha 0.1 \
    --epochs 40
```

### Ablation Plan
Run with each technique disabled individually to measure contribution:
```bash
# No contrastive
--no-contrastive

# No label smoothing
--no-label-smooth

# No temporal consistency
--no-temporal
```

### What to Measure
1. **Validation loss** -- should decrease faster and plateau lower
2. **Train-val gap** -- should be smaller (less overfitting)
3. **Token edit distance** on held-out test set
4. **Loss breakdown** -- track each component separately
5. **Ablation**: which technique contributes most?

### References
- [6] SignCLIP: EMNLP 2024
- [7] MoCLIP: arXiv:2505.10810, 2025

---

## Experiment 4: Motion Token Augmentation

### Location
`exp4_motion_augmentation/`

### Motivation
The effective training set is small (~2K unique sentences x 4 templates = 8K samples).
With 40 epochs, the model sees each exact pattern ~40 times, enough to memorise.

### Architecture
Two levels of augmentation:

**(A) VQ-VAE Feature-Space Augmentation** (during VQ-VAE training):
- Temporal warping (0.8x-1.2x speed)
- Gaussian noise (sigma=0.02)
- Hand-specific jittering (sigma=0.03 on hand dims only)
- Left-right hand mirroring

**(B) Token-Level Augmentation** (during LLM training):
- Token dropout (p=0.15): randomly remove motion tokens
- Codebook-nearest-neighbour substitution (p=0.1): replace token with its k-NN
- Sequence cropping (p=0.2): use random sub-sequences
- Local permutation (window=3): permute within small windows

### Key Innovations
- **Codebook-aware substitution**: uses actual codebook L2 distances to find
  semantically similar replacement tokens, not random. This teaches the LLM that
  nearby codes are interchangeable.
- **On-the-fly augmentation**: augmented variants are generated dynamically each
  epoch, so the model never sees the exact same augmented pattern twice.
- **Effective dataset multiplication**: with 2 augmented variants per sample, the
  effective training set triples (base + 2 augmented).

### Expected Impact
- Larger effective training set reduces memorisation
- Token dropout forces learning of temporal context (predict from neighbours)
- Code substitution directly teaches noise tolerance

### How to Run
```bash
python -m agent_experiments.exp4_motion_augmentation.train \
    --dataset-path /path/to/sentence_dataset.json \
    --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
    --output-dir ./agent_experiments/outputs/exp4 \
    --dropout-prob 0.15 \
    --substitute-prob 0.1 \
    --aug-per-sample 2 \
    --epochs 40
```

### What to Measure
1. **Validation loss** vs baseline (no augmentation) -- expect lower
2. **Token accuracy** on held-out test data
3. **Training loss convergence** -- should be slower (harder training) but val improves
4. **Augmentation ablation**: which augmentation type helps most?

### References
- [8] EnsAug: arXiv:2603.06661, 2025
- [9] Cross-modality Data Augmentation (XmDA): EMNLP 2023

---

## Experiment 5: Retrieval-Augmented Generation (RAG)

### Location
`exp5_retrieval_augmented/`

### Motivation
The word-level dataset (ASL Citizen, ~83K samples) contains rich per-word motion
knowledge that is only used indirectly through VQ-VAE pre-training. At sentence level,
the LLM has no mechanism to decompose sentences into known word-level signs.

### Architecture
1. **Build Word-Sign Dictionary**: extract word -> motion_token_sequence mappings from
   word-level data. For each word, store the median-length variant as representative.

2. **Retrieval at Training Time**: for each sentence, extract content words, look up
   their motion patterns, and inject as context:
   ```
   [SIGN_CONTEXT] hello: <M42> <M18> | world: <M91> <M205> [/SIGN_CONTEXT]
   Instruction: Generate sign language motion for: 'hello world'
   (Length: <|LEN_MEDIUM|>)
   Motion: <M_START> <M42> <M18> <M91> <M205> <M_END>
   ```

3. **Retrieval at Inference Time**: same mechanism, providing compositional hints for
   unseen sentences. Even if "good morning" was never in sentence-level training, the
   individual words "good" and "morning" may have word-level patterns.

### Key Innovations
- **Stochastic context inclusion** (p=0.8 during training): sometimes omit context so
  the model does not become fully dependent on it. This maintains standalone capability.
- **Stop word filtering**: only retrieve content words, avoiding noise from function words
  that rarely have distinct sign patterns.
- **Median-length representative**: for words with multiple signer variants, use the
  median-length version to avoid outliers.

### Expected Impact
- Direct knowledge transfer from word-level to sentence-level
- Compositional generation for unseen word combinations
- Reduced search space during generation (context provides strong priors)

### How to Run
```bash
python -m agent_experiments.exp5_retrieval_augmented.train \
    --sentence-dataset /path/to/sentence_dataset.json \
    --word-dataset /path/to/word_level_dataset.json \
    --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
    --output-dir ./agent_experiments/outputs/exp5 \
    --context-prob 0.8 \
    --max-context-words 5 \
    --epochs 40
```

### What to Measure
1. **Validation loss** with and without RAG context
2. **Token edit distance** on held-out test sentences
3. **Vocabulary coverage**: what % of test sentence words are in the dictionary?
4. **Ablation**: context-prob=0.8 vs 1.0 vs 0.5 vs 0.0 (no context = baseline)

### References
- [5] SOKE: ICCV 2025
- [10] Lewis et al., "Retrieval-Augmented Generation", NeurIPS 2020

---

## Recommended Experiment Order

Based on expected impact and implementation complexity:

### Phase 1: Quick Wins (LLM-side, reuse existing VQ-VAE)
1. **Experiment 4** (Motion Augmentation) -- easiest to implement, directly increases
   effective training set. Run first as a sanity check.
2. **Experiment 5** (RAG) -- leverages existing word-level data, no model changes needed.

### Phase 2: Architectural (VQ-VAE changes, requires retraining)
3. **Experiment 2** (Part-Aware VQ-VAE) -- strongest theoretical justification for sign
   language specifically. Independent hand codebooks are a clear win.
4. **Experiment 1** (RQ-VAE) -- general improvement to codebook expressiveness.

### Phase 3: Advanced Regularisation (combine with best VQ-VAE from Phase 2)
5. **Experiment 3** (Semantic Regularisation) -- combine with the best VQ-VAE and best
   augmentation strategy from earlier experiments.

### Phase 4: Full Combination
6. **Combine the winning techniques** from each experiment. For example:
   Part-Aware VQ-VAE (Exp 2) + RAG (Exp 5) + Token Augmentation (Exp 4) + Contrastive
   Alignment (Exp 3-A).

---

## Interpreting Results

### What "success" looks like (from most to least important):
1. **Val loss decreasing when train loss decreases** (no overfitting)
2. **Token edit distance on test set improving** (actual generalization)
3. **DTW-JPE on test set improving** (motion quality on unseen data)
4. **FID on test set improving** (distribution-level quality)

### Red flags to watch for:
- **Train loss decreasing but val loss increasing** = overfitting (increase regularisation)
- **All codebook perplexities < 50** = codebook collapse (reduce commitment loss, add noise)
- **Val loss stuck at high value** = model not learning (increase LR or model capacity)
- **Generated sequences all identical** = mode collapse (lower temperature, add diversity loss)

### Comparing experiments:
Always compare against the same baseline (current pipeline) on the same held-out test set.
Use the same random seed (42) and the same train/val split for fair comparison.

---

## Directory Structure

```
agent_experiments/
    README.md                          # This file
    shared/
        __init__.py
        base_config.py                 # Common configuration
        logging_utils.py               # Structured logging
        data_utils.py                  # Data loading utilities
    exp1_residual_vq/
        __init__.py
        model.py                       # RQ-VAE model + ResidualQuantizer
        train.py                       # Training script
    exp2_part_aware_vq/
        __init__.py
        model.py                       # Part-Aware VQ-VAE + CrossPartFusion
        train.py                       # Training script
    exp3_semantic_regularization/
        __init__.py
        model.py                       # ContrastiveAlignment + LabelSmoothing + TemporalConsistency
        train.py                       # Training script
    exp4_motion_augmentation/
        __init__.py
        model.py                       # FeatureSpaceAugmentor + TokenAugmentor
        train.py                       # Training script
    exp5_retrieval_augmented/
        __init__.py
        model.py                       # WordSignDictionary + SignRetriever + RAGInference
        train.py                       # Training script
    outputs/                           # Created at runtime
        exp1/ exp2/ exp3/ exp4/ exp5/  # Per-experiment outputs
```

---

## Full Reference List

1. **MOGO**: "Residual Quantized Hierarchical Causal Transformer", arXiv:2506.05952, 2025
2. **SoundStream**: "An End-to-End Neural Audio Codec", IEEE/ACM TASLP, 2022
3. **MoSa**: "Motion Generation with Scalable Autoregressive Modeling", arXiv:2511.01200, 2024
4. **MotionGPT-2**: "A General-Purpose Motion-Language Model", arXiv:2410.21747, 2024
5. **SOKE**: "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator", ICCV 2025
6. **SignCLIP**: "Connecting Text and Sign Language by Contrastive Learning", EMNLP 2024
7. **MoCLIP**: "Motion-Aware Fine-Tuning and Distillation of CLIP", arXiv:2505.10810, 2025
8. **EnsAug**: "Augmentation-Driven Ensembles for Human Motion Sequence Analysis", arXiv:2603.06661, 2025
9. **XmDA**: "Cross-modality Data Augmentation for sign language translation", EMNLP 2023
10. **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
11. **GenM3**: "Generative Pretrained Multi-path Motion Model", ICCV 2025
12. **T2S-GPT**: "Dynamic Vector Quantization for Autoregressive Sign Language Production", ACL 2024
13. **MotionBase**: "Scaling Large Motion Models with Million-Level Human Motions", arXiv:2410.03311, 2024
14. **MoMask**: "Generative Masked Modeling of 3D Human Motions", CVPR 2024
15. **wSignGen**: "Word-Conditioned 3D American Sign Language Motion Generation", EMNLP Findings 2024
