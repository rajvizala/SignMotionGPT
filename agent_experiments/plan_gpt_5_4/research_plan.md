# Research Plan: Compositional Generalization for ASL Motion Generation

## 1. Problem statement

Current behavior:

- Word-level pretraining helps.
- Sentence-level How2Sign training fits the training distribution.
- Unseen How2Sign test performance remains poor.

This strongly suggests that the current system is learning sentence-pattern memorization more easily than reusable sign composition.

The target is not just better in-domain fitting. The target is:

1. reuse isolated-word sign knowledge inside unseen sentence contexts,
2. remain stable on longer or slightly different sentence structures,
3. improve text-motion semantic alignment instead of only next-token likelihood.

## 2. Hypothesis

The current failure is likely a combination of three bottlenecks:

### Bottleneck A: isolated-word knowledge is not explicitly available at sentence time

Sequential training assumes the sentence model will automatically preserve and compose the word-level knowledge it learned earlier. In practice, sentence fine-tuning can overwrite that structure. This is especially likely when the sentence objective only sees teacher-forced next-token prediction.

### Bottleneck B: the sentence objective over-rewards local continuation, not robust recovery

Autoregressive loss alone can learn how to continue familiar token trajectories without learning how to recover from partial corruption or slight prompt shifts. T2M-GPT already added a corruption strategy to reduce train-test discrepancy, and masked motion modeling later showed even stronger generalization benefits in motion generation. [R2][R4]

### Bottleneck C: text-motion alignment is underconstrained

Language-conditioned motion generation often suffers when text features are good for language understanding but weak for motion discrimination. LaMP argues that standard text embeddings do not align well with motion semantics and improves results with stronger language-motion pretraining. KMM also shows that fine-grained alignment losses help preserve critical details in long motion generation. [R6][R7]

## 3. Proposed contribution

Instead of copying one published method, this workspace proposes a combined ASL-specific strategy:

### Compositional Lexicon Memory + Robust Motion Training

1. Build a lexicon memory from Microsoft ASL Citizen word-level samples.
2. Retrieve matching word entries for each sentence during How2Sign training.
3. Inject compact motion anchors from those word entries into the sentence prompt.
4. Keep replaying word-level batches during sentence training so isolated-word knowledge is not forgotten.
5. Add masked motion-token denoising so the model learns to reconstruct sign sequences from partially corrupted motion code streams.
6. Add a contrastive text-motion alignment loss so prompts and target motions share a tighter latent structure.
7. Randomize positional offsets to reduce reliance on fixed absolute positions and improve length generalization. [R8]

This keeps the same data and the same high-level pipeline:

`raw motion -> VQ-VAE tokens -> LLM -> tokens -> VQ-VAE decode`

but changes how the LLM is conditioned and optimized.

## 4. Why these ideas are relevant

### 4.1 Lexicon memory and replay

Why it should help:

- How2Sign contains many sentence combinations whose component words are already known from isolated-word data.
- If sentence prompts are augmented with compact lexical motion anchors, the model can treat word-level knowledge as a non-parametric support signal rather than relying only on what remains in its weights.
- Continuing word replay reduces catastrophic forgetting during sentence fine-tuning.

Why this is paper-grounded:

- Spoken2Sign shows that dictionary-like composition is a viable route for sign production systems, even though its final pipeline differs from yours. [R9]
- Retrieval-Pretrained Transformer shows that retrieval works best when treated as part of sequence modeling rather than a late add-on. [R10]
- MotionGPT shows prompt-based multitask motion-language training can unify motion behaviors inside a single language-model view. [R3]

What is novel here:

- The retrieved memory comes from isolated-word ASL supervision.
- The retrieved content is not plain text retrieval; it is compressed motion-token evidence.
- The goal is compositional sentence generalization, not just in-domain retrieval wins.

### 4.2 Masked denoising

Why it should help:

- At inference, the model must generate without teacher forcing.
- A model trained only on clean prefixes often overfits local next-token continuation.
- Masked denoising forces recovery from missing spans, which should help with unseen sentence phrasing and imperfect motion planning.

Why this is paper-grounded:

- T2M-GPT used corruption to reduce train-test discrepancy in discrete motion generation. [R2]
- MoMask shows masked motion modeling is especially effective for semantic motion generation. [R4]

What is novel here:

- The denoising objective is added as an auxiliary loss inside an autoregressive sign-motion LLM rather than replacing the LLM with a masked-only generator.

### 4.3 Contrastive text-motion alignment

Why it should help:

- You want a sentence embedding that reflects sign-relevant semantics, not only language similarity.
- When the target sentence is unseen, stronger latent alignment should reduce drift toward fluent but semantically wrong motions.

Why this is paper-grounded:

- LaMP improves motion generation by moving from generic language-vision embeddings toward language-motion structure. [R6]
- KMM shows fine-grained alignment helps long motion generation keep critical textual details. [R7]

What is novel here:

- The alignment is computed directly over prompt and target-motion regions inside the same autoregressive model, making it cheap enough to test in the current codebase.

### 4.4 Randomized positional offsets

Why it should help:

- Sentence motions are longer and more variable than isolated words.
- A model trained with rigid absolute position ranges can become brittle when test sequences require different effective position patterns.

Why this is paper-grounded:

- Randomized positional encodings improved out-of-distribution length generalization in Transformers by exposing models to a wider range of position indices during training. [R8]

What is novel here:

- The method is adapted to RoPE-style motion-token generation by randomizing batch-level position offsets at train time rather than modifying the tokenizer or data.

## 5. Experiment matrix

## E0: Baseline reproduction inside this workspace

Purpose:

- Verify the new runner reproduces the current training style.
- Establish a fair comparison against the new objectives.

Changes from current repo:

- none in principle,
- but adds stronger coverage analysis and reporting.

Expected outcome:

- similar train behavior,
- similar unseen-set weakness.

Interpretation:

- if E0 already improves a lot, the issue may be training hygiene or evaluation differences rather than architecture.

## E1: Lexicon memory + word replay

Components:

- sentence prompt includes retrieved word entries,
- retrieved entries provide approximate length and compact motion anchors,
- word-level replay batches are mixed into sentence training.

Expected benefit:

- biggest gains on sentences where most words are seen individually but their combination is unseen,
- better controllability and more stable hand configuration reuse.

Failure mode:

- if gains only appear on very high-coverage sentences, memory may be helping lookup but not true composition.

## E2: Denoising + contrastive alignment + randomized positions

Components:

- denoising auxiliary loss on corrupted motion tokens,
- prompt-to-motion contrastive loss,
- randomized positional offsets.

Expected benefit:

- stronger robustness on unseen sentence structures,
- better long-sentence stability,
- less train-test collapse.

Failure mode:

- if train loss rises but unseen metrics improve, keep it,
- if both worsen, the auxiliary losses may be too strong and need smaller weights.

## E3: Full stack

Components:

- E1 + E2 together.

Expected benefit:

- best overall candidate.
- Especially promising if the current problem is both forgetting and weak robustness.

Interpretation:

- If E3 beats E1 and E2 individually, the effects are complementary.
- If E3 underperforms E1, the alignment or denoising loss is probably over-regularizing.

## 6. Recommended run order

1. Run `analyze_generalization.py` first.
2. Run E0 for a short sanity schedule.
3. Run E1.
4. Run E2.
5. Run E3.
6. Compare by lexical-coverage buckets and sentence-length buckets, not only one aggregate score.

## 7. Evaluation protocol that can support a paper claim

The claim you want is compositional generalization, so average loss alone is not enough. Report:

- unseen How2Sign aggregate metrics,
- metrics on sentences whose words are individually seen in Microsoft ASL Citizen,
- metrics on high lexical overlap but unseen sentence combinations,
- metrics on low lexical overlap,
- metrics by length bucket,
- qualitative examples for:
  - exact lexical coverage but unseen composition,
  - partially novel wording,
  - longer-than-typical sentence motions.

The analysis script in this folder produces lexical-coverage and novelty summaries so that the experiment can be argued scientifically rather than by anecdote.

## 8. What result patterns would mean

### Pattern A: E1 helps, E2 does not

Meaning:

- the main bottleneck is missing access to isolated-word knowledge during sentence training.

Next direction:

- strengthen lexicon planning,
- add phrase-level memory,
- try separate planner tokens before motion decoding.

### Pattern B: E2 helps, E1 does not

Meaning:

- the model knows enough words, but inference robustness and text-motion binding are the weak links.

Next direction:

- increase denoising strength slowly,
- add stronger length planning,
- test a separate planner head.

### Pattern C: both help, E3 helps most

Meaning:

- you have evidence for a real combined explanation:
  preserving word knowledge and improving sentence robustness are both necessary.

Next direction:

- scale the most successful variant,
- run 2 to 3 seeds,
- curate a compositional holdout split for a stronger paper.

### Pattern D: none help

Meaning:

- the failure may come earlier in the stack, especially tokenization or sentence-domain VQ representation.

Next direction:

- revisit VQ-VAE adaptation first,
- recompute normalization stats on combined data,
- check codebook usage and sentence reconstruction error before blaming the LLM.

## 9. VQ-VAE note

This workspace keeps the LLM experiments separate, but your repository already has a useful sentence VQ-VAE fine-tuning path. If results suggest a tokenizer bottleneck, first measure:

- word vs sentence reconstruction gap,
- codebook usage,
- hand reconstruction error,
- velocity loss.

That is important because poor sentence tokenization can make any LLM look non-generalizing even when the root cause is representational mismatch.

## 10. References

[R1] Duarte et al., "How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language," CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html

[R2] Zhang et al., "T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations," CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html

[R3] Jiang et al., "MotionGPT: Human Motion as a Foreign Language," NeurIPS 2023. https://arxiv.org/abs/2306.14795

[R4] Guo et al., "MoMask: Generative Masked Modeling of 3D Human Motions," CVPR 2024. https://arxiv.org/abs/2312.00063

[R5] Kong et al., "Priority-Centric Human Motion Generation in Discrete Latent Space," ICCV 2023. https://arxiv.org/abs/2308.14480

[R6] Li et al., "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning," ICLR 2025. https://arxiv.org/abs/2410.07093

[R7] Zhang et al., "KMM: Key Frame Mask Mamba for Extended Motion Generation," 2024. https://arxiv.org/abs/2411.06481

[R8] Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers," ACL 2023. https://aclanthology.org/2023.acl-short.161/

[R9] Zuo et al., "A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars," ECCV 2024. https://arxiv.org/abs/2401.04730

[R10] Rubin and Berant, "Retrieval-Pretrained Transformer: Long-range Language Modeling with Self-retrieval," TACL 2024. https://aclanthology.org/2024.tacl-1.66/

[R11] Fang et al., "SignLLM: Sign Languages Production Large Language Models," ICCV Workshop 2025. https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/html/Fang_SignLLM_Sign_Language_Production_Large_Language_Models_ICCVW_2025_paper.html
