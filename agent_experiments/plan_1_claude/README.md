# Plan 1 (Claude): VQ-VAE First, Then LLM

## Core Position
The confirmed diagnosis says VQ-VAE is the primary bottleneck (not the LLM).
This plan addresses that bottleneck first with two alternative VQ-VAE architectures,
then improves the LLM only after the tokenizer ceiling is raised.

## Experiment Sequence

### Exp 0: Baseline Evaluation (~10 min, CPU)
Establishes per-bucket metrics for the current VQ-VAE checkpoint.
All subsequent experiments compare against these numbers.

### Exp 1: Part-Aware VQ-VAE + Feature Augmentation (~3-5 hours GPU)
HIGHEST PRIORITY. Splits 182-dim SMPL-X into 4 body-part codebooks with
cross-part attention fusion. Feature augmentation during training.
Kill: worst-group no improve for 30 epochs, any codebook perplexity < 10.

### Exp 2: Residual VQ-VAE (RQ-VAE) (~3-5 hours GPU)
Alternative VQ-VAE: 4-level residual quantization with level dropout.
Controlled comparison against Exp 1.
Kill: same criteria as Exp 1.

### DECISION POINT
Compare Exp 1 and Exp 2 reconstruction on val data. Pick winner.
If neither beats baseline, the problem requires different approach.

### Exp 3: LLM with Denoising + RAG + SWA (~2-3 hours GPU)
After selecting best VQ-VAE, retrain LLM with:
- Masked motion-token denoising (from Plan 2 consensus)
- RAG context injection from word-level data
- SWA over final 20% of training
- Word-level replay (10% of batches)
Kill: val loss no improve for 10 epochs.

## Run Commands
```bash
# Exp 0: Baseline
python -m agent_experiments.plan_1_claude.exp_0.train \
    --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
    --val-dir /path/to/val_npz \
    --stats-path computed_stats.pt

# Exp 1: Part-Aware VQ-VAE
python -m agent_experiments.plan_1_claude.exp_1.train \
    --data-dir /path/to/train_npz \
    --val-dir /path/to/val_npz \
    --stats-path computed_stats.pt \
    --epochs 200

# Exp 2: RQ-VAE
python -m agent_experiments.plan_1_claude.exp_2.train \
    --data-dir /path/to/train_npz \
    --val-dir /path/to/val_npz \
    --stats-path computed_stats.pt \
    --epochs 200

# Exp 3: LLM (after selecting best VQ-VAE)
python -m agent_experiments.plan_1_claude.exp_3.train \
    --sentence-data /path/to/sentence_dataset.json \
    --word-data /path/to/word_dataset.json \
    --vqvae-ckpt ./agent_experiments/plan_1_claude/outputs/exp_1/best_worst_group.pt
```

## Consensus Features (all three plans agree)
- Worst-group checkpoint selection (best_worst_group.pt saved separately)
- SWA over final 20% of training epochs
- Masked motion-token denoising as auxiliary LLM loss (Exp 3)
- Word-level replay during sentence LLM training (Exp 3)
- Standardized eval_harness.py with identical JSON schema

## Kill Criteria (hard-coded in every train.py)
- VQ loss > 3.0 at any epoch: warning printed
- Worst-group val no improvement for [patience] epochs: early stop
- Codebook coverage drops below 60%: collapse warning
- Hand MSE not improving while body MSE is: diagnostic printed

## References
1. MotionGPT-2 (arXiv:2410.21747, 2024) -- Part-Aware VQVAE
2. SOKE (ICCV 2025) -- Signs as Tokens, multi-head decoding
3. MOGO (arXiv:2506.05952, 2025) -- Residual quantization for motion
4. SoundStream (IEEE TASLP 2022) -- Residual VQ with level dropout
5. MoMask (CVPR 2024) -- Masked motion modeling
6. RAG (NeurIPS 2020) -- Retrieval-augmented generation
7. SWA (UAI 2018) -- Stochastic weight averaging
