# Inference and visualization

This note uses the cleaned entrypoints in `scripts/`.

## Word-level inference

Generate motion tokens from a trained word-level checkpoint:

```bash
python scripts/infer_motion.py --prompt "walking forward" --stage 3
```

Useful variants:

```bash
python scripts/infer_motion.py --prompt "jumping" --output my_motion.txt
python scripts/infer_motion.py --prompt "yoga pose" --pid P40
python scripts/infer_motion.py --prompt "dance" --stage 2
```

## Visualization

Render motion tokens into an HTML animation:

```bash
python scripts/visualize_motion.py --input my_motion.txt
```

You can also pass tokens directly:

```bash
python scripts/visualize_motion.py --tokens "<M_START><M177><M135><M_END>"
```

Or generate and render in one call:

```bash
python scripts/visualize_motion.py --prompt "walking" --stage 3
```

## Common environment variables

```bash
export VQVAE_CHECKPOINT=/path/to/vqvae_model.pt
export VQVAE_STATS_PATH=/path/to/vqvae_stats.pt
export SMPLX_MODEL_DIR=/path/to/smplx_models
```

## Troubleshooting

### Inference

- If the model checkpoint is missing, train the pipeline first with `python scripts/train_word_pipeline.py`.
- Inference rebuilds the motion vocabulary from the dataset, so make sure `DATA_JSON_PATH` points at the same dataset family used during training.

### Visualization

- The VQ-VAE checkpoint, normalization stats, and SMPL-X assets must match the tokenization setup used during training.
- If decoding fails, verify the codebook size and SMPL-X dimensionality expected by `signmotion_gpt/visualization/visualize.py`.
