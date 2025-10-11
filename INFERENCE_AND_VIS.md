# Inference & Visualization Quick Reference

## Overview
After training your 3-stage SignMotionGPT model, use these scripts to generate and visualize motions.

---

## 1. Inference (Generate Motion Tokens)

### Basic Usage
```bash
# Generate from Stage 3 model (recommended)
python inference.py --prompt "walking forward"

# Try different stages
python inference.py --prompt "dancing" --stage 1  # Motion-only LM
python inference.py --prompt "dancing" --stage 2  # Multi-task
python inference.py --prompt "dancing" --stage 3  # T2M SFT (best quality)
```

### Save Output
```bash
python inference.py --prompt "jumping" --output my_motion.txt
```

### With Participant ID
```bash
python inference.py --prompt "yoga pose" --pid P40
```

### Expected Output
```
============================================================
Motion Generation Inference - Stage 3
============================================================
Prompt: 'walking forward'
Device: cuda

Loading Stage 3 model from: /kaggle/working/SignMotionGPT/stage3_t2m_sft
✅ Stage 3 model loaded successfully

Generating motion for: 'walking forward'

============================================================
Generated Motion:
============================================================
<MOT_BEGIN><motion_224><motion_39><motion_76>...<MOT_END>
============================================================
```

---

## 2. Visualization (Motion Tokens → 3D Animation)

### Prerequisites

#### Option A: Use Google Drive (Colab/Kaggle)
Edit `setup_env.sh` and add your Google Drive file IDs:
```bash
VQVAE_MODEL_ID="1AbCdEfGhIj"           # VQ-VAE checkpoint (.pt)
VQVAE_STATS_ID="2KlMnOpQrSt"          # Normalization stats (.pt)
SMPLX_MODELS_ID="3UvWxYzAbCd"         # SMPL-X models (.zip)
```

Then run:
```bash
bash setup_env.sh
```

#### Option B: Manual Setup (Local)
```bash
export VQVAE_CHECKPOINT=/path/to/vqvae_model.pt
export VQVAE_STATS_PATH=/path/to/vqvae_stats.pt
export SMPLX_MODEL_DIR=/path/to/smplx_models
```

### Basic Usage

```bash
# Visualize token string
python visualize.py --tokens "<MOT_BEGIN><motion_177><motion_135>...<MOT_END>"

# Visualize from file
python visualize.py --input my_motion.txt

# Generate + visualize in one command
python visualize.py --prompt "walking" --stage 3
```

### Custom Output
```bash
python visualize.py \
  --input motion_tokens.txt \
  --output walk_animation.html \
  --title "Walking Forward" \
  --fps 30
```

### With Custom Paths
```bash
python visualize.py \
  --tokens "<MOT_BEGIN>..." \
  --vqvae-ckpt /custom/vqvae.pt \
  --stats /custom/stats.pt \
  --smplx-dir /custom/smplx_models \
  --output animation.html
```

### Expected Output
```
============================================================
Motion Visualization Pipeline
============================================================

[1/5] Parsing tokens...
   Parsed 15 tokens

[2/5] Loading VQ-VAE...
✅ VQ-VAE loaded (codebook size: 512)

[3/5] Loading normalization stats...
✅ Stats loaded (mean shape: (182,))

[4/5] Loading SMPL-X model...
✅ SMPL-X loaded

[5/5] Decoding and rendering...
   Decoding tokens to SMPL-X parameters...
   Decoded params shape: (16, 182)
   Converting parameters to vertices...
   Vertices shape: (16, 10475, 3), Faces: (20908, 3)
   Creating animation...
✅ Animation saved to: motion_animation.html

============================================================
✅ Visualization complete!
============================================================
```

---

## 3. Complete Workflow Example

### A. Train (already done)
```bash
python train_pipeline.py
```

### B. Generate Motion Tokens
```bash
python inference.py --prompt "college" --stage 3 --output college_motion.txt
```

### C. Visualize
```bash
python visualize.py --input college_motion.txt --output college_animation.html
```

### D. View Animation
Open `college_animation.html` in a browser. You'll see an interactive 3D SMPL-X character performing the motion. Use mouse to rotate/zoom, and click Play/Pause buttons.

---

## 4. Troubleshooting

### Inference Issues

**"Checkpoint not found"**
- Ensure you've trained all stages first: `python train_pipeline.py`
- Check that `OUT_S1`, `OUT_S2`, `OUT_S3` directories exist in `WORK_DIR`

**"Dataset not found"**
- Inference needs the dataset to build vocabulary
- Set `DATA_JSON_PATH` in `config.py` or via environment variable

### Visualization Issues

**"VQ-VAE checkpoint not found"**
- Download VQ-VAE model or set `VQVAE_CHECKPOINT` path
- The VQ-VAE is separate from LLM training (used to decode tokens to SMPL-X params)

**"SMPL-X models not found"**
- Download SMPL-X models from https://smpl-x.is.tue.mpg.de/
- Extract to a directory and set `SMPLX_MODEL_DIR`

**"No tokens to visualize"**
- Check token format: should contain `<motion_ID>` tags or space-separated numbers
- Example valid formats:
  - `<MOT_BEGIN><motion_177><motion_135><MOT_END>`
  - `177 135 152 200 46 142`

**"Shape mismatch" or "Decoding errors"**
- Ensure VQ-VAE checkpoint matches the codebook size used in LLM training
- Check `CODEBOOK_SIZE`, `CODE_DIM`, `SMPL_DIM` in `visualize.py` match training

---

## 5. Configuration

### Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VQVAE_CHECKPOINT` | VQ-VAE model path | `./data/vqvae_model.pt` |
| `VQVAE_STATS_PATH` | Normalization stats | `./data/vqvae_stats.pt` |
| `SMPLX_MODEL_DIR` | SMPL-X models directory | `./data/smplx_models` |
| `VIS_OUTPUT_DIR` | Output directory for animations | `WORK_DIR` |

### VQ-VAE Architecture (must match training)
In `visualize.py`:
```python
SMPL_DIM = 182           # SMPL-X parameter dimension
CODEBOOK_SIZE = 512      # Motion vocabulary size
CODE_DIM = 512           # Latent code dimension
VQ_ARGS = dict(
    width=512,
    depth=3,
    down_t=2,
    stride_t=2,
    ...
)
```

---

## 6. Tips

### Inference
- **Stage 3** generally produces best quality for text-to-motion
- **Stage 2** can handle M2T and denoising (but inference.py only does T2M)
- **Stage 1** generates motion without text conditioning (still needs prompt for length)
- Use `--no-per-prompt-vocab` to allow novel combinations (less constrained)

### Visualization
- **FPS 20-30** works well for most motions
- Longer sequences may take a few seconds to render
- The HTML file is self-contained and can be shared
- 3D mesh has ~10K vertices; animations can be large for long sequences

### Performance
- Inference: ~1-2 seconds per generation (depends on length)
- Visualization: ~3-10 seconds (depends on sequence length and batch size)
- Both run on GPU if available, fall back to CPU otherwise

---

## 7. Next Steps

- **Batch Inference**: Loop over multiple prompts and save outputs
- **Evaluate Quality**: Compare generated tokens to ground truth using edit distance
- **Fine-tune Generation**: Adjust `GEN_TEMPERATURE`, `GEN_TOP_P` in `config.py`
- **Export to Other Formats**: Extend `visualize.py` to export BVH, FBX, or USD

