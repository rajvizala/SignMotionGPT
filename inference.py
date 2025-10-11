"""
Inference script for generating motion tokens from text prompts.
Run after training to generate motion sequences from any text description.

Usage:
    python inference.py --prompt "walking forward" --stage 3
    python inference.py --prompt "dancing" --stage 2 --output motion_output.txt
"""
import os
import argparse
import torch
from pathlib import Path

from config import (
    OUT_S1, OUT_S2, OUT_S3, MAX_SEQ_LEN, DATA_JSON_PATH,
    WORK_DIR
)
from data import (
    load_dataset, compute_length_stats, build_prompt_vocab,
    check_has_participant_id
)
from model import setup_model_and_tokenizer, get_motion_token_info
from generate import generate_t2m


def load_trained_model(stage: int, device: torch.device):
    """
    Load a trained model from a specific stage checkpoint.
    
    Args:
        stage: Stage number (1, 2, or 3)
        device: Device to load model on
    
    Returns:
        model, tokenizer, motion_token_ids, mot_begin_id, mot_end_id
    """
    stage_dirs = {1: OUT_S1, 2: OUT_S2, 3: OUT_S3}
    stage_dir = stage_dirs.get(stage)
    
    if not stage_dir or not os.path.exists(stage_dir):
        raise FileNotFoundError(
            f"Stage {stage} checkpoint not found at {stage_dir}. "
            f"Train stage {stage} first."
        )
    
    print(f"\nLoading Stage {stage} model from: {stage_dir}")
    
    # Load dataset to build vocab (needed for model setup)
    if not os.path.exists(DATA_JSON_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_JSON_PATH}")
    
    raw_ds = load_dataset(DATA_JSON_PATH)
    
    # Build motion vocab
    def max_token_in_example(ex):
        return max(int(x) for x in ex["motion_tokens"].split())
    
    global_max_id = max(max_token_in_example(ex) for ex in raw_ds)
    codebook_size = global_max_id + 1
    
    # Check for participant IDs
    has_pid = check_has_participant_id(raw_ds)
    unique_pids = None
    if has_pid:
        unique_pids = sorted({str(ex["participant_id"]) for ex in raw_ds})
    
    # Setup model and tokenizer with same config as training
    model, tokenizer, _ = setup_model_and_tokenizer(codebook_size, unique_pids)
    
    # Load trained weights from checkpoint
    # Try different checkpoint naming patterns
    possible_ckpts = [
        os.path.join(stage_dir, "pytorch_model.bin"),
        os.path.join(stage_dir, "model.safetensors"),
        os.path.join(stage_dir, "adapter_model.bin"),
    ]
    
    loaded = False
    for ckpt_path in possible_ckpts:
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            # Unsloth/PEFT models save adapters separately
            # The model will auto-load from the directory
            loaded = True
            break
    
    if not loaded:
        print(f"⚠️  No explicit checkpoint file found, using model directory: {stage_dir}")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Get motion token info
    motion_token_ids, mot_begin_id, mot_end_id = get_motion_token_info(
        tokenizer, codebook_size
    )
    
    print(f"✅ Stage {stage} model loaded successfully")
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   Motion tokens: {len(motion_token_ids)}")
    
    return model, tokenizer, motion_token_ids, mot_begin_id, mot_end_id, raw_ds


def inference(
    prompt: str,
    stage: int = 3,
    pid: str = None,
    output_file: str = None,
    per_prompt_vocab: bool = True,
    device: torch.device = None
):
    """
    Generate motion tokens from a text prompt.
    
    Args:
        prompt: Text description of desired motion
        stage: Which training stage model to use (1, 2, or 3)
        pid: Optional participant ID for personalization
        output_file: Optional file to save output tokens
        per_prompt_vocab: Whether to use per-prompt vocabulary constraints
        device: Device to run inference on
    
    Returns:
        Generated motion token string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"Motion Generation Inference - Stage {stage}")
    print("="*60)
    print(f"Prompt: '{prompt}'")
    print(f"Device: {device}")
    
    # Load model and dataset
    model, tokenizer, motion_token_ids, mot_begin_id, mot_end_id, raw_ds = load_trained_model(stage, device)
    
    # Compute length stats and prompt vocab
    print("\nComputing dataset statistics...")
    length_stats_by_text, global_median_len = compute_length_stats(raw_ds)
    prompt_vocab = build_prompt_vocab(raw_ds)
    has_pid = check_has_participant_id(raw_ds)
    
    # Generate motion tokens
    print(f"\nGenerating motion for: '{prompt}'")
    print(f"Per-prompt vocabulary: {per_prompt_vocab}")
    
    generated = generate_t2m(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        mot_begin_id=mot_begin_id,
        mot_end_id=mot_end_id,
        motion_token_ids=motion_token_ids,
        length_stats_by_text=length_stats_by_text,
        global_median_len=global_median_len,
        prompt_vocab=prompt_vocab,
        has_pid=has_pid,
        per_prompt_vocab=per_prompt_vocab,
        pid=pid
    )
    
    print("\n" + "="*60)
    print("Generated Motion:")
    print("="*60)
    print(generated)
    print("="*60)
    
    # Optionally save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(generated)
        print(f"\n✅ Output saved to: {output_file}")
    
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion tokens from text prompts using trained SignMotionGPT model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the desired motion (e.g., 'walking forward', 'dancing')"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Which training stage model to use (1=motion-only, 2=multi-task, 3=T2M SFT, default=3)"
    )
    parser.add_argument(
        "--pid",
        type=str,
        default=None,
        help="Optional participant ID for personalized generation (e.g., 'P40')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file to save generated tokens"
    )
    parser.add_argument(
        "--no-per-prompt-vocab",
        action="store_true",
        help="Disable per-prompt vocabulary constraints (allows all motion tokens)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to run inference on (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run inference
    inference(
        prompt=args.prompt,
        stage=args.stage,
        pid=args.pid,
        output_file=args.output,
        per_prompt_vocab=not args.no_per_prompt_vocab,
        device=device
    )


if __name__ == "__main__":
    main()
