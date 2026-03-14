"""
Inference script for generating motion tokens from text prompts.
Updated to match the train_pipeline.py / test_overfit.py logic.

Usage:
    python inference.py --prompt "walking forward" --stage 3
"""
import os
import argparse
import torch
import random
from pathlib import Path

from config import (
    MODEL_NAME, PIPELINE_OUTPUT_DIR, CHECKPOINTS_DIR,
    OUT_S1, OUT_S2, OUT_S3, M_START, M_END,
    INFERENCE_TEMPERATURE, INFERENCE_TOP_K, INFERENCE_REPETITION_PENALTY
)
from data import read_json_data, deduplicate_and_prepare_data
from model import setup_model_and_tokenizer_raw, ensure_tokenizer_has_motion_tokens
from metrics import generate_motion, build_instruction_prompt

def load_trained_model(stage: int, device: torch.device):
    """
    Load a trained model from a specific stage checkpoint.
    """
    # 1. Load data to get all motion tokens (needed for vocabulary resizing)
    from config import DATA_JSON_PATH
    all_entries = read_json_data(DATA_JSON_PATH)
    cleaned_data, all_motion_tokens = deduplicate_and_prepare_data(all_entries)

    # 2. Determine which directory to load from
    stage_dirs = {1: OUT_S1, 2: OUT_S2, 3: OUT_S3}
    # Fallback to PIPELINE_OUTPUT_DIR if stage-specific dir doesn't exist
    load_dir = stage_dirs.get(stage)
    
    if not load_dir or not os.path.exists(load_dir):
        print(f"⚠️  Stage {stage} specific directory not found at {load_dir}. Trying {PIPELINE_OUTPUT_DIR}...")
        load_dir = PIPELINE_OUTPUT_DIR

    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"No model found at {load_dir}. Please train the model first.")

    print(f"\nLoading model from: {load_dir}")
    
    # Load model and tokenizer using standard transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(load_dir, trust_remote_code=True)
    
    # Ensure all motion tokens are present
    ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, cleaned_data

def inference(
    prompt: str,
    stage: int = 3,
    pid: str = None,
    output_file: str = None,
    device: torch.device = None,
    per_prompt_vocab: bool = False  # Added for compatibility with visualize.py
):
    """
    Generate motion tokens from a text prompt.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"Motion Generation Inference - Stage {stage}")
    print("="*60)
    print(f"Prompt: '{prompt}'")
    print(f"Device: {device}")
    
    # Load model and data
    model, tokenizer, cleaned_data = load_trained_model(stage, device)
    
    # Build prompt
    # Stage 3 uses word-only prompts; Stage 1 & 2 can use participant IDs
    include_participant = (stage != 3)
    
    # Logic for random PID selection for Stage 1 & 2 if not provided
    if include_participant and pid is None:
        word_lower = prompt.lower().strip()
        available_pids = sorted({
            item.get("participant_id", "") 
            for item in cleaned_data 
            if str(item.get("word", "")).lower().strip() == word_lower
        })
        if available_pids:
            pid = random.choice(available_pids)
            print(f"🎲 No PID provided for Stage {stage}. Randomly selected '{pid}' from {len(available_pids)} variants.")
        else:
            print(f"⚠️  Word '{prompt}' not found in training dataset. Defaulting to empty PID.")
            pid = ""

    full_prompt = build_instruction_prompt(
        word=prompt,
        participant_id=pid,
        include_participant=include_participant
    )
    
    # Generate
    print(f"\nGenerating motion for: '{prompt}'...")
    generated = generate_motion(model, tokenizer, full_prompt, device)
    
    print("\n" + "="*60)
    print("Generated Motion:")
    print("="*60)
    print(generated)
    print("="*60)
    
    # Optionally save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding="utf-8") as f:
            f.write(generated)
        print(f"\n✅ Output saved to: {output_file}")
    
    return generated

def main():
    parser = argparse.ArgumentParser(description="Generate motion tokens from text prompts")
    parser.add_argument("--prompt", type=str, required=True, help="Text description (e.g., 'walking')")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3], help="Stage model (1, 2, or 3)")
    parser.add_argument("--pid", type=str, default=None, help="Optional participant ID")
    parser.add_argument("--output", type=str, default=None, help="Optional output file")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    # Added for compatibility with visualize.py calling with per_prompt_vocab=True
    parser.add_argument("--per-prompt-vocab", action="store_true", help="Deprecated/Ignored (kept for compat)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inference(
        prompt=args.prompt,
        stage=args.stage,
        pid=args.pid,
        output_file=args.output,
        device=device
    )

if __name__ == "__main__":
    main()
