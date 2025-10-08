"""
Main training pipeline for Motion LLM
Run this script to execute the full 3-stage training process
"""
import os
import random
import torch
import warnings
from collections import defaultdict

# Set seeds
from config import SEED
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")

# Import modules
from config import (
    DATA_JSON_PATH, OUT_S1, OUT_S2, OUT_S3,
    EPOCHS_S1, EPOCHS_S2, EPOCHS_S3,
    MAX_TRAIN_SAMPLES_S1, MAX_TRAIN_SAMPLES_S2, MAX_TRAIN_SAMPLES_S3,
    MAX_SEQ_LEN
)
from data import (
    load_dataset, build_motion_vocab, compute_length_stats,
    build_prompt_vocab, make_splits, check_has_participant_id
)
from model import setup_model_and_tokenizer, get_motion_token_info
from templates import create_mapper
from collators import AssistantSpanCollator
from train import train_stage
from generate import generate_t2m
from metrics import eval_t2m_set, build_text_to_refs


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("Motion LLM Training Pipeline")
    print("="*60)
    
    # ==========================================
    # 1. Load and prepare dataset
    # ==========================================
    print("\n[1/8] Loading dataset...")
    assert os.path.exists(DATA_JSON_PATH), f"Missing: {DATA_JSON_PATH}"
    
    raw_ds = load_dataset(DATA_JSON_PATH)
    print(f"Total samples: {len(raw_ds)}")
    
    # Build motion vocabulary
    print("\n[2/8] Building motion vocabulary...")
    codebook_size, global_max_id = build_motion_vocab(raw_ds)
    
    # Compute length statistics
    print("\n[3/8] Computing length statistics...")
    length_stats_by_text, global_median_len = compute_length_stats(raw_ds)
    print(f"Global median length: {global_median_len}")
    
    # Build per-prompt vocabulary
    print("\nBuilding per-prompt vocabulary...")
    prompt_vocab = build_prompt_vocab(raw_ds)
    
    # Check for participant IDs
    has_pid = check_has_participant_id(raw_ds)
    print(f"Has participant IDs: {has_pid}")
    
    # Get unique PIDs if they exist
    unique_pids = None
    if has_pid:
        unique_pids = sorted({str(ex["participant_id"]) for ex in raw_ds})
        print(f"Unique participants: {len(unique_pids)}")
    
    # ==========================================
    # 2. Setup model and tokenizer
    # ==========================================
    print("\n[4/8] Setting up model and tokenizer...")
    model, tokenizer, new_token_ids = setup_model_and_tokenizer(
        codebook_size, unique_pids
    )
    print(f"Model initialized with {len(tokenizer)} tokens")
    
    # Get motion token IDs
    motion_token_ids, mot_begin_id, mot_end_id = get_motion_token_info(
        tokenizer, codebook_size
    )
    
    # ==========================================
    # 3. Prepare datasets for all stages
    # ==========================================
    print("\n[5/8] Preparing datasets for all stages...")
    
    # Stage 1: Motion-only LM
    mapper_s1 = create_mapper(1, has_pid)
    train_s1, val_s1 = make_splits(raw_ds, mapper_s1, MAX_TRAIN_SAMPLES_S1)
    print(f"Stage 1 - Train: {len(train_s1)}, Val: {len(val_s1)}")
    # Show 1 sample (Stage 1)
    if len(train_s1) > 0:
        s1_example = train_s1[0]
        print("\n[Sample] Stage 1 mapped example:")
        print({k: (v[:240] + '...') if isinstance(v, str) and len(v) > 240 else v for k, v in s1_example.items()})
    
    # Stage 2: Multi-task
    mapper_s2 = create_mapper(2, has_pid)
    train_s2, val_s2 = make_splits(raw_ds, mapper_s2, MAX_TRAIN_SAMPLES_S2)
    print(f"Stage 2 - Train: {len(train_s2)}, Val: {len(val_s2)}")
    # Show 1 sample (Stage 2)
    if len(train_s2) > 0:
        s2_example = train_s2[0]
        print("\n[Sample] Stage 2 mapped example:")
        print({k: (v[:240] + '...') if isinstance(v, str) and len(v) > 240 else v for k, v in s2_example.items()})
    
    # Stage 3: T2M SFT
    mapper_s3 = create_mapper(3, has_pid)
    train_s3, val_s3 = make_splits(raw_ds, mapper_s3, MAX_TRAIN_SAMPLES_S3)
    print(f"Stage 3 - Train: {len(train_s3)}, Val: {len(val_s3)}")
    # Show 1 sample (Stage 3)
    if len(train_s3) > 0:
        s3_example = train_s3[0]
        print("\n[Sample] Stage 3 mapped example:")
        print({k: (v[:240] + '...') if isinstance(v, str) and len(v) > 240 else v for k, v in s3_example.items()})
    
    # Create data collator
    collator = AssistantSpanCollator(tokenizer, MAX_SEQ_LEN)
    
    # Build text-to-refs mapping for evaluation
    text_to_refs = build_text_to_refs(raw_ds)
    t2m_pairs = [(k, v) for k, v in text_to_refs.items()]
    
    # ==========================================
    # 4. Stage 1: Motion-only LM
    # ==========================================
    print("\n[6/8] Stage 1: Motion-only Language Model")
    metrics_s1 = train_stage(
        stage_name="Stage1_MotionOnlyLM",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_s1,
        eval_dataset=val_s1,
        data_collator=collator,
        out_dir=OUT_S1,
        epochs=EPOCHS_S1
    )
    
    # Move to GPU for inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Quick test generations
    print("\n[Stage 1 Quick Test Generations]")
    test_prompts = ["walking", "running", "college", "witch"]
    for prompt in test_prompts:
        gen = generate_t2m(
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
            per_prompt_vocab=False
        )
        print(f"{prompt} => {gen}")
    
    # ==========================================
    # 5. Stage 2: Multi-task Pretrain
    # ==========================================
    print("\n[7/8] Stage 2: Multi-task Pretrain (T2M/M2T/Denoise)")
    metrics_s2 = train_stage(
        stage_name="Stage2_MultiTask",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_s2,
        eval_dataset=val_s2,
        data_collator=collator,
        out_dir=OUT_S2,
        epochs=EPOCHS_S2
    )
    
    # Evaluate T2M
    print("\n[Stage 2 T2M Evaluation]")
    eval_t2m_set(
        model=model,
        tokenizer=tokenizer,
        sample_pairs=t2m_pairs,
        mot_begin_id=mot_begin_id,
        mot_end_id=mot_end_id,
        motion_token_ids=motion_token_ids,
        length_stats_by_text=length_stats_by_text,
        global_median_len=global_median_len,
        prompt_vocab=prompt_vocab,
        has_pid=has_pid,
        per_prompt_vocab=True,
        n_eval=50
    )
    
    # ==========================================
    # 6. Stage 3: T2M SFT
    # ==========================================
    print("\n[8/8] Stage 3: Text-to-Motion Supervised Fine-Tuning")
    metrics_s3 = train_stage(
        stage_name="Stage3_T2M_SFT",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_s3,
        eval_dataset=val_s3,
        data_collator=collator,
        out_dir=OUT_S3,
        epochs=EPOCHS_S3
    )
    
    # Move to GPU for final evaluation
    model.to(device)
    
    # Final evaluation
    print("\n[Stage 3 Final T2M Evaluation]")
    eval_t2m_set(
        model=model,
        tokenizer=tokenizer,
        sample_pairs=t2m_pairs,
        mot_begin_id=mot_begin_id,
        mot_end_id=mot_end_id,
        motion_token_ids=motion_token_ids,
        length_stats_by_text=length_stats_by_text,
        global_median_len=global_median_len,
        prompt_vocab=prompt_vocab,
        has_pid=has_pid,
        per_prompt_vocab=True,
        n_eval=100
    )
    
    # Final test generations
    print("\n[Final Sample Generations]")
    final_prompts = ["college", "president", "witch", "dance", "yoga"]
    for prompt in final_prompts:
        gen = generate_t2m(
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
            per_prompt_vocab=True
        )
        print(f"{prompt} => {gen}")
    
    # ==========================================
    # Training complete
    # ==========================================
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)
    print(f"\nStage 1 eval loss: {metrics_s1.get('eval_loss', 'N/A')}")
    print(f"Stage 2 eval loss: {metrics_s2.get('eval_loss', 'N/A')}")
    print(f"Stage 3 eval loss: {metrics_s3.get('eval_loss', 'N/A')}")
    print("\nModels saved to:")
    print(f"  - {OUT_S1}")
    print(f"  - {OUT_S2}")
    print(f"  - {OUT_S3}")


if __name__ == "__main__":
    main()