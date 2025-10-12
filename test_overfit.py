"""
Overfitting test script to verify model can learn the data.
Trains on only 50 words (~1500 samples) with early stopping at loss < 0.1

Usage:
    python test_overfit.py
"""
import os
import random
import torch
import warnings
import math
from collections import defaultdict

# Set seeds
from config import SEED
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

warnings.filterwarnings("ignore")

# Import modules
from config import (
    DATA_JSON_PATH, WORK_DIR, MAX_SEQ_LEN, BATCH_TRAIN, BATCH_EVAL,
    GRAD_ACCUM, LR, WARMUP, LOG_STEPS, SAVE_STEPS
)
from data import (
    load_dataset, build_motion_vocab, compute_length_stats,
    build_prompt_vocab, check_has_participant_id
)
from model import setup_model_and_tokenizer, get_motion_token_info
from templates import create_mapper
from collators import AssistantSpanCollator
from train import make_training_args
from generate import generate_t2m
from transformers import Trainer, TrainerCallback
from datasets import Dataset


# Overfitting test config
NUM_WORDS = 50
TARGET_WORD = "passport"  # Must include this word
OUT_DIR = os.path.join(WORK_DIR, "overfit_test")
EPOCHS_PER_STAGE = 50  # High number, rely on early stopping
EARLY_STOP_LOSS = 0.1  # Stop when eval loss drops below this


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when eval loss drops below threshold"""
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.best_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", float('inf'))
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            print(f"   📉 New best eval loss: {eval_loss:.4f}")
        
        if eval_loss < self.threshold:
            print(f"\n   ✅ Early stopping triggered! Eval loss {eval_loss:.4f} < {self.threshold}")
            control.should_training_stop = True
        
        return control


def create_small_dataset(raw_ds, num_words=50, target_word="passport"):
    """
    Create a small dataset with only num_words unique prompts.
    Ensures target_word is included.
    """
    print(f"\nCreating small dataset with {num_words} words (including '{target_word}')...")
    
    # Group by text_query
    by_text = defaultdict(list)
    for ex in raw_ds:
        by_text[ex["text_query"]].append(ex)
    
    # Get unique prompts
    all_prompts = list(by_text.keys())
    print(f"Total unique prompts in dataset: {len(all_prompts)}")
    
    # Ensure target_word is in the selection
    selected_prompts = []
    if target_word in by_text:
        selected_prompts.append(target_word)
        print(f"✅ Target word '{target_word}' found with {len(by_text[target_word])} samples")
    else:
        print(f"⚠️  Target word '{target_word}' not found in dataset!")
    
    # Randomly select remaining prompts
    remaining = [p for p in all_prompts if p != target_word]
    random.shuffle(remaining)
    selected_prompts.extend(remaining[:num_words - len(selected_prompts)])
    
    print(f"Selected {len(selected_prompts)} prompts")
    
    # Collect all samples for selected prompts
    small_samples = []
    for prompt in selected_prompts:
        small_samples.extend(by_text[prompt])
    
    print(f"Total samples in small dataset: {len(small_samples)}")
    print(f"Sample distribution:")
    for prompt in selected_prompts[:10]:  # Show first 10
        print(f"  - '{prompt}': {len(by_text[prompt])} samples")
    if len(selected_prompts) > 10:
        print(f"  ... and {len(selected_prompts) - 10} more prompts")
    
    return Dataset.from_list(small_samples)


def train_stage_with_early_stop(
    stage_name, model, tokenizer, train_ds, val_ds, collator, epochs, early_stop_threshold=0.1
):
    """Train a stage with early stopping"""
    print(f"\n{'='*60}")
    print(f"Training {stage_name} (Early Stop @ loss < {early_stop_threshold})")
    print(f"{'='*60}")
    
    args = make_training_args(
        out_dir=os.path.join(OUT_DIR, stage_name.lower().replace(" ", "_")),
        epochs=epochs,
        two_point_hub=False  # No Hub checkpointing for overfit test
    )
    
    # Override some args for faster training
    args.eval_strategy = "steps"
    args.eval_steps = 20  # Evaluate frequently to catch early stop
    args.logging_steps = 10
    args.save_steps = 10000  # Don't save intermediate checkpoints
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(threshold=early_stop_threshold)]
    )
    
    print(f"Starting training for {stage_name}...")
    trainer.train()
    
    print(f"\nEvaluating {stage_name}...")
    metrics = trainer.evaluate()
    
    eval_loss = metrics.get("eval_loss", float("nan"))
    ppl = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")
    
    print(f"\n{stage_name} Final Results:")
    print(f"  eval_loss: {eval_loss:.4f}")
    print(f"  perplexity: {ppl:.3f}")
    
    return metrics


def main():
    print("="*70)
    print("OVERFITTING TEST - Small Dataset Training")
    print("="*70)
    print(f"Target: Train on {NUM_WORDS} words to verify model can learn")
    print(f"Early stopping: loss < {EARLY_STOP_LOSS}")
    print(f"Test word: '{TARGET_WORD}'")
    
    # ==========================================
    # 1. Load and filter dataset
    # ==========================================
    print("\n[1/8] Loading dataset...")
    assert os.path.exists(DATA_JSON_PATH), f"Missing: {DATA_JSON_PATH}"
    
    raw_ds = load_dataset(DATA_JSON_PATH)
    print(f"Full dataset size: {len(raw_ds)}")
    
    # Create small dataset
    small_ds = create_small_dataset(raw_ds, num_words=NUM_WORDS, target_word=TARGET_WORD)
    
    # ==========================================
    # 2. Build vocab from small dataset
    # ==========================================
    print("\n[2/8] Building motion vocabulary...")
    codebook_size, global_max_id = build_motion_vocab(small_ds)
    
    print("\n[3/8] Computing length statistics...")
    length_stats_by_text, global_median_len = compute_length_stats(small_ds)
    print(f"Global median length: {global_median_len}")
    
    print("\nBuilding per-prompt vocabulary...")
    prompt_vocab = build_prompt_vocab(small_ds)
    
    has_pid = check_has_participant_id(small_ds)
    print(f"Has participant IDs: {has_pid}")
    
    unique_pids = None
    if has_pid:
        unique_pids = sorted({str(ex["participant_id"]) for ex in small_ds})
        print(f"Unique participants: {len(unique_pids)}")
    
    # ==========================================
    # 3. Setup model and tokenizer
    # ==========================================
    print("\n[4/8] Setting up model and tokenizer...")
    model, tokenizer, new_token_ids = setup_model_and_tokenizer(
        codebook_size, unique_pids
    )
    print(f"Model initialized with {len(tokenizer)} tokens")
    
    motion_token_ids, mot_begin_id, mot_end_id = get_motion_token_info(
        tokenizer, codebook_size
    )
    
    # ==========================================
    # 4. Prepare datasets for all stages
    # ==========================================
    print("\n[5/8] Preparing datasets for all stages...")
    
    # Use small dataset for all splits (intentional overfitting)
    # Split 90/10 for train/val
    split = small_ds.train_test_split(test_size=0.1, seed=SEED)
    
    # Stage 1: Motion-only LM
    mapper_s1 = create_mapper(1, has_pid)
    train_s1 = split["train"].map(mapper_s1, remove_columns=split["train"].column_names, num_proc=1)
    val_s1 = split["test"].map(mapper_s1, remove_columns=split["test"].column_names, num_proc=1)
    print(f"Stage 1 - Train: {len(train_s1)}, Val: {len(val_s1)}")
    
    # Stage 2: Multi-task
    mapper_s2 = create_mapper(2, has_pid)
    train_s2 = split["train"].map(mapper_s2, remove_columns=split["train"].column_names, num_proc=1)
    val_s2 = split["test"].map(mapper_s2, remove_columns=split["test"].column_names, num_proc=1)
    print(f"Stage 2 - Train: {len(train_s2)}, Val: {len(val_s2)}")
    
    # Stage 3: T2M SFT
    mapper_s3 = create_mapper(3, has_pid)
    train_s3 = split["train"].map(mapper_s3, remove_columns=split["train"].column_names, num_proc=1)
    val_s3 = split["test"].map(mapper_s3, remove_columns=split["test"].column_names, num_proc=1)
    print(f"Stage 3 - Train: {len(train_s3)}, Val: {len(val_s3)}")
    
    collator = AssistantSpanCollator(tokenizer, MAX_SEQ_LEN)
    
    # ==========================================
    # 5. Stage 1: Motion-only LM
    # ==========================================
    print("\n[6/8] Stage 1: Motion-only Language Model")
    metrics_s1 = train_stage_with_early_stop(
        "Stage1_MotionOnlyLM",
        model, tokenizer, train_s1, val_s1, collator,
        epochs=EPOCHS_PER_STAGE,
        early_stop_threshold=EARLY_STOP_LOSS
    )
    
    # ==========================================
    # 6. Stage 2: Multi-task Pretrain
    # ==========================================
    print("\n[7/8] Stage 2: Multi-task Pretrain")
    metrics_s2 = train_stage_with_early_stop(
        "Stage2_MultiTask",
        model, tokenizer, train_s2, val_s2, collator,
        epochs=EPOCHS_PER_STAGE,
        early_stop_threshold=EARLY_STOP_LOSS
    )
    
    # ==========================================
    # 7. Stage 3: T2M SFT
    # ==========================================
    print("\n[8/8] Stage 3: Text-to-Motion SFT")
    metrics_s3 = train_stage_with_early_stop(
        "Stage3_T2M_SFT",
        model, tokenizer, train_s3, val_s3, collator,
        epochs=EPOCHS_PER_STAGE,
        early_stop_threshold=EARLY_STOP_LOSS
    )
    
    # ==========================================
    # 8. Test generation on target word
    # ==========================================
    print("\n" + "="*70)
    print("OVERFITTING TEST COMPLETE")
    print("="*70)
    print(f"\nFinal losses:")
    print(f"  Stage 1: {metrics_s1.get('eval_loss', 'N/A')}")
    print(f"  Stage 2: {metrics_s2.get('eval_loss', 'N/A')}")
    print(f"  Stage 3: {metrics_s3.get('eval_loss', 'N/A')}")
    
    # Move model to GPU for generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Testing generation on '{TARGET_WORD}'")
    print(f"{'='*70}")
    
    # Generate for target word
    generated = generate_t2m(
        model=model,
        tokenizer=tokenizer,
        prompt_text=TARGET_WORD,
        mot_begin_id=mot_begin_id,
        mot_end_id=mot_end_id,
        motion_token_ids=motion_token_ids,
        length_stats_by_text=length_stats_by_text,
        global_median_len=global_median_len,
        prompt_vocab=prompt_vocab,
        has_pid=has_pid,
        per_prompt_vocab=True
    )
    
    print(f"\nGenerated for '{TARGET_WORD}':")
    print(generated)
    
    # Save for visualization
    output_file = os.path.join(OUT_DIR, f"{TARGET_WORD}_tokens.txt")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(generated)
    print(f"\n✅ Saved tokens to: {output_file}")
    
    # Also test a few other words from training set
    print(f"\n{'='*70}")
    print("Testing other training words")
    print(f"{'='*70}")
    
    test_words = list(prompt_vocab.keys())[:5]  # First 5 words
    for word in test_words:
        if word == TARGET_WORD:
            continue
        gen = generate_t2m(
            model=model,
            tokenizer=tokenizer,
            prompt_text=word,
            mot_begin_id=mot_begin_id,
            mot_end_id=mot_end_id,
            motion_token_ids=motion_token_ids,
            length_stats_by_text=length_stats_by_text,
            global_median_len=global_median_len,
            prompt_vocab=prompt_vocab,
            has_pid=has_pid,
            per_prompt_vocab=True
        )
        print(f"\n'{word}': {gen}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Review final losses above - should be < 0.1 if model can learn")
    print(f"2. Visualize '{TARGET_WORD}':")
    print(f"   python visualize.py --input {output_file} --output {OUT_DIR}/{TARGET_WORD}_vis.html")
    print(f"3. If visualization looks good, the model CAN learn the data")
    print(f"4. If not, there may be issues with:")
    print(f"   - VQ-VAE decoder mismatch")
    print(f"   - Token vocabulary mismatch")
    print(f"   - SMPL-X parameter layout mismatch")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

