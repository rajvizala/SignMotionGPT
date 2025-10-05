"""
Training utilities and functions
"""
import math
import torch
from transformers import TrainingArguments, Trainer
from config import (
    BATCH_TRAIN, BATCH_EVAL, GRAD_ACCUM, LR, WARMUP,
    LOG_STEPS, EVAL_STEPS, SAVE_STEPS, SEED, DTYPE
)


def make_training_args(out_dir: str, epochs: int) -> TrainingArguments:
    """
    Create TrainingArguments for a training stage
    """
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=epochs,
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        warmup_ratio=WARMUP,
        bf16=(DTYPE == torch.bfloat16),
        fp16=(DTYPE == torch.float16),
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )


def train_stage(
    stage_name: str,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    out_dir: str,
    epochs: int
):
    """
    Train a single stage
    """
    print(f"\n{'='*60}")
    print(f"Training {stage_name}")
    print(f"{'='*60}")
    
    args = make_training_args(out_dir, epochs)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        data_collator=data_collator,
    )
    
    # Train
    print(f"Starting training for {stage_name}...")
    trainer.train()
    
    # Evaluate
    print(f"Evaluating {stage_name}...")
    metrics = trainer.evaluate()
    
    # Compute perplexity
    eval_loss = metrics.get("eval_loss", float("nan"))
    ppl = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")
    
    print(f"\n{stage_name} Results:")
    print(f"  eval_loss: {eval_loss:.4f}")
    print(f"  perplexity: {ppl:.3f}")
    
    # Save model (optional - can be commented out to save space)
    # trainer.save_model(out_dir)
    # print(f"Model saved to {out_dir}")
    
    return metrics


def save_model_to_hub(model, tokenizer, repo_id: str, stage_name: str):
    """
    Save model and tokenizer to HuggingFace Hub
    """
    print(f"\nSaving {stage_name} to HuggingFace Hub: {repo_id}")
    model.push_to_hub(repo_id, commit_message=f"Upload {stage_name}")
    tokenizer.push_to_hub(repo_id, commit_message=f"Upload {stage_name}")
    print(f"Successfully saved {stage_name}")


def load_model_from_hub(repo_id: str):
    """
    Load model and tokenizer from HuggingFace Hub
    """
    from unsloth import FastLanguageModel
    from config import MAX_SEQ_LEN, DTYPE
    
    print(f"\nLoading model from HuggingFace Hub: {repo_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo_id,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=True,
    )
    print(f"Successfully loaded model from {repo_id}")
    return model, tokenizer