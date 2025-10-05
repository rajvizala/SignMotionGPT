"""
Model and tokenizer initialization
"""
import torch
from unsloth import FastLanguageModel
from typing import List, Set
from config import (
    MODEL_NAME, MAX_SEQ_LEN, DTYPE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, LORA_MODULES_TO_SAVE
)


def build_special_tokens(codebook_size: int, unique_pids: List[str] = None) -> List[str]:
    """
    Build all special tokens for motion vocabulary
    """
    # Motion tokens
    motion_tokens = [f"<motion_{i}>" for i in range(codebook_size)]
    
    # Boundary tokens
    boundary_tokens = ["<MOT_BEGIN>", "<MOT_END>"]
    
    # Task tokens
    task_tokens = ["<T2M>", "<M2T>", "<DENOISE>", "<MOTION_MASK>"]
    
    # Participant ID tokens
    pid_tokens = []
    if unique_pids:
        pid_tokens = ["<PID_NULL>"] + [f"<PID_{pid}>" for pid in unique_pids]
    
    return boundary_tokens + motion_tokens + task_tokens + pid_tokens


def setup_model_and_tokenizer(codebook_size: int, unique_pids: List[str] = None):
    """
    Initialize model and tokenizer with custom tokens
    Returns: (model, tokenizer, new_token_ids)
    """
    # Build special tokens
    additional_special_tokens = build_special_tokens(codebook_size, unique_pids)
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    
    # Configure tokenizer
    tokenizer.padding_side = "right"
    
    # Add special tokens
    existing = set(tokenizer.special_tokens_map_extended.get("additional_special_tokens", []))
    to_add = [t for t in additional_special_tokens if t not in existing]
    
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=LORA_MODULES_TO_SAVE,
        use_gradient_checkpointing="unsloth",
    )
    
    # Get new token IDs for gradient masking
    new_token_ids = set(tokenizer.convert_tokens_to_ids(additional_special_tokens))
    
    # Apply gradient mask to prevent base vocab drift
    apply_gradient_mask(model, new_token_ids)
    
    return model, tokenizer, new_token_ids


def apply_gradient_mask(model, new_token_ids: Set[int]):
    """
    Apply gradient mask so only new token embeddings are updated
    """
    def mask_rows_hook(param, rows: set):
        mask = torch.zeros(param.size(0), device=param.device, dtype=param.dtype)
        idxs = sorted(list(rows))
        if len(idxs) > 0:
            mask[idxs] = 1.0
        param.register_hook(lambda g: g * mask.unsqueeze(1))
    
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        
        mask_rows_hook(emb, new_token_ids)
        mask_rows_hook(head, new_token_ids)


def get_motion_token_info(tokenizer, codebook_size: int):
    """
    Get motion token IDs and boundary token IDs
    Returns: (motion_token_ids, mot_begin_id, mot_end_id)
    """
    motion_token_strs = [f"<motion_{i}>" for i in range(codebook_size)]
    motion_token_ids = tokenizer.convert_tokens_to_ids(motion_token_strs)
    mot_begin_id = tokenizer.convert_tokens_to_ids("<MOT_BEGIN>")
    mot_end_id = tokenizer.convert_tokens_to_ids("<MOT_END>")
    
    return motion_token_ids, mot_begin_id, mot_end_id