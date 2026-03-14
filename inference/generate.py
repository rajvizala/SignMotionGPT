"""
Generation and inference utilities with constrained decoding
"""
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Dict
from config import (
    SYSTEM_MSG, GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE,
    GEN_TOP_P, GEN_TOP_K, GEN_NO_REPEAT_NGRAM_SIZE,
    GEN_REPETITION_PENALTY, GEN_END_LOGIT_SLOPE
)


class LengthAwareMotionLogitsProcessor(LogitsProcessor):
    """
    Constrained decoding processor that:
    1. Enforces motion token vocabulary
    2. Controls sequence length (min/soft_target/max)
    3. Biases toward ending at soft_target length
    """
    
    def __init__(self, prompt_len, mot_begin_id, mot_end_id, motion_ids,
                 hard_min, soft_target, hard_max, end_logit_slope=0.25):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.mot_begin_id = int(mot_begin_id)
        self.mot_end_id = int(mot_end_id)
        self.motion_ids = torch.tensor(sorted(set(int(x) for x in motion_ids)))
        self.motion_plus_end = torch.tensor(
            sorted(set(list(self.motion_ids.tolist()) + [self.mot_end_id]))
        )
        self.hard_min = int(hard_min)
        self.soft_target = int(soft_target)
        self.hard_max = int(hard_max)
        self.end_logit_slope = float(end_logit_slope)
    
    def __call__(self, input_ids, scores):
        device = scores.device
        bs = scores.size(0)
        mask = torch.full_like(scores, float("-inf"))
        
        for b in range(bs):
            gen = input_ids[b, self.prompt_len:]
            
            # No tokens generated yet - must start with MOT_BEGIN
            if gen.numel() == 0:
                allowed = torch.tensor([self.mot_begin_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
                continue
            
            # Find MOT_BEGIN position
            begin_pos = (gen == self.mot_begin_id).nonzero(as_tuple=True)[0]
            if begin_pos.numel() == 0:
                allowed = torch.tensor([self.mot_begin_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
                continue
            
            # Already generated MOT_END - force EOS
            if (gen == self.mot_end_id).any():
                allowed = torch.tensor([self.mot_end_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
                continue
            
            # Count motion tokens after MOT_BEGIN
            after_begin = gen[begin_pos[0].item() + 1:]
            cur_len = after_begin.numel()
            
            # Before minimum length - only allow motion tokens
            if cur_len < self.hard_min:
                allowed = self.motion_ids.to(device)
                mask[b].index_fill_(0, allowed, 0.0)
            
            # After maximum length - force end
            elif cur_len >= self.hard_max:
                allowed = torch.tensor([self.mot_end_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
            
            # Between min and max - allow motion tokens or end
            else:
                allowed = self.motion_plus_end.to(device)
                mask[b].index_fill_(0, allowed, 0.0)
                
                # Bias toward ending at soft_target
                distance = max(0, cur_len - self.soft_target)
                bias = self.end_logit_slope * float(distance)
                scores[b, self.mot_end_id] = scores[b, self.mot_end_id] + bias
        
        return scores + mask


def get_len_controls(prompt_text: str, length_stats_by_text: Dict, global_median_len: int):
    """
    Get length controls (min/soft_target/max) for a given prompt
    """
    s = length_stats_by_text.get(prompt_text)
    if s is None:
        med = global_median_len
    else:
        med = s["median"]
    
    hard_min = max(1, int(0.6 * med))
    soft_tgt = med
    hard_max = max(hard_min + 4, int(1.4 * med))
    
    return hard_min, soft_tgt, hard_max


def generate_t2m(
    model,
    tokenizer,
    prompt_text: str,
    mot_begin_id: int,
    mot_end_id: int,
    motion_token_ids: list,
    length_stats_by_text: Dict,
    global_median_len: int,
    prompt_vocab: Dict = None,
    pid: str = None,
    has_pid: bool = False,
    max_new_tokens: int = None,
    per_prompt_vocab: bool = True
):
    """
    Generate motion sequence from text prompt with constrained decoding
    """
    model.eval()
    device = next(model.parameters()).device
    
    if max_new_tokens is None:
        max_new_tokens = GEN_MAX_NEW_TOKENS
    
    # Build prompt
    pid_tok = ""
    if has_pid and pid is not None:
        pid_tok = f"<PID_{pid}>"
    
    user_text = f"<T2M>{pid_tok}\n\n" + prompt_text
    prompt = (
        "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
        + "<|im_start|>user\n" + user_text + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].size(1)
    
    # Get length controls
    hard_min, soft_tgt, hard_max = get_len_controls(
        prompt_text, length_stats_by_text, global_median_len
    )
    
    # Get allowed motion tokens
    if per_prompt_vocab and prompt_vocab:
        allowed_motion_ids = prompt_vocab.get(prompt_text, motion_token_ids)
    else:
        allowed_motion_ids = motion_token_ids
    
    # Setup constrained decoding
    processors = LogitsProcessorList([
        LengthAwareMotionLogitsProcessor(
            prompt_len=prompt_len,
            mot_begin_id=mot_begin_id,
            mot_end_id=mot_end_id,
            motion_ids=allowed_motion_ids,
            hard_min=hard_min,
            soft_target=soft_tgt,
            hard_max=hard_max,
            end_logit_slope=GEN_END_LOGIT_SLOPE,
        )
    ])
    
    # Generate
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=min(max_new_tokens, hard_max + 4),
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=GEN_REPETITION_PENALTY,
            logits_processor=processors,
            eos_token_id=mot_end_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    reply = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    
    return reply