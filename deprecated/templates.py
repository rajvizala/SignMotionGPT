"""
Prompt templates and mapping functions for different training stages
"""
import random
from data import ids_to_motion_specials
from config import SYSTEM_MSG, SEED

random.seed(SEED)


def pid_token_from_example(ex, has_pid: bool):
    """Get participant ID token from example"""
    if not has_pid:
        return ""
    
    pid = ex.get("participant_id", None)
    if pid is not None:
        return f"<PID_{pid}>"
    return "<PID_NULL>"


def map_stage1(ex, has_pid: bool):
    """
    Stage 1: Word + optional PID conditioning to learn motion language.
    The user explicitly provides the word (+PID); assistant outputs motion span.
    """
    mot = ids_to_motion_specials(ex["motion_tokens"])
    assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
    pid_tok = pid_token_from_example(ex, has_pid)
    word = ex.get("word", ex.get("text_query", ""))

    # Word + PID conditioning (no natural language chatter to keep it compact)
    user = f"<T2M>{pid_tok}\nword: {word}"
    text = (
        "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
    )

    return {"text": text, "where": "mot"}


def map_stage2(ex, has_pid: bool):
    """
    Stage 2: Multi-task (T2M/M2T/DENOISE)
    Randomly choose between text-to-motion, motion-to-text, or denoising
    """
    t = ex["text_query"]
    mot = ids_to_motion_specials(ex["motion_tokens"])
    pid_tok = pid_token_from_example(ex, has_pid)
    
    # Sample task type
    task = random.choices(["t2m", "m2t", "denoise"], weights=[0.5, 0.3, 0.2], k=1)[0]
    
    if task == "t2m":
        # Text to motion
        assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
        text = (
            "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
            + "<|im_start|>user\n" + f"<T2M>{pid_tok}\n\n" + t + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
        )
        where = "mot"
    
    elif task == "m2t":
        # Motion to text
        user = f"<M2T>{pid_tok}\n\n<MOT_BEGIN> {mot} <MOT_END>"
        text = (
            "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
            + "<|im_start|>user\n" + user + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n" + t + "\n<|im_end|>\n"
        )
        where = "text"
    
    else:
        # Denoising
        toks = mot.split()
        noisy = []
        for tok in toks:
            if random.random() < 0.15:
                noisy.append("<MOTION_MASK>")
            else:
                noisy.append(tok)
        
        user = f"<DENOISE>{pid_tok}\n\n<MOT_BEGIN> {' '.join(noisy)} <MOT_END>"
        assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
        text = (
            "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
            + "<|im_start|>user\n" + user + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
        )
        where = "mot"
    
    return {"text": text, "where": where, "text_query": t}


def map_stage3(ex, has_pid: bool):
    """
    Stage 3 (Instruct): Word-only request, no participant ID.
    The system prompt directs: "Output motion tokens for the given word".
    """
    t = ex["text_query"]
    mot = ids_to_motion_specials(ex["motion_tokens"])
    assistant = f"<MOT_BEGIN> {mot} <MOT_END>"

    # Instruct-style, no PID
    user = f"<T2M>\nword: {t}"
    text = (
        "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
    )

    return {
        "text": text,
        "where": "mot",
        "text_query": t,
        "motion_tokens": ex["motion_tokens"]
    }


def create_mapper(stage: int, has_pid: bool):
    """
    Create a mapper function for a specific stage
    """
    if stage == 1:
        return lambda ex: map_stage1(ex, has_pid)
    elif stage == 2:
        return lambda ex: map_stage2(ex, has_pid)
    elif stage == 3:
        return lambda ex: map_stage3(ex, has_pid)
    else:
        raise ValueError(f"Unknown stage: {stage}")