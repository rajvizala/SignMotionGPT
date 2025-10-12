"""
Overfitting test script to verify model can learn the data.
Trains on only 50 words (~1500 samples) with early stopping at loss < 0.5

Usage:
    python test_overfit.py
"""
# ======================
# 1) Imports and config
# ======================
import os, math, json, random, warnings, time
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import (
    TrainingArguments, Trainer, TrainerCallback,
    LogitsProcessor, LogitsProcessorList,
)
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from rapidfuzz.distance import Levenshtein

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings("ignore")

# Import base config
from config import DATA_JSON_PATH, WORK_DIR

# Overfit test config
NUM_WORDS = 50
TARGET_WORD = "passport"  # Must include this word (use a word that exists in dataset)
OUT_DIR = os.path.join(WORK_DIR, "overfit_test")
EARLY_STOP_LOSS = 0.5  # Stop when eval loss drops below this

# Model/Training
MODEL_NAME     = "unsloth/Qwen3-0.6B"
MAX_SEQ_LEN    = 512
BATCH_TRAIN    = 8
BATCH_EVAL     = 8
GRAD_ACCUM     = 8  # Keep same as full training for faster convergence
LR             = 1e-5
WARMUP         = 0.1
LOG_STEPS      = 20
EVAL_STEPS     = 50  # Evaluate more frequently for early stopping
SAVE_STEPS     = 10000  # Don't save checkpoints

# Epochs (high number, rely on early stopping)
EPOCHS_S1 = 100
EPOCHS_S2 = 100
EPOCHS_S3 = 100

# ======================
# 2) Early Stopping Callback
# ======================
class EarlyStoppingCallback(TrainerCallback):
    """Stop training when eval loss drops below threshold"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.best_loss = float("inf")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        
        eval_loss = metrics.get("eval_loss", float("inf"))
        
        # Track best
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            print(f"📉 New best eval loss: {eval_loss:.4f}")
        
        # Early stop if below threshold
        if eval_loss < self.threshold:
            print(f"\n🎯 Early stopping triggered! eval_loss={eval_loss:.4f} < {self.threshold}")
            control.should_training_stop = True
        
        return control


# ======================
# 3) Load dataset and build motion vocab
# ======================
print("="*70)
print("OVERFITTING TEST - Small Dataset Training")
print("="*70)
print(f"Target: Train on {NUM_WORDS} words to verify model can learn")
print(f"Early stopping: loss < {EARLY_STOP_LOSS}")
print(f"Test word: '{TARGET_WORD}'")
print()

print("[1/8] Loading dataset...")
with open(DATA_JSON_PATH, "r") as f:
    data = json.load(f)
raw_ds = Dataset.from_list(data)
print(f"Full dataset size: {len(raw_ds)}")

# Create small dataset with only NUM_WORDS unique words
def create_small_dataset(raw_ds, num_words=50, target_word="passport"):
    """
    Create a small dataset with only num_words unique words.
    Ensures target_word is included.
    Groups by the 'word' field (not 'text_query' which has variations).
    """
    print(f"\nCreating small dataset with {num_words} words (including '{target_word}')...")
    
    # Group by 'word' field (the actual word being signed)
    by_word = defaultdict(list)
    for ex in raw_ds:
        word = ex.get("word", ex["text_query"])  # Fallback to text_query if no 'word' field
        by_word[word].append(ex)
    
    # Get unique words
    all_words = list(by_word.keys())
    print(f"Total unique words in dataset: {len(all_words)}")
    
    # Ensure target_word is in the selection
    selected_words = []
    if target_word in by_word:
        selected_words.append(target_word)
        print(f"✅ Target word '{target_word}' found with {len(by_word[target_word])} samples")
    else:
        print(f"⚠️  Target word '{target_word}' not found in dataset!")
        print(f"   Available words sample: {list(all_words)[:20]}")
        # Pick first available word as fallback
        if all_words:
            target_word = all_words[0]
            selected_words.append(target_word)
            print(f"   Using fallback word: '{target_word}'")
    
    # Randomly select remaining words
    remaining = [w for w in all_words if w != target_word]
    random.shuffle(remaining)
    selected_words.extend(remaining[:num_words - len(selected_words)])
    
    print(f"Selected {len(selected_words)} words")
    
    # Collect all samples for selected words
    small_samples = []
    for word in selected_words:
        small_samples.extend(by_word[word])
    
    print(f"Total samples in small dataset: {len(small_samples)}")
    print(f"Sample distribution:")
    for word in selected_words[:10]:  # Show first 10
        print(f"  - '{word}': {len(by_word[word])} samples")
    if len(selected_words) > 10:
        print(f"  ... and {len(selected_words) - 10} more words")
    
    return Dataset.from_list(small_samples), target_word

raw_ds, TARGET_WORD = create_small_dataset(raw_ds, NUM_WORDS, TARGET_WORD)

print("\n[2/8] Building motion vocabulary...")
def _max_token_in_example(ex):
    return max(int(x) for x in ex["motion_tokens"].split())

global_max_id = 0
for ex in raw_ds:
    global_max_id = max(global_max_id, _max_token_in_example(ex))
CODEBOOK_SIZE = global_max_id + 1
print(f"Max motion token id found: {global_max_id}")
print(f"Codebook size: {CODEBOOK_SIZE}")

# Utilities
def ids_to_motion_specials(s: str) -> str:
    return " ".join(f"<motion_{int(x)}>" for x in s.split())

def motion_specials_to_ids(s: str) -> List[int]:
    toks = s.strip().split()
    ids = []
    for t in toks:
        if t.startswith("<motion_") and t.endswith(">"):
            try:
                ids.append(int(t[8:-1]))
            except:
                pass
    return ids

# Build prompt-wise stats
print("\n[3/8] Computing length statistics...")
def compute_length_stats(dataset) -> Tuple[Dict[str, Dict[str, int]], int]:
    by_text = defaultdict(list)
    for ex in dataset:
        by_text[ex["text_query"]].append(len(ex["motion_tokens"].split()))
    stats = {}
    all_lens = []
    for k, arr in by_text.items():
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        median = arr_sorted[n//2] if n % 2 == 1 else (arr_sorted[n//2 - 1] + arr_sorted[n//2]) // 2
        stats[k] = {"median": median, "min": arr_sorted[0], "max": arr_sorted[-1]}
        all_lens.extend(arr_sorted)
    all_lens = sorted(all_lens) or [16]
    gmed = all_lens[len(all_lens)//2]
    return stats, gmed

length_stats_by_text, global_median_len = compute_length_stats(raw_ds)
print(f"Global median length: {global_median_len}")

# Per-prompt whitelist (for decode-time restriction)
print("\nBuilding per-prompt vocabulary...")
def build_prompt_vocab(dataset) -> Dict[str, List[int]]:
    table = defaultdict(set)
    for ex in dataset:
        for x in ex["motion_tokens"].split():
            table[ex["text_query"]].add(int(x))
    return {k: sorted(v) for k, v in table.items()}

prompt_vocab = build_prompt_vocab(raw_ds)

# Participant IDs (optional)
has_pid = "participant_id" in raw_ds.column_names
print(f"Has participant IDs: {has_pid}")
if has_pid:
    unique_pids = sorted({str(ex["participant_id"]) for ex in raw_ds if ex.get("participant_id") is not None})
    print(f"Unique participants: {len(unique_pids)}")

# ======================
# 4) Build tokenizer/model with Unsloth
# ======================
print("\n[4/8] Setting up model and tokenizer...")
MOTION_TOKENS  = [f"<motion_{i}>" for i in range(CODEBOOK_SIZE)]
BOUNDARY_TOKENS  = ["<MOT_BEGIN>", "<MOT_END>"]
TASK_TOKENS   = ["<MOT_LM>", "<T2M>", "<M2T>", "<MOT_DENOISE>"]
PID_TOKENS    = []
if has_pid:
    unique_pids = sorted({str(ex["participant_id"]) for ex in raw_ds if ex.get("participant_id") is not None})
    PID_TOKENS = ["<PID_NULL>"] + [f"<PID_{pid}>" for pid in unique_pids]

ADDITIONAL_SPECIAL_TOKENS = BOUNDARY_TOKENS + MOTION_TOKENS + TASK_TOKENS + PID_TOKENS

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = MODEL_NAME,
    max_seq_length   = MAX_SEQ_LEN,
    dtype            = dtype,
    load_in_4bit     = True,
    trust_remote_code= True,
)

tokenizer.padding_side = "right"
existing = set(tokenizer.special_tokens_map_extended.get("additional_special_tokens", []))
to_add = [t for t in ADDITIONAL_SPECIAL_TOKENS if t not in existing]
if to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Attach LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    modules_to_save=["embed_tokens","lm_head"],
    use_gradient_checkpointing="unsloth",
)

print(f"Model initialized with {len(tokenizer)} tokens")

# ======================
# 5) Templates
# ======================
SYSTEM_MSG = (
    "You are a MotionGPT-style assistant for joint text–motion modeling.\n"
    "Follow these rules:\n"
    "1) For motion outputs, respond only with <MOT_BEGIN> <motion_*> ... <MOT_END> using space-separated <motion_ID> tokens; never output raw numbers.\n"
    "2) Respect task markers: <T2M> for text-to-motion, <M2T> for motion-to-text, <DENOISE> for masked motion reconstruction.\n"
    "3) Use participant token <PID_*> when present to personalize outputs.\n"
    "4) For T2M/DENOISE, output only the motion span; for M2T, output only fluent text.\n"
    "5) Do not echo system/user content; avoid extraneous text outside the required span.\n"
    "6) Prefer realistic lengths and smoothness consistent with the dataset's typical sequences."
)

def pid_token_from_example(ex):
    if not has_pid: return "<PID_NULL>"
    return f"<PID_{ex['participant_id']}>" if ex.get("participant_id", None) is not None else "<PID_NULL>"

# Stage 1: Motion-only LM (assistant predicts motion span)
def map_stage1(ex):
    mot = ids_to_motion_specials(ex["motion_tokens"])
    assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
    pid_tok = pid_token_from_example(ex)
    text = (
        "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
        + "<|im_start|>user\n" + f"{pid_tok}\n<MOT_LM>\n" + "<|im_end|>\n"
        + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
    )
    return {"text": text, "where": "mot"}

# Stage 2: Multi-task (T2M/M2T/DENOISE)
def map_stage2(ex):
    t = ex["text_query"]
    mot = ids_to_motion_specials(ex["motion_tokens"])
    pid_tok = pid_token_from_example(ex)
    task = random.choices(["t2m","m2t","denoise"], weights=[0.5,0.3,0.2], k=1)[0]
    if task == "t2m":
        assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
        text = (
            "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
            + "<|im_start|>user\n" + f"{pid_tok}\n<T2M>\n" + t + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
        )
        where = "mot"
    elif task == "m2t":
        user = f"<MOT_BEGIN> {mot} <MOT_END>"
        text = (
            "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
            + "<|im_start|>user\n" + f"{pid_tok}\n<M2T>\n" + user + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n" + t + "\n<|im_end|>\n"
        )
        where = "text"
    else:
        toks = mot.split()
        noisy = []
        for tok in toks:
            if random.random() < 0.15:
                noisy.append("<motion_0>")
            else:
                noisy.append(tok)
        user = f"<MOT_BEGIN> {' '.join(noisy)} <MOT_END>"
        assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
        text = (
            "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
            + "<|im_start|>user\n" + f"{pid_tok}\n<MOT_DENOISE>\n" + user + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
        )
        where = "mot"
    return {"text": text, "where": where, "text_query": t}

# Stage 3: T2M SFT (assistant predicts motion span)
def map_stage3(ex):
    t = ex["text_query"]
    mot = ids_to_motion_specials(ex["motion_tokens"])
    assistant = f"<MOT_BEGIN> {mot} <MOT_END>"
    pid_tok = pid_token_from_example(ex)
    text = (
        "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
        + "<|im_start|>user\n" + f"{pid_tok}\n<T2M>\n" + t + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n" + assistant + "\n<|im_end|>\n"
    )
    return {"text": text, "where": "mot", "text_query": t, "motion_tokens": ex["motion_tokens"]}

# ======================
# 6) Collators (label masking)
# ======================
class AssistantSpanCollator:
    # where=="mot": labels only inside <MOT_BEGIN>.. <MOT_END> in assistant
    # where=="text": labels entire assistant span (for M2T)
    def __init__(self, tokenizer, max_length):
        self.tok = tokenizer
        self.max_len = max_length
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end   = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.mot_beg  = self.tok.convert_tokens_to_ids("<MOT_BEGIN>")
        self.mot_end  = self.tok.convert_tokens_to_ids("<MOT_END>")

    def __call__(self, examples):
        texts = [e["text"] for e in examples]
        wheres = [e["where"] for e in examples]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        input_ids = enc["input_ids"]
        labels = input_ids.clone().fill_(-100)

        for i, w in enumerate(wheres):
            seq = input_ids[i]
            starts = (seq == self.im_start).nonzero(as_tuple=True)[0]
            if starts.numel() == 0:
                continue
            a_start = int(starts[-1].item())
            sub = seq[a_start+1:]
            ends = (sub == self.im_end).nonzero(as_tuple=True)[0]
            a_end = (a_start+1+int(ends[0].item())) if ends.numel() > 0 else (seq.size(0)-1)

            if w == "text":
                labels[i, a_start+1:a_end] = seq[a_start+1:a_end]
            else:
                asst = seq[a_start+1:a_end]
                bpos = (asst == self.mot_beg).nonzero(as_tuple=True)[0]
                epos = (asst == self.mot_end).nonzero(as_tuple=True)[0]
                if bpos.numel() > 0 and epos.numel() > 0 and epos[0] >= bpos[0]:
                    b = a_start+1+int(bpos[0].item())
                    e = a_start+1+int(epos[0].item())
                    labels[i, b:e+1] = seq[b:e+1]

        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}

# ======================
# 7) Prepare splits per stage
# ======================
print("\n[5/8] Preparing datasets for all stages...")
def make_splits(mapper):
    split = raw_ds.train_test_split(test_size=0.1, seed=SEED)
    train = split["train"].map(mapper, remove_columns=split["train"].column_names, num_proc=1)
    val   = split["test" ].map(mapper, remove_columns=split["test" ].column_names, num_proc=1)
    return train, val

train_s1, val_s1 = make_splits(map_stage1)
print(f"Stage 1 - Train: {len(train_s1)}, Val: {len(val_s1)}")
train_s2, val_s2 = make_splits(map_stage2)
print(f"Stage 2 - Train: {len(train_s2)}, Val: {len(val_s2)}")
train_s3, val_s3 = make_splits(map_stage3)
print(f"Stage 3 - Train: {len(train_s3)}, Val: {len(val_s3)}")

collator = AssistantSpanCollator(tokenizer, MAX_SEQ_LEN)

# ======================
# 8) Train helpers
# ======================
def make_args(out_dir, epochs):
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
        bf16=(dtype==torch.bfloat16),
        fp16=(dtype==torch.float16),
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

def train_stage(stage_name, out_dir, train_ds, val_ds, epochs):
    print(f"\n{'='*60}")
    print(f"Training {stage_name}")
    print(f"Early Stop: eval_loss < {EARLY_STOP_LOSS} OR max {epochs} epochs")
    print(f"{'='*60}")
    
    args = make_args(out_dir, epochs)
    tr = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(threshold=EARLY_STOP_LOSS)]
    )
    
    print(f"Starting training for {stage_name}...")
    tr.train()
    
    print(f"\nEvaluating {stage_name}...")
    metrics = tr.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
    print(f"{stage_name} eval_loss={metrics.get('eval_loss', 0.0):.4f}, ppl={ppl:.3f}")
    
    return metrics

# ======================
# 9) Constrained decoding (motion-only) with length-awareness
# ======================
motion_token_strs = [f"<motion_{i}>" for i in range(CODEBOOK_SIZE)]
motion_token_ids  = tokenizer.convert_tokens_to_ids(motion_token_strs)
mot_begin_id = tokenizer.convert_tokens_to_ids("<MOT_BEGIN>")
mot_end_id   = tokenizer.convert_tokens_to_ids("<MOT_END>")

class LengthAwareMotionLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len, mot_begin_id, mot_end_id, motion_ids, hard_min, soft_target, hard_max, end_logit_slope=0.25):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.mot_begin_id = int(mot_begin_id)
        self.mot_end_id = int(mot_end_id)
        self.motion_ids = torch.tensor(sorted(set(int(x) for x in motion_ids)))
        self.motion_plus_end = torch.tensor(sorted(set(list(self.motion_ids.tolist()) + [self.mot_end_id])))
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
            if gen.numel() == 0:
                allowed = torch.tensor([self.mot_begin_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
                continue
            begin_pos = (gen == self.mot_begin_id).nonzero(as_tuple=True)[0]
            if begin_pos.numel() == 0:
                allowed = torch.tensor([self.mot_begin_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
                continue
            if (gen == self.mot_end_id).any():
                allowed = torch.tensor([self.mot_end_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
                continue

            after_begin = gen[begin_pos[0].item() + 1:]
            cur_len = after_begin.numel()
            if cur_len < self.hard_min:
                allowed = self.motion_ids.to(device)
                mask[b].index_fill_(0, allowed, 0.0)
            elif cur_len >= self.hard_max:
                allowed = torch.tensor([self.mot_end_id], device=device)
                mask[b].index_fill_(0, allowed, 0.0)
            else:
                allowed = self.motion_plus_end.to(device)
                mask[b].index_fill_(0, allowed, 0.0)
                distance = max(0, cur_len - self.soft_target)
                bias = self.end_logit_slope * float(distance)
                scores[b, self.mot_end_id] = scores[b, self.mot_end_id] + bias

        return scores + mask

def get_len_controls(prompt_text: str):
    s = length_stats_by_text.get(prompt_text)
    if s is None:
        med = global_median_len
    else:
        med = s["median"]
    hard_min   = max(1, int(0.6 * med))
    soft_tgt   = med
    hard_max   = max(hard_min + 4, int(1.4 * med))
    return hard_min, soft_tgt, hard_max

def generate_t2m(prompt_text: str, pid: str = None, max_new_tokens: int = 256, per_prompt_vocab=True):
    model.eval()
    device = next(model.parameters()).device
    pid_tok = f"<PID_{pid}>" if (has_pid and pid is not None) else "<PID_NULL>"
    user_text = pid_tok + "\n<T2M>\n" + prompt_text

    prompt = (
        "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
        + "<|im_start|>user\n" + user_text + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].size(1)
    hard_min, soft_tgt, hard_max = get_len_controls(prompt_text)

    allowed_motion_ids = prompt_vocab.get(prompt_text, motion_token_ids) if per_prompt_vocab else motion_token_ids
    processors = LogitsProcessorList([
        LengthAwareMotionLogitsProcessor(
            prompt_len=prompt_len,
            mot_begin_id=mot_begin_id,
            mot_end_id=mot_end_id,
            motion_ids=allowed_motion_ids,
            hard_min=hard_min,
            soft_target=soft_tgt,
            hard_max=hard_max,
            end_logit_slope=0.25,
        )
    ])

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=min(max_new_tokens, hard_max + 4),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            no_repeat_ngram_size=6,
            repetition_penalty=1.2,
            logits_processor=processors,
            eos_token_id=mot_end_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    reply = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return reply

# ======================
# 10) Main training loop
# ======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Stage 1: Motion-only LM
    print("\n[6/8] Stage 1: Motion-only Language Model")
    metrics_s1 = train_stage("Stage1_MotionOnlyLM", os.path.join(OUT_DIR, "stage1"), train_s1, val_s1, EPOCHS_S1)
    
    # Stage 2: Multi-task
    print("\n[7/8] Stage 2: Multi-task Training")
    metrics_s2 = train_stage("Stage2_Multitask", os.path.join(OUT_DIR, "stage2"), train_s2, val_s2, EPOCHS_S2)
    
    # Stage 3: T2M SFT
    print("\n[8/8] Stage 3: T2M Fine-tuning")
    metrics_s3 = train_stage("Stage3_T2M_SFT", os.path.join(OUT_DIR, "stage3"), train_s3, val_s3, EPOCHS_S3)
    
    # Test generation on target word
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    
    # Get test prompts for target word
    target_examples = [ex for ex in raw_ds if ex.get("word", ex["text_query"]) == TARGET_WORD]
    if target_examples:
        test_prompt = target_examples[0]["text_query"]
        print(f"\nTest word: '{TARGET_WORD}'")
        print(f"Test prompt: '{test_prompt}'")
        print(f"\nGenerating motion for '{TARGET_WORD}'...")
        
        generated = generate_t2m(test_prompt, pid=None, per_prompt_vocab=True)
        print(f"\nGenerated output:\n{generated}")
        
        # Extract motion tokens
        if "<MOT_BEGIN>" in generated and "<MOT_END>" in generated:
            span = generated.split("<MOT_BEGIN>")[-1].split("<MOT_END>")[0]
            pred_ids = motion_specials_to_ids(span)
            print(f"\nExtracted motion token IDs ({len(pred_ids)} tokens):")
            print(" ".join(str(x) for x in pred_ids))
            
            # Save to file
            out_file = os.path.join(OUT_DIR, f"{TARGET_WORD}_tokens.txt")
            with open(out_file, "w") as f:
                f.write(" ".join(str(x) for x in pred_ids))
            print(f"\n✅ Saved tokens to: {out_file}")
        else:
            print("\n⚠️  Warning: Generated output missing <MOT_BEGIN>/<MOT_END> markers!")
    else:
        print(f"\n⚠️  Warning: No examples found for target word '{TARGET_WORD}'")
    
    print("\n" + "="*70)
    print("OVERFIT TEST COMPLETE")
    print("="*70)
    print(f"Final metrics:")
    print(f"  Stage 1: loss={metrics_s1.get('eval_loss', 0.0):.4f}")
    print(f"  Stage 2: loss={metrics_s2.get('eval_loss', 0.0):.4f}")
    print(f"  Stage 3: loss={metrics_s3.get('eval_loss', 0.0):.4f}")


if __name__ == "__main__":
    main()
