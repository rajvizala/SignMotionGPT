"""
Data collators with label masking for training
"""
import torch


class AssistantSpanCollator:
    """
    Collator that masks labels to only train on assistant responses.
    
    For where=="mot": labels only inside <MOT_BEGIN>...<MOT_END> in assistant
    For where=="text": labels entire assistant span (for M2T tasks)
    """
    
    def __init__(self, tokenizer, max_length):
        self.tok = tokenizer
        self.max_len = max_length
        
        # Get special token IDs
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.mot_beg = self.tok.convert_tokens_to_ids("<MOT_BEGIN>")
        self.mot_end = self.tok.convert_tokens_to_ids("<MOT_END>")
    
    def __call__(self, examples):
        texts = [e["text"] for e in examples]
        wheres = [e["where"] for e in examples]
        
        # Tokenize
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        )
        
        input_ids = enc["input_ids"]
        labels = input_ids.clone().fill_(-100)
        
        # Apply label masking per example
        for i, w in enumerate(wheres):
            seq = input_ids[i]
            
            # Find last <|im_start|> (start of assistant)
            starts = (seq == self.im_start).nonzero(as_tuple=True)[0]
            if starts.numel() == 0:
                continue
            
            a_start = int(starts[-1].item())
            
            # Find corresponding <|im_end|>
            sub = seq[a_start+1:]
            ends = (sub == self.im_end).nonzero(as_tuple=True)[0]
            a_end = (a_start + 1 + int(ends[0].item())) if ends.numel() > 0 else (seq.size(0) - 1)
            
            if w == "text":
                # Label entire assistant span
                labels[i, a_start+1:a_end] = seq[a_start+1:a_end]
            else:
                # Label only motion tokens between <MOT_BEGIN> and <MOT_END>
                asst = seq[a_start+1:a_end]
                bpos = (asst == self.mot_beg).nonzero(as_tuple=True)[0]
                epos = (asst == self.mot_end).nonzero(as_tuple=True)[0]
                
                if bpos.numel() > 0 and epos.numel() > 0 and epos[0] >= bpos[0]:
                    b = a_start + 1 + int(bpos[0].item())
                    e = a_start + 1 + int(epos[0].item())
                    labels[i, b:e+1] = seq[b:e+1]
        
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }