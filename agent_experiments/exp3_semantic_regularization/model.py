"""
Experiment 3 -- Semantic-Aware Regularisation for LLM Training

REFERENCES
----------
[1] Duquenne et al., "SignCLIP: Connecting Text and Sign Language by
    Contrastive Learning", EMNLP 2024.
[2] MoCLIP: "Motion-Aware Fine-Tuning and Distillation of CLIP for Human
    Motion Generation", arXiv:2505.10810, 2025.
[3] Cross-modality Data Augmentation (XmDA) for sign language translation,
    EMNLP 2023.
[4] MOGO: "Textual Condition Alignment" via LLM chain-of-thought reasoning
    for bridging real-world prompts to training data, arXiv:2506.05952, 2025.

CORE IDEA
---------
The current LLM training memorises exact (text -> motion_token_sequence)
mappings seen during training.  On unseen test sentences, the LLM produces
incoherent or default motion because it never learned the *semantic structure*
that connects text meaning to motion patterns.

We introduce three complementary regularisation techniques:

(A) **Contrastive Motion-Text Alignment Loss**
    Inspired by SignCLIP [1] and MoCLIP [2].  During LLM training, we add an
    auxiliary loss that pulls the LLM's hidden representation of the text
    prompt *close* to the hidden representation of the ground-truth motion
    tokens, and pushes it *away* from motion representations of other samples
    in the batch.  This forces the LLM to encode semantic meaning in its
    representations, not just next-token prediction.

(B) **Motion Token Label Smoothing**
    Standard cross-entropy treats all wrong tokens as equally bad.  But
    motion codes that are *neighbours* in the VQ-VAE codebook (small L2
    distance) should receive less penalty.  We define a soft target
    distribution that assigns probability mass proportional to codebook
    proximity.  This teaches the LLM that producing a "close" code is much
    better than a random code, which directly helps on unseen data where
    the exact code may not be seen but a close one would suffice.

(C) **Temporal Consistency Regularisation**
    Motion tokens at adjacent timesteps should be correlated.  We add a loss
    term that penalises large changes in the predicted logit distribution
    between consecutive timesteps, encouraging smooth, coherent generation.

WHY THIS SHOULD HELP GENERALIZATION
------------------------------------
1. Contrastive alignment creates a semantic bridge: similar text -> similar
   motion hidden state -> similar generated motion, even for unseen text.
2. Label smoothing with codebook distances relaxes the exact-match objective,
   letting the model learn that nearby codes are acceptable alternatives.
3. Temporal consistency reduces the likelihood of degenerate outputs
   (repeated or random tokens) that plague autoregressive models on unseen data.
"""

import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# (A) Contrastive Motion-Text Alignment
# ---------------------------------------------------------------------------

class ContrastiveAlignmentLoss(nn.Module):
    """
    InfoNCE-style contrastive loss between text and motion hidden states.

    For each sample in the batch, we extract:
    - text_repr: mean of LLM hidden states over the text/prompt tokens
    - motion_repr: mean of LLM hidden states over the motion tokens

    The loss maximises cosine similarity between matched (text_i, motion_i)
    pairs while minimising similarity with all other pairs in the batch.

    This is directly inspired by SignCLIP [1] and CLIP, adapted to work
    within the LLM's own hidden space rather than requiring a separate encoder.
    """

    def __init__(self, hidden_dim: int, projection_dim: int = 256,
                 temperature: float = 0.07):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.temperature = temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_lengths: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states: (B, seq_len, hidden_dim) from LLM
        prompt_lengths: (B,) number of prompt tokens per sample
        motion_mask: (B, seq_len) bool mask where True = motion token
        """
        B, S, D = hidden_states.shape

        text_reprs = []
        motion_reprs = []
        for i in range(B):
            pl = prompt_lengths[i].item()
            text_h = hidden_states[i, :pl].mean(dim=0)
            m_mask = motion_mask[i]
            if m_mask.any():
                motion_h = hidden_states[i][m_mask].mean(dim=0)
            else:
                motion_h = hidden_states[i, pl:].mean(dim=0)
            text_reprs.append(text_h)
            motion_reprs.append(motion_h)

        text_reprs = torch.stack(text_reprs)     # (B, D)
        motion_reprs = torch.stack(motion_reprs)  # (B, D)

        text_z = F.normalize(self.text_proj(text_reprs), dim=-1)
        motion_z = F.normalize(self.motion_proj(motion_reprs), dim=-1)

        logits = text_z @ motion_z.t() / self.temperature  # (B, B)
        labels = torch.arange(B, device=logits.device)

        loss_t2m = F.cross_entropy(logits, labels)
        loss_m2t = F.cross_entropy(logits.t(), labels)
        return (loss_t2m + loss_m2t) / 2


# ---------------------------------------------------------------------------
# (B) Codebook-Aware Label Smoothing
# ---------------------------------------------------------------------------

class CodebookLabelSmoother:
    """
    Replace hard one-hot targets for motion tokens with a soft distribution
    based on L2 distances in the VQ-VAE codebook.

    Given target code i, the soft distribution is:
        p(j) ~ exp(-dist(i,j) / tau)   for all j in motion vocab
        p(j) = 0                         for non-motion tokens

    Then blend: target = (1-alpha)*one_hot + alpha*soft_target

    This is a novel contribution combining ideas from:
    - Standard label smoothing (Szegedy et al., 2016)
    - Codebook-distance awareness from motion generation literature
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        motion_token_ids: List[int],
        vocab_size: int,
        alpha: float = 0.1,
        tau: float = 1.0,
    ):
        """
        codebook: (K, code_dim) VQ-VAE codebook vectors
        motion_token_ids: list of token IDs in the LLM vocab that correspond
                          to motion tokens (e.g. IDs for <M0>...<M511>)
        vocab_size: total LLM vocabulary size
        alpha: smoothing weight (0 = no smoothing, 1 = full soft targets)
        tau: temperature for distance-based distribution
        """
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.motion_token_ids = sorted(motion_token_ids)
        self.motion_set = set(motion_token_ids)

        K = codebook.shape[0]
        dist_matrix = torch.cdist(codebook.float(), codebook.float(), p=2)
        soft = torch.softmax(-dist_matrix / tau, dim=-1)  # (K, K)
        self.soft_targets = soft  # row i = soft target for code i

    def smooth_labels(
        self,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        labels: (B, S) integer target IDs (may contain -100 for masked)
        Returns: (B, S, vocab_size) soft target tensor
        """
        B, S = labels.shape
        soft = torch.zeros(B, S, self.vocab_size, device=device)

        valid_mask = labels != -100
        valid_labels = labels.clone()
        valid_labels[~valid_mask] = 0

        one_hot = F.one_hot(valid_labels, self.vocab_size).float()
        soft = (1 - self.alpha) * one_hot

        motion_ids_tensor = torch.tensor(self.motion_token_ids, device=device)
        id_to_code_idx = {tid: i for i, tid in enumerate(self.motion_token_ids)}

        for b in range(B):
            for s in range(S):
                if not valid_mask[b, s]:
                    soft[b, s] = 0
                    continue
                tid = labels[b, s].item()
                if tid in id_to_code_idx:
                    code_idx = id_to_code_idx[tid]
                    st = self.soft_targets[code_idx].to(device)
                    for j, mid in enumerate(self.motion_token_ids):
                        soft[b, s, mid] += self.alpha * st[j]

        soft[~valid_mask] = 0
        return soft


# ---------------------------------------------------------------------------
# (C) Temporal Consistency Regularisation
# ---------------------------------------------------------------------------

class TemporalConsistencyLoss(nn.Module):
    """
    Penalise large changes in predicted logit distributions between adjacent
    timesteps within the motion token region.

    L_tc = mean_{t in motion_region} KL(p_t || p_{t-1})

    This encourages smooth transitions and prevents the degenerate case
    where the model oscillates between unrelated codes.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (B, S, V) raw logits from LLM
        motion_mask: (B, S) bool mask where True = motion token position
        """
        B, S, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        shifted_probs = probs[:, :-1].detach()
        current_log_probs = log_probs[:, 1:]
        shifted_mask = motion_mask[:, :-1] & motion_mask[:, 1:]

        kl = F.kl_div(current_log_probs, shifted_probs, reduction='none').sum(-1)
        kl = kl * shifted_mask.float()

        n_valid = shifted_mask.float().sum().clamp(min=1)
        return self.weight * kl.sum() / n_valid


# ---------------------------------------------------------------------------
# Combined Regularised Training Wrapper
# ---------------------------------------------------------------------------

class SemanticRegularisedTrainer:
    """
    Wraps the standard LLM training loop with three additional loss terms.

    Total loss = CE_loss
               + lambda_contrastive * L_contrastive
               + lambda_temporal * L_temporal
               + (label smoothing replaces raw CE when enabled)
    """

    def __init__(
        self,
        model,
        tokenizer,
        codebook: Optional[torch.Tensor] = None,
        hidden_dim: int = 1536,
        lambda_contrastive: float = 0.1,
        lambda_temporal: float = 0.01,
        label_smooth_alpha: float = 0.1,
        label_smooth_tau: float = 1.0,
        use_contrastive: bool = True,
        use_label_smoothing: bool = True,
        use_temporal_consistency: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_contrastive = lambda_contrastive
        self.lambda_temporal = lambda_temporal
        self.use_contrastive = use_contrastive
        self.use_label_smoothing = use_label_smoothing
        self.use_temporal_consistency = use_temporal_consistency

        device = next(model.parameters()).device

        if use_contrastive:
            self.contrastive_loss = ContrastiveAlignmentLoss(
                hidden_dim=hidden_dim,
            ).to(device)

        if use_temporal_consistency:
            self.temporal_loss = TemporalConsistencyLoss(weight=lambda_temporal)

        if use_label_smoothing and codebook is not None:
            motion_token_ids = []
            for i in range(codebook.shape[0]):
                tid = tokenizer.convert_tokens_to_ids(f"<M{i}>")
                if tid != tokenizer.unk_token_id:
                    motion_token_ids.append(tid)
            self.label_smoother = CodebookLabelSmoother(
                codebook=codebook,
                motion_token_ids=motion_token_ids,
                vocab_size=len(tokenizer),
                alpha=label_smooth_alpha,
                tau=label_smooth_tau,
            )
        else:
            self.label_smoother = None
            if use_label_smoothing:
                print("[WARNING] Label smoothing requested but no codebook provided.")
                self.use_label_smoothing = False

        # Motion token detection
        self.m_start_id = tokenizer.convert_tokens_to_ids("<M_START>")
        self.m_end_id = tokenizer.convert_tokens_to_ids("<M_END>")

    def _get_motion_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Identify positions that are motion tokens (between M_START and M_END)."""
        B, S = input_ids.shape
        mask = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            in_motion = False
            for s in range(S):
                tid = input_ids[b, s].item()
                if tid == self.m_start_id:
                    in_motion = True
                    continue
                if tid == self.m_end_id:
                    in_motion = False
                    continue
                if in_motion:
                    mask[b, s] = True
        return mask

    def _get_prompt_lengths(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find prompt length (position of M_START token)."""
        B, S = input_ids.shape
        lengths = torch.full((B,), S, dtype=torch.long, device=input_ids.device)
        for b in range(B):
            positions = (input_ids[b] == self.m_start_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                lengths[b] = positions[0].item()
        return lengths

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with all regularisation terms.

        Returns: (total_loss, loss_breakdown_dict)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=self.use_contrastive,
        )

        loss_breakdown = {}

        # Base CE loss (or label-smoothed CE)
        if self.use_label_smoothing and self.label_smoother is not None:
            logits = outputs.logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            soft_targets = self.label_smoother.smooth_labels(
                shifted_labels, input_ids.device
            )
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(soft_targets * log_probs).sum(-1)
            valid = shifted_labels != -100
            ce_loss = ce_loss[valid].mean()
        else:
            ce_loss = outputs.loss

        total_loss = ce_loss
        loss_breakdown["ce_loss"] = ce_loss.item()

        # Contrastive alignment
        if self.use_contrastive and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
            motion_mask = self._get_motion_mask(input_ids)
            prompt_lengths = self._get_prompt_lengths(input_ids)
            cl = self.contrastive_loss(hidden, prompt_lengths, motion_mask)
            total_loss = total_loss + self.lambda_contrastive * cl
            loss_breakdown["contrastive_loss"] = cl.item()

        # Temporal consistency
        if self.use_temporal_consistency:
            motion_mask = self._get_motion_mask(input_ids)
            tc = self.temporal_loss(outputs.logits, motion_mask)
            total_loss = total_loss + tc
            loss_breakdown["temporal_loss"] = tc.item()

        loss_breakdown["total_loss"] = total_loss.item()
        return total_loss, loss_breakdown
