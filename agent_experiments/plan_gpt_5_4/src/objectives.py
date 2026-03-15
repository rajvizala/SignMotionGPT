from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def build_randomized_position_ids(
    attention_mask: torch.Tensor,
    max_position_offset: int,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    base = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1)
    offsets = torch.randint(
        low=0,
        high=max(1, max_position_offset),
        size=(batch_size, 1),
        device=attention_mask.device,
    )
    position_ids = base + offsets
    return position_ids * attention_mask.long()


def masked_mean(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    summed = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def contrastive_alignment_loss(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    prompt_mask = (labels == -100) & attention_mask.bool()
    motion_mask = (labels != -100) & attention_mask.bool()

    prompt_repr = masked_mean(hidden_states, prompt_mask)
    motion_repr = masked_mean(hidden_states, motion_mask)

    prompt_repr = F.normalize(prompt_repr, dim=-1)
    motion_repr = F.normalize(motion_repr, dim=-1)

    logits = prompt_repr @ motion_repr.transpose(0, 1)
    logits = logits / max(temperature, 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_a + loss_b)


def maybe_position_ids(
    attention_mask: torch.Tensor,
    enabled: bool,
    max_position_offset: int,
) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    return build_randomized_position_ids(
        attention_mask=attention_mask,
        max_position_offset=max_position_offset,
    )
