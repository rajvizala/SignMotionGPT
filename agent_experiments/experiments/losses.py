from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


def label_smoothed_nll_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    if smoothing <= 0:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)

    vocab = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)

    target = labels.clone()
    mask = target.ne(ignore_index)
    target = target.masked_fill(~mask, 0)

    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth
    loss = loss.masked_fill(~mask, 0.0)

    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    logp_a = F.log_softmax(logits_a, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()

    kl_ab = F.kl_div(logp_a, p_b, reduction="none").sum(-1)
    kl_ba = F.kl_div(logp_b, p_a, reduction="none").sum(-1)
    kl = 0.5 * (kl_ab + kl_ba)

    if mask is not None:
        kl = kl.masked_fill(~mask, 0.0)
        denom = mask.float().sum().clamp_min(1.0)
        return kl.sum() / denom
    return kl.mean()


@dataclass
class EWCState:
    fisher: Dict[str, torch.Tensor]
    params: Dict[str, torch.Tensor]


def estimate_fisher_diagonal(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int = 128,
) -> EWCState:
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    params: Dict[str, torch.Tensor] = {}

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for n, p in named_params:
        fisher[n] = torch.zeros_like(p, device=device)
        params[n] = p.detach().clone()

    processed = 0
    for batch in dataloader:
        if processed >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()

        for n, p in named_params:
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        processed += 1

    scale = max(1, processed)
    for n in fisher:
        fisher[n] = fisher[n] / float(scale)

    model.train()
    return EWCState(fisher=fisher, params=params)


def ewc_penalty(model: torch.nn.Module, state: EWCState) -> torch.Tensor:
    loss = torch.zeros((), device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n not in state.fisher:
            continue
        fisher_n = state.fisher[n]
        theta_star = state.params[n]
        loss = loss + (fisher_n * (p - theta_star).pow(2)).sum()
    return 0.5 * loss

