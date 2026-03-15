from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **base_kwargs):
        if rho <= 0:
            raise ValueError("rho must be > 0")
        self.rho = rho
        self.base_optimizer = base_optimizer_cls(params, **base_kwargs)
        defaults = self.base_optimizer.defaults
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].pop("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM needs a closure with two forward-backward passes.")
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norms.append(p.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


@dataclass
class SWAAccumulator:
    start_step: int
    num_updates: int = 0
    averages: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module, global_step: int):
        if global_step < self.start_step:
            return
        state = model.state_dict()
        if self.averages is None:
            self.averages = {k: v.detach().clone() for k, v in state.items()}
            self.num_updates = 1
            return
        self.num_updates += 1
        beta = 1.0 / float(self.num_updates)
        for k in self.averages.keys():
            self.averages[k].mul_(1.0 - beta).add_(state[k], alpha=beta)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        if self.averages is None or self.num_updates == 0:
            return
        model.load_state_dict(self.averages, strict=False)

