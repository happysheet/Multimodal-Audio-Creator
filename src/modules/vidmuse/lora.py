"""
Lightweight LoRA injection utilities for VidMuse LMModel (audiocraft).

We avoid HF PEFT since LMModel does not implement `prepare_inputs_for_generation`.
This module:
  - Adds trainable low-rank adapters to selected nn.Linear layers.
  - Keeps base weights frozen.
  - Supports saving/loading LoRA state dicts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze base params
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        x_d = self.dropout(x)
        update = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return result + update * self.scaling


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def add_lora_adapters(model: nn.Module, target_modules: List[str], rank: int, alpha: int, dropout: float = 0.05):
    """
    Replace target Linear layers with LoRALinear wrappers.
    target_modules: list of substrings; any nn.Linear whose name contains one of them will be wrapped.
    """
    lora_names = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_module(model, name, wrapped)
        lora_names.append(name)
    return lora_names


def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]):
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


__all__ = ["add_lora_adapters", "extract_lora_state", "load_lora_state", "LoRALinear"]
