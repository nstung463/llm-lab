"""Shared optimizer construction for comparable training runs."""

from __future__ import annotations

import torch
from torch import nn

from config import TrainingConfig


def build_adamw(model: nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    """Build AdamW with explicit decay/no-decay parameter groups.

    Biases, normalization parameters, embeddings, and other one-dimensional
    parameters are excluded from decoupled weight decay.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered_name = name.lower()
        if (
            parameter.ndim < 2
            or "bias" in lowered_name
            or "norm" in lowered_name
            or "emb" in lowered_name
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.adam_eps,
    )


__all__ = ["build_adamw"]
