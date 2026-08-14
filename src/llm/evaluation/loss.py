"""Shared token-weighted causal-language-model evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _logits(model: nn.Module, inputs: Tensor) -> Tensor:
    output = model(inputs, use_cache=False)
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def evaluate_loss_stats(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Return token-weighted loss, token count and batch count."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    batch_count = 0
    iterator = batches.evaluation_iterator(max_batches) if hasattr(batches, "evaluation_iterator") else batches
    for batch_index, (inputs, targets) in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        targets = targets.to(device)
        logits = _logits(model, inputs.to(device))
        loss_sum += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        ).item()
        token_count += targets.numel()
        batch_count += 1
    model.train(was_training)
    if not token_count:
        raise ValueError("Evaluation loader produced no batches")
    loss = loss_sum / token_count
    return {"loss": loss, "tokens": token_count, "batches": batch_count}


def token_weighted_loss(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Return natural-log next-token cross entropy averaged by target token."""
    return float(evaluate_loss_stats(model, batches, device, max_batches)["loss"])
