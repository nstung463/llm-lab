"""Shared model contracts used by training and evaluation entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class DecoderModel(Protocol):
    """Minimal decoder-only interface shared by all registered models."""

    training_cfg: Mapping[str, object]

    def __call__(self, input_ids: Tensor, use_cache: bool = False) -> Tensor:
        ...

    def reset_kv_cache(self) -> None:
        ...


@dataclass(frozen=True)
class ModelMetadata:
    """Stable metadata for checkpoints, benchmark tables and manifests."""

    architecture: str
    family: str
    module: str
    parameters: int
    context_length: int
    layers: int
    embedding_dim: int
    vocabulary_size: int


def as_module(model: DecoderModel) -> nn.Module:
    """Narrow a protocol model to an nn.Module for optimizer/count utilities."""
    if not isinstance(model, nn.Module):
        raise TypeError("Registered decoder models must inherit torch.nn.Module")
    return model
