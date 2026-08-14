"""Model interfaces and architecture registry."""

from .base import DecoderModel, ModelMetadata
from .baseline import GPTModel, MultiHeadAttention, count_parameters
from .registry import (
    MODEL_REGISTRY,
    available_architectures,
    build_model,
    model_metadata,
    parameter_count,
)

__all__ = [
    "DecoderModel",
    "GPTModel",
    "MultiHeadAttention",
    "ModelMetadata",
    "MODEL_REGISTRY",
    "available_architectures",
    "build_model",
    "model_metadata",
    "parameter_count",
    "count_parameters",
]
