"""Backward-compatible import path for the Phase 01 architecture runner.

New code should import from ``llm.models.registry``.  Keeping this shim lets
existing notebooks and commands continue to work while Phase 02 introduces
the domain-oriented models package.
"""

from models.registry import (
    MODEL_REGISTRY,
    available_architectures,
    build_model,
    model_metadata,
    parameter_count,
)

__all__ = [
    "MODEL_REGISTRY",
    "available_architectures",
    "build_model",
    "model_metadata",
    "parameter_count",
]
