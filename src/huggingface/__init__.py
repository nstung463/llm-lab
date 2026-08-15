"""Hugging Face export and loading helpers for the educational models."""

from __future__ import annotations


def register_auto_classes() -> None:
    """Register the project model with the Transformers Auto classes."""
    from .configuration_llm import LLMConfig
    from .modeling_llm import LLMForCausalLM

    LLMConfig.register_auto_class()
    LLMForCausalLM.register_auto_class("AutoModelForCausalLM")
