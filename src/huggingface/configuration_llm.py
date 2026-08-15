"""Transformers configuration for the five educational decoder architectures."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class LLMConfig(PretrainedConfig):
    """Configuration persisted in a Hugging Face model directory.

    ``model_config`` deliberately keeps the project's original configuration
    dictionary intact.  The adapter also exposes common Transformers names so
    generic tooling can inspect vocabulary, width, depth and context length.
    """

    model_type = "tiny_llm"

    def __init__(
        self,
        architecture: str = "mha",
        model_config: dict[str, Any] | None = None,
        tokenizer_kind: str | None = None,
        **kwargs: Any,
    ) -> None:
        model_config = dict(model_config or {})
        self.architecture = architecture
        self.model_config = model_config
        self.tokenizer_kind = tokenizer_kind

        kwargs.setdefault("vocab_size", model_config.get("vocab_size", 0))
        kwargs.setdefault("hidden_size", model_config.get("emb_dim", 0))
        kwargs.setdefault("num_hidden_layers", model_config.get("n_layers", 0))
        kwargs.setdefault(
            "max_position_embeddings", model_config.get("context_length", 0)
        )
        kwargs.setdefault("tie_word_embeddings", model_config.get("tie_embeddings", False))
        kwargs.setdefault("use_cache", False)
        super().__init__(**kwargs)
