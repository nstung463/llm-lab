"""Hugging Face causal-LM wrapper for the project architectures.

The module has a project-installed path through ``models.registry`` and a
standalone export path through ``architectures/`` next to this file.  The
latter makes a converted directory usable with ``trust_remote_code=True``
without importing the training repository.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_llm import LLMConfig
except ImportError:  # pragma: no cover - used by Transformers remote-code loading
    from configuration_llm import LLMConfig


def _tie_embeddings(model: nn.Module) -> None:
    input_embedding = getattr(model, "tok_emb", None)
    output_head = getattr(model, "out_head", None) or getattr(model, "lm_head", None)
    if input_embedding is None or output_head is None:
        raise ValueError("Model does not expose compatible embedding layers")
    if input_embedding.weight.shape != output_head.weight.shape:
        raise ValueError("Input and output embedding shapes do not match")
    output_head.weight = input_embedding.weight


def _standalone_build_model(architecture: str, model_config: dict[str, Any]) -> nn.Module:
    """Build a model from architecture files copied into an HF export."""
    export_root = Path(__file__).resolve().parent
    architecture_dir = export_root / "architectures"
    source_name = {
        "mha": "standard_mha.py",
        "gqa": "gqa.py",
        "mla": "mla.py",
        "moe": "moe.py",
        "v4": "v4.py",
    }.get(architecture)
    if source_name is None:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    source_path = architecture_dir / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Export is missing architecture source: {source_path}")

    # The educational GQA/MLA/MoE files import tiktoken only for their CLI
    # examples.  Keep inference-only HF exports free of that optional package.
    try:
        import tiktoken  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
    sys.path.insert(0, str(export_root))
    spec = importlib.util.spec_from_file_location(
        f"tiny_llm_export_{architecture}", source_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load exported architecture: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = dict(model_config)
    cfg["dtype"] = torch.float32
    cfg.setdefault("drop_rate", cfg.get("dropout", 0.0))
    cfg.setdefault("qkv_bias", False)
    if architecture in {"gqa", "moe"}:
        cfg.setdefault("n_kv_groups", cfg["n_heads"])
    if architecture == "moe":
        cfg.setdefault("num_experts", 4)
        cfg.setdefault("num_experts_per_tok", 2)
        cfg.setdefault("shared_expert_hidden_dim", cfg["hidden_dim"])
    if architecture == "mla":
        cfg.setdefault("latent_dim", max(16, cfg["emb_dim"] // 4))
        cfg.setdefault("num_experts", 0)
        cfg.setdefault("num_experts_per_tok", 0)

    model = module.GPTModel(cfg)
    if cfg.get("tie_embeddings", False):
        _tie_embeddings(model)
    model.training_cfg = cfg
    return model


def _build_decoder(architecture: str, model_config: dict[str, Any]) -> nn.Module:
    try:
        from models.registry import build_model
    except ModuleNotFoundError:  # pragma: no cover - exercised by standalone export
        return _standalone_build_model(architecture, model_config)
    return build_model(architecture, model_config)


class LLMForCausalLM(PreTrainedModel, GenerationMixin):
    """Causal-LM interface shared by MHA, GQA, MLA, MoE and V4."""

    config_class = LLMConfig
    base_model_prefix = "transformer"
    _supports_cache_class = False

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.transformer = _build_decoder(config.architecture, config.model_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.tok_emb

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer.tok_emb = value

    def get_output_embeddings(self) -> nn.Module:
        output_head = getattr(self.transformer, "out_head", None)
        if output_head is None:
            output_head = getattr(self.transformer, "lm_head", None)
        if output_head is None:
            raise AttributeError("Model does not expose an output head")
        return output_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        if hasattr(self.transformer, "out_head"):
            self.transformer.out_head = value
        else:
            self.transformer.lm_head = value

    def _check_attention_mask(self, attention_mask: Tensor | None, input_ids: Tensor) -> None:
        if attention_mask is not None and not torch.all(attention_mask.bool()):
            raise ValueError(
                "The educational decoders currently support unpadded batches only; "
                "provide an all-ones attention_mask."
            )
        if input_ids.shape[1] > int(self.config.max_position_embeddings):
            raise ValueError("input_ids exceed the configured context length")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Tensor, ...]:
        if input_ids is None:
            raise ValueError("input_ids are required")
        if inputs_embeds is not None:
            raise ValueError("inputs_embeds are not supported by the educational decoders")
        if position_ids is not None or past_key_values is not None:
            raise ValueError("position_ids and past_key_values are not supported yet")
        if token_type_ids is not None and not torch.all(token_type_ids == 0):
            raise ValueError("Only zero token_type_ids are supported")
        self._check_attention_mask(attention_mask, input_ids)

        # The custom cache objects are architecture-specific and cannot be
        # represented as HF DynamicCache yet.  Full-context recomputation is
        # slower but preserves exact logits and keeps generate() correct.
        logits = self.transformer(input_ids, use_cache=False)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(loss=loss, logits=logits)
        if return_dict is False:
            return output.to_tuple()
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Any | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "use_cache": False,
        }

    @staticmethod
    def _reorder_cache(past_key_values: Any, beam_idx: Tensor) -> Any:
        return past_key_values
