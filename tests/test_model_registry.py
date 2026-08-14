import torch
from torch import nn

from architectures import available_architectures as legacy_architectures
from models import (
    available_architectures,
    build_model,
    model_metadata,
    parameter_count,
)


def small_config() -> dict[str, object]:
    return {
        "vocab_size": 512,
        "context_length": 16,
        "emb_dim": 64,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "tie_embeddings": True,
        "n_kv_groups": 2,
        "latent_dim": 16,
        "head_dim": 16,
        "q_lora_rank": 16,
        "rope_dim": 8,
        "rope_base": 10000.0,
        "window_size": 4,
        "compress_ratios": [0, 2],
        "index_topk": 2,
        "moe_hidden_dim": 64,
        "shared_hidden_dim": 64,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "shared_expert_hidden_dim": 64,
        "norm_eps": 1e-6,
    }


def config_for(architecture: str) -> dict[str, object]:
    cfg = small_config()
    if architecture == "mla":
        cfg.update({"num_experts": 0, "num_experts_per_tok": 0})
    return cfg


def test_registry_is_stable_and_legacy_import_is_compatible() -> None:
    expected = ("mha", "gqa", "mla", "moe", "v4")
    assert available_architectures() == expected
    assert legacy_architectures() == expected


def test_all_registered_models_share_forward_and_metadata_contract() -> None:
    tokens = torch.randint(0, 512, (2, 16))
    for architecture in available_architectures():
        model = build_model(architecture, config_for(architecture))
        assert isinstance(model, nn.Module)
        logits = model(tokens, use_cache=False)
        assert logits.shape == (2, 16, 512)
        assert torch.isfinite(logits).all()
        metadata = model_metadata(architecture, model)
        assert metadata.architecture == architecture
        assert metadata.parameters == parameter_count(model)
        assert metadata.parameters > 0
        assert metadata.context_length == 16
        assert metadata.vocabulary_size == 512


def test_registry_rejects_unknown_architecture() -> None:
    try:
        build_model("unknown", small_config())
    except ValueError as error:
        assert "mha" in str(error)
    else:
        raise AssertionError("unknown architecture must be rejected")
