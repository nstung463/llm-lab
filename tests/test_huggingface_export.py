from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("transformers")

from data.tokenizer import ByteLevelBPE
from huggingface.export import export_checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from models.registry import available_architectures, build_model


TINY_MODEL_CONFIG = {
    "context_length": 32,
    "emb_dim": 64,
    "hidden_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "n_kv_groups": 2,
    "latent_dim": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_expert_hidden_dim": 128,
    "head_dim": 16,
    "q_lora_rank": 16,
    "rope_dim": 8,
    "rope_base": 10_000.0,
    "window_size": 8,
    "compress_ratios": [0, 2],
    "index_topk": 2,
    "moe_hidden_dim": 128,
    "shared_hidden_dim": 128,
    "norm_eps": 1e-6,
}


@pytest.mark.parametrize("architecture", available_architectures())
def test_export_loads_with_huggingface_auto_classes(tmp_path, architecture):
    tokenizer = ByteLevelBPE.fit(
        ["A small test document.", "Một tài liệu kiểm thử nhỏ."], vocab_size=263
    )
    model_config = {**TINY_MODEL_CONFIG, "vocab_size": tokenizer.vocab_size}
    torch.manual_seed(7)
    model = build_model(architecture, model_config).eval()
    checkpoint = {
        "architecture": architecture,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_state(),
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    output_path = tmp_path / "hf"

    metadata = export_checkpoint(checkpoint_path, output_path, validate=True)

    assert metadata["architecture"] == architecture
    assert json.loads((output_path / "config.json").read_text())[
        "model_type"
    ] == "tiny_llm"
    assert (output_path / "model.safetensors").exists()
    assert (output_path / "configuration_llm.py").exists()
    assert (output_path / "modeling_llm.py").exists()

    config = AutoConfig.from_pretrained(
        output_path, trust_remote_code=True, local_files_only=True
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        output_path, trust_remote_code=True, local_files_only=True
    ).eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(output_path, local_files_only=True)
    assert config.architecture == architecture

    input_ids = torch.tensor(
        [tokenizer.encode("A small")], dtype=torch.long
    )
    with torch.no_grad():
        expected = model(input_ids, use_cache=False)
        actual = loaded(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    generated = loaded.generate(
        **hf_tokenizer("A small", return_tensors="pt"),
        max_new_tokens=2,
        do_sample=False,
    )
    assert generated.shape[0] == 1
    assert generated.shape[1] > input_ids.shape[1]
