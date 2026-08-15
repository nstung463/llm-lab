from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from data.tokenizer import ByteLevelBPE
from evaluation.lm_eval_adapter import TinyLLMEval
from models.registry import build_model


def _write_checkpoint(tmp_path):
    tokenizer = ByteLevelBPE.fit(
        ["A small story about a cat.", "The child walked home."],
        vocab_size=270,
        min_frequency=1,
    )
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "rope_dim": 8,
        "rope_base": 10_000.0,
        "tie_embeddings": False,
    }
    model = build_model("mha", model_config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "architecture": "mha",
            "model_config": model_config,
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_state(),
        },
        checkpoint,
    )
    return checkpoint


def test_custom_model_implements_lm_eval_requests(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)
    model = TinyLLMEval(checkpoint, device="cpu", batch_size=2, prefer_best=False)

    requests = [
        SimpleNamespace(args=("The ", "cat")),
        SimpleNamespace(args=("A small ", "story")),
    ]
    scores = model.loglikelihood(requests)
    rolling = model.loglikelihood_rolling(
        [SimpleNamespace(args=("A small story about a cat.",))]
    )
    generated = model.generate_until(
        [SimpleNamespace(args=("A small", {"max_gen_toks": 2, "until": []}))]
    )

    assert len(scores) == 2
    assert all(isinstance(score, float) and isinstance(greedy, bool) for score, greedy in scores)
    assert len(rolling) == 1 and isinstance(rolling[0], float)
    assert len(generated) == 1 and isinstance(generated[0], str)
