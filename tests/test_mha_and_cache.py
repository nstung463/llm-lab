import torch

from llm.config import ModelConfig
from llm.model import GPTModel, MultiHeadAttention


def config() -> ModelConfig:
    return ModelConfig(vocab_size=41, context_length=16, emb_dim=32, n_heads=4, n_layers=2, dropout=0.0)


def test_mha_is_causal() -> None:
    torch.manual_seed(0)
    attention = MultiHeadAttention(config()).eval()
    tokens = torch.randn(2, 6, 32)
    changed_future = tokens.clone()
    changed_future[:, 4:] = torch.randn_like(changed_future[:, 4:]) * 100
    original, _ = attention(tokens)
    changed, _ = attention(changed_future)
    torch.testing.assert_close(original[:, :4], changed[:, :4])


def test_cached_logits_equal_full_prefix_logits() -> None:
    torch.manual_seed(1)
    model = GPTModel(config()).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 8))
    full_logits, _ = model(tokens)
    _, cache = model(tokens[:, :5], use_cache=True)
    cached_logits, cache = model(tokens[:, 5:], past_key_values=cache, use_cache=True)
    assert cache is not None
    torch.testing.assert_close(full_logits[:, 5:], cached_logits, rtol=1e-5, atol=1e-6)


def test_cached_and_uncached_generation_match_inside_and_across_window() -> None:
    torch.manual_seed(2)
    model = GPTModel(config()).eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 7))
    expected = model.generate_uncached(prompt, max_new_tokens=20)
    actual = model.generate_cached(prompt, max_new_tokens=20)
    torch.testing.assert_close(expected, actual)


def test_generation_temporarily_disables_dropout_and_restores_train_mode() -> None:
    torch.manual_seed(4)
    cfg = ModelConfig(vocab_size=41, context_length=16, emb_dim=32, n_heads=4, n_layers=2, dropout=0.25)
    model = GPTModel(cfg).train()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    expected = model.generate_uncached(prompt, 5)
    assert model.training
    actual = model.generate_cached(prompt, 5)
    assert model.training
    torch.testing.assert_close(expected, actual)


def test_inconsistent_layer_cache_is_rejected() -> None:
    torch.manual_seed(5)
    model = GPTModel(config()).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, 4))
    _, cache = model(tokens, use_cache=True)
    assert cache is not None
    malformed = list(cache)
    malformed[1] = None
    try:
        model(tokens[:, :1], past_key_values=tuple(malformed), use_cache=True)
    except ValueError as error:
        assert "every layer" in str(error)
    else:
        raise AssertionError("mixed layer cache should be rejected")
