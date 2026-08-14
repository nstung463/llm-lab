import torch

from llm.benchmarking.compute import active_parameter_count, estimate_flops, estimate_kv_cache, estimate_training_flops


class _ToyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 8, bias=False)
        self.head = torch.nn.Linear(8, 16, bias=False)
        self.training_cfg = {
            "context_length": 8,
            "emb_dim": 8,
            "n_layers": 2,
        }


def test_flop_estimate_uses_linear_and_attention_convention() -> None:
    model = _ToyDecoder()
    estimate = estimate_flops(model, "mha")
    linear = 2 * (4 * 8 + 8 * 16)
    attention = 4 * 8 * 8 * 2
    assert estimate.forward_flops_per_token == linear + attention
    assert estimate.training_flops_per_token == 3 * estimate.forward_flops_per_token
    assert estimate_training_flops(model, "mha", 10) == 10 * estimate.training_flops_per_token


def test_mha_kv_cache_estimate_accumulates_k_and_v_per_layer() -> None:
    model = _ToyDecoder()
    estimate = estimate_kv_cache(model, "mha", sequence_length=8)
    expected_bytes_per_token = 2 * model.training_cfg["n_layers"] * model.training_cfg["emb_dim"] * 4
    assert estimate.cache_kind == "full_kv"
    assert estimate.bytes_per_token == expected_bytes_per_token
    assert estimate.total_bytes == expected_bytes_per_token * 8


def test_dense_active_parameter_count_equals_total() -> None:
    model = _ToyDecoder()
    assert active_parameter_count(model, "mha") == sum(p.numel() for p in model.parameters())
