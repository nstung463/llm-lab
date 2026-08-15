import torch

from benchmarking.compute import active_parameter_count, estimate_flops, estimate_kv_cache, estimate_training_flops


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


class _ToyExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.out_proj = torch.nn.Linear(4, 4, bias=False)


class _ToyV4Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.ffn = torch.nn.Module()
        block.ffn.experts = torch.nn.ModuleList([_ToyExpert() for _ in range(4)])
        block.ffn.shared_expert = _ToyExpert()
        self.blocks = torch.nn.ModuleList([block])
        self.training_cfg = {
            "context_length": 8,
            "emb_dim": 8,
            "n_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "window_size": 4,
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


def test_v4_active_parameters_and_flops_only_count_top_k_experts() -> None:
    model = _ToyV4Decoder()
    total_parameters = sum(p.numel() for p in model.parameters())
    active_parameters = active_parameter_count(model, "v4")
    assert active_parameters < total_parameters

    estimate = estimate_flops(model, "v4", sequence_length=8)
    all_expert_flops = 4 * 3 * (2 * 4 * 4)
    active_expert_flops = 3 * (2 * 4 * 4)
    linear_flops = sum(2 * module.in_features * module.out_features for module in model.modules() if isinstance(module, torch.nn.Linear))
    expected_linear_flops = linear_flops - all_expert_flops + active_expert_flops
    attention_flops = 4 * 4 * 8 * 1
    assert estimate.forward_flops_per_token == expected_linear_flops + attention_flops
