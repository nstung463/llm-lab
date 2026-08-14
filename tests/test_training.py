import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compare_architectures import evaluate as compare_evaluate
from config import ModelConfig, TrainingConfig
from data import NextTokenDataset
from model import GPTModel
from train_architectures import evaluate as architecture_evaluate
from training import evaluate, loss_for_batch, train


def test_training_uses_uncached_path_and_updates_weights() -> None:
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=17, context_length=8, emb_dim=16, n_heads=4, n_layers=1)
    model = GPTModel(cfg)
    tokens = [index % cfg.vocab_size for index in range(200)]
    loader = DataLoader(NextTokenDataset(tokens, cfg.context_length, stride=cfg.context_length), batch_size=4)
    before = model.lm_head.weight.detach().clone()
    history = train(model, loader, loader, TrainingConfig(max_steps=3, eval_every=1, eval_batches=1), torch.device("cpu"))
    assert len(history) == 3
    assert not torch.equal(before, model.lm_head.weight)
    inputs, targets = next(iter(loader))
    assert torch.isfinite(loss_for_batch(model, inputs, targets))


def test_evaluate_is_token_weighted_for_partial_final_batch() -> None:
    torch.manual_seed(6)
    cfg = ModelConfig(vocab_size=17, context_length=8, emb_dim=16, n_heads=4, n_layers=1)
    model = GPTModel(cfg).eval()
    tokens = [index % cfg.vocab_size for index in range(200)]
    loader = DataLoader(NextTokenDataset(tokens, cfg.context_length, stride=cfg.context_length), batch_size=3)
    value = evaluate(model, loader, torch.device("cpu"), max_batches=100)
    losses, counts = [], []
    for inputs, targets in loader:
        losses.append(loss_for_batch(model, inputs, targets).item())
        counts.append(targets.numel())
    expected = sum(loss * count for loss, count in zip(losses, counts)) / sum(counts)
    assert abs(value - expected) < 1e-7


class _BatchSensitiveModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, *, use_cache: bool = False) -> torch.Tensor:
        logits = torch.zeros((*inputs.shape, 2), dtype=torch.float32)
        logits[..., 0] = inputs[:, :1].float() * 4.0
        return logits


def test_architecture_evaluators_are_token_weighted() -> None:
    model = _BatchSensitiveModel().eval()
    batches = [
        (torch.zeros((3, 2), dtype=torch.long), torch.zeros((3, 2), dtype=torch.long)),
        (torch.ones((1, 2), dtype=torch.long), torch.zeros((1, 2), dtype=torch.long)),
    ]
    expected_loss = sum(
        F.cross_entropy(
            model(inputs).reshape(-1, 2), targets.reshape(-1), reduction="sum"
        ).item()
        for inputs, targets in batches
    ) / sum(targets.numel() for _, targets in batches)

    assert abs(architecture_evaluate(model, batches, torch.device("cpu"), 100) - expected_loss) < 1e-7
    result = compare_evaluate(model, batches, torch.device("cpu"), 100)
    assert abs(result["test_loss"] - expected_loss) < 1e-7
    assert result["tokens"] == 8
