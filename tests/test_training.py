import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compare_architectures import evaluate as compare_evaluate
from config import ModelConfig, TrainingConfig, validate_resume_config
from data import NextTokenDataset
from model import GPTModel
from train_architectures import evaluate as architecture_evaluate
from training import evaluate, loss_for_batch, train
from training.schedule import cosine_lr
from train_architectures import _validate_global_tokens_per_update


def test_cosine_lr_uses_one_based_optimizer_updates() -> None:
    kwargs = {
        "warmup_steps": 3,
        "max_steps": 10,
        "lr": 3e-4,
        "min_lr": 3e-5,
    }
    assert cosine_lr(1, **kwargs) == pytest.approx(1e-4)
    assert cosine_lr(3, **kwargs) == pytest.approx(3e-4)
    assert cosine_lr(4, **kwargs) < 3e-4
    assert cosine_lr(10, **kwargs) == pytest.approx(3e-5)
    assert cosine_lr(11, **kwargs) == pytest.approx(3e-5)


def test_cosine_lr_without_warmup_starts_at_peak() -> None:
    kwargs = {"warmup_steps": 0, "max_steps": 10, "lr": 3e-4, "min_lr": 3e-5}
    assert cosine_lr(1, **kwargs) == pytest.approx(3e-4)
    assert cosine_lr(10, **kwargs) == pytest.approx(3e-5)


def test_training_config_rejects_warmup_without_decay() -> None:
    with pytest.raises(ValueError, match="smaller than max_steps"):
        TrainingConfig(max_steps=3, warmup_steps=3)


def test_training_accumulates_microbatches_and_tracks_tokens(monkeypatch) -> None:
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.forward_calls = 0

        def forward(self, inputs: torch.Tensor, *, use_cache: bool = False):
            self.forward_calls += 1
            first = self.weight.expand(*inputs.shape)
            second = torch.zeros_like(first)
            return torch.stack((first, second), dim=-1), None

    monkeypatch.setattr("training.loop.evaluate", lambda *args, **kwargs: 0.0)
    model = CountingModel()
    inputs = torch.zeros((2, 1), dtype=torch.long)
    targets = torch.zeros((2, 1), dtype=torch.long)
    loader = DataLoader(torch.utils.data.TensorDataset(inputs, targets), batch_size=1)
    callback_tokens = []

    train(
        model,
        loader,
        loader,
        TrainingConfig(
            max_steps=1,
            gradient_accumulation_steps=2,
            eval_every=99,
            eval_batches=1,
        ),
        torch.device("cpu"),
        checkpoint_callback=lambda step, optimizer, history, tokens: callback_tokens.append(tokens),
    )

    assert model.forward_calls == 2
    assert callback_tokens == [2]


def test_architecture_token_update_config_matches_batch_shape() -> None:
    cfg = TrainingConfig(batch_size=2, gradient_accumulation_steps=3)
    _validate_global_tokens_per_update({"global_tokens_per_update": 48}, cfg, context_length=8)
    with pytest.raises(ValueError, match="global_tokens_per_update"):
        _validate_global_tokens_per_update({"global_tokens_per_update": 47}, cfg, context_length=8)


def test_training_config_rejects_invalid_numeric_types() -> None:
    with pytest.raises(TypeError, match="max_steps must be an integer"):
        TrainingConfig(max_steps=3.0)
    with pytest.raises(ValueError, match="learning_rate must be finite"):
        TrainingConfig(learning_rate=float("nan"))


def test_resume_config_requires_same_dynamics_and_more_steps() -> None:
    saved = TrainingConfig(max_steps=3)
    with pytest.raises(ValueError, match="learning_rate"):
        validate_resume_config(
            {**saved.__dict__, "learning_rate": 1e-4},
            saved,
            saved_step=2,
        )
    with pytest.raises(ValueError, match="greater than the checkpoint step"):
        validate_resume_config(saved.__dict__, saved, saved_step=3)


def test_training_sets_the_current_lr_before_each_optimizer_update() -> None:
    torch.manual_seed(4)
    cfg = ModelConfig(vocab_size=17, context_length=8, emb_dim=16, n_heads=4, n_layers=1)
    model = GPTModel(cfg)
    tokens = [index % cfg.vocab_size for index in range(200)]
    loader = DataLoader(NextTokenDataset(tokens, cfg.context_length, stride=cfg.context_length), batch_size=4)
    training_cfg = TrainingConfig(max_steps=3, warmup_steps=1, eval_every=3, eval_batches=1, save_every=1)
    observed: list[float] = []

    def checkpoint_callback(step, optimizer, history, tokens_seen) -> None:
        observed.append(optimizer.param_groups[0]["lr"])

    train(model, loader, loader, training_cfg, torch.device("cpu"), checkpoint_callback=checkpoint_callback)
    expected = [
        cosine_lr(
            step,
            warmup_steps=training_cfg.warmup_steps,
            max_steps=training_cfg.max_steps,
            lr=training_cfg.learning_rate,
            min_lr=training_cfg.learning_rate * training_cfg.min_lr_ratio,
        )
        for step in range(1, training_cfg.max_steps + 1)
    ]
    assert observed == pytest.approx(expected)


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
