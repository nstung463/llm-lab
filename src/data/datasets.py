"""Next-token datasets for the current in-memory TinyStories pipeline."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


def take_token_budget(token_ids: Sequence[int], target_tokens: int) -> list[int]:
    """Return exactly ``target_tokens`` IDs from a token stream.

    The function deliberately does not pad or repeat data.  A preparation
    command must collect enough source data before calling it, otherwise a
    short-corpus mistake could silently change the effective training budget.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if len(token_ids) < target_tokens:
        raise ValueError(
            f"Token stream has {len(token_ids)} tokens, fewer than target budget {target_tokens}"
        )
    return [int(token_id) for token_id in token_ids[:target_tokens]]


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create fixed-length ``(input_ids, labels)`` windows from token IDs."""

    def __init__(
        self, token_ids: Sequence[int], context_length: int, stride: int | None = None
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if len(token_ids) <= context_length:
            raise ValueError("Need more tokens than context_length")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")

        # Preserve numpy memmaps; materialize only the requested window.
        self.tokens = token_ids
        self.context_length = context_length
        self.stride = context_length if stride is None else stride
        self.starts = list(range(0, len(self.tokens) - context_length, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        end = start + self.context_length
        inputs = np.asarray(self.tokens[start:end], dtype=np.int64)
        targets = np.asarray(self.tokens[start + 1 : end + 1], dtype=np.int64)
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    @property
    def signature(self) -> str:
        """Identify the exact token stream used by this dataset."""
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(np.asarray(self.tokens)).view(np.uint8))
        return digest.hexdigest()


class StatefulBatchLoader:
    """Small OLMo-style resumable batch loader for the in-memory pipeline.

    The state includes the current shuffled order and cursor, so a checkpoint
    resumes at the same batch instead of starting a new permutation. ``epoch``
    is one-based and identifies the epoch currently being consumed.
    """

    def __init__(
        self,
        dataset: NextTokenDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 1
        self.batch_index = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self._start_epoch()

    def _start_epoch(self) -> None:
        if self.shuffle:
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
        else:
            self.order = torch.arange(len(self.dataset), dtype=torch.long)
        self.batch_index = 0

    def _batch_count(self) -> int:
        if self.drop_last:
            return len(self.order) // self.batch_size
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self._batch_count()

    @property
    def batches_in_epoch(self) -> int:
        """Number of batches in one loader epoch."""
        return self._batch_count()

    @property
    def tokens_per_epoch(self) -> int:
        """Number of target tokens exposed by one complete training epoch."""
        samples = len(self.order)
        if self.drop_last:
            samples = (samples // self.batch_size) * self.batch_size
        return samples * self.dataset.context_length

    def _make_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        items = [self.dataset[int(index)] for index in indices]
        return torch.stack([item[0] for item in items]), torch.stack([item[1] for item in items])

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while self.batch_index < self._batch_count():
            start = self.batch_index * self.batch_size
            end = min(start + self.batch_size, len(self.order))
            indices = self.order[start:end]
            self.batch_index += 1
            if self.drop_last and len(indices) < self.batch_size:
                break
            yield self._make_batch(indices)
        self.epoch += 1
        self._start_epoch()

    def evaluation_iterator(self, max_batches: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate without changing training cursor, permutation or RNG state."""
        count = 0
        for start in range(0, len(self.dataset), self.batch_size):
            if max_batches is not None and count >= max_batches:
                break
            end = min(start + self.batch_size, len(self.dataset))
            if self.drop_last and end - start < self.batch_size:
                break
            count += 1
            yield self._make_batch(torch.arange(start, end, dtype=torch.long))

    def state_dict(self) -> dict[str, Any]:
        """Serialize loader position, order, RNG and dataset identity."""
        return {
            "state_version": 2,
            "dataset_signature": self.dataset.signature,
            "dataset_size": len(self.dataset),
            "context_length": self.dataset.context_length,
            "stride": self.dataset.stride,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "batches_in_epoch": self.batches_in_epoch,
            "tokens_per_epoch": self.tokens_per_epoch,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state and reject a loader built from different data/config."""
        checks = {
            "dataset_signature": self.dataset.signature,
            "dataset_size": len(self.dataset),
            "context_length": self.dataset.context_length,
            "stride": self.dataset.stride,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }
        for key, expected in checks.items():
            if state.get(key) != expected:
                raise ValueError(f"Loader state mismatch for {key}: {state.get(key)!r} != {expected!r}")
        order = torch.as_tensor(state["order"], dtype=torch.long)
        if order.numel() != len(self.dataset):
            raise ValueError("Loader state has an invalid sample order")
        # Version 1 stored completed epochs with a zero-based counter. Map it
        # to the one-based current-epoch convention when resuming old runs.
        if int(state.get("state_version", 1)) < 2:
            self.epoch = int(state["epoch"]) + 1
        else:
            self.epoch = int(state["epoch"])
        if self.epoch <= 0:
            raise ValueError("Loader state has an invalid epoch")
        self.batch_index = int(state["batch_index"])
        if not 0 <= self.batch_index <= self._batch_count():
            raise ValueError("Loader state has an invalid batch cursor")
        self.order = order.clone()
        self.generator.set_state(torch.as_tensor(state["generator_state"], dtype=torch.uint8))


def make_loaders(
    train_token_ids: Sequence[int],
    validation_token_ids: Sequence[int],
    context_length: int,
    batch_size: int,
    seed: int,
) -> tuple[StatefulBatchLoader, StatefulBatchLoader]:
    """Build resumable train/validation loaders for the current in-memory pipeline."""
    train_data = NextTokenDataset(train_token_ids, context_length)
    validation_data = NextTokenDataset(validation_token_ids, context_length)
    return (
        StatefulBatchLoader(
            train_data, batch_size, shuffle=True, seed=seed, drop_last=False
        ),
        StatefulBatchLoader(
            validation_data, batch_size, shuffle=False, seed=seed, drop_last=False
        ),
    )
