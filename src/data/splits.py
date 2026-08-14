"""Deterministic document-level dataset splitting."""

from __future__ import annotations

import random


def deduplicate_documents(documents: list[str]) -> list[str]:
    """Remove exact duplicate documents while preserving first-seen order."""
    unique: list[str] = []
    seen: set[str] = set()
    for document in documents:
        if document not in seen:
            seen.add(document)
            unique.append(document)
    return unique


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def split_documents(
    documents: list[str], train_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split documents into deterministic train and validation partitions."""
    documents = deduplicate_documents(documents)
    _validate_fraction("train_fraction", train_fraction)
    if len(documents) < 2:
        raise ValueError("At least two documents are required for a train/validation split")

    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    split_index = min(max(1, int(len(indices) * train_fraction)), len(indices) - 1)
    train_indices, validation_indices = indices[:split_index], indices[split_index:]
    return (
        [documents[index] for index in train_indices],
        [documents[index] for index in validation_indices],
    )


def _make_three_smoke_documents(documents: list[str]) -> list[str]:
    """Create three non-overlapping text chunks for tiny smoke corpora."""
    text = "\n\n".join(documents)
    if len(text) < 3:
        raise ValueError("At least three characters are required for a three-way split")
    first_end = max(1, len(text) // 3)
    second_end = max(first_end + 1, (2 * len(text)) // 3)
    second_end = min(second_end, len(text) - 1)
    return [text[:first_end], text[first_end:second_end], text[second_end:]]


def split_documents_three(
    documents: list[str], train_fraction: float, validation_fraction: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Split documents into deterministic train, validation and test partitions."""
    documents = deduplicate_documents(documents)
    _validate_fraction("train_fraction", train_fraction)
    _validate_fraction("validation_fraction", validation_fraction)
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if len(documents) < 3:
        documents = _make_three_smoke_documents(documents)

    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    train_end = min(max(1, int(len(indices) * train_fraction)), len(indices) - 2)
    validation_end = min(
        max(train_end + 1, int(len(indices) * (train_fraction + validation_fraction))),
        len(indices) - 1,
    )
    return (
        [documents[index] for index in indices[:train_end]],
        [documents[index] for index in indices[train_end:validation_end]],
        [documents[index] for index in indices[validation_end:]],
    )


__all__ = ["deduplicate_documents", "split_documents", "split_documents_three"]
