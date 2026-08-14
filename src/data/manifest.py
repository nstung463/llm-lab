"""Dataset provenance and split metadata."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetManifest:
    """Provenance and split details stored with every training run."""

    source_id: str
    source_url: str
    license: str
    license_url: str
    document_count: int
    document_sha256: str
    split_seed: int
    train_fraction: float
    train_document_count: int
    validation_document_count: int
    test_document_count: int = 0
    tokenizer_kind: str | None = None
    tokenizer_vocab_size: int | None = None
    tokenizer_sha256: str | None = None
    train_token_count: int | None = None
    validation_token_count: int | None = None
    test_token_count: int | None = None
    target_train_tokens: int | None = None
    target_validation_tokens: int | None = None
    target_test_tokens: int | None = None
    token_id_dtype: str | None = None
    train_token_sha256: str | None = None
    validation_token_sha256: str | None = None
    test_token_sha256: str | None = None
    validation_fraction: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def document_sha256(documents: list[str]) -> str:
    """Hash documents in order, including separators between documents."""
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest(
    documents: list[str],
    train_documents: list[str],
    validation_documents: list[str],
    source: dict[str, str],
    seed: int,
    train_fraction: float,
    test_documents: list[str] | None = None,
    *,
    tokenizer_kind: str | None = None,
    tokenizer_vocab_size: int | None = None,
    tokenizer_sha256: str | None = None,
    train_token_count: int | None = None,
    validation_token_count: int | None = None,
    test_token_count: int | None = None,
    target_train_tokens: int | None = None,
    target_validation_tokens: int | None = None,
    target_test_tokens: int | None = None,
    token_id_dtype: str | None = None,
    train_token_sha256: str | None = None,
    validation_token_sha256: str | None = None,
    test_token_sha256: str | None = None,
    validation_fraction: float | None = None,
) -> DatasetManifest:
    """Build a manifest from raw documents and their deterministic partitions."""
    required_source_fields = ("id", "url", "license", "license_url")
    missing = [field for field in required_source_fields if field not in source]
    if missing:
        raise ValueError(f"Source metadata is missing: {', '.join(missing)}")
    return DatasetManifest(
        source_id=source["id"],
        source_url=source["url"],
        license=source["license"],
        license_url=source["license_url"],
        document_count=len(documents),
        document_sha256=document_sha256(documents),
        split_seed=seed,
        train_fraction=train_fraction,
        train_document_count=len(train_documents),
        validation_document_count=len(validation_documents),
        test_document_count=len(test_documents or []),
        tokenizer_kind=tokenizer_kind,
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_sha256=tokenizer_sha256,
        train_token_count=train_token_count,
        validation_token_count=validation_token_count,
        test_token_count=test_token_count,
        target_train_tokens=target_train_tokens,
        target_validation_tokens=target_validation_tokens,
        target_test_tokens=target_test_tokens,
        token_id_dtype=token_id_dtype,
        train_token_sha256=train_token_sha256,
        validation_token_sha256=validation_token_sha256,
        test_token_sha256=test_token_sha256,
        validation_fraction=validation_fraction,
    )
