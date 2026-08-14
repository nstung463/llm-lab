"""Small tokenizer boundary used by the learning pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class ByteLevelBPE:
    """Trainable byte-level BPE with an explicit document-boundary token."""

    eos_token = "<|endoftext|>"
    unk_token = "<|unk|>"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.token_to_id(self.eos_token)
        self.unk_id = tokenizer.token_to_id(self.unk_token)
        if self.eos_id is None:
            raise ValueError("Tokenizer is missing its end-of-text token")
        if self.unk_id is None:
            raise ValueError("Tokenizer is missing its unknown token")

    @classmethod
    def fit(
        cls, documents: list[str], vocab_size: int, min_frequency: int = 2
    ) -> "ByteLevelBPE":
        """Fit the tokenizer on the supplied documents only."""
        if vocab_size < 258:
            raise ValueError("Byte-level BPE vocab_size must be at least 258")
        if not documents:
            raise ValueError("Cannot train a tokenizer without documents")

        tokenizer = Tokenizer(models.BPE(unk_token=cls.unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[cls.unk_token, cls.eos_token],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(documents, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "ByteLevelBPE":
        """Restore a tokenizer serialized by :meth:`to_state`."""
        tokenizer_json = state.get("tokenizer_json")
        if not isinstance(tokenizer_json, str):
            raise ValueError("Tokenizer state is missing 'tokenizer_json'")
        return cls(Tokenizer.from_str(tokenizer_json))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode one document without adding EOS."""
        return self.tokenizer.encode(text).ids

    def encode_documents(self, documents: list[str]) -> list[int]:
        """Encode documents and append EOS after every document."""
        token_ids: list[int] = []
        for document in documents:
            token_ids.extend(self.encode(document))
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs, including special tokens when present."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def to_state(self) -> dict[str, object]:
        """Return a self-contained tokenizer state for checkpoints."""
        return {
            "kind": "byte_level_bpe",
            "tokenizer_json": self.tokenizer.to_str(),
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }

    def save(self, path: Path) -> None:
        """Save the raw tokenizer JSON for inspection and reuse."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.tokenizer.to_str(), encoding="utf-8")


# Compatibility name for the existing training scripts and checkpoints.
BPETokenizer = ByteLevelBPE


class TiktokenTokenizer:
    """Wrapper around a fixed OpenAI ``tiktoken`` encoding such as GPT-2."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)
        self.eos_id = self.tokenizer.eot_token
        self.unk_id = None

    @classmethod
    def from_name(cls, name: str = "r50k_base") -> "TiktokenTokenizer":
        """Load a built-in tiktoken encoding by name."""
        if name not in {"gpt2", "r50k_base"}:
            raise ValueError("Supported tiktoken encodings are 'gpt2' and 'r50k_base'")
        return cls(name)

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "TiktokenTokenizer":
        """Restore a fixed encoding from checkpoint metadata."""
        name = state.get("name")
        if not isinstance(name, str):
            raise ValueError("Tiktoken state is missing 'name'")
        return cls.from_name(name)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode text without interpreting special strings as control tokens."""
        return self.tokenizer.encode_ordinary(text)

    def encode_documents(self, documents: list[str]) -> list[int]:
        """Encode documents and append the encoding's end-of-text token."""
        token_ids: list[int] = []
        for document in documents:
            token_ids.extend(self.encode(document))
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs using the original tiktoken vocabulary."""
        return self.tokenizer.decode(token_ids)

    def to_state(self) -> dict[str, Any]:
        """Return the encoding name and special-token metadata for a checkpoint."""
        return {
            "kind": "tiktoken",
            "name": self.name,
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
        }

    def save(self, path: Path) -> None:
        """Save a small JSON descriptor; tiktoken supplies the vocabulary on load."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(), indent=2), encoding="utf-8")


def tokenizer_from_state(state: dict[str, object]) -> ByteLevelBPE | TiktokenTokenizer:
    """Restore either the trainable BPE or fixed tiktoken tokenizer."""
    kind = state.get("kind")
    if kind == "tiktoken":
        return TiktokenTokenizer.from_state(state)
    if kind == "byte_level_bpe":
        return ByteLevelBPE.from_state(state)
    raise ValueError(f"Unknown tokenizer kind: {kind!r}")


def build_tokenizer(
    kind: str,
    documents: list[str],
    vocab_size: int = 512,
    min_frequency: int = 2,
    name: str = "r50k_base",
) -> ByteLevelBPE | TiktokenTokenizer:
    """Build the selected tokenizer for a new training run."""
    if kind == "bpe":
        return ByteLevelBPE.fit(documents, vocab_size, min_frequency)
    if kind == "tiktoken":
        return TiktokenTokenizer.from_name(name)
    raise ValueError("tokenizer kind must be 'bpe' or 'tiktoken'")
