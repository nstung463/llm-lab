"""Conversion from project tokenizers to Hugging Face tokenizer objects."""

from __future__ import annotations

from typing import Any

from data.tokenizer import ByteLevelBPE, TiktokenTokenizer


def to_huggingface_tokenizer(
    tokenizer: ByteLevelBPE | TiktokenTokenizer,
    model_max_length: int,
    tiktoken_model: str = "openai-community/gpt2",
) -> Any:
    """Return a tokenizer that supports the standard HF save/load API."""
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

    if isinstance(tokenizer, ByteLevelBPE):
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer.tokenizer,
            model_max_length=model_max_length,
            unk_token=tokenizer.unk_token,
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.eos_token,
            clean_up_tokenization_spaces=False,
        )

    if isinstance(tokenizer, TiktokenTokenizer):
        # r50k_base/gpt2 uses the GPT-2 vocabulary.  Loading the corresponding
        # HF tokenizer preserves the exact IDs while providing tokenizer.json.
        hf_tokenizer = GPT2TokenizerFast.from_pretrained(tiktoken_model)
        if hf_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "The selected HF GPT-2 tokenizer does not match the checkpoint "
                f"vocab size: {hf_tokenizer.vocab_size} != {tokenizer.vocab_size}"
            )
        hf_tokenizer.model_max_length = model_max_length
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        return hf_tokenizer

    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer).__name__}")
