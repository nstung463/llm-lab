"""Export a project checkpoint to a Hugging Face model directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from data.tokenizer import tokenizer_from_state
from models.registry import MODEL_DIR, build_model

from .configuration_llm import LLMConfig
from .modeling_llm import LLMForCausalLM
from .tokenizer import to_huggingface_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_FILES = {
    "standard_mha.py",
    "gqa.py",
    "mla.py",
    "moe.py",
    "v4.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", choices=("best", "final"), default="best")
    parser.add_argument(
        "--tiktoken-model",
        default="openai-community/gpt2",
        help="HF tokenizer ID used when exporting a tiktoken checkpoint",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the original-vs-HF logits and AutoClass round-trip check",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Project checkpoint must contain a dictionary payload")
    required = {"architecture", "model_config", "model_state", "tokenizer"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(missing)}")
    return checkpoint


def _select_state(
    checkpoint_path: Path, checkpoint: dict[str, Any], weights: str, device: torch.device
) -> tuple[dict[str, torch.Tensor], Path | None]:
    best_path = checkpoint_path.with_name("best_model.pt")
    if weights == "best" and best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"Best model file is not a state dict: {best_path}")
        return state, best_path
    return checkpoint["model_state"], None


def _copy_standalone_sources(output: Path) -> None:
    """Copy the minimal architecture sources needed by remote HF loading."""
    architecture_output = output / "architectures"
    architecture_output.mkdir(parents=True, exist_ok=True)
    for filename in ARCHITECTURE_FILES:
        source = MODEL_DIR / filename
        if not source.exists():
            raise FileNotFoundError(f"Architecture source does not exist: {source}")
        shutil.copy2(source, architecture_output / filename)
    shutil.copy2(PROJECT_ROOT / "src" / "rope.py", output / "rope.py")


def _make_config(checkpoint: dict[str, Any], tokenizer: Any) -> LLMConfig:
    model_config = dict(checkpoint["model_config"])
    model_config["vocab_size"] = tokenizer.vocab_size
    return LLMConfig(
        architecture=str(checkpoint["architecture"]),
        model_config=model_config,
        tokenizer_kind=str(checkpoint["tokenizer"].get("kind", "unknown")),
        vocab_size=int(tokenizer.vocab_size),
        hidden_size=int(model_config["emb_dim"]),
        num_hidden_layers=int(model_config["n_layers"]),
        max_position_embeddings=int(model_config["context_length"]),
        eos_token_id=int(tokenizer.eos_token_id),
        unk_token_id=(
            int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None
        ),
        pad_token_id=int(tokenizer.eos_token_id),
        torch_dtype=torch.float32,
        architectures=["LLMForCausalLM"],
    )


def _validate_logits(
    original: torch.nn.Module,
    exported: LLMForCausalLM,
    vocab_size: int,
    context_length: int,
    device: torch.device,
) -> None:
    torch.manual_seed(1234)
    input_ids = torch.randint(
        0, vocab_size, (2, min(8, context_length)), device=device, dtype=torch.long
    )
    original.eval()
    exported.eval()
    with torch.no_grad():
        original_logits = original(input_ids, use_cache=False)
        exported_logits = exported(input_ids=input_ids, return_dict=True).logits
    torch.testing.assert_close(exported_logits, original_logits, rtol=1e-5, atol=1e-5)


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    device: str | torch.device = "cpu",
    weights: str = "best",
    tiktoken_model: str = "openai-community/gpt2",
    validate: bool = True,
) -> dict[str, Any]:
    """Convert one project checkpoint and return export metadata."""
    torch_device = torch.device(device)
    checkpoint = _load_checkpoint(checkpoint_path, torch_device)
    architecture = str(checkpoint["architecture"])
    model_config = dict(checkpoint["model_config"])
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    hf_tokenizer = to_huggingface_tokenizer(
        tokenizer,
        model_max_length=int(model_config["context_length"]),
        tiktoken_model=tiktoken_model,
    )
    model_config["vocab_size"] = int(hf_tokenizer.vocab_size)
    checkpoint["model_config"] = model_config
    state, selected_path = _select_state(checkpoint_path, checkpoint, weights, torch_device)

    original = build_model(architecture, model_config).to(torch_device)
    original.load_state_dict(state)
    config = _make_config(checkpoint, hf_tokenizer)
    model = LLMForCausalLM(config).to(torch_device)
    model.transformer.load_state_dict(state)
    model.eval()

    output_path.mkdir(parents=True, exist_ok=True)
    # Registering these classes makes save_pretrained() add the AutoClass map
    # and copy configuration_llm.py/modeling_llm.py into the export directory.
    LLMConfig.register_for_auto_class()
    LLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(output_path, safe_serialization=True)
    hf_tokenizer.save_pretrained(output_path)
    _copy_standalone_sources(output_path)

    metadata = {
        "format": "huggingface-transformers",
        "architecture": architecture,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "selected_weights": str(selected_path.resolve()) if selected_path else "checkpoint.model_state",
        "vocab_size": int(hf_tokenizer.vocab_size),
        "context_length": int(model_config["context_length"]),
        "tokenizer_class": type(hf_tokenizer).__name__,
        "tokenizer_kind": checkpoint["tokenizer"].get("kind"),
        "validated": bool(validate),
    }
    (output_path / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    if validate:
        _validate_logits(
            original,
            model,
            int(hf_tokenizer.vocab_size),
            int(model_config["context_length"]),
            torch_device,
        )
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        reloaded_config = AutoConfig.from_pretrained(
            output_path, local_files_only=True, trust_remote_code=True
        )
        if reloaded_config.model_type != LLMConfig.model_type:
            raise AssertionError("AutoConfig did not resolve the project model type")
        reloaded = AutoModelForCausalLM.from_pretrained(
            output_path,
            trust_remote_code=True,
            local_files_only=True,
        ).to(torch_device)
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            output_path, local_files_only=True
        )
        if reloaded_tokenizer.encode("HF round trip") != hf_tokenizer.encode("HF round trip"):
            raise AssertionError("AutoTokenizer round trip changed token IDs")
        _validate_logits(
            original,
            reloaded,
            int(hf_tokenizer.vocab_size),
            int(model_config["context_length"]),
            torch_device,
        )
        metadata["auto_config_class"] = type(reloaded_config).__name__
        metadata["auto_model_class"] = type(reloaded).__name__
        metadata["auto_tokenizer_class"] = type(reloaded_tokenizer).__name__
        (output_path / "export_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    return metadata


def main() -> None:
    args = _parse_args()
    metadata = export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        weights=args.weights,
        tiktoken_model=args.tiktoken_model,
        validate=not args.no_validate,
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
