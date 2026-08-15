"""Run standard lm-evaluation-harness tasks on a project checkpoint.

Example:
    python src/evaluate_benchmarks.py \
        --checkpoint runs/mha_50m/checkpoint.pt \
        --suite core_zero_shot \
        --output runs/mha_50m/eval
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_PATH = PROJECT_ROOT / "configs" / "evaluation" / "deepseek_suite.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--backend",
        choices=("auto", "custom", "hf"),
        default="auto",
        help="Use the project adapter or the standard lm-eval Hugging Face backend",
    )
    parser.add_argument("--suite", default="core_zero_shot")
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--no-log-samples", action="store_true")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument(
        "--truncation",
        action="store_true",
        help="Allow HF backend to truncate overlong prompts; use for smoke tests only",
    )
    return parser.parse_args()


def load_suite(path: Path, name: str) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    suites = config.get("suites", {})
    if not isinstance(suites, dict) or name not in suites:
        known = ", ".join(sorted(suites)) if isinstance(suites, dict) else "none"
        raise ValueError(f"Unknown suite {name!r}; choose from {known}")
    suite = dict(suites[name])
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Suite {name!r} must contain a non-empty tasks list")
    return suite


def main() -> None:
    args = parse_args()
    suite = load_suite(args.suite_config, args.suite)
    tasks = [str(task) for task in suite["tasks"]]
    num_fewshot = args.num_fewshot
    if num_fewshot is None:
        num_fewshot = int(suite.get("num_fewshot", 0))
    limit = args.limit if args.limit is not None else suite.get("limit")
    log_samples = not args.no_log_samples and bool(suite.get("log_samples", True))
    args.output.mkdir(parents=True, exist_ok=True)

    from lm_eval import simple_evaluate

    backend = args.backend
    if backend == "auto":
        backend = "hf" if args.model_dir is not None else "custom"
    if backend == "hf" and args.model_dir is None:
        raise ValueError("--backend hf requires --model-dir")
    if backend == "custom" and args.checkpoint is None:
        raise ValueError("--backend custom requires --checkpoint")

    model = None
    model_args = None
    model_dir = None
    if backend == "custom":
        from evaluation.lm_eval_adapter import TinyLLMEval

        model = TinyLLMEval(
            checkpoint=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        model_dir = args.model_dir.resolve()
        from transformers import AutoConfig, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        model_args = {
            "pretrained": str(model_dir),
            "backend": "causal",
            "revision": None,
            "trust_remote_code": True,
            "dtype": "float32",
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "tokenizer": str(model_dir),
            "truncation": args.truncation,
            "max_length": int(getattr(hf_config, "max_position_embeddings", 2048)),
        }

    results = simple_evaluate(
        model="hf" if backend == "hf" else model,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=args.batch_size,
        limit=limit,
        log_samples=log_samples,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        random_seed=int(suite.get("random_seed", 42)),
        numpy_random_seed=int(suite.get("numpy_random_seed", 42)),
        torch_random_seed=int(suite.get("torch_random_seed", 42)),
        fewshot_random_seed=int(suite.get("fewshot_random_seed", 42)),
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")

    if backend == "custom":
        checkpoint_name = str(args.checkpoint.resolve())
        selected_checkpoint = str(model.selected_checkpoint)
        architecture = model.architecture
        tokenizer_name = model.tokenizer_name
        context_length = model.max_length
    else:
        checkpoint_name = None
        selected_checkpoint = str(model_dir)
        architecture = getattr(hf_config, "architecture", None)
        tokenizer_name = type(hf_tokenizer).__name__
        context_length = getattr(hf_config, "max_position_embeddings", None)
    evaluation_device = str(model.device) if backend == "custom" else str(model_args["device"])

    metadata = {
        "backend": backend,
        "checkpoint": checkpoint_name,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "selected_checkpoint": selected_checkpoint,
        "architecture": architecture,
        "tasks": tasks,
        "suite": args.suite,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "truncation": bool(args.truncation) if backend == "hf" else None,
        "device": evaluation_device,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_name,
        "context_length": context_length,
    }
    payload = {"metadata": metadata, "results": results}
    (args.output / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(json.dumps(results.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
