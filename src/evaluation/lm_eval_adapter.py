"""Adapter from the educational decoders to ``lm-evaluation-harness``.

The project models are deliberately kept independent from Hugging Face.  This
module implements the three request types used by ``lm-eval`` so the same
benchmark task and scoring code can evaluate all registered architectures.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.tokenizer import tokenizer_from_state
from models.registry import build_model

try:
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.model import LM
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by installation errors
    raise ImportError(
        "lm-evaluation-harness is required for benchmark evaluation. "
        "Install it with: pip install -e '.[eval]'"
    ) from exc


def _as_int_list(values: Iterable[int]) -> list[int]:
    return [int(value) for value in values]


class TinyLLMEval(LM):
    """Run a project checkpoint through the standard lm-eval interface."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
        batch_size: int = 1,
        prefer_best: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._batch_size = max(1, int(batch_size))
        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=self._device, weights_only=False
        )
        self.architecture = str(self.checkpoint["architecture"])
        self.model_config = dict(self.checkpoint["model_config"])
        self.tokenizer = tokenizer_from_state(self.checkpoint["tokenizer"])
        self._max_length = int(self.model_config["context_length"])
        self._eot_token_id = getattr(self.tokenizer, "eos_id", None)

        self.model = build_model(self.architecture, self.model_config).to(self._device)
        state_path = self.checkpoint_path.parent / "best_model.pt"
        if prefer_best and state_path.exists():
            state = torch.load(state_path, map_location=self._device, weights_only=True)
            self.selected_checkpoint = state_path
        else:
            state = self.checkpoint["model_state"]
            self.selected_checkpoint = self.checkpoint_path
        self.model.load_state_dict(state)
        self.model.eval()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def eot_token_id(self) -> int | None:
        return int(self._eot_token_id) if self._eot_token_id is not None else None

    @property
    def prefix_token_id(self) -> int:
        if self.eot_token_id is None:
            return 0
        return self.eot_token_id

    @property
    def tokenizer_name(self) -> str:
        kind = getattr(self.tokenizer, "name", None) or "byte_level_bpe"
        return f"tiny-llm:{kind}:v1"

    def tok_encode(self, string: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text using the exact tokenizer stored in the checkpoint."""
        del add_special_tokens
        return _as_int_list(self.tokenizer.encode(string))

    def tok_decode(self, tokens: list[int] | torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return self.tokenizer.decode(_as_int_list(tokens))

    def _encode_context(self, context: str) -> list[int]:
        ids = self.tok_encode(context)
        if not ids:
            if self.eot_token_id is None:
                raise ValueError("The tokenizer cannot score an empty context without an EOS token")
            ids = [self.eot_token_id]
        return ids

    def _prepare_token_pair(
        self, context_ids: list[int], continuation_ids: list[int]
    ) -> tuple[list[int], int]:
        """Return a bounded token sequence and the first continuation label index."""
        context_ids = list(context_ids) or [self.prefix_token_id]
        continuation_ids = list(continuation_ids)
        if not continuation_ids:
            return context_ids[: self.max_length], len(context_ids) - 1

        full = context_ids + continuation_ids
        # lm-eval allows max_length input tokens, so the input+label sequence
        # can contain max_length + 1 tokens.
        keep_start = max(0, len(full) - (self.max_length + 1))
        full = full[keep_start:]
        context_len = max(0, len(context_ids) - keep_start)
        return full, max(0, context_len - 1)

    def _prepare_pair(self, context: str, continuation: str) -> tuple[list[int], int]:
        return self._prepare_token_pair(
            self._encode_context(context), self.tok_encode(continuation)
        )

    @torch.inference_mode()
    def _score_token_pairs(
        self, pairs: list[tuple[list[int], list[int]]]
    ) -> list[tuple[float, bool]]:
        prepared = [self._prepare_token_pair(context, continuation) for context, continuation in pairs]
        results: list[tuple[float, bool]] = []

        for offset in range(0, len(prepared), self.batch_size):
            group = prepared[offset : offset + self.batch_size]
            input_sequences = [tokens[:-1] for tokens, _ in group]
            max_input = max(len(tokens) for tokens in input_sequences)
            pad_id = self.eot_token_id or 0
            input_ids = torch.full(
                (len(input_sequences), max_input),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, tokens in enumerate(input_sequences):
                input_ids[row, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=self.device
                )

            output = self.model(input_ids, use_cache=False)
            logits = output[0] if isinstance(output, tuple) else output
            log_probs = F.log_softmax(logits.float(), dim=-1)

            for row, (full_tokens, continuation_start) in enumerate(group):
                labels = torch.tensor(
                    full_tokens[1:], dtype=torch.long, device=self.device
                )
                label_start = min(continuation_start, labels.numel())
                if label_start >= labels.numel():
                    results.append((0.0, True))
                    continue
                label_positions = torch.arange(
                    label_start, labels.numel(), device=self.device
                )
                row_log_probs = log_probs[row, label_positions, :]
                selected = row_log_probs.gather(
                    -1, labels[label_positions, None]
                ).squeeze(-1)
                greedy = row_log_probs.argmax(dim=-1).eq(labels[label_positions]).all()
                results.append((float(selected.sum().item()), bool(greedy.item())))

        return results

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        token_pairs = [
            (self._encode_context(context), self.tok_encode(continuation))
            for context, continuation in pairs
        ]
        return self._score_token_pairs(token_pairs)

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        pairs = [(str(request.args[0]), str(request.args[1])) for request in requests]
        return self._score_pairs(pairs)

    def _rolling_score(self, text: str) -> float:
        token_ids = self.tok_encode(text)
        if not token_ids:
            return 0.0
        windows = [
            lm_eval_utils.make_disjoint_window(window)
            for window in lm_eval_utils.get_rolling_token_windows(
                token_list=token_ids,
                prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            )
        ]
        scores = self._score_token_pairs(windows)
        return sum(score for score, _ in scores)

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        return [self._rolling_score(str(request.args[0])) for request in requests]

    @torch.inference_mode()
    def _generate_one(self, context: str, gen_kwargs: dict[str, Any]) -> str:
        input_ids = self.tok_encode(context)
        input_ids = input_ids[-self.max_length :]
        max_gen_toks = int(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", 128)))
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        temperature = float(gen_kwargs.get("temperature", 0.0) or 0.0)
        top_k = int(gen_kwargs.get("top_k", 0) or 0)
        top_p = float(gen_kwargs.get("top_p", 1.0) or 1.0)
        generated: list[int] = []

        for _ in range(max_gen_toks):
            window = input_ids[-self.max_length :]
            if not window:
                window = [self.eot_token_id or 0]
            tensor = torch.tensor(window, dtype=torch.long, device=self.device)[None]
            output = self.model(tensor, use_cache=False)
            logits = output[0] if isinstance(output, tuple) else output
            next_logits = logits[:, -1, :].float()
            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < values[:, -1, None]] = -torch.inf
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    filtered_sorted = sorted_logits.masked_fill(remove, -torch.inf)
                    next_logits = torch.zeros_like(next_logits).scatter_(
                        1, sorted_indices, filtered_sorted
                    )
                next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1).item()
            else:
                next_token = next_logits.argmax(dim=-1).item()

            generated.append(int(next_token))
            input_ids.append(int(next_token))
            text = self.tok_decode(generated)
            if self.eot_token_id is not None and next_token == self.eot_token_id:
                break
            if any(stop and stop in text for stop in until):
                break

        text = self.tok_decode(generated)
        for stop in until:
            if stop:
                text = text.split(stop, 1)[0]
        return text

    def generate_until(self, requests: list[Any]) -> list[str]:
        return [self._generate_one(str(request.args[0]), dict(request.args[1])) for request in requests]
