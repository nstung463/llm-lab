# Tiny LLM Lab: MHA baseline and KV cache

This lab is deliberately separate from `../LLMs-from-scratch`, which stays as the
upstream reference. It provides a small, configurable decoder-only GPT with:

- causal Multi-Head Attention (MHA);
- a byte-level BPE tokenizer, deterministic document split, source manifest, and resumable next-token training loop;
- a functional per-request KV cache for inference only; and
- tests that prove causal masking and cached/uncached numerical equivalence.

## Quick start

```powershell
cd D:\tiny-llm-lab\llm-lab
$env:UV_CACHE_DIR = 'D:\tiny-llm-lab\.uv-cache' # prevents a large cache on C:
uv sync --extra dev                              # creates .venv inside this folder
uv run python -m train --config configs/baseline_tiny.json --text data\sample.txt
uv run python -m benchmark --checkpoint runs\baseline\checkpoint.pt --new-tokens 64
uv run pytest
```

The sample corpus is intentionally tiny: it proves the end-to-end pipeline, not
language capability. Replace it with a licensed corpus after the smoke run.

## Package layout

New code should use the domain-oriented APIs:

```text
src/
├── data/          # readers, tokenizers, manifests, fixed token artifacts
├── models/        # baseline model API and architecture registry
├── training/      # baseline loop and architecture runner boundary
├── evaluation/   # shared token-weighted loss and comparison boundary
├── benchmarking/  # FLOPs, KV cache, inference and system metrics
└── cli/           # command adapters
```

The flat modules are the command entrypoints used by `python -m <command>`.
Shared logic belongs in the domain subpackages rather than in duplicate
top-level implementations.

## TinyStories data

TinyStories is available at `roneneldan/TinyStories` under CDLA-Sharing-1.0.
The preparation command streams only the requested subset and records the source
and license beside the JSONL file:

```powershell
$env:UV_CACHE_DIR = 'D:\tiny-llm-lab\.uv-cache'
$env:HF_HOME = 'D:\tiny-llm-lab\.hf-cache'
$env:HF_DATASETS_CACHE = 'D:\tiny-llm-lab\.hf-cache\datasets'
$env:HF_HUB_CACHE = 'D:\tiny-llm-lab\.hf-cache\hub'
uv sync --extra dev --extra data
uv run python -m prepare_tinystories --max-examples 10000
uv run python -m train --config configs\tinystories_tiny.json --text data\tinystories_sample.jsonl --output runs\tinystories
```

To continue a run, increase `training.max_steps` in a config and pass
`--resume runs\tinystories\checkpoint.pt`. The checkpoint verifies the document
SHA-256, split seed, and train fraction before continuing.

## Cache boundary

Training always calls the model with `use_cache=False`. The cache is supplied as
an explicit `past_key_values` argument and returned to the caller, so requests
cannot leak state into one another. `generate_cached` rebuilds the cache when the
absolute-position context window is full; this preserves the same sliding-window
semantics as uncached generation.

## Fair MHA vs KV-cache comparison

KV cache changes inference work and memory, not model weights or training. For a
fair comparison, use the same checkpoint, prompt, decoding parameters, dtype,
hardware, warm-up count, and number of generated tokens. Compare:

- output equality (the test suite checks this);
- decode tokens/s and total latency;
- peak CUDA memory, if CUDA is available; and
- cache bytes, reported analytically in the benchmark JSON.

## Comparing the five educational architectures

The standalone models in `learning_assets/model/` can be trained with the shared runner. The
runner follows the chapter 5 workflow from `LLMs-from-scratch`: deterministic
document splitting, a fitted tokenizer saved in the checkpoint, next-token
cross-entropy, periodic train/validation evaluation, token counts, samples,
best-model weights, and a final held-out test loss.

Use the small configuration for an end-to-end smoke run:

```powershell
uv run python -m train_architectures `
  --architecture v4 `
  --config configs/architecture_tiny.json `
  --text data/sample.txt `
  --output runs/v4_tiny
```

Replace `v4` with `mha`, `gqa`, `mla`, or `moe`. For a real experiment, use a
licensed corpus with many documents and increase `context_length`, model size,
and `training.max_steps` together. The default split is 80% train, 10%
validation, and 10% test; the small-corpus fallback exists only for smoke tests.

Evaluate one checkpoint:

```powershell
uv run python -m evaluate_architectures `
  --checkpoint runs/v4_tiny/checkpoint.pt `
  --text data/sample.txt
```

Compare checkpoints trained with the same tokenizer, split seed, data, and
training budget:

```powershell
uv run python -m compare_architectures `
  --checkpoint-root runs `
  --text data/sample.txt
```

For a fixed-token architecture comparison, pass the validated artifact
manifest instead of raw text. This loads the exact `.npy` token streams and
rejects checkpoints with a different data contract:

```powershell
uv run python -m train_architectures `
  --architecture gqa `
  --config configs/colab_architecture_10m_smoke.json `
  --artifact-manifest data/tinystories_pilot_10m.manifest.json `
  --output runs/architecture_compare_gqa `
  --device cuda

uv run python -m compare_architectures `
  --checkpoint-root runs `
  --run-prefix architecture_compare_ `
  --artifact-manifest data/tinystories_pilot_10m.manifest.json `
  --device cuda
```

With an artifact manifest, comparison evaluates the full held-out token
stream by default. Dense models target roughly 10M-12M resident parameters.
MoE reports both total resident parameters and estimated active
parameters/FLOPs per token; its total resident capacity is intentionally larger
while its active compute is kept near the dense baseline.

The primary metrics are held-out cross-entropy and perplexity. Also record
parameter count, tokens seen, estimated FLOPs/token, estimated total training
FLOPs, training throughput, inference latency, accumulated KV-cache bytes,
KV-cache bytes/token, peak memory, long-context retrieval accuracy, and (for
MoE) expert load balance.
The FLOPs fields use an analytical matrix-multiply plus attention estimate and
the training estimate is 3x forward FLOPs; they are comparable under this
project convention, not a hardware-profiler measurement. Loss and perplexity
compare language-modeling quality; compute, cache, and throughput metrics
compare efficiency and should not be collapsed into a single score without an
explicit trade-off.

## Standard downstream benchmarks

The project can run the public tasks used in the DeepSeekMoE-style comparison
through `lm-evaluation-harness`. Install the optional evaluation dependencies:

```powershell
uv sync --extra eval
```

### Hugging Face export

Training checkpoints remain in the project's resumable `.pt` format. Export a
checkpoint separately when you need the standard Hugging Face API:

```powershell
uv run python -m huggingface.export `
  --checkpoint runs/architecture_compare_mha/checkpoint.pt `
  --output runs/architecture_compare_mha/hf `
  --device cpu
```

The exporter writes `config.json`, `model.safetensors`, HF tokenizer files and
the custom AutoClass source files. It validates logits before and after
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. The
exported directory can be loaded with `AutoModelForCausalLM` and
`AutoTokenizer`, and supports `generate()` without importing the training
checkpoint format.

Run a small smoke suite first:

```powershell
uv run python -m evaluate_benchmarks `
  --checkpoint runs/architecture_compare_mha/checkpoint.pt `
  --suite smoke_core `
  --output runs/architecture_compare_mha/eval `
  --device cuda
```

For the standard Hugging Face evaluation path, evaluate the exported model
with the harness `hf` backend:

```powershell
uv run python -m evaluate_benchmarks `
  --model-dir runs/architecture_compare_mha/hf `
  --backend hf `
  --suite smoke_core `
  --output runs/architecture_compare_mha/eval_hf `
  --device cuda
```

The custom checkpoint adapter remains available with `--checkpoint` for fast
development checks. For final benchmark tables, use the exported directory and
the `hf` backend. The model context length must be large enough for the chosen
benchmark; `--truncation` is intended only for smoke tests.

The standard suites are configured in
`configs/evaluation/deepseek_suite.json`:

- `core_zero_shot`: HellaSwag, PIQA, ARC-easy and ARC-challenge;
- `reading_five_shot`: RACE with five examples;
- `generation`: HumanEval, MBPP, TriviaQA and NaturalQuestions (`nq_open`).

Run the main multiple-choice suite with:

```powershell
uv run python -m evaluate_benchmarks `
  --checkpoint runs/architecture_compare_mha/checkpoint.pt `
  --suite core_zero_shot `
  --output runs/architecture_compare_mha/eval_core `
  --device cuda `
  --batch-size 4
```

The runner writes `benchmark_results.json` with task scores and
reproducibility metadata: architecture, selected checkpoint, tokenizer,
context length, few-shot count, device, PyTorch version and timestamp. Use the
same suite, checkpoint policy, tokenizer, context length and batch protocol for
MHA, GQA, MLA, MoE and V4. HumanEval/MBPP execute generated Python code; run
that suite only in an isolated environment and pass
`--confirm-run-unsafe-code` explicitly.

## Colab 10M architecture benchmark

The Colab protocol is stored in `configs/colab_architecture_10m.json`:

- byte-level BPE target vocabulary 16K;
- context length 256;
- tied input/output embeddings;
- 32,768 global tokens per optimizer update;
- AdamW with gradient accumulation and automatic BF16/FP16 on CUDA;
- 100M-token target, represented by 3,052 optimizer steps.

The config includes per-architecture overrides so MHA, GQA, MLA, MoE, and V4 stay in a similar 10M–12M parameter range when the full tokenizer reaches 16K tokens. The tiny local validation config is `configs/colab_architecture_10m_smoke.json`.

Run a local smoke test:

```powershell
uv run python -m train_architectures `
  --architecture mha `
  --config configs/colab_architecture_10m_smoke.json `
  --text data/tinystories_architecture.jsonl `
  --output runs/colab_smoke_mha `
  --device cpu
```

Run the full Colab profile after copying the dataset to the runtime-local filesystem:

```powershell
uv run python -m train_architectures `
  --architecture gqa `
  --config configs/colab_architecture_10m.json `
  --text data/tinystories_architecture.jsonl `
  --output runs/colab_gqa_10m `
  --device cuda
```

Replace `gqa` with `mha`, `mla`, `moe`, or `v4`. Save checkpoints and metrics to Google Drive, but copy the active dataset to `/content` before training to avoid Drive I/O becoming the bottleneck.
