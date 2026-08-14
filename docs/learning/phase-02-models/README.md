# Phase 02 — Model registry and common interface

## Mục tiêu

Tách việc chọn và khởi tạo architecture khỏi các CLI training/evaluation. Mọi
decoder model được benchmark phải đi qua một registry và có cùng interface:

```python
model = build_model(name, config)
logits = model(input_ids, use_cache=False)
model.reset_kv_cache()
```

## Đã triển khai

```text
src/llm/models/
├── __init__.py
├── base.py       # DecoderModel protocol và ModelMetadata
└── registry.py   # ModelSpec, registry, factory, parameter metadata
```

Registry hiện có:

```text
mha  -> standard_mha.py
gqa  -> learning asset GQA implementation
mla  -> learning asset latent-attention implementation
moe  -> learning asset routed-expert implementation
v4   -> learning asset compressed-sparse implementation
```

Các learning asset vẫn được giữ nguyên để bảo toàn notebook/standalone demo.
Registry là boundary mới; không tạo bản copy thứ hai của các model implementation.

`src/llm/architectures.py` chỉ còn là compatibility shim cho code cũ.

## Common contract

- `forward(input_ids, use_cache=False)` trả về logits có shape `(batch, tokens, vocab)`.
- `reset_kv_cache()` xóa cache inference nếu architecture hỗ trợ cache.
- `training_cfg` lưu normalized model config.
- `parameter_count(model)` đếm trainable parameters và không đếm tied weight hai lần.
- `model_metadata(name, model)` trả về architecture, family, module, params, context, layers, hidden size và vocab size.

## Không làm trong phase này

- Không đổi logic attention/FFN bên trong từng architecture.
- Không thêm optimizer, scheduler hoặc DDP.
- Không thay đổi format data/token store của Phase 01.
- Không xóa compatibility shim hoặc standalone learning assets.

## Test và command

`tests/test_model_registry.py` kiểm tra:

- registry order và legacy import path;
- forward shape/finite logits cho cả năm model;
- parameter metadata;
- unknown architecture bị reject.

Chạy test:

```powershell
uv run pytest -q
```

Smoke command cũ vẫn giữ nguyên:

```powershell
uv run python -m llm.train_architectures `
  --architecture gqa `
  --config configs/colab_architecture_10m_smoke.json `
  --text data/tinystories_architecture.jsonl `
  --output runs/phase02_gqa_smoke `
  --device cpu
```

## Tiêu chí hoàn thành

- Runner/evaluator/benchmark không import trực tiếp adapter cũ.
- Mọi architecture được build qua `llm.models.registry`.
- Legacy `llm.architectures` vẫn hoạt động.
- Registry tests và toàn bộ test suite pass.
