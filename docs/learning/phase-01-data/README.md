# Phase 01 — Data pipeline

## Mục tiêu

Tách pipeline dữ liệu thành các trách nhiệm nhỏ nhưng vẫn giữ nguyên output
hiện tại: một batch gồm `input_ids` và `labels` cho bài toán next-token
prediction.

## Flow

```text
raw file
  -> reader
  -> document split
  -> tokenizer
  -> token store
  -> next-token dataset
  -> dataloader
```

## File sẽ tạo trong phase này

Chỉ tạo các file sau khi bắt đầu code phase:

```text
src/llm/data/
├── __init__.py
├── readers.py
├── splits.py
├── tokenizer.py
├── datasets.py
└── manifest.py
```

Chưa tạo `collators.py`, HDF5/memmap backend hoặc composable data source vì
TinyStories hiện tại chưa cần chúng. Stateful loader hiện vẫn nằm trong
`datasets.py`; chỉ tách thành file riêng khi backend loader phát triển đủ lớn.

## Trách nhiệm từng file

### `readers.py`

- Đọc plain text và JSONL.
- Trả về `list[str]` document.
- Không tokenize và không split trong reader.

### `splits.py`

- Deterministic train/validation/test split.
- Nhận `seed` và fractions.
- Không phụ thuộc PyTorch.

### `tokenizer.py`

- Chứa interface tokenizer tối thiểu: `fit`, `encode`, `decode`, `save`,
  `from_state`.
- Hỗ trợ byte-level BPE train trên train documents.
- Hỗ trợ tokenizer có sẵn `tiktoken` với `gpt2` hoặc `r50k_base`.
- Lưu rõ `vocab_size`, `eos_id`, `unk_id` và tokenizer state.

### `datasets.py`

- Tạo input/target lệch nhau một token.
- Hỗ trợ `context_length` và `stride`.
- Lưu/khôi phục batch order, epoch, cursor, RNG và dataset signature.
- Không biết model và optimizer.

### `manifest.py`

- Lưu source, license, document hash, split seed và số lượng document.
- Manifest phải được lưu cùng run.

## API dự kiến

```python
documents = read_documents(path)
train_docs, val_docs, test_docs = split_documents(documents, seed=42)
tokenizer = ByteLevelBPE.fit(train_docs, vocab_size=512)
train_ids = tokenizer.encode_documents(train_docs)
dataset = NextTokenDataset(train_ids, context_length=128, stride=128)
```

## Không làm trong phase này

- Không đổi architecture MHA/GQA/MLA/MoE/V4.
- Không thêm scheduler, AMP, DDP hoặc checkpoint resume.
- Không đổi format checkpoint hiện tại ngoài việc cập nhật data metadata nếu cần.
- Không tối ưu HDF5/memmap trước khi có benchmark chứng minh cần thiết.

## Test bắt buộc

- Reader đọc đúng JSONL/plain text.
- Split deterministic và không overlap document.
- Tokenizer fit chỉ trên train documents.
- Encode/decode có EOS.
- Dataset tạo đúng cặp input/target.
- Dataset không tạo sample vượt context length.

## Tiêu chí hoàn thành

- Pipeline TinyStories hiện tại chạy qua API mới.
- Toàn bộ test cũ vẫn pass.
- Có ít nhất một test integration từ JSONL đến batch.
- Chưa tạo module nào không được liệt kê ở phần “File sẽ tạo”.

## Kết quả triển khai

- `prepare_tinystories` hỗ trợ `--target-train-tokens`, validation/test token budgets và deterministic seed.
- Artifact token được lưu dạng NumPy `uint32`, không thêm HDF5/memmap backend.
- Manifest ghi tokenizer kind/vocabulary/hash, token counts, target budgets, dtype và artifact paths.
- Pilot artifact đã được tạo tại `data/tinystories_pilot_10m.*` với 10M train tokens và 500K tokens cho mỗi validation/test split.
- Data contract test suite hiện có 13 tests pass.
