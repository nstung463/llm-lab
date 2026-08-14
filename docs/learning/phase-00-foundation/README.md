# Phase 00 — Foundation

## Mục tiêu

Ổn định tên project và tạo baseline để những phase sau không làm mất pipeline
đang chạy được.

## Đã làm

- Package Python chính dùng tên `llm`.
- Command module dùng dạng `python -m llm.<module>`.
- Project metadata dùng tên `llm`.
- Giữ nguyên tên `TinyStories` vì đó là tên dataset, không phải tên project.
- Chưa di chuyển logic model/data/training sang folder mới.

## Phạm vi code

Trong phase này chỉ có các thay đổi nền:

```text
src/llm/                 # package hiện tại, đổi tên từ llm_lab
pyproject.toml           # project name = llm
tests/                   # import từ llm
```

Không tạo các module mới như `llm.data`, `llm.training` hay `llm.models` dạng
package cho đến khi phase tương ứng bắt đầu.

## Tiêu chí hoàn thành

- `python -m pytest -q` chạy pass.
- `python -m llm.train --help` chạy được.
- `python -m llm.train_architectures --help` chạy được.
- Các import cũ trong test được đổi sang `llm`.

## Việc còn lại

Các file trong package compatibility cũ `src/llm_lab` đã được dọn; chỉ còn
thư mục rỗng do language-server giữ handle và sẽ biến mất sau khi process đó
khởi động lại.
