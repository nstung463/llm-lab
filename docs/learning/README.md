# Learning and refactor roadmap

Mục tiêu của thư mục này là chia dự án thành các phase nhỏ, mỗi phase có một
ranh giới rõ ràng và có thể kiểm tra độc lập.

## Nguyên tắc tạo code

- Chỉ tạo file implementation khi phase hiện tại đã xác định rõ API và test.
- Không tạo trước các folder `optimizer`, `distributed`, `callbacks` hoặc
  `evaluation/tasks` nếu chưa có code sử dụng chúng.
- Mỗi phase phải có test tối thiểu và một command chạy được.
- Không thay đổi logic model nếu phase đó không liên quan đến model.
- Giữ một baseline để so sánh trước và sau mỗi refactor.

## Các phase

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 00 | Đổi tên package, baseline và quy ước project | Hoàn thành |
| 01 | Data source, split, tokenizer, token store và next-token dataset | Hoàn thành |
| 02 | Model registry và interface chung cho MHA/GQA/MLA/MoE/V4 | Hoàn thành |
| 03 | Training loop tối thiểu | Tiếp theo |
| 04 | Optimizer, scheduler, accumulation và AMP | Chưa bắt đầu |
| 05 | Checkpoint/resume đầy đủ | Chưa bắt đầu |
| 06 | Evaluation, generation và architecture comparison | Chưa bắt đầu |
| 07 | Performance và distributed training | Chưa bắt đầu |

## Pattern tham khảo

- `LLMs-from-scratch`: cách giải thích từng khái niệm và test nhỏ.
- `train-llm-from-scratch`: cách chia workflow theo stage và script.
- `OLMo-core`: cách tách data, model, train, checkpoint và config thành module.

## Cấu trúc đích

Chỉ tạo từng phần khi phase tương ứng bắt đầu:

```text
src/llm/
├── data/
├── models/
├── training/
├── evaluation/
├── benchmarking/
└── cli/
```

Các module top-level cũ vẫn được giữ làm compatibility entrypoint. Logic dùng
chung mới phải đi vào subpackage tương ứng; README của từng phase xác định API
và test cho phần đó.
