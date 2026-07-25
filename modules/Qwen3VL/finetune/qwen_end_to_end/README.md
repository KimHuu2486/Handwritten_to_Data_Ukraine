# Qwen3-VL End-to-End

Pipeline này fine-tune Qwen3-VL để xử lý toàn bộ trang tài liệu: phát hiện vùng,
phân loại `type` và đọc nội dung văn bản. Model không phụ thuộc vào một detector
layout riêng.

## Luồng chính

```text
RUKOPYS silver
      ↓
Stage 1 — Silver warm-up
      ↓ LoRA Stage 1
Stage 2 — Gold fine-tune
      ↓ LoRA Stage 2
Inference — Full page + crop OCR
      ↓
submission.csv
```

## Thứ tự đọc

1. [`rukopys_qwen3vl_stage1_silver_finetune.ipynb`](./rukopys_qwen3vl_stage1_silver_finetune.ipynb)
2. [`rukopys_qwen3vl_stage2_gold_finetune.ipynb`](./rukopys_qwen3vl_stage2_gold_finetune.ipynb)
3. [`rukopys_qwen3vl_inference_submit.ipynb`](./rukopys_qwen3vl_inference_submit.ipynb)

Thông tin nguồn dữ liệu được lưu tại [`dataset.md`](./dataset.md).

## Vai trò từng notebook

### Stage 1 — Silver warm-up

Huấn luyện LoRA đầu tiên trên split `silver`. Dữ liệu này lớn nhưng được gán nhãn
tự động, phù hợp để model học cấu trúc output và thích nghi ban đầu với tài liệu
RUKOPYS.

Đầu ra chính:

```text
qwen3vl_rukopys_stage1_silver/
└── qwen3vl_silver_lora_final/
```

### Stage 2 — Gold fine-tune

Nạp adapter Stage 1 rồi tiếp tục huấn luyện trên split `train` đã được con người
xác thực. Đây là bước tạo adapter chính cho inference.

Input quan trọng:

- RUKOPYS dataset.
- `STAGE1_LORA_DIR` trỏ đến adapter Stage 1.

Đầu ra chính:

```text
qwen3vl_rukopys_stage2_gold/
├── gold_validation_records.jsonl
└── qwen3vl_rukopys_lora_final/
```

### Inference và submission

Nạp adapter Stage 2 và chạy hai pass:

1. Full page để sinh danh sách region gồm `bbox`, `type` và `text`.
2. Crop OCR tùy chọn để đọc lại các vùng văn bản và thay thế transcription yếu.

`CROP_OCR_MODE` điều khiển pass thứ hai:

| Giá trị | Ý nghĩa |
|---|---|
| `none` | Chỉ dùng kết quả full-page |
| `smart` | Chỉ đọc lại các vùng cần thiết |
| `all_text` | Đọc lại toàn bộ vùng văn bản; chậm hơn nhưng thường ổn định hơn |

Notebook hỗ trợ chia ảnh theo GPU, lưu partial CSV để resume và tạo
`submission.csv`.

## Cách đọc code trong mỗi notebook

Đọc theo thứ tự: **config → prompt/schema → chuẩn bị dataset → load model/LoRA →
training hoặc inference → lưu artifact**.

> [!NOTE]
> Pipeline này để Qwen3-VL tự xử lý layout. Nếu muốn dùng YOLO cho `bbox/type` và
> chỉ dùng Qwen cho OCR crop, xem các thư mục `yolo_ocr` hoặc
> `yolo_qwen_end_to_end`.
