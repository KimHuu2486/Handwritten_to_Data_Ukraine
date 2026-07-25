# YOLO + Qwen3-VL OCR

Pipeline này tách layout detection và OCR thành hai phần:

- DocLayout-YOLO dự đoán `bbox` và `type`.
- Qwen3-VL LoRA chỉ đọc nội dung từ từng crop.

Qwen không phải học lại bài toán phát hiện vùng, nên dữ liệu fine-tune của pipeline
này được tạo ở cấp region crop.

## Luồng chính

```text
Silver region crops + crop jitter
              ↓
Stage 1 — OCR Silver warm-up
              ↓ OCR LoRA Stage 1
Gold region crops
              ↓
Stage 2 — OCR Gold fine-tune
              ↓ OCR LoRA Stage 2
DocLayout-YOLO bbox/type + Qwen crop OCR
              ↓
submission.csv
```

## Thứ tự đọc

1. [`rukopys_qwen3vl_ocr_stage1_silver_finetune.ipynb`](./rukopys_qwen3vl_ocr_stage1_silver_finetune.ipynb)
2. [`rukopys_qwen3vl_ocr_stage2_gold_finetune.ipynb`](./rukopys_qwen3vl_ocr_stage2_gold_finetune.ipynb)
3. [`rukopys_yolo_qwen3vl_ocr_submit.ipynb`](./rukopys_yolo_qwen3vl_ocr_submit.ipynb)

[`dataset.md`](./dataset.md) hiện là ghi chú giữ chỗ, chưa phải tài liệu nguồn dữ
liệu hoàn chỉnh.

## Vai trò từng notebook

### Stage 1 — OCR Silver warm-up

Tạo mẫu OCR từ các region trong split `silver`. Crop được thêm jitter để mô phỏng
sai lệch bbox thực tế của detector. Model học prompt theo loại vùng và chỉ trả về
transcription, không sinh region JSON cho toàn trang.

Đầu ra chính:

```text
qwen3vl_yolo_ocr_stage1_silver/
└── qwen3vl_yolo_ocr_silver_lora_final/
```

### Stage 2 — OCR Gold fine-tune

Nạp adapter Stage 1 rồi tiếp tục huấn luyện trên crop từ dữ liệu gold. Notebook
tạo validation samples và adapter OCR cuối dùng cho submission.

Input quan trọng:

- RUKOPYS gold data.
- `STAGE1_LORA_DIR` chứa chính xác adapter Stage 1.

Đầu ra chính:

```text
qwen3vl_yolo_ocr_stage2_gold/
├── gold_ocr_validation_samples.jsonl
└── qwen3vl_yolo_ocr_lora_final/
```

### YOLO + OCR-LoRA submit

Notebook submit thực hiện:

1. DocLayout-YOLO phát hiện bbox và loại region.
2. Các region văn bản được crop từ ảnh gốc.
3. Qwen3-VL cùng OCR LoRA Stage 2 đọc từng crop.
4. Kết quả được chuẩn hóa và ghi vào `submission.csv`.

Notebook hỗ trợ validation/test split, nhiều GPU và partial CSV để resume.

## Khác với Qwen end-to-end

| Thành phần | Qwen end-to-end | YOLO + OCR |
|---|---|---|
| Phát hiện bbox | Qwen3-VL | DocLayout-YOLO |
| Phân loại region | Qwen3-VL | DocLayout-YOLO |
| OCR | Qwen3-VL full page + crop | Qwen3-VL crop-only |
| Dữ liệu fine-tune | Trang đầy đủ | Region crops |
| Điểm phụ thuộc chính | Chất lượng output full-page | Chất lượng bbox/type của YOLO |

> [!NOTE]
> Adapter trong thư mục này là OCR-only. Không nên coi nó như adapter end-to-end
> có khả năng tự phát hiện layout toàn trang.
