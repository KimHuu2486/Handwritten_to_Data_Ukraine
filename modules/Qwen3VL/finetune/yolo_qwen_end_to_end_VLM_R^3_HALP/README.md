# YOLO + Qwen3-VL VLM-R3 + HALP

Thư mục này là biến thể end-to-end có thêm **VLM-R3** và **HALP**. Pipeline vẫn
dùng DocLayout-YOLO để tạo region, nhưng Qwen3-VL được huấn luyện/inference với
luồng VLM-R3; HALP hỗ trợ nhận diện region rủi ro để áp dụng xử lý/refinement phù
hợp. Đây là nhánh khác với workflow curriculum clean và nhánh hybrid cơ bản.

## Luồng tổng quát

```text
Silver annotations
    ↓ Stage 1 VLM-R3 + HALP warm-up
Gold annotations
    ↓ Stage 2 VLM-R3 + HALP fine-tune
DocLayout-YOLO bbox/type
    ↓
Qwen3-VL VLM-R3 + HALP inference/refinement
    ↓
submission.csv hoặc local validation score
```

## Bản đồ file

| Thứ tự | File | Tác dụng |
|---:|---|---|
| 1 | [`rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb`](./rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb) | Silver warm-up cho LoRA VLM-R3 + HALP. Tạo adapter Stage 1 để làm điểm khởi đầu cho gold fine-tune. |
| 2 | [`rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb`](./rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb) | Fine-tune tiếp trên gold data; tạo adapter cuối và validation records dùng cho kiểm tra local. |
| 3 | [`rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb`](./rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb) | Inference YOLO + Qwen3-VL, có chế độ chạy test hoặc validation, tạo CSV và có thể gọi local metric nếu đã cung cấp evaluator. |
| — | [`dataset.md`](./dataset.md) | Liên kết tới dataset nguồn. |

## Thứ tự đọc/chạy khuyến nghị

1. Đọc `dataset.md`, sau đó đọc cell config của Stage 1 để xác định silver data,
   base model, output directory và GPU profile.
2. Chạy Stage 1 để có LoRA warm-up. Có thể dùng lại adapter đã có nếu cấu hình
   Stage 2 chấp nhận checkpoint đó.
3. Chạy Stage 2 trên gold data; lưu lại adapter cuối cùng và
   `gold_validation_records.jsonl` được tạo từ split validation.
4. Chạy notebook submit. Dùng `RUN_SPLIT="validation"` khi cần kiểm tra local
   trên validation records; dùng `RUN_SPLIT="test"` khi tạo submission.

## Khác biệt giữa ba notebook

| Notebook | Dữ liệu đầu vào chính | Output chính | Khác biệt |
|---|---|---|---|
| Stage 1 silver | Silver annotations | Stage 1 LoRA | Học khởi tạo từ nhãn/nguồn silver trước khi thấy gold. |
| Stage 2 gold | Gold annotations + Stage 1 LoRA | Final LoRA + validation records | Tinh chỉnh chất lượng và format trên dữ liệu đáng tin cậy hơn. |
| Submit | Ảnh test hoặc validation records + YOLO + final LoRA | `submission.csv`, partial CSV, metric local tùy cấu hình | Không update trọng số; đây là notebook đánh giá/inference full-system. |

## VLM-R3 + HALP khác gì hybrid cơ bản?

| Khía cạnh | Hybrid cơ bản | VLM-R3 + HALP |
|---|---|---|
| Luồng chính | YOLO region → Qwen crop OCR | YOLO + Qwen với VLM-R3 và tín hiệu HALP/risk-aware handling |
| Mục tiêu | Pipeline OCR/layout tối giản | Xử lý region khó và theo dõi refinement/risk trong pipeline end-to-end |
| Kiểm tra local | Tùy notebook | Notebook submit có `RUN_SPLIT="validation"` và dùng validation records từ Stage 2 |

> [!IMPORTANT]
> Không thay lẫn adapter giữa nhánh này và các notebook trong
> `yolo_qwen_end_to_end` hoặc `ukrainian_curriculum_clean` nếu prompt/config
> không tương thích. Kiểm tra model path, LoRA path, YOLO weights và validation
> records trong cell config trước khi chạy.
