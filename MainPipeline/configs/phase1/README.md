# Phase 1 Configs

Các config này phục vụ triển khai Phase 1 trên VM A6000 48GB.

Luồng khuyến nghị:

```text
build_silver_a1.json  -> silver_a1_layout.jsonl
build_silver_b1.json  -> silver_b1_crop_ocr.jsonl
mix_silver_b1_only.json -> silver_b1_train.jsonl + silver_b1_val.jsonl
mix_silver_a1_only.json -> silver_a1_train.jsonl + silver_a1_val.jsonl
mix_silver.json       -> silver_train.jsonl + silver_val.jsonl
train_silver_b1_a6000.json
train_silver_a1_a6000.json
train_silver_a6000.json

build_gold_a1.json    -> gold_a1_layout.jsonl
build_gold_b1.json    -> gold_b1_crop_ocr.jsonl
mix_gold.json         -> gold_train.jsonl + gold_val.jsonl
train_gold_a6000.json

inference_val.json
inference_test.json
inference_test_yolo_layout.json
validate_val.json
analyze_val.json
```

Path VM mặc định hiện tại:

- Model: `/mnt/models/Qwen3-VL-8B-Instruct`
- Dataset root: `/mnt/data/rukopys`
- Gold metadata/images: `/mnt/data/rukopys/train/metadata.jsonl`, `/mnt/data/rukopys/train/images`
- Silver metadata/images: `/mnt/data/rukopys/silver/metadata.jsonl`, `/mnt/data/rukopys/silver/images`
- Test metadata/images: `/mnt/data/rukopys/test/metadata.jsonl`, `/mnt/data/rukopys/test/images`

Nếu VM khác mount, chỉnh các path sau:

- `model.base_model_path`
- `metadata_path`
- `image_roots`
- `model.adapter_path` trong inference

Lưu ý: các config Phase 1 dùng đúng thư mục `silver`.

Phase 1 không bật Stage C/D, không dùng `text_draft`, không dùng `ocr_context_light`.
`inference_test.json` đọc `/mnt/data/rukopys/test/metadata.jsonl` và tạo file submission tại `artifacts/main_pipeline/phase1/submissions/submission_phase1.csv`.
`inference_test_yolo_layout.json` đọc `dataset/test/metadata.jsonl`, lấy bbox/type có sẵn từ `MainPipeline/Bbox - DocLayoutYOLOv4.csv`, bỏ qua Stage A và chỉ chạy Stage B crop OCR + Stage E guardrail. Output submission riêng là `artifacts/main_pipeline/phase1/submissions/submission_phase1_yolo_layout.csv`.
Trong inference, `generation.image_batch_size` là số ảnh gom trước khi OCR crop chung; `generation.stage_b_batch_size` là số crop trong mỗi lần gọi generate; `output.progress_every` điều khiển log tiến độ sau bao nhiêu ảnh hoàn tất.
`logging.transformers_verbosity` đang để `error` để tắt warning lặp từ Transformers trong lúc inference; đổi về `warning` hoặc `info` khi cần debug sâu model/processor.
