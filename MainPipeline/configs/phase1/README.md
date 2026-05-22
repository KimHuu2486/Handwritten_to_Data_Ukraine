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
validate_val.json
analyze_val.json
```

Trên VM, chỉnh các path sau cho đúng mount thực tế:

- `model.base_model_path`
- `metadata_path`
- `image_roots`
- `model.adapter_path` trong inference

Phase 1 không bật Stage C/D, không dùng `text_draft`, không dùng `ocr_context_light`.
