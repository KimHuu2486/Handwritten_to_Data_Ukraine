# Runbook triển khai Phase 1

Phase 1 triển khai baseline rút gọn từ `finetune_phase1.md`:

```text
Stage A = page_layout_only
Stage B = crop OCR theo type
Stage C = OFF
Stage D = OFF
Stage E = assembly + schema guardrail
```

## 1. Cấu trúc thư mục

```text
MainPipeline/
  configs/phase1/          # Config JSON cho build data, train, inference, validation
  prompts/phase1/          # Tài liệu prompt để người đọc đối chiếu
  src/common/              # Utility dùng chung: IO/schema/bbox/prompt/parser/scoring
  src/phase1/              # Builder, trainer, inference, validation, analysis cho Phase 1

artifacts/main_pipeline/
  phase1/datasets/         # SFT JSONL A1/B1/Mixed đã sinh và crop images
  phase1/checkpoints/      # LoRA checkpoints và final adapters
  phase1/predictions/      # Validation predictions và raw model outputs
  phase1/scores/           # Metric summaries
  phase1/error_reports/    # Báo cáo breakdown theo source/type
```

`artifacts/main_pipeline/` đã được ignore trong git vì sẽ chứa crops, adapters, raw outputs và scores.

## 2. Build dữ liệu

Silver warm-up:

```text
python -m MainPipeline.src.phase1.build_a1_layout_dataset --config MainPipeline/configs/phase1/build_silver_a1.json
python -m MainPipeline.src.phase1.build_b1_crop_dataset --config MainPipeline/configs/phase1/build_silver_b1.json
python -m MainPipeline.src.phase1.mix_phase1_tasks --config MainPipeline/configs/phase1/mix_silver_b1_only.json
python -m MainPipeline.src.phase1.mix_phase1_tasks --config MainPipeline/configs/phase1/mix_silver_a1_only.json
python -m MainPipeline.src.phase1.mix_phase1_tasks --config MainPipeline/configs/phase1/mix_silver.json
```

Gold fine-tune:

```text
python -m MainPipeline.src.phase1.build_a1_layout_dataset --config MainPipeline/configs/phase1/build_gold_a1.json
python -m MainPipeline.src.phase1.build_b1_crop_dataset --config MainPipeline/configs/phase1/build_gold_b1.json
python -m MainPipeline.src.phase1.mix_phase1_tasks --config MainPipeline/configs/phase1/mix_gold.json
```

## 3. Train trên A6000

Trước khi train trên VM, chỉnh `model.base_model_path` trong các config train cho đúng đường dẫn model thực tế.

```text
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_b1_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_a1_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_gold_a6000.json
```

Profile mặc định là FP16 LoRA cho A6000 48GB. Chỉ nên dùng 4-bit như fallback khi VM bị OOM.

## 4. Validate

Nếu final adapter không nằm ở path mặc định, chỉnh `model.adapter_path` trong `inference_val.json`.

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_val.json
python -m MainPipeline.src.phase1.validate_phase1 --config MainPipeline/configs/phase1/validate_val.json
python -m MainPipeline.src.phase1.analyze_errors --config MainPipeline/configs/phase1/analyze_val.json
```

Chạy smoke test nhanh:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_val.json --limit 2
```

## 5. Guardrail Phase 1

- Stage A target chỉ có `bbox,type`.
- Stage B không sửa bbox hoặc type.
- `image` và `graph` luôn trả text rỗng.
- Stage C và Stage D không được gọi trong Phase 1.
- Stage E chỉ sort, normalize schema, preserve markers và enforce structural empty text.

