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

## 2. Chuẩn bị VM: thư viện và đường dẫn

Luôn chạy các lệnh từ repo root để mọi relative path trong config được resolve đúng.

### 2.1. Thư viện cần có

Cài PyTorch theo đúng CUDA/driver của VM trước. Với A6000 48GB, ưu tiên CUDA build chính thức tương thích với driver đang có trên VM.

Các package Python Phase 1 cần dùng:

```text
torch
torchvision
torchaudio
transformers
accelerate
datasets
trl
peft
qwen-vl-utils
pillow
pandas
pyyaml
safetensors
sentencepiece
```

Gợi ý cài các package không phụ thuộc CUDA:

```text
python -m pip install -U transformers accelerate datasets trl peft qwen-vl-utils pillow pandas pyyaml safetensors sentencepiece
```

`bitsandbytes` chỉ cần cài nếu bật `model_load.load_in_4bit=true`. Mặc định Phase 1 trên A6000 đang dùng FP16 + SDPA, nên chưa cần `bitsandbytes` hoặc `flash-attn`.

### 2.2. Các file cần chỉnh nếu path trên VM khác

| Nhóm | File | Field cần kiểm tra/chỉnh |
| --- | --- | --- |
| Silver dataset | `MainPipeline/configs/phase1/build_silver_a1.json` | `metadata_path`, `image_roots` |
| Silver dataset | `MainPipeline/configs/phase1/build_silver_b1.json` | `metadata_path`, `image_roots`, `crops_dir` nếu muốn lưu crop ra ổ khác |
| Gold dataset | `MainPipeline/configs/phase1/build_gold_a1.json` | `metadata_path`, `image_roots` |
| Gold dataset | `MainPipeline/configs/phase1/build_gold_b1.json` | `metadata_path`, `image_roots`, `crops_dir` nếu muốn lưu crop ra ổ khác |
| Mix dataset | `MainPipeline/configs/phase1/mix_silver_b1_only.json` | `inputs[].path`, `output_train_jsonl`, `output_val_jsonl` nếu đổi thư mục artifacts |
| Mix dataset | `MainPipeline/configs/phase1/mix_silver_a1_only.json` | `inputs[].path`, `output_train_jsonl`, `output_val_jsonl` nếu đổi thư mục artifacts |
| Mix dataset | `MainPipeline/configs/phase1/mix_silver.json` | `inputs[].path`, `output_train_jsonl`, `output_val_jsonl` nếu đổi thư mục artifacts |
| Mix dataset | `MainPipeline/configs/phase1/mix_gold.json` | `inputs[].path`, `exclude_manifest_path`, `output_train_jsonl`, `output_val_jsonl` nếu đổi thư mục artifacts |
| Train | `MainPipeline/configs/phase1/train_silver_b1_a6000.json` | `model.base_model_path`, `training.output_dir`, `training.final_adapter_dir` |
| Train | `MainPipeline/configs/phase1/train_silver_a1_a6000.json` | `model.base_model_path`, `model.resume_adapter_path`, `training.output_dir`, `training.final_adapter_dir` |
| Train | `MainPipeline/configs/phase1/train_silver_a6000.json` | `model.base_model_path`, `model.resume_adapter_path`, `training.output_dir`, `training.final_adapter_dir` |
| Train | `MainPipeline/configs/phase1/train_gold_a6000.json` | `model.base_model_path`, `model.resume_adapter_path`, `training.output_dir`, `training.final_adapter_dir` |
| Inference | `MainPipeline/configs/phase1/inference_val.json` | `input.metadata_path`, `input.image_roots`, `model.base_model_path`, `model.adapter_path`, `output.*` nếu đổi thư mục artifacts |
| Validate | `MainPipeline/configs/phase1/validate_val.json` | `manifest_path`, `predictions_csv`, `metric_notebook_path`, `metric_module_path`, `score_json` |
| Analyze | `MainPipeline/configs/phase1/analyze_val.json` | `manifest_path`, `predictions_csv`, `source_breakdown_csv`, `type_breakdown_csv` |

Path mặc định hiện đang giả định:

```text
Qwen3-VL-8B-Instruct: /mnt/models/Qwen3-VL-8B-Instruct
Dataset root: /mnt/data/rukopys
Gold images/metadata: /mnt/data/rukopys/train
Silver images/metadata: /mnt/data/rukopys/silver
Frozen validation: artifacts/manifests/frozen_validation_manifest.v1.jsonl
Phase 1 artifacts: artifacts/main_pipeline/phase1/
```

Các config Phase 1 đã dùng đúng thư mục `silver`.

`model.resume_adapter_path` ở các stage sau nên giữ đúng checkpoint stage trước nếu train tuần tự theo runbook. Chỉ chỉnh field này khi bạn copy adapter sang một thư mục khác.

## 3. Build dữ liệu

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

## 4. Train trên A6000

Trước khi train trên VM, kiểm tra lại `model.base_model_path` trong các config train cho đúng đường dẫn Qwen3-VL-8B-Instruct thực tế.

```text
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_b1_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_a1_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_gold_a6000.json
```

Profile mặc định là FP16 LoRA cho A6000 48GB. Chỉ nên dùng 4-bit như fallback khi VM bị OOM.

## 5. Validate

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

## 6. Guardrail Phase 1

- Stage A target chỉ có `bbox,type`.
- Stage B không sửa bbox hoặc type.
- `image` và `graph` luôn trả text rỗng.
- Stage C và Stage D không được gọi trong Phase 1.
- Stage E chỉ sort, normalize schema, preserve markers và enforce structural empty text.
