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

`build_b1_crop_dataset` dùng `num_workers=8` trong `build_silver_b1.json` và `build_gold_b1.json` để crop ảnh song song bằng CPU/I/O. Đây không phải batch GPU; nếu VM/local bị nghẽn disk hoặc RAM, giảm `num_workers` về `4` hoặc `1`.

## 4. Train trên A6000

Trước khi train trên VM, kiểm tra lại `model.base_model_path` trong các config train cho đúng đường dẫn Qwen3-VL-8B-Instruct thực tế.

```text
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_b1_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_a1_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_silver_a6000.json
python -m MainPipeline.src.phase1.train_phase1 --config MainPipeline/configs/phase1/train_gold_a6000.json
```

Profile mặc định là FP16 LoRA cho A6000 48GB với `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` và `per_device_eval_batch_size=2`.
Effective train batch là `4 * 4 = 16`, nhưng VRAM chỉ chịu micro-batch 4 mỗi forward/backward.
Log train đã tắt progress bar mặc định của Hugging Face để chỉ còn log Phase 1 gọn: loss/eval_loss, lr, ETA, tốc độ, VRAM, checkpoint path và best checkpoint.
Chỉ nên dùng 4-bit như fallback khi VM bị OOM.

## 5. Validate

Nếu final adapter không nằm ở path mặc định, chỉnh `model.adapter_path` trong `inference_val.json`.
Inference Phase 1 có `generation.stage_a_batch_size` cho Stage A full-page và `generation.stage_b_batch_size` cho Stage B crop OCR. `generation.image_batch_size` là số page được gom trong lô bao ngoài, nên cần >= `stage_a_batch_size` nếu muốn Stage A batch thật.
Nếu OOM ở Stage A, giảm `stage_a_batch_size` về `1`. Nếu OOM khi OCR crop/table lớn, giảm `stage_b_batch_size` về `32`, `16`, hoặc thấp hơn.

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_val.json
python -m MainPipeline.src.phase1.validate_phase1 --config MainPipeline/configs/phase1/validate_val.json
python -m MainPipeline.src.phase1.analyze_errors --config MainPipeline/configs/phase1/analyze_val.json
```

### 5.1. Smoke test validation inference

Chạy 2 ảnh validation trước để kiểm tra model/adapter load được, path ảnh resolve đúng, Stage A/B generate được và output CSV ghi được:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_val.json --limit 2
```

Kiểm tra output validation smoke:

```text
head -n 3 artifacts/main_pipeline/phase1/predictions/phase1_validation_predictions.csv
tail -n 5 artifacts/main_pipeline/phase1/predictions/phase1_validation_raw_outputs.jsonl
```

Nếu smoke validation ổn, chạy full validation rồi score:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_val.json
python -m MainPipeline.src.phase1.validate_phase1 --config MainPipeline/configs/phase1/validate_val.json
python -m MainPipeline.src.phase1.analyze_errors --config MainPipeline/configs/phase1/analyze_val.json
```

Nếu job bị ngắt giữa chừng, chạy tiếp bằng `--resume`:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_val.json --resume
```

### 5.2. Smoke test test inference và submission

Trước khi chạy toàn bộ test, chạy 2 ảnh test để kiểm tra submission path:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_test.json --limit 2
```

`inference_test.json` đọc metadata test từ `/mnt/data/rukopys/test/metadata.jsonl`. Nếu một bản test khác không có metadata JSONL, `infer_phase1.py` vẫn hỗ trợ fallback qua `sample_submission_csv`, nhưng cấu hình mặc định ưu tiên metadata.

Kiểm tra file submission smoke:

```text
head -n 3 artifacts/main_pipeline/phase1/submissions/submission_phase1.csv
python - <<'PY'
import csv, json
path = "artifacts/main_pipeline/phase1/submissions/submission_phase1.csv"
with open(path, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
print("rows:", len(rows))
print("columns:", rows[0].keys() if rows else [])
json.loads(rows[0]["regions"] if rows else "[]")
print("submission smoke parse ok")
PY
```

Khi smoke test submission ổn, chạy full test:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_test.json
```

Nếu full test bị ngắt giữa chừng, chạy tiếp bằng:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_test.json --resume
```

Output test inference gồm:

```text
artifacts/main_pipeline/phase1/predictions/phase1_test_predictions_debug.csv
artifacts/main_pipeline/phase1/submissions/submission_phase1.csv
artifacts/main_pipeline/phase1/predictions/phase1_test_raw_outputs.jsonl
```

File nộp competition là `submission_phase1.csv`, chỉ có 2 cột `image,regions`. Cột `text` nằm bên trong JSON của `regions`, không phải cột CSV riêng.

### 5.3. Test inference với bbox/type có sẵn từ DocLayoutYOLO

Nếu muốn bỏ qua Stage A Qwen layout detection và chỉ dùng bbox/type có sẵn từ `MainPipeline/Bbox - DocLayoutYOLOv4.csv`, chạy config riêng:

```text
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_test_yolo_layout.json --limit 2
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_test_yolo_layout.json
```

Luồng này đọc metadata test từ `dataset/test/metadata.jsonl`, map theo tên file ảnh trong CSV YOLO, rồi chạy Stage B crop OCR và Stage E guardrail. Output submission riêng là:

```text
artifacts/main_pipeline/phase1/submissions/submission_phase1_yolo_layout.csv
```

Với config YOLO, `generation.image_batch_size` quyết định số ảnh được gom trong một lô page trước khi OCR crop chung; `generation.stage_b_batch_size` vẫn là số crop trong mỗi lần `generate`; `generation.stage_a_batch_size` không có tác dụng vì YOLO đã bỏ qua Stage A; `output.progress_every` mặc định in tiến độ sau mỗi 10 ảnh hoàn tất.
Các config inference có `logging.transformers_verbosity="error"` để giảm noise warning Transformers trên terminal. Đổi về `warning` hoặc `info` nếu cần debug processor/model.

Nếu một số ảnh bị `error_type` trong predictions CSV, xóa riêng các row lỗi rồi resume:

```text
python -m MainPipeline.src.phase1.clean_prediction_errors --config MainPipeline/configs/phase1/inference_test_yolo_layout.json --dry-run
python -m MainPipeline.src.phase1.clean_prediction_errors --config MainPipeline/configs/phase1/inference_test_yolo_layout.json
python -m MainPipeline.src.phase1.infer_phase1 --config MainPipeline/configs/phase1/inference_test_yolo_layout.json --resume
```

Utility này backup predictions/submission trước khi ghi lại, giữ các ảnh OK và chỉ để `--resume` chạy lại ảnh lỗi.

## 6. Guardrail Phase 1

- Stage A target chỉ có `bbox,type`.
- Stage B không sửa bbox hoặc type.
- `image` và `graph` luôn trả text rỗng.
- Stage C và Stage D không được gọi trong Phase 1.
- Stage E chỉ sort, normalize schema, preserve markers và enforce structural empty text.
