# Phase B2 Baseline Runner

Thư mục này chứa runtime `.py only` để chạy B2 baseline đầu tiên:

```text
b2_stage2_gold__crop-none__prompt-v1__val-v1
```

Runner chỉ chạy `crop_mode=none`. Không chạy crop ablation, risk label, candidate, crop/zoom, trajectory, B3/B4/B5 trước khi B2 baseline cache pass và được lock.

## 1. Chọn VM

Khuyến nghị ban đầu:

```text
Template: None (Base ML Environment)
GPU: RTX A6000 48GB hoặc GPU >= 48GB VRAM
GPU count: 1
RAM: 32GB trở lên
Disk: 150-200GB nếu có thể
```

Disk 100GB có thể chạy được nhưng hơi sát vì cần chứa dataset, model, môi trường Python, cache và output.

## 2. Cấu trúc thư mục chuẩn

Repo code nằm trong `/workspace`. Model và dataset nằm ngoài repo trong `/mnt`.

```text
/workspace/Handwritten_to_Data_Ukraine/
  phaseB2/
  artifacts/
    manifests/
      frozen_validation_manifest.v1.jsonl
      frozen_validation_manifest.v1.config.json
      frozen_validation_manifest.v1.gate_report.json
  qwen3vl_rukopys_lora_final/
    adapter_config.json
    adapter_model.safetensors
    rukopys_prompt_config.json

/mnt/models/
  Qwen3-VL-8B-Instruct/

/mnt/data/
  rukopys/
    train/
      images/
      metadata.jsonl
    silver/
      images/
      metadata.jsonl
    test/
      images/
      metadata.jsonl
    sample_submission.csv
```

Manifest validation dùng ảnh từ:

```text
/mnt/data/rukopys/train/images/{uuid}.jpg
```

## 3. Chuẩn bị repo trên VM

Sau khi tạo VM, vào thư mục làm việc:

```text
cd /workspace
```

Clone repo `Handwritten_to_Data_Ukraine` về `/workspace/Handwritten_to_Data_Ukraine`.

Sau khi clone, kiểm tra repo có các mục sau:

```text
phaseB2/
artifacts/manifests/
qwen3vl_rukopys_lora_final/
official-evaluation-metric-text-normalization.ipynb
```

Nếu `qwen3vl_rukopys_lora_final/` chưa nằm trong repo sau khi clone, copy folder đó vào repo root.

## 4. Tải dataset RUKOPYS

Tải Hugging Face dataset `UkrainianCatholicUniversity/rukopys` và giữ nguyên cấu trúc:

```text
/mnt/data/rukopys/train/images/
/mnt/data/rukopys/train/metadata.jsonl
/mnt/data/rukopys/silver/
/mnt/data/rukopys/test/
/mnt/data/rukopys/sample_submission.csv
```

Điểm quan trọng: `frozen_validation_manifest.v1.jsonl` có `file_name` dạng:

```text
images/<uuid>.jpg
```

Vì vậy checker sẽ resolve thành:

```text
/mnt/data/rukopys/train/images/<uuid>.jpg
```

## 5. Tải Qwen3-VL base model

Tải model `Qwen/Qwen3-VL-8B-Instruct` vào:

```text
/mnt/models/Qwen3-VL-8B-Instruct
```

Không đặt base model trong repo. Repo chỉ nên chứa code, manifest và LoRA adapter B2.

## 6. Chuẩn bị Python environment

Base ML Environment thường đã có Python/CUDA/PyTorch. Bạn vẫn cần đảm bảo các thư viện runtime có mặt:

```text
torch
transformers
peft
bitsandbytes
qwen-vl-utils
pandas
pillow
tqdm
```

`check_runtime_ready.py` sẽ báo thiếu module nào. Cài thiếu gì thì cài bổ sung trong environment của VM.

## 7. Tạo runtime config

Từ repo root:

```text
cd /workspace/Handwritten_to_Data_Ukraine
```

Tạo file config local từ template:

```text
phaseB2/b2_runtime_config.json
```

Nội dung mặc định đã trỏ tới layout chuẩn:

```json
{
  "base_model_path": "/mnt/models/Qwen3-VL-8B-Instruct",
  "lora_adapter_path": "qwen3vl_rukopys_lora_final",
  "image_roots": [
    "/mnt/data/rukopys/train",
    "/mnt/data/rukopys"
  ]
}
```

Nếu VM của bạn dùng path khác, chỉ sửa trong `phaseB2/b2_runtime_config.json`, không sửa template.

## 8. Kiểm tra readiness

Chạy:

```text
python phaseB2/check_runtime_ready.py --config phaseB2/b2_runtime_config.json --check-images
```

Chỉ chạy baseline khi các nhóm sau đều pass:

```text
manifest_exists
manifest_schema
gate_report_pass
manifest_checksum_lf
base_model_path_exists
lora_adapter_exists
lora_adapter_config
lora_adapter_weights
prompt_config
image_paths_all
python_module_torch
python_module_transformers
python_module_peft
python_module_qwen_vl_utils
official_metric_notebook
```

Nếu `image_paths_all` fail, gần như chắc dataset chưa nằm đúng `/mnt/data/rukopys/train/images`.

## 9. Chạy smoke test

Trước khi chạy đủ 159 ảnh validation, chạy thử 3 ảnh:

```text
python phaseB2/run_b2_baseline_cache.py --config phaseB2/b2_runtime_config.json --limit 3 --skip-score --output-dir artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1_smoke --overwrite
```

Smoke test cần tạo được:

```text
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1_smoke/config.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1_smoke/validation_predictions.csv
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1_smoke/validation_raw_outputs.jsonl
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1_smoke/failed_rows.jsonl
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1_smoke/runtime_summary.json
```

Smoke test dùng `--skip-score` vì chỉ kiểm tra model load, image path, prompt, raw output và parser.

## 10. Chạy B2 baseline thật

Khi smoke test ổn, chạy:

```text
python phaseB2/run_b2_baseline_cache.py --config phaseB2/b2_runtime_config.json
```

Output chính:

```text
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/config.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_predictions.csv
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_raw_outputs.jsonl
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_score.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/failed_rows.jsonl
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/runtime_summary.json
```

Nếu VM bị ngắt giữa chừng và đã có partial `validation_predictions.csv`, chạy tiếp bằng:

```text
python phaseB2/run_b2_baseline_cache.py --config phaseB2/b2_runtime_config.json --resume
```

Chỉ dùng `--overwrite` khi bạn chắc chắn muốn thay toàn bộ output trong thư mục đích.

## 11. Kiểm tra B2 pass gate

Sau khi chạy xong, mở:

```text
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_score.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/runtime_summary.json
```

Điều kiện pass theo plan:

```text
prediction row count = manifest row count
parse fail <= 0.5% hoặc <= 3 rows
validation_score.json có total_score, detection_f1, class_acc, region_cer, page_cer
runtime_per_page_sec được ghi
config.json có checkpoint, prompt, generation params, metric version, source manifest
```

Nếu pass, bước tiếp theo mới là lock dòng B2 baseline vào:

```text
artifacts/ablations/ablation_results.csv
```

Không chạy crop ablation, risk labels, candidates, crop/zoom, trajectory, B3, B4 hoặc B5 trước khi B2 baseline được lock.
