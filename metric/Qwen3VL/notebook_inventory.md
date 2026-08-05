# Danh Sách Notebook

File này dùng để audit nhanh các notebook đã tạo cho phần kết quả của paper. Tất cả notebook đều có cùng cơ chế:

- Import logic gốc từ `INFERENCE_PY`, mặc định đang để placeholder `/kaggle/input/your-code/fillpaper/inference.py`.
- Đọc input từ `TEST_JSONL`, mặc định placeholder `/kaggle/input/your-data/fillpaper/test.jsonl`.
- Chia ảnh thành 2 shard: `records[0::2] -> cuda:0`, `records[1::2] -> cuda:1`.
- Ghi output theo format `sample_submission.csv`: cột `image,regions`.
- Output của mỗi notebook: `/kaggle/working/<RUN_NAME>/<RUN_NAME>_submission.csv`.

## Cần Config Trước Khi Chạy

Tất cả notebook hiện đang dùng placeholder path. Nghĩa là **chưa thể nói là đang dùng đúng trọng số trên Kaggle của bạn**, vì bạn sẽ tự config chính xác đường dẫn.

Cần sửa trong cell config đầu mỗi notebook:

| Biến | Hiện tại | Ghi chú |
|---|---|---|
| `INFERENCE_PY` | `/kaggle/input/your-code/fillpaper/inference.py` | File gốc mạnh nhất của team. Notebook import file này, không copy lại code. |
| `TEST_JSONL` | `/kaggle/input/your-data/fillpaper/test.jsonl` | File jsonl bạn đặt trong `fillpaper/`. |
| `IMAGE_ROOT` | `/kaggle/input/your-data/fillpaper` | Root để resolve `file_name`, vì `test.jsonl` đang có dạng `images/*.jpg`. |
| `HPA_MODEL_DIR` | `/kaggle/input/your-models/trocr_model` | TrOCR/HPA model. |
| `QWEN_BASE_DIR` | `/kaggle/input/your-models/qwen3vl_8b_instruct` | Qwen3-VL base. |
| `QWEN_LORA_DIR` | `/kaggle/input/your-models/qwen3vl_lora_adapter` | Qwen LoRA adapter. Không dùng trong notebook base zero-shot. |
| `YOLO_WEIGHTS` | `/kaggle/input/your-models/DoclayoutYoloV4.1.pt` | Primary YOLO checkpoint. |
| `YOLO_EXTRA_WEIGHTS` | `/kaggle/input/your-models/DoclayoutYoloV4.2.pt` | Extra YOLO checkpoint để ensemble giống base `inference.py`. |

Config runtime đang giữ theo `fillpaper/inference.py`:

```text
YOLO_IMG_SIZE=1280
YOLO_CONF=0.26
YOLO_MAX_DET=200
YOLO_IOU_NMS=0.60
YOLO_AGNOSTIC_NMS=True
YOLO_DEDUP_IOU=0.90
CROP_BATCH_SIZE=2
MAX_PIXELS_CROP=262144
MAX_NEW_TOKENS_QWEN=192
HPA_CROP_PAD_RATIO=0.0
QWEN_CROP_PAD_RATIO=0.04
```

Các notebook có chạy Qwen đã có thêm cell cài `qwen-vl-utils` và `bitsandbytes>=0.46.1` ở đầu notebook. Nếu Kaggle tắt Internet, sửa `QWEN_VL_UTILS_PACKAGE` và `BITSANDBYTES_PACKAGE` trong cell đó thành đường dẫn chính xác tới file `.whl` local.

## Ý Nghĩa Các Mode

| `RUN_MODE` | Dùng bbox nào? | Load branch nào? | Mục đích |
|---|---|---|---|
| `empty_prediction` | Không dùng bbox | Không cần model | Baseline rỗng. |
| `bbox_only` | YOLO predicted boxes | YOLO only | Bbox-only ablation. Vì submission schema bắt buộc có `type`, notebook ghi một placeholder type cố định. |
| `detector_only` | YOLO predicted boxes | YOLO only | Kiểm tra detector/full-system với text rỗng. |
| `trocr_all` | YOLO predicted boxes | YOLO + TrOCR | TrOCR đọc tất cả text-like regions: handwritten, printed, formula, table, annotation. |
| `qwen_all` | YOLO predicted boxes | YOLO + Qwen | Qwen đọc tất cả text-like regions. |
| `hpa_only` | YOLO predicted boxes | YOLO + TrOCR | TrOCR chỉ đọc handwritten/printed/annotation; formula/table để text rỗng. |
| `qwen_only` | YOLO predicted boxes | YOLO + Qwen | Qwen chỉ đọc formula/table; HPA để text rỗng. |
| `hybrid` | YOLO predicted boxes | YOLO + TrOCR + Qwen | Final routing: TrOCR cho HPA, Qwen cho formula/table. |
| `hpa_gt` | GT boxes từ `test.jsonl` | TrOCR only | Module-level HPA OCR, không liên quan YOLO bbox. |
| `qwen_gt` | GT boxes từ `test.jsonl` | Qwen only | Module-level/Qwen ablation formula-table, không liên quan YOLO bbox. |

## Qwen Ablation

Folder: `fillpaper/qwen_ablation/`

Các notebook này dùng **GT boxes** từ `test.jsonl`, lọc theo `formula/table`, và chỉ thay model/prompt của nhánh Qwen. Dùng cho `tab:qwen-ablation` gồm Formula CER, Table CER, Struct. Acc., Invalid Rate. Nếu cần Official Score, cần evaluator của bạn tính trên CSV output.

| File | GT? | YOLO? | Branch/trọng số cần dùng | Prompt | Output CSV | Dùng để làm gì |
|---|---:|---:|---|---|---|---|
| `01_base_qwen_generic_gt_formula_table.ipynb` | Yes | No | Qwen base only, **không LoRA** | `generic` | `01_base_qwen_generic_gt_formula_table_submission.csv` | Base Qwen3-VL zero-shot generic prompt. |
| `02_lora_generic_gt_formula_table.ipynb` | Yes | No | Qwen base + LoRA | `generic` | `02_lora_generic_gt_formula_table_submission.csv` | LoRA với generic prompt. |
| `03_lora_type_specific_gt_formula_table.ipynb` | Yes | No | Qwen base + LoRA | `type_specific` | `03_lora_type_specific_gt_formula_table_submission.csv` | LoRA với prompt riêng cho formula/table, chưa source, chưa guardrails đầy đủ. |
| `04_lora_source_type_gt_formula_table.ipynb` | Yes | No | Qwen base + LoRA | `source_type` | `04_lora_source_type_gt_formula_table_submission.csv` | LoRA với source-aware + type-specific prompt. |
| `05_lora_source_guardrails_gt_formula_table.ipynb` | Yes | No | Qwen base + LoRA | `source_guardrails` | `05_lora_source_guardrails_gt_formula_table_submission.csv` | LoRA với source-aware + guardrails, nhưng generic/default type prompt. |
| `06_lora_final_gt_formula_table.ipynb` | Yes | No | Qwen base + LoRA | `final` | `06_lora_final_gt_formula_table_submission.csv` | Prompt final giống logic `fillpaper/inference.py`. |

Trọng số:

- Qwen base path cần dùng: `QWEN_BASE_DIR`.
- LoRA path cần dùng: `QWEN_LORA_DIR`, trừ notebook `01_base_qwen_generic...` không load LoRA.
- YOLO/HPA paths có trong config cell nhưng **không được load** ở các notebook `qwen_gt`.

## Routing Ablation

Folder: `fillpaper/routing_ablation/`

Các notebook này dùng **YOLO predicted boxes**, vì routing/full-system có sự hiện diện của bbox detector.

| File | GT? | YOLO? | Branch/trọng số cần dùng | Output CSV | Dùng để làm gì |
|---|---:|---:|---|---|---|
| `01_yolo_trocr_all_text_like.ipynb` | No | Yes | YOLO + TrOCR/HPA | `01_yolo_trocr_all_text_like_submission.csv` | Routing ablation: TrOCR đọc mọi text-like region. |
| `02_yolo_qwen_all_text_like.ipynb` | No | Yes | YOLO + Qwen base + LoRA | `02_yolo_qwen_all_text_like_submission.csv` | Routing ablation: Qwen đọc mọi text-like region. |
| `03_yolo_trocr_hpa_empty_formula_table.ipynb` | No | Yes | YOLO + TrOCR/HPA | `03_yolo_trocr_hpa_empty_formula_table_submission.csv` | HPA-only: đọc handwritten/printed/annotation, formula/table rỗng. |
| `04_yolo_qwen_formula_table_empty_hpa.ipynb` | No | Yes | YOLO + Qwen base + LoRA | `04_yolo_qwen_formula_table_empty_hpa_submission.csv` | Qwen-only structured branch: đọc formula/table, HPA rỗng. |
| `05_yolo_trocr_qwen_hybrid.ipynb` | No | Yes | YOLO + TrOCR/HPA + Qwen base + LoRA | `05_yolo_trocr_qwen_hybrid_submission.csv` | Final hybrid routing. |

Trọng số:

- Tất cả notebook routing cần `YOLO_WEIGHTS` và `YOLO_EXTRA_WEIGHTS`.
- Notebook có TrOCR cần `HPA_MODEL_DIR`.
- Notebook có Qwen cần `QWEN_BASE_DIR` và `QWEN_LORA_DIR`.
- Tất cả prompt Qwen trong routing ablation đang là `final`.

## Full-System Results

Folder: `fillpaper/full_system_results/`

Đây là các dòng trong `tab:full-system-results`. Các dòng có detector sẽ dùng YOLO predicted boxes.

Lưu ý cho dòng bbox-only: format submission bắt buộc mỗi region phải có `bbox`, `type`, `text`. Vì vậy notebook `00` vẫn phải ghi một `type` hợp lệ. Nó dùng placeholder cố định `handwritten`, không phải type prediction thật.

| File | GT? | YOLO? | Branch/trọng số cần dùng | Output CSV | Dùng để làm gì |
|---|---:|---:|---|---|---|
| `00_yolo_bbox_only_empty_text.ipynb` | No | Yes | YOLO only | `00_yolo_bbox_only_empty_text_submission.csv` | YOLO bbox-only. `type` là placeholder cố định, `text` rỗng. |
| `01_yolo_bbox_type_empty_text.ipynb` | No | Yes | YOLO only | `01_yolo_bbox_type_empty_text_submission.csv` | YOLO bbox + predicted type, `text` rỗng. |
| `02_yolo_trocr_all_text_like.ipynb` | No | Yes | YOLO + TrOCR/HPA | `02_yolo_trocr_all_text_like_submission.csv` | YOLO + TrOCR for all text-like regions. |
| `03_yolo_qwen_all_text_like.ipynb` | No | Yes | YOLO + Qwen base + LoRA | `03_yolo_qwen_all_text_like_submission.csv` | YOLO + Qwen3-VL for all text-like regions. |
| `04_yolo_trocr_hpa_empty_formula_table.ipynb` | No | Yes | YOLO + TrOCR/HPA | `04_yolo_trocr_hpa_empty_formula_table_submission.csv` | YOLO + TrOCR HPA, formula/table rỗng. |
| `05_yolo_qwen_formula_table_empty_hpa.ipynb` | No | Yes | YOLO + Qwen base + LoRA | `05_yolo_qwen_formula_table_empty_hpa_submission.csv` | YOLO + Qwen formula/table, HPA rỗng. |
| `06_yolo_trocr_qwen_hybrid.ipynb` | No | Yes | YOLO + TrOCR/HPA + Qwen base + LoRA | `06_yolo_trocr_qwen_hybrid_submission.csv` | Final full hybrid system. |

Trọng số:

- `00_yolo_bbox_only_empty_text` chỉ cần YOLO weights.
- `01_yolo_bbox_type_empty_text` chỉ cần YOLO weights.
- TrOCR rows cần `HPA_MODEL_DIR`.
- Qwen rows cần `QWEN_BASE_DIR` và `QWEN_LORA_DIR`.
- Hybrid row cần tất cả weights.

## Module Results

Folder: `fillpaper/module_results/`

Dùng cho `tab:module-results` và các metric module-level. Rule: số liệu không liên quan bbox thì dùng GT; có detector thì dùng YOLO.

| File | GT? | YOLO? | Branch/trọng số cần dùng | Output CSV | Dùng để làm gì |
|---|---:|---:|---|---|---|
| `01_detector_yolo_predictions_empty_text.ipynb` | No | Yes | YOLO only | `01_detector_yolo_predictions_empty_text_submission.csv` | Lấy YOLO predictions để tính detector/module detection metrics. |
| `02_hpa_gt_crops.ipynb` | Yes | No | TrOCR/HPA only | `02_hpa_gt_crops_submission.csv` | HPA OCR trên GT handwritten/printed/annotation crops. |
| `03_qwen_final_gt_formula_table.ipynb` | Yes | No | Qwen base + LoRA | `03_qwen_final_gt_formula_table_submission.csv` | Qwen final prompt trên GT formula/table crops. |

Trọng số:

- Detector notebook cần `YOLO_WEIGHTS` và `YOLO_EXTRA_WEIGHTS`.
- HPA notebook cần `HPA_MODEL_DIR`.
- Qwen notebook cần `QWEN_BASE_DIR` và `QWEN_LORA_DIR`.

## Notebook Đánh Giá

Folder: `fillpaper/evaluate/`

| File | Dùng GT? | Dùng submission? | Output | Dùng để làm gì |
|---|---:|---:|---|---|
| `evaluate_paper_submissions.ipynb` | Yes, đọc GT từ `test.jsonl` | Yes, đọc mọi file `.csv` trong các folder kết quả đặt cạnh notebook chấm điểm | `paper_metric_outputs/` | Tính official metric và các bảng thống kê để điền paper. |
Notebook này copy metric gốc từ `official-evaluation-metric-text-normalization.ipynb`, build solution CSV từ `test.jsonl`, rồi tính:

- `all_submission_metrics.csv`: official score, Detection F1, Precision, Recall, Classification Accuracy, Region CER, Page CER, FP/FN.
- `per_type_metrics.csv`: metric theo từng type GT như handwritten, printed, formula, table, annotation.
- `per_source_metrics.csv`: metric theo source nếu trong `test.jsonl` có field source.
- `qwen_invalid_and_cer_details.csv`: thống kê Formula CER, Table CER, Struct. Acc., Invalid Rate cho Qwen ablation.
- `paper_full_system_results.csv/md`, `paper_routing_ablation.csv/md`, `paper_qwen_ablation.csv/md`: bảng đã format để copy sang paper.
- `paper_module_results.csv/md`: hai dòng Qwen3-VL module-level cho `tab:module-results`, gồm Formula CER và Table CER. `N` là số formula/table scorable matched regions thật sự được tính CER.
- `paper_per_type_results.csv/md`: bảng `tab:per-type-results` cho full-system/routing final nếu có submission tương ứng.
- `paper_qwen_ablation_full_official_debug.csv/md`: full-page official score cho Qwen ablation, chỉ để debug. Không nên dùng làm kết luận chính vì các notebook Qwen ablation cố tình để rỗng text của handwritten/printed/annotation.

- `evaluation_errors.csv`: các submission bị lỗi format hoặc thiếu image.

## Quick Sanity Checklist

Trước khi chạy trên Kaggle:

- [ ] Sửa `INFERENCE_PY` đến đúng file `fillpaper/inference.py` trong Kaggle dataset code.
- [ ] Sửa `TEST_JSONL` đến đúng `fillpaper/test.jsonl`.
- [ ] Sửa `IMAGE_ROOT` sao cho `IMAGE_ROOT / record["file_name"]` tồn tại.
- [ ] Sửa `HPA_MODEL_DIR`, `QWEN_BASE_DIR`, `QWEN_LORA_DIR`, `YOLO_WEIGHTS`, `YOLO_EXTRA_WEIGHTS`.
- [ ] Đảm bảo Kaggle notebook bật GPU T4 x2.
- [ ] Chạy trước một notebook nhẹ, ví dụ `full_system_results/00_yolo_bbox_only_empty_text.ipynb`, để confirm output CSV format.
- [ ] Sau đó chạy `module_results/01_detector_yolo_predictions_empty_text.ipynb` để confirm YOLO path.
- [ ] Sau đó chạy một notebook Qwen GT, ví dụ `qwen_ablation/06_lora_final_gt_formula_table.ipynb`, để confirm Qwen path/VRAM.
