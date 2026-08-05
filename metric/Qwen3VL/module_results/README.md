# Module Results

Folder này tách đóng góp detector và OCR khỏi full-system. Metric liên quan detector dùng YOLO prediction; metric OCR riêng dùng GT crop để không bị nhiễu bởi lỗi layout.

| Notebook | Bbox/type | Mục đích | Model cần load | Output CSV |
|---|---|---|---|---|
| `01_detector_yolo_predictions_empty_text.ipynb` | YOLO predicted | Đo detector/module detection metric; text rỗng. | YOLO | `01_detector_yolo_predictions_empty_text_submission.csv` |
| `02_hpa_gt_crops.ipynb` | GT | Đo TrOCR/HPA trên handwritten/printed/annotation crop. | TrOCR/HPA | `02_hpa_gt_crops_submission.csv` |
| [`03_qwen_final_gt_formula_table.ipynb`](./03_qwen_final_gt_formula_table.ipynb) | GT | Đo Qwen final prompt trên formula/table crop. | Qwen base + LoRA | `03_qwen_final_gt_formula_table_submission.csv` |

## Cách chạy notebook hiện có

1. Cấu hình `TEST_JSONL`, `IMAGE_ROOT`, `QWEN_BASE_DIR`, `QWEN_LORA_DIR`.
2. Chạy `03_qwen_final_gt_formula_table.ipynb` để tạo CSV.
3. Chạy [`../evaluate/evaluate_paper_submissions.ipynb`](../evaluate/evaluate_paper_submissions.ipynb); evaluator dùng run ID `03_qwen_final_gt_formula_table` để xuất Formula CER và Table CER vào `paper_module_results.csv/md`.

Vì bbox/type là GT, không diễn giải Detection F1/ClassAcc của output này như năng lực detector.
