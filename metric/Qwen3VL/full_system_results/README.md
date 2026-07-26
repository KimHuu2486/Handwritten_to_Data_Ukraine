# Full-System Results

Các notebook trong folder này là ablation **toàn bộ pipeline**: YOLO dự đoán `bbox/type`, sau đó các nhánh OCR điền text. Đây là nhóm dùng để báo Detection F1, Classification Accuracy, CER, PageCER và official score.

| Notebook | Pipeline | Model cần load | Output CSV |
|---|---|---|---|
| [`00_yolo_bbox_only_empty_text.ipynb`](./00_yolo_bbox_only_empty_text.ipynb) | YOLO bbox-only, text rỗng; type placeholder `handwritten` | YOLO | `00_yolo_bbox_only_empty_text_submission.csv` |
| [`01_yolo_bbox_type_empty_text.ipynb`](./01_yolo_bbox_type_empty_text.ipynb) | YOLO bbox + type, text rỗng | YOLO | `01_yolo_bbox_type_empty_text_submission.csv` |
| [`02_yolo_trocr_all_text_like.ipynb`](./02_yolo_trocr_all_text_like.ipynb) | YOLO + TrOCR cho mọi text-like region | YOLO + TrOCR/HPA | `02_yolo_trocr_all_text_like_submission.csv` |
| [`03_yolo_qwen_all_text_like.ipynb`](./03_yolo_qwen_all_text_like.ipynb) | YOLO + Qwen cho mọi text-like region | YOLO + Qwen base + LoRA | `03_yolo_qwen_all_text_like_submission.csv` |
| [`04_yolo_trocr_hpa_empty_formula_table.ipynb`](./04_yolo_trocr_hpa_empty_formula_table.ipynb) | YOLO + TrOCR cho HPA; formula/table rỗng | YOLO + TrOCR/HPA | `04_yolo_trocr_hpa_empty_formula_table_submission.csv` |
| [`05_yolo_qwen_formula_table_empty_hpa.ipynb`](./05_yolo_qwen_formula_table_empty_hpa.ipynb) | YOLO + Qwen cho formula/table; HPA rỗng | YOLO + Qwen base + LoRA | `05_yolo_qwen_formula_table_empty_hpa_submission.csv` |
| [`06_yolo_trocr_qwen_hybrid.ipynb`](./06_yolo_trocr_qwen_hybrid.ipynb) | Hybrid final: TrOCR cho HPA, Qwen cho formula/table | YOLO + TrOCR/HPA + Qwen + LoRA | `06_yolo_trocr_qwen_hybrid_submission.csv` |

## Thứ tự chạy

Chạy `00` → `06` để thấy đóng góp lần lượt của bbox, type, HPA, Qwen và hybrid. `00` không phải classifier baseline vì type chỉ là placeholder để thỏa schema submission.

## Cần cấu hình

Tất cả notebook cần `INFERENCE_PY`, `TEST_JSONL`, `IMAGE_ROOT`, `YOLO_WEIGHTS` và `YOLO_EXTRA_WEIGHTS`. Bổ sung HPA/Qwen paths theo bảng trên.

Sau khi tạo CSV, chạy [`../evaluate/evaluate_paper_submissions.ipynb`](../evaluate/evaluate_paper_submissions.ipynb). Khác với `qwen_ablation`, nhóm này không dùng GT bbox/type nên phù hợp để báo metric full-system.
