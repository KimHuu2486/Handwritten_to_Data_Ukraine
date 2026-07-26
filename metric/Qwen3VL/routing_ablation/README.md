# Routing Ablation

Folder này so sánh chiến lược routing khi **YOLO tự dự đoán** `bbox/type`.
Kết quả dùng để chứng minh lợi ích của phân luồng TrOCR/HPA và Qwen3-VL.

| Notebook | Routing | Model cần load | Output CSV |
|---|---|---|---|
| [`01_yolo_trocr_all_text_like.ipynb`](./01_yolo_trocr_all_text_like.ipynb) | TrOCR đọc mọi text-like region | YOLO + TrOCR/HPA | `01_yolo_trocr_all_text_like_submission.csv` |
| [`02_yolo_qwen_all_text_like.ipynb`](./02_yolo_qwen_all_text_like.ipynb) | Qwen đọc mọi text-like region | YOLO + Qwen base + LoRA | `02_yolo_qwen_all_text_like_submission.csv` |
| [`03_yolo_trocr_hpa_empty_formula_table.ipynb`](./03_yolo_trocr_hpa_empty_formula_table.ipynb) | TrOCR đọc HPA; formula/table rỗng | YOLO + TrOCR/HPA | `03_yolo_trocr_hpa_empty_formula_table_submission.csv` |
| [`04_yolo_qwen_formula_table_empty_hpa.ipynb`](./04_yolo_qwen_formula_table_empty_hpa.ipynb) | Qwen đọc formula/table; HPA rỗng | YOLO + Qwen base + LoRA | `04_yolo_qwen_formula_table_empty_hpa_submission.csv` |
| [`05_yolo_trocr_qwen_hybrid.ipynb`](./05_yolo_trocr_qwen_hybrid.ipynb) | Hybrid: TrOCR cho HPA, Qwen cho formula/table | YOLO + TrOCR/HPA + Qwen + LoRA | `05_yolo_trocr_qwen_hybrid_submission.csv` |

Chạy theo số thứ tự để đi từ single-model baseline đến hybrid final. Tất cả
notebook routing dùng Qwen prompt `final` khi có nhánh Qwen.

## Cần cấu hình

- Mọi notebook: `INFERENCE_PY`, `TEST_JSONL`, `IMAGE_ROOT`, `YOLO_WEIGHTS`, `YOLO_EXTRA_WEIGHTS`.
- `HPA_MODEL_DIR`: `01`, `03`, `05`.
- `QWEN_BASE_DIR`, `QWEN_LORA_DIR`: `02`, `04`, `05`.

Đưa các CSV sinh ra vào thư mục evaluator tìm kiếm, rồi chạy
[`../evaluate/evaluate_paper_submissions.ipynb`](../evaluate/evaluate_paper_submissions.ipynb).
