# Qwen3-VL Formula/Table Ablation

Các notebook trong folder này đo riêng nhánh Qwen3-VL trên `formula` và `table`.
Chúng dùng **GT bbox/type từ `test.jsonl`**, không dùng YOLO, nên phù hợp để so
sánh OCR/prompt Qwen chứ không phải full-system score.

## Thứ tự chạy

Chạy `01` → `06`; mỗi notebook chỉ đổi model/prompt, còn GT crop giữ cố định.

| Notebook | Model / prompt | Mục đích | Output CSV |
|---|---|---|---|
| [`01_base_qwen_generic_gt_formula_table.ipynb`](./01_base_qwen_generic_gt_formula_table.ipynb) | Qwen base, generic | Zero-shot baseline, không LoRA. | `01_base_qwen_generic_gt_formula_table_submission.csv` |
| [`02_lora_generic_gt_formula_table.ipynb`](./02_lora_generic_gt_formula_table.ipynb) | Base + LoRA, generic | Đo lợi ích LoRA với prompt giữ nguyên. | `02_lora_generic_gt_formula_table_submission.csv` |
| [`03_lora_type_specific_gt_formula_table.ipynb`](./03_lora_type_specific_gt_formula_table.ipynb) | Base + LoRA, type-specific | Prompt riêng cho formula/table. | `03_lora_type_specific_gt_formula_table_submission.csv` |
| [`04_lora_source_type_gt_formula_table.ipynb`](./04_lora_source_type_gt_formula_table.ipynb) | Base + LoRA, source-aware + type-specific | Thử thêm ngữ cảnh source. | `04_lora_source_type_gt_formula_table_submission.csv` |
| [`05_lora_source_guardrails_gt_formula_table.ipynb`](./05_lora_source_guardrails_gt_formula_table.ipynb) | Base + LoRA, source-aware + guardrails | Đo guardrail chống giải/sửa/hallucinate. | `05_lora_source_guardrails_gt_formula_table_submission.csv` |
| [`06_lora_final_gt_formula_table.ipynb`](./06_lora_final_gt_formula_table.ipynb) | Base + LoRA, final | Cấu hình cuối, đồng bộ inference pipeline. | `06_lora_final_gt_formula_table_submission.csv` |

## Cần cấu hình

- `TEST_JSONL` và `IMAGE_ROOT` phải trỏ tới GT và ảnh.
- `QWEN_BASE_DIR` cần cho mọi notebook.
- `QWEN_LORA_DIR` cần cho `02`–`06`; `01` không load LoRA.

Sau khi có CSV, chạy
[`../evaluate/evaluate_paper_submissions.ipynb`](../evaluate/evaluate_paper_submissions.ipynb).
Ưu tiên Formula CER, Table CER, Structure Accuracy và Invalid Rate; không dùng
official full-page score nhóm này để kết luận full-system vì các loại HPA được
để text rỗng có chủ đích.
