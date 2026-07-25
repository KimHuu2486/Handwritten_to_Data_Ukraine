# Qwen3-VL Formula/Table Specialist

Thư mục này chứa Stage 2C chuyên biệt cho hai loại vùng khó: `formula` và
`table`. Notebook chạy trên Google Colab A100, bắt đầu từ một LoRA đã fine-tune
và tiếp tục huấn luyện trên crop dataset tăng cường.

## File trong thư mục

- [`formula_table_aug_colab.ipynb`](./formula_table_aug_colab.ipynb): notebook
  huấn luyện dùng chung cho formula và table.
- [`dataset.md`](./dataset.md): liên kết tới dữ liệu trên Google Drive.

## Cách đọc notebook

Đọc theo thứ tự:

1. Cấu trúc Google Drive và đường dẫn dữ liệu.
2. `RUN_TASK` và các cấu hình riêng cho từng task.
3. Kiểm tra starting LoRA, crop dataset và resume checkpoint.
4. Tạo mẫu prompt/answer từ `labels.jsonl`.
5. Load Qwen3-VL cùng LoRA hiện có.
6. Huấn luyện, đánh giá và lưu adapter chuyên biệt.

## Hai lượt huấn luyện độc lập

Notebook chỉ xử lý một task trong mỗi lần chạy:

```text
RUN_TASK = "formula" → adapter chuyên formula
RUN_TASK = "table"   → adapter chuyên table
```

Sau khi hoàn thành một task, cần giải phóng VRAM hoặc khởi động lại runtime trước
khi chạy task còn lại. Hai lượt không nối tiếp nhau; cả hai có thể bắt đầu từ
cùng một starting LoRA.

## Input

Cấu trúc chính mà notebook mong đợi:

```text
/content/drive/MyDrive/rukopys/
├── data/       # crop dataset zip
├── model/      # tùy chọn: bản sao base model
├── adapter/    # starting LoRA
├── resume/     # tùy chọn: checkpoint riêng theo task
└── output/     # checkpoint và adapter cuối
```

Crop dataset sau khi giải nén:

```text
ocr_region_crops_wrong_aug/
├── labels.jsonl
├── formula/*.jpg
├── table/*.jpg
└── stats.json
```

Notebook lọc `labels.jsonl` theo `RUN_TASK`, dùng crop làm input và trường `text`
làm đáp án.

## Output

| Task | Thư mục checkpoint | Adapter cuối |
|---|---|---|
| Formula | `output/stage2c_formula_aug` | `output/qwen3vl_rukopys_stage2c_formula_aug_lora_final` |
| Table | `output/stage2c_table_aug` | `output/qwen3vl_rukopys_stage2c_table_aug_lora_final` |

> [!IMPORTANT]
> `START_LORA_DIR` phải chứa `adapter_config.json`. Thư mục crop phải chứa
> `labels.jsonl`; nếu thiếu một trong hai, notebook sẽ dừng trước khi huấn luyện.

## Khi nào dùng pipeline này

Dùng các adapter specialist khi cần cải thiện riêng formula hoặc table. Đây là
nhánh fine-tune bổ sung, không thay thế quy trình Stage 1 Silver → Stage 2 Gold
cho năng lực OCR tổng quát.
