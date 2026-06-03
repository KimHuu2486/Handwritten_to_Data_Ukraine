# Runbook fine-tune 2 adapter Qwen3-VL cho Formula và Table OCR

## 1. Mục tiêu gần nhất

Notebook hiện tại `formula_table_aug_colab_a100_4bit_bf16.ipynb` đang chạy theo hướng **một adapter chung** cho cả `formula` và `table`:

```text
TARGET_TYPES = {"formula", "table"}
output/stage2c_formula_table_aug
output/qwen3vl_rukopys_stage2c_formula_table_aug_lora_final
```

Quy trình tiếp theo nên chuyển thành **hai lượt fine-tune riêng**, dùng cùng base model và cùng starting LoRA nếu cần, nhưng tách dataset, prompt, checkpoint, output và đánh giá:

```text
Run A: formula only -> qwen3vl_rukopys_stage2c_formula_aug_lora_final
Run B: table only   -> qwen3vl_rukopys_stage2c_table_aug_lora_final
```

Mục tiêu không phải là train lâu hơn trên cùng dữ liệu, mà là giảm nhiễu format giữa hai task. `formula` cần sinh công thức/ký hiệu chính xác; `table` cần giữ cấu trúc hàng, cột, ô trống và pipe-separated format.

---

## 2. Vì sao phải tách adapter

Không nên dùng một adapter chung cho cả `formula` và `table` trong giai đoạn này.

```text
YOLO detect bbox/type
  ↓
if type == formula:
    use Qwen formula adapter
if type == table:
    use Qwen table adapter
```

Lý do chính:

- `formula` ưu tiên LaTeX, chỉ số trên/dưới, phân số, căn, vector, ma trận, ký hiệu toán/hóa.
- `table` ưu tiên pipe format, số hàng/cột, ô trống, thứ tự cell, text trong từng cell.
- Hai task có lỗi thường gặp khác nhau, nên nếu train chung model dễ học lẫn format.
- Metric kiểm tra cũng khác nhau: formula cần đúng ký hiệu; table cần đúng cấu trúc trước rồi mới đến text.

---

## 3. Mapping từ notebook hiện tại sang 2 run

Các điểm trong notebook cần được tham số hóa theo từng run:

| Hạng mục | Run formula | Run table |
|---|---|---|
| Target type | `formula` only | `table` only |
| Training output | `output/stage2c_formula_aug` | `output/stage2c_table_aug` |
| Final adapter | `output/qwen3vl_rukopys_stage2c_formula_aug_lora_final` | `output/qwen3vl_rukopys_stage2c_table_aug_lora_final` |
| Callback name | `stage2c-formula-aug` | `stage2c-table-aug` |
| Eval chính | formula CER/exact/sanity LaTeX | table CER + row/column consistency |

Điểm quan trọng: không để hai run ghi chung checkpoint hoặc final adapter. Nếu resume training, mỗi run cũng phải có resume root riêng để tránh load nhầm checkpoint của task còn lại.

---

## 4. Layout Drive/Colab đề xuất

Giữ layout Drive hiện có, nhưng tách output rõ ràng:

```text
/content/drive/MyDrive/rukopys/
  data/
    ocr-region-crops-wrong-aug.zip
  model/
    qwen-3-vl-8b-instruct/
  adapter/
    adapter_config.json              # starting LoRA chung, nếu dùng Stage trước làm nền
  resume/
    formula/
    table/
  output/
    stage2c_formula_aug/
    stage2c_table_aug/
    qwen3vl_rukopys_stage2c_formula_aug_lora_final/
    qwen3vl_rukopys_stage2c_table_aug_lora_final/
```

Nếu starting adapter hiện tại là adapter chung từ stage trước, cả hai run có thể bắt đầu từ cùng adapter đó. Sau khi train xong, không tiếp tục dùng adapter chung `formula_table` làm adapter chính nữa; nó chỉ nên là baseline để so sánh.

---

## 5. Run A: fine-tune Formula adapter

Run formula chỉ lấy sample có `type == "formula"` từ `labels.jsonl`.

Pseudo-config:

```text
RUN_TASK: formula
TARGET_TYPES: formula only
output_dir: output/stage2c_formula_aug
final_dir: output/qwen3vl_rukopys_stage2c_formula_aug_lora_final
resume_dir: resume/formula
```

Prompt formula nên giữ tinh thần hiện tại:

- Đọc đúng công thức như ảnh.
- Dùng LaTeX khi đó là biểu diễn rõ nhất.
- Giữ chỉ số trên/dưới, phân số, căn, mũi tên, ma trận, định thức, dấu câu, đánh số, correction marker.
- Không giải bài, không rút gọn, không chuẩn hóa ký hiệu cũ thành kiểu khác.

Đánh giá sau run:

- So với starting adapter.
- So với adapter chung `formula_table` hiện tại.
- Kiểm tra riêng các lỗi: mất `=`, mất phân số, nhầm chỉ số, nhầm dấu `+/-`, bỏ ký hiệu, tự giải/rút gọn.

---

## 6. Run B: fine-tune Table adapter

Run table chỉ lấy sample có `type == "table"` từ `labels.jsonl`.

Pseudo-config:

```text
RUN_TASK: table
TARGET_TYPES: table only
output_dir: output/stage2c_table_aug
final_dir: output/qwen3vl_rukopys_stage2c_table_aug_lora_final
resume_dir: resume/table
```

Prompt table nên tập trung vào cấu trúc:

- Một visual row tương ứng một output line.
- Cell cách nhau bằng `|`.
- Giữ ô trống bằng empty field.
- Giữ thứ tự hàng/cột, text nhiều từ trong cell, wrapped cell, số, đơn vị, dấu câu, lỗi chính tả nhìn thấy.
- Không tự cân bằng cột, không suy luận ô thiếu, không biến bảng thành đoạn văn.

Đánh giá sau run:

- Số dòng output có khớp số hàng nhìn thấy không.
- Mỗi dòng có số cột hợp lý không.
- Có giữ pipe format không.
- Có giữ ô trống không.
- Có biến table thành paragraph không.

---

## 7. Hyperparameter khởi điểm

Vì notebook hiện tại đang fine-tune tiếp từ starting LoRA bằng QLoRA 4-bit BF16 trên A100, nên cấu hình ban đầu nên ưu tiên ổn định trước:

| Hạng mục | Giá trị khởi điểm |
|---|---|
| Base model | Qwen3-VL-8B-Instruct hiện tại |
| Starting adapter | Stage adapter hiện có trong `adapter/` |
| Precision | 4-bit NF4 + BF16 compute |
| Learning rate | giữ gần notebook hiện tại, khoảng `8e-6` đến `2e-5` |
| Epoch | 2-3 epoch, tùy số sample từng task |
| Batch/grad accum | giữ theo notebook trước, chỉ giảm khi OOM |
| Save | checkpoint riêng cho từng task |

Không nên tăng learning rate mạnh ngay ở lần đầu tách adapter, vì starting LoRA đã mang năng lực OCR chung. Sau khi có baseline riêng, mới thử sweep nhỏ theo từng task.

---

## 8. Inference sau khi có 2 adapter

Inference nên dispatch theo `type` từ YOLO:

```text
YOLO bbox/type
  ↓
formula crop -> Qwen base + formula adapter -> formula output
table crop   -> Qwen base + table adapter   -> table output
```

Nếu inference nhiều crop trong một trang, nên gom crop theo type trước để giảm chi phí đổi adapter:

```text
batch formula crops -> formula adapter
batch table crops   -> table adapter
merge outputs theo thứ tự bbox/page
```

Adapter chung cũ chỉ nên giữ làm baseline hoặc fallback tạm thời. Final pipeline nên dùng adapter chuyên biệt.

---

## 9. Những việc chưa làm ở phase này

Phase hiện tại chỉ tập trung vào **tách fine-tune thành 2 adapter riêng**. Các cải tiến sau nên để phase kế tiếp:

1. Formula multi-candidate + LaTeX validator.
2. Table row/cell decomposition trước OCR.
3. Qwen verifier chọn giữa nhiều candidate.
4. Hard-negative correction prompt từ chính lỗi model.
5. Synthetic formula/table có kiểm soát tỷ lệ.

Không nên trộn các thay đổi trên vào cùng lần tách adapter đầu tiên, vì sẽ khó biết điểm tăng/giảm đến từ đâu.

---

## 10. Acceptance checklist

Sau khi chỉnh notebook và chạy xong, cần có các dấu hiệu hoàn tất sau:

- Có hai final adapter riêng:
  - `qwen3vl_rukopys_stage2c_formula_aug_lora_final`
  - `qwen3vl_rukopys_stage2c_table_aug_lora_final`
- Mỗi adapter có `rukopys_prompt_config.json` riêng.
- Config của formula adapter chỉ ghi `target_types = ["formula"]`.
- Config của table adapter chỉ ghi `target_types = ["table"]`.
- Checkpoint formula và table không nằm chung thư mục.
- Log train in ra counts by type chỉ có một type cho từng run.
- Eval riêng cho formula và table đều so với baseline adapter chung hiện tại.

---

## 11. Thứ tự thực hiện đề xuất

```text
1. Tham số hóa notebook bằng RUN_TASK.
2. Chạy smoke test formula với MAX_TRAIN_SAMPLES nhỏ để kiểm tra filter/output path.
3. Chạy full formula adapter.
4. Restart runtime hoặc giải phóng model/VRAM sạch.
5. Chạy smoke test table với MAX_TRAIN_SAMPLES nhỏ.
6. Chạy full table adapter.
7. Đánh giá riêng từng adapter.
8. Cập nhật inference để dispatch adapter theo YOLO type.
```

Kết quả mong muốn của bước này là có hai specialist adapter ổn định, dễ so sánh, dễ rollback và đủ sạch để các phase validator/reranker/hard-negative phía sau có nền đo lường rõ ràng.
