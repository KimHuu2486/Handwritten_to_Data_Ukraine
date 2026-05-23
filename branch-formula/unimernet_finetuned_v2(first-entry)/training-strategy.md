# UniMERNet Finetuned V2 - Training Strategy

## Mục tiêu

Model `unimernet_finetuned_v2(first-entry)` là nhánh nhận dạng công thức cho pipeline RUKOPYS. Mục tiêu của lần train này không phải chỉ ép model học thêm toàn bộ dữ liệu formula một lần, mà là fine-tune theo curriculum để giảm nguy cơ hallucination và giữ model ổn định khi đi từ nhãn sạch sang nhãn nhiễu hơn.

Chiến lược chính:

- Bắt đầu bằng gold formula ngắn, sạch, dễ học.
- Sau đó mở rộng sang công thức dài vừa phải.
- Dùng silver đã lọc như một bước mở rộng dữ liệu, nhưng luôn chọn checkpoint dựa trên gold validation.
- Recovery lại bằng gold để kéo model về phân phối nhãn sạch.
- Cuối cùng train structured formula với learning rate thấp để model học thêm ma trận, bảng, determinant, công thức nhiều dòng mà không phá các phase trước.

## Dữ liệu đầu vào

Metadata gốc có format:

```json
{"image": "images/xxx.jpg", "label": "formula", "text": "..."}
```

Script chuẩn bị metadata:

```text
branch-formula\unimernet_finetuned_v2(first-entry)\prepare_formula_curriculum_metadata.py
```

Script này lọc `label == "formula"` và chia công thức thành 3 nhóm:

- `inline_simple`: công thức ngắn, không có layout phức tạp.
- `medium_long`: công thức dài vừa phải.
- `structured`: công thức có layout bảng/ma trận/nhiều dòng hoặc quá dài.

Các ngưỡng hiện dùng:

```python
SIMPLE_MAX_CHARS = 80
MEDIUM_MAX_CHARS = 180
MAX_SILVER_CHARS = 120
SILVER_FILTER = True
```

Output cho gold:

```text
train_formula_curriculum/formula_all.jsonl
train_formula_curriculum/formula_inline_simple.jsonl
train_formula_curriculum/formula_medium_long.jsonl
train_formula_curriculum/formula_structured.jsonl
train_formula_curriculum/phase1_gold_inline_simple.jsonl
train_formula_curriculum/phase2_gold_inline_medium.jsonl
train_formula_curriculum/phase3_gold_recovery.jsonl
train_formula_curriculum/optional_gold_structured.jsonl
```

Output cho silver:

```text
silver_formula_curriculum/formula_all.jsonl
silver_formula_curriculum/formula_inline_simple.jsonl
silver_formula_curriculum/formula_medium_long.jsonl
silver_formula_curriculum/formula_structured.jsonl
silver_formula_curriculum/phase3_silver_filtered.jsonl
```

Silver structured bị loại khỏi phase silver vì nhóm này thường nhiễu hơn và dễ làm model sinh công thức ảo.

## Model và cấu hình nền

Script train chính:

```text
branch-formula\unimernet_finetuned_v2(first-entry)\train_unimernet_formula_5phase.py
```

Model khởi tạo:

```python
MODEL_DIR = "/workspace/models/unimernet_base"
PRETRAINED = MODEL_DIR / "unimernet_base.pth"
```

Các cấu hình ảnh và sequence:

```python
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 672
MAX_SEQ_LEN = 384
MAX_TEXT_CHARS = 1536
```

Batch mặc định cho GPU lớn như RTX A6000 48GB:

```python
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
AMP = True
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
```

Nếu OOM, giảm `BATCH_SIZE` xuống 16 và tăng `GRAD_ACCUM_STEPS` lên 2 để giữ effective batch size gần tương đương.

## Pipeline train

### Phase 1 - Gold Inline Simple

Metadata:

```text
phase1_gold_inline_simple.jsonl
```

Cấu hình:

```python
PHASE1_EPOCHS = 5
PHASE1_LR = 3e-6
PHASE1_WARMUP_STEPS = 50
```

Mục tiêu:

- Làm model thích nghi với domain handwriting/crop của dataset.
- Chỉ dùng công thức ngắn và sạch để tránh nhiễu sớm.
- Giảm rủi ro model học sai format output ngay từ đầu.

Phase này là nền móng. Nếu phase 1 CER xấu bất thường, không nên kỳ vọng các phase sau cứu được hoàn toàn.

### Phase 2 - Gold Inline + Medium

Metadata:

```text
phase2_gold_inline_medium.jsonl
```

Cấu hình:

```python
PHASE2_EPOCHS = 3
PHASE2_LR = 1e-6
PHASE2_WARMUP_STEPS = 30
```

Mục tiêu:

- Resume từ best checkpoint của phase 1.
- Mở rộng từ công thức ngắn sang công thức dài vừa phải.
- Học thêm các chuỗi có nhiều toán tử, biến, phân số, căn, logic/set notation.

Learning rate thấp hơn phase 1 để tránh làm model quên các pattern cơ bản vừa học.

### Phase 3A - Silver Filtered

Metadata:

```text
phase3_silver_filtered.jsonl
```

Cấu hình:

```python
PHASE3_SILVER_EPOCHS = 1
PHASE3_SILVER_LR = 5e-6
PHASE3_SILVER_WARMUP_STEPS = 50
```

Mục tiêu:

- Tận dụng silver để tăng coverage.
- Chỉ dùng silver simple/medium đã lọc.
- Không dùng silver structured vì rủi ro nhiễu cao.

Điểm quan trọng: phase silver vẫn validate bằng gold phase 2 validation, không validate bằng silver. Điều này giúp checkpoint được chọn theo dữ liệu sạch thay vì tối ưu theo nhãn tự động nhiễu.

### Phase 3B - Gold Recovery

Metadata:

```text
phase3_gold_recovery.jsonl
```

Cấu hình:

```python
PHASE3_GOLD_EPOCHS = 3
PHASE3_GOLD_LR = 1e-6
PHASE3_GOLD_WARMUP_STEPS = 30
```

Mục tiêu:

- Resume từ best checkpoint sau silver.
- Fine-tune lại trên gold clean để giảm drift do silver.
- Khôi phục độ tin cậy của output trên distribution gold.

Đây là bước chống "silver poisoning". Nếu bỏ gold recovery, model có thể học thêm coverage nhưng dễ sinh lỗi lạ hơn.

### Phase 5 - Gold Structured

Metadata:

```text
optional_gold_structured.jsonl
```

Cấu hình:

```python
PHASE5_STRUCTURED_EPOCHS = 2
PHASE5_STRUCTURED_LR = 5e-7
PHASE5_STRUCTURED_WARMUP_STEPS = 20
```

Mục tiêu:

- Học thêm công thức structured: ma trận, bảng, determinant, array, nhiều dòng, layout có dấu `|`, `/`, `\begin{array}`, `\hline`.
- Dùng learning rate thấp để hạn chế phá các capability đã học ở phase trước.
- Chỉ dùng gold structured, không dùng silver structured.

Phase này đặt cuối pipeline vì structured formula phức tạp hơn và dễ làm model bị lệch nếu đưa vào quá sớm.

## Validation và checkpoint

Mỗi stage gọi `run_train_stage(...)` và lưu:

```text
checkpoint_<stage>_best.pth
checkpoint_<stage>_latest.pth
<stage>_log.jsonl
<stage>_epochX_predictions.jsonl
```

Best checkpoint được chọn theo `val_cer`.

Lưu ý quan trọng: validation hiện không phải một tập cố định chung cho mọi phase.

- Phase 1 validate trên split từ `phase1_gold_inline_simple.jsonl`.
- Phase 2 validate trên split từ `phase2_gold_inline_medium.jsonl`.
- Phase 3A silver validate trên split từ `phase2_gold_inline_medium.jsonl`.
- Phase 3B gold recovery validate trên split từ `phase3_gold_recovery.jsonl`.
- Phase 5 structured validate trên split từ `optional_gold_structured.jsonl`.

Vì metadata khác nhau, `val_cer` giữa các phase không nên so sánh trực tiếp như cùng một benchmark. Nó chủ yếu dùng để chọn best checkpoint trong từng phase.

Nếu cần benchmark công bằng xuyên suốt pipeline, nên tạo thêm một fixed validation set riêng gồm simple, medium và structured, rồi evaluate mọi checkpoint trên cùng file đó.

## Early stopping

Cấu hình:

```python
EARLY_STOP_PATIENCE = 2
```

Sau mỗi epoch, nếu `val_cer` không cải thiện đủ số epoch theo patience, stage hiện tại dừng sớm. Điều này giúp tránh overfit, nhất là với gold nhỏ và structured ít mẫu.

## Output cuối

Sau khi chạy xong các phase, script lưu:

```text
checkpoint_curriculum_final_best_loaded.pth
curriculum_manifest.json
```

Manifest chứa:

- đường dẫn checkpoint cuối,
- best checkpoint của từng phase,
- cấu hình train đã dùng.

Ví dụ key phase:

```json
{
  "phase1": "...",
  "phase2": "...",
  "phase3_silver": "...",
  "phase3_gold": "...",
  "phase5_structured": "..."
}
```

## Cách chạy

Trước tiên tạo metadata curriculum:

```bash
python branch-formula/prepare_formula_curriculum_metadata.py
```

Sau đó train:

```bash
python branch-formula/train_unimernet_formula_3phase.py
```

Nếu chạy trên Thunder Compute với layout khác, dùng environment variables:

```bash
MODEL_DIR=/home/ubuntu/unimernet_base \
PRETRAINED=/home/ubuntu/unimernet_base/pytorch_model.pth \
GOLD_ROOT=/home/ubuntu/dataset/train \
SILVER_ROOT=/home/ubuntu/dataset/silver \
GOLD_CURRICULUM_DIR=/home/ubuntu/dataset/train_formula_curriculum \
SILVER_CURRICULUM_DIR=/home/ubuntu/dataset/silver_formula_curriculum \
OUTPUT_DIR=/home/ubuntu/unimernet_formula_curriculum_3phase \
python train_unimernet_formula_3phase.py
```

## Rủi ro cần theo dõi

### Validation không cố định

CER giữa các phase không hoàn toàn comparable. Nên có thêm fixed validation set nếu muốn báo cáo model version một cách sạch sẽ.

### Structured phase có thể làm lệch model

Structured formula nên train cuối, LR thấp. Nếu sau phase 5 model giảm chất lượng trên simple/medium, có thể:

- giảm `PHASE5_STRUCTURED_EPOCHS` còn 1,
- giảm `PHASE5_STRUCTURED_LR` xuống `3e-7`,
- hoặc chọn checkpoint phase 3 gold làm final cho general use và giữ phase 5 như checkpoint chuyên structured.

### Silver nhiễu

Silver chỉ nên dùng sau khi lọc. Nếu phase 3 silver làm CER gold tăng mạnh, cần giảm LR hoặc bỏ silver stage.

### OOM

Nếu gặp OOM:

```python
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
```

Nếu vẫn OOM, giảm tiếp `MAX_SEQ_LEN` hoặc `EVAL_BATCH_SIZE`.

## Kết luận

Chiến lược v2 dùng curriculum theo độ khó:

```text
gold simple
-> gold simple + medium
-> silver filtered
-> gold recovery
-> gold structured low-LR
```

Thiết kế này ưu tiên sự ổn định: học từ sạch đến khó, dùng silver có kiểm soát, rồi luôn quay lại gold để giữ model không bị trôi khỏi nhãn tin cậy.