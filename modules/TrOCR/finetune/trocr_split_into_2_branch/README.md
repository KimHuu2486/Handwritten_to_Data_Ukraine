# TrOCR 2-Branch Fine-Tuning Strategy

Pipeline này thực hiện chiến lược **tách HPA thành 2 mô hình nhánh riêng biệt (2-Branch Fine-Tuning)** cho bài toán nhận diện văn bản tiếng Ukraine trên tập dữ liệu RUKOPYS:

1. **Nhánh 1 (`handwritten` & `printed`)**: Dành riêng cho chữ viết tay và chữ in (dòng dài, nhiều ngữ cảnh) sử dụng base model [`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model).
2. **Nhánh 2 (`annotation`)**: Dành riêng cho các ký tự chú thích ngắn (thường dạng ký tự đơn hoặc số chú giải) sử dụng base model [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten).

Dữ liệu huấn luyện được hợp nhất từ 3 tệp metadata (`metadata_part1.jsonl`, `metadata_part2.jsonl`, `metadata_part3.jsonl`) trích xuất từ dataset gốc (`train`, `silver`), đồng thời **lọc bỏ hoàn toàn các trang bị gắn nhãn lỗi** dựa trên danh sách audit [`manifest_labelerror_prederror_or_labelerror_predtrue.csv`](./manifest_labelerror_prederror_or_labelerror_predtrue.csv).

## Luồng chính

```text
Metadata hợp nhất (metadata_part1.jsonl, part2.jsonl, part3.jsonl)
                           ↓
Lọc bỏ các trang lỗi nhãn (manifest_labelerror_prederror_or_labelerror_predtrue.csv)
                           ↓
  crop_bboxes_from_metadata.py (Cắt ảnh crop theo bbox & tạo manifest.csv)
                           ↓
     ┌─────────────────────┴─────────────────────┐
     ↓                                           ↓
Nhánh 1: Handwritten + Printed             Nhánh 2: Annotation Specialist
Base: Kansallisarkisto/cyrillic-htr-model  Base: microsoft/trocr-base-handwritten
Config: num_beams=2, max_tokens=96         Config: num_beams=1, max_tokens=32
     ↓                                           ↓
Checkpoint & Artifacts 1                    Checkpoint & Artifacts 2
```

## Thứ tự đọc

1. [`crop_bboxes_from_metadata.py`](./crop_bboxes_from_metadata.py)
2. [`train_hpa_handwritten_printed_single_phase.py`](./train_hpa_handwritten_printed_single_phase.py)
3. [`run_train_hpa_handwritten_printed_single_phase.sh`](./run_train_hpa_handwritten_printed_single_phase.sh)
4. [`train_hpa_annotation_single_phase.py`](./train_hpa_annotation_single_phase.py)
5. [`run_train_hpa_annotation_single_phase.sh`](./run_train_hpa_annotation_single_phase.sh)
6. [`dataset.md`](./dataset.md)

## Vai trò từng file

### `crop_bboxes_from_metadata.py`

Python script xử lý trích xuất ảnh crop. Script đọc file metadata, tự động điều chỉnh và giới hạn tọa độ bounding box (`clamp_bbox`), cắt các ảnh dòng chữ từ ảnh trang gốc và tạo ra tệp `manifest.csv` kèm cấu trúc thư mục ảnh phân loại theo nhãn (`--save-by-type`).

### `manifest_labelerror_prederror_or_labelerror_predtrue.csv`

Tệp CSV danh sách các trang ảnh bị gắn nhãn lỗi (Label Errors) qua quá trình rà soát audit out-of-fold. Toàn bộ các trang xuất hiện trong danh sách này bị loại bỏ khỏi tập huấn luyện để đảm bảo độ sạch của dữ liệu.

### `metadata_part1.jsonl`, `metadata_part2.jsonl`, `metadata_part3.jsonl`

Ba tệp siêu dữ liệu JSONL cấp trang chứa danh sách các vùng chọn (`regions`) được tổng hợp từ dữ liệu chuẩn gốc (`train`) và dữ liệu tăng cường (`silver`).

### `train_hpa_handwritten_printed_single_phase.py`

Python script huấn luyện chuyên biệt cho Nhánh 1 (`handwritten` và `printed`). Script mở rộng Tokenizer với các ký tự đặc thù tiếng Ukraine, hỗ trợ các nhãn ký hiệu đặc biệt (`~~`, `{`, `}`, `[illegible]`) và chạy sinh chuỗi với `num_beams=2`.

Đầu ra chính:
```text
OUTPUT_DIR/
├── splits/                           # Các file split cố định (train_fixed_hpa.jsonl, val_fixed_hpa.jsonl)
└── final_cyrillic_htr_model/         # Trọng số mô hình thành phẩm Nhánh 1
```

### `run_train_hpa_handwritten_printed_single_phase.sh`

Bash runner script tự động tạo môi trường ảo `venv`, nạp file môi trường `.env.hpa` và khởi chạy huấn luyện cho Nhánh 1.

### `train_hpa_annotation_single_phase.py`

Python script huấn luyện chuyên biệt cho Nhánh 2 (`annotation`). Script tối ưu hóa cho các vùng chú thích ngắn, sử dụng Greedy Search (`num_beams=1`, `max_new_tokens=32`) để tăng tốc độ suy luận và giữ độ chính xác cao.

Đầu ra chính:
```text
OUTPUT_DIR/
├── splits/                           # Các file split cố định
└── final_cyrillic_htr_model/         # Trọng số mô hình thành phẩm Nhánh 2
```

### `run_train_hpa_annotation_single_phase.sh`

Bash runner script cho Nhánh 2.

### `dataset.md`

Tài liệu tham chiếu liên kết nguồn dữ liệu RUKOPYS Dataset trên Kaggle.

---

## Kiến trúc Tách 2 Nhánh (2-Branch Strategy)

| Đặc điểm / Tham số | Nhánh 1: Handwritten & Printed | Nhánh 2: Annotation Specialist |
|---|---|---|
| Nhãn mục tiêu (`TARGET_LABELS`) | `handwritten`, `printed` | `annotation` |
| Base Model (`MODEL_NAME`) | `Kansallisarkisto/cyrillic-htr-model` | `microsoft/trocr-base-handwritten` |
| Base Processor | `microsoft/trocr-base-handwritten` | `microsoft/trocr-base-handwritten` |
| Beam Search (`num_beams`) | `2` | `1` (Greedy Search) |
| Max New Tokens | `96` | `32` |
| Length Penalty | `1.5` | `1.0` |
| Repetition Penalty | `1.15` | `1.2` |
| Bảng ký tự đặc trưng | Bổ sung ký tự tiếng Ukraine + Text Markers | Bổ sung ký tự tiếng Ukraine + Text Markers |

> [!NOTE]
> Việc tách riêng `annotation` thành một mô hình độc lập giúp tránh hiện tượng xung đột phân phối độ dài (Length bias) giữa các dòng viết tay dài và các nốt chú thích cực ngắn.

---

## Làm sạch Dữ liệu (Data Audit & Filtering)

Tập dữ liệu đầu vào sử dụng 3 phần `metadata_part1.jsonl`, `metadata_part2.jsonl`, `metadata_part3.jsonl` thu thập từ cả hai nguồn `train` và `silver`. 

Để nâng cao chất lượng nhãn, tiến trình tiền xử lý tiến hành đối chiếu với `manifest_labelerror_prederror_or_labelerror_predtrue.csv` và tự động loại bỏ tất cả các trang nằm trong danh sách nhãn nhiễu/lỗi.

---

## Hướng dẫn Chạy Tiến Trình Huấn Luyện (Run Guide)

### Bước 1: Cắt ảnh Bounding Box từ Metadata đã lọc

```bash
python crop_bboxes_from_metadata.py \
  --metadata metadata_part1.jsonl \
  --image-root /path/to/dataset \
  --output-dir dataset/train \
  --save-by-type \
  --padding 0
```

---

### Bước 2: Chuẩn bị File Môi Trường (`.env.hpa`)

Tạo file cấu hình môi trường `.env.hpa`:

```bash
cat << 'EOF' > .env.hpa
# Token HuggingFace (nếu cần tải hoặc push mô hình)
HF_TOKEN=hf_your_huggingface_token_here

# Đường dẫn tới thư mục dữ liệu đã crop và thư mục đầu ra
HPA_DATA_ROOT=/home/ubuntu/dataset
HPA_TRAIN_DIR=/home/ubuntu/dataset/train
HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_handwritten_printed

# Siêu tham số huấn luyện
HPA_TRAIN_EPOCHS=10.0
HPA_TRAIN_LR=1e-5
HPA_TRAIN_BATCH_SIZE=32
HPA_EVAL_BATCH_SIZE=16

# Trình thông dịch Python
SYSTEM_PYTHON_BIN=python3
EOF
```

---

### Bước 3: Khởi chạy huấn luyện Nhánh 1 (Handwritten & Printed)

#### Cách 1: Chạy qua Bash Script (Khuyến nghị)

```bash
chmod +x run_train_hpa_handwritten_printed_single_phase.sh
ENV_FILE=.env.hpa ./run_train_hpa_handwritten_printed_single_phase.sh
```

#### Cách 2: Chạy trực tiếp Python script

```bash
python train_hpa_handwritten_printed_single_phase.py
```

---

### Bước 4: Khởi chạy huấn luyện Nhánh 2 (Annotation Specialist)

Tạo file môi trường cho Nhánh 2 (hoặc thay đổi `HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_annotation`):

```bash
# Khởi chạy qua Bash Script
chmod +x run_train_hpa_annotation_single_phase.sh
HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_annotation ENV_FILE=.env.hpa ./run_train_hpa_annotation_single_phase.sh
```

Hoặc chạy bằng Python script:

```bash
python train_hpa_annotation_single_phase.py
```

---

## Các Biến Môi Trường Hỗ Trợ

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `/home/ubuntu/dataset` | Đường dẫn thư mục gốc chứa `train/`. |
| `HPA_TRAIN_DIR` | `$DATA_ROOT/train` | Thư mục chứa `manifest.csv` (hoặc `metadata.jsonl`) và các thư mục ảnh. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `/home/ubuntu/outputs_hpa_...` | Thư mục lưu checkpoint và mô hình xuất ra của từng nhánh. |
| `HPA_TRAIN_EPOCHS` | `10.0` | Số lượng Epoch huấn luyện tối đa. |
| `HPA_TRAIN_LR` | `1e-5` | Tốc độ học (Learning Rate) ban đầu. |
| `HPA_TRAIN_BATCH_SIZE` | `32` | Batch size huấn luyện trên mỗi GPU. |
| `HPA_EVAL_BATCH_SIZE` | `32` | Batch size kiểm định trên mỗi GPU. |
| `HPA_VAL_RATIO` | `0.10` | Tỷ lệ trích xuất tập Validation (10%). |
| `HF_TOKEN` | `None` | Token xác thực HuggingFace API. |
| `SYSTEM_PYTHON_BIN` | `python3` | Trình thông dịch Python gốc hệ thống. |
