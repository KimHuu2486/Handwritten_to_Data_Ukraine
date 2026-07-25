# TrOCR Single-Phase Fine-Tuning with Scanned Dataset

Pipeline này thực hiện cắt ảnh dòng (Line cropping) từ tập dữ liệu trang quét gốc và huấn luyện tinh chỉnh đơn tầng (Single-phase Fine-tuning) mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) trên tập dữ liệu: **[rukopys-scanned-dataset](https://www.kaggle.com/datasets/quii29/rukopys-scanned-dataset)**.

## Luồng chính

```text
Trang ảnh quét gốc + metadata.jsonl (cấp trang)
                         ↓
  crop_dataset.py (Trích xuất crop dòng chữ theo bbox)
                         ↓
Tập dữ liệu ảnh dòng (dataset/train/metadata.jsonl & images/)
                         ↓
train_kansallisarkisto_hpa_single_train.py (Seq2SeqTrainer + EarlyStopping)
                         ↓
  Typed Evaluation & Checkpoint tối ưu (final_cyrillic_htr_model)
```

## Thứ tự đọc

1. [`crop_dataset.py`](./crop_dataset.py)
2. [`train_kansallisarkisto_hpa_single_train.py`](./train_kansallisarkisto_hpa_single_train.py)
3. [`run_train_kansallisarkisto_hpa_single_train_cloud.sh`](./run_train_kansallisarkisto_hpa_single_train_cloud.sh)
4. [`requirements.txt`](./requirements.txt)
5. [`dataset.md`](./dataset.md)

## Vai trò từng file

### `crop_dataset.py`

Python script xử lý tiền dữ liệu. Script đọc `metadata.jsonl` cấp trang quét gốc, cắt các vùng chữ theo `bbox` (`handwritten`, `printed`, `annotation`) và đóng gói thành tập dữ liệu dòng chữ chuẩn cho script huấn luyện.

### `train_kansallisarkisto_hpa_single_train.py`

Python script chính huấn luyện tinh chỉnh TrOCR đơn tầng. Script tự động chia tập train/val phân tầng 90/10, mở rộng tokenizer cho tiếng Ukraine và lưu checkpoint có chỉ số CER tốt nhất thông qua HuggingFace Trainer.

Đầu ra chính:
```text
OUTPUT_DIR/
├── splits/                           # Các file split cố định (train_fixed_hpa.jsonl, val_fixed_hpa.jsonl)
├── checkpoint-xxxx/                  # Checkpoints lưu theo epoch
└── final_cyrillic_htr_model/         # Mô hình thành phẩm xuất cuối cùng
```

### `run_train_kansallisarkisto_hpa_single_train_cloud.sh`

Script Bash Wrapper tự động khởi tạo môi trường ảo `venv`, nạp file môi trường `.env.hpa`, kiểm tra cấu trúc thư mục dữ liệu và khởi chạy tiến trình huấn luyện trên máy chủ/Cloud.

### `requirements.txt`

Khai báo các thư viện Python cần thiết (`torch`, `transformers`, `Pillow`, `jiwer`, `accelerate`, ...).

### `dataset.md`

Tài liệu tham chiếu liên kết nguồn dữ liệu trang quét gốc trên Kaggle.

---

## Quy trình Xử lý Dữ liệu & Cắt Ảnh Dòng (Line Cropping)

Tập dữ liệu trang quét gốc chứa thông tin cấp trang ảnh:
```json
{
  "file_name": "images/964a31fc-0bee-5096-9359-2371641869eb.jpg",
  "image_width": 2200,
  "image_height": 1617,
  "regions": [
    {
      "bbox": [195, 983, 657, 1046],
      "type": "handwritten",
      "text": "Електростанції:"
    }
  ]
}
```

### Lệnh thực hiện Cắt ảnh (Crop Command)

Chạy lệnh sau để trích xuất toàn bộ dòng ảnh theo bounding box:

```bash
python crop_dataset.py \
  --input-metadata metadata.jsonl \
  --input-dir . \
  --output-dir dataset \
  --split-name train \
  --target-types handwritten,printed,annotation \
  --margin 0 \
  --image-format jpg
```

### Cấu trúc dữ liệu sau khi cắt

Sau khi cắt xong, thư mục đầu ra có dạng:
```text
dataset/
└── train/
    ├── metadata.jsonl
    └── images/
        ├── 964a31fc-0bee-5096-9359-2371641869eb_crop_0_printed.jpg
        ├── 964a31fc-0bee-5096-9359-2371641869eb_crop_1_handwritten.jpg
        └── ...
```

---

## Hướng dẫn Chạy Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 2: Tạo File Môi Trường (`.env.hpa`)

```bash
cat << 'EOF' > .env.hpa
# Token HuggingFace (nếu tải hoặc lưu mô hình private)
HF_TOKEN=hf_your_huggingface_token_here

# Đường dẫn dữ liệu đã cắt (chứa thư mục train/metadata.jsonl và train/images/)
HPA_DATA_ROOT=/home/ubuntu/dataset
HPA_TRAIN_DIR=/home/ubuntu/dataset/train
HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_single

# Siêu tham số huấn luyện
HPA_TRAIN_EPOCHS=10
HPA_TRAIN_LR=1e-5
HPA_TRAIN_BATCH_SIZE=32
HPA_EVAL_BATCH_SIZE=32
HPA_VAL_RATIO=0.10

# Đường dẫn Python
PYTHON_BIN=python3
EOF
```

### Bước 3: Kích hoạt Tiến trình Huấn Luyện

Bạn có thể lựa chọn 1 trong 3 cách kích hoạt sau:

#### Cách 1: Chạy qua Bash Script với file `.env.hpa` (Khuyến nghị)

```bash
# Gán quyền thực thi
chmod +x run_train_kansallisarkisto_hpa_single_train_cloud.sh

# Khởi chạy huấn luyện
ENV_FILE=.env.hpa ./run_train_kansallisarkisto_hpa_single_train_cloud.sh
```

#### Cách 2: Truyền trực tiếp các biến môi trường qua dòng lệnh

```bash
HPA_DATA_ROOT=/home/ubuntu/dataset \
HPA_OUTPUT_DIR=/home/ubuntu/outputs_hpa_single \
./run_train_kansallisarkisto_hpa_single_train_cloud.sh
```

#### Cách 3: Chạy trực tiếp Python script

```bash
python train_kansallisarkisto_hpa_single_train.py
```

---

## Các Biến Môi Trường Hỗ Trợ

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `/home/ubuntu/dataset` | Thư mục gốc chứa thư mục con `train/`. |
| `HPA_TRAIN_DIR` | `$DATA_ROOT/train` | Thư mục chứa `metadata.jsonl` và ảnh dòng `images/`. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `/home/ubuntu/outputs_hpa_single` | Thư mục xuất kết quả đánh giá, checkpoint và mô hình thành phẩm. |
| `HPA_TRAIN_EPOCHS` | `10.0` | Số lượng Epoch huấn luyện tối đa. |
| `HPA_TRAIN_LR` | `1e-5` | Tốc độ học (Learning Rate) ban đầu. |
| `HPA_TRAIN_BATCH_SIZE` | `32` | Batch size huấn luyện trên mỗi GPU. |
| `HPA_EVAL_BATCH_SIZE` | `32` | Batch size kiểm định trên mỗi GPU. |
| `HPA_VAL_RATIO` | `0.10` | Tỷ lệ trích xuất tập Validation phân tầng (10%). |
| `HPA_REBUILD_SPLITS` | `false` | Đặt `true` nếu muốn tạo lại phân tách train/val mới. |
| `HF_TOKEN` | `None` | Token API HuggingFace. |
| `PYTHON_BIN` | `python` | Trình thông dịch Python được chỉ định. |
