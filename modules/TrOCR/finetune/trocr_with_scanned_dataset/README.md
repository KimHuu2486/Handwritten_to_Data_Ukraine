# TrOCR Single-Phase Fine-Tuning with Scanned Dataset

Thư mục này chứa toàn bộ mã nguồn, kịch bản cắt ảnh dòng và kịch bản huấn luyện tinh chỉnh đơn tầng (Single-phase Fine-tuning) mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) cho bài toán nhận diện chữ viết tay tiếng Ukraine trên tập dữ liệu trang quét: **[rukopys-scanned-dataset](https://www.kaggle.com/datasets/quii29/rukopys-scanned-dataset)**.

---

## Cấu trúc thư mục

| File / Thư mục | Mô tả |
| --- | --- |
| [`crop_dataset.py`](./crop_dataset.py) | Python script đọc tệp `metadata.jsonl` cấp trang quét gốc và tiến hành trích xuất/cắt (crop) các vùng ảnh dòng chữ theo bounding box (`bbox`), tự động đóng gói cấu trúc thư mục đầu ra chuẩn cho script huấn luyện. |
| [`train_kansallisarkisto_hpa_single_train.py`](./train_kansallisarkisto_hpa_single_train.py) | Python script chính thực hiện toàn bộ pipeline huấn luyện đơn tầng, tự động chia tập train/val phân tầng, lưu checkpoint có CER tốt nhất và đánh giá theo loại vùng (Typed Evaluation). |
| [`run_train_kansallisarkisto_hpa_single_train_cloud.sh`](./run_train_kansallisarkisto_hpa_single_train_cloud.sh) | Script Bash tự động tạo môi trường ảo `venv`, nạp file cấu hình môi trường `.env` và kích hoạt quá trình huấn luyện trên máy chủ/Cloud. |
| [`metadata.jsonl`](./metadata.jsonl) | Tệp nhãn mẫu gốc cấp trang quét chứa danh sách các vùng chọn (`regions`) kèm tọa độ `bbox`, loại vùng (`type`) và văn bản giải mã (`text`). |
| [`requirements.txt`](./requirements.txt) | Danh sách các thư viện Python cần thiết (`torch`, `transformers`, `Pillow`, `jiwer`, `accelerate`, ...). |
| [`dataset.md`](./dataset.md) | Liên kết tham chiếu tới tập dữ liệu trang quét gốc trên Kaggle ([rukopys-scanned-dataset](https://www.kaggle.com/datasets/quii29/rukopys-scanned-dataset)). |
| [`README.md`](./README.md) | Tài liệu hướng dẫn chi tiết quy trình xử lý dữ liệu và chạy tiến trình. |

---

## Quy trình Xử lý Dữ liệu & Cắt Ảnh Dòng (Line Cropping)

### Đặt vấn đề & Yêu cầu đầu vào

Tập dữ liệu trang quét gốc lưu giữ thông tin ở cấp độ nguyên trang ảnh kèm tệp `metadata.jsonl` biểu diễn các vùng chữ dưới dạng danh sách `regions`:

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

Mô hình TrOCR yêu cầu đầu vào là từng ảnh dòng chữ riêng biệt (Line crops) kèm thông tin nhãn văn bản. Kịch bản [`crop_dataset.py`](./crop_dataset.py) được xây dựng để thực hiện tự động công đoạn này.

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

Sau khi chạy xong `crop_dataset.py`, thư mục dữ liệu đầu ra sẽ được tạo theo đúng quy chuẩn yêu cầu của script huấn luyện:

```
dataset/
└── train/
    ├── metadata.jsonl
    └── images/
        ├── 964a31fc-0bee-5096-9359-2371641869eb_crop_0_printed.jpg
        ├── 964a31fc-0bee-5096-9359-2371641869eb_crop_1_handwritten.jpg
        └── ...
```

Mỗi dòng trong `dataset/train/metadata.jsonl` mới có định dạng tinh gọn:
```json
{
  "file_name": "images/964a31fc-0bee-5096-9359-2371641869eb_crop_1_handwritten.jpg",
  "image": "images/964a31fc-0bee-5096-9359-2371641869eb_crop_1_handwritten.jpg",
  "label": "handwritten",
  "type": "handwritten",
  "text": "Електростанції:"
}
```

---

## Hướng dẫn Chạy Tiến Trình Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---

### Bước 2: Tạo File Môi Trường (`.env.hpa`)

Tạo một file `.env.hpa` để thiết lập các đường dẫn và tham số huấn luyện:

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

---

### Bước 3: Kích hoạt Tiến trình Huấn Luyện

Bạn có thể lựa chọn 1 trong 3 cách kích hoạt sau:

#### Cách 1: Chạy qua Bash Script với file `.env.hpa` (Khuyến nghị)

Bash script `run_train_kansallisarkisto_hpa_single_train_cloud.sh` sẽ tự động thiết lập virtualenv và kiểm tra cấu trúc dữ liệu trước khi chạy:

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
