# TrOCR Fine-Tuning with Augmented Dataset

Thư mục này chứa toàn bộ mã nguồn và kịch bản huấn luyện tinh chỉnh (Fine-tuning) mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) cho bài toán nhận diện chữ viết tay tiếng Ukraine trên tập dữ liệu đã qua tăng cường dữ liệu (Data Augmentation): **[kuropys-augmented-cropped-train](https://www.kaggle.com/datasets/habao2603/kuropys-augmented-cropped-train)**.

---

## Cấu trúc thư mục

| File / Thư mục | Mô tả |
| --- | --- |
| [`train_kansallisarkisto_hpa_with_augmented.py`](./train_kansallisarkisto_hpa_with_augmented.py) | Python script chính thực hiện quá trình tinh chỉnh TrOCR đơn tầng (Single-stage), hỗ trợ Custom Collator, bổ sung ký tự tiếng Ukraine, đánh giá phân loại (Typed Eval) và lưu checkpoint tự động. |
| [`run_train_kansallisarkisto_hpa_cloud.sh`](./run_train_kansallisarkisto_hpa_cloud.sh) | Script Bash giúp nạp file môi trường `.env` và tự động khởi chạy tiến trình huấn luyện trên máy chủ Cloud/GPU. |
| [`requirements.txt`](./requirements.txt) | Danh sách thư viện Python phụ thuộc cần thiết (`transformers`, `torch`, `jiwer`, `accelerate`, ...). |
| [`dataset.md`](./dataset.md) | Liên kết tham chiếu tới tập dữ liệu Augment trên Kaggle ([kuropys-augmented-cropped-train](https://www.kaggle.com/datasets/habao2603/kuropys-augmented-cropped-train)). |
| [`README.md`](./README.md) | Tài liệu hướng dẫn sử dụng và thông tin chi tiết về thư mục. |

---

## Đặc điểm & Kiến trúc Huấn luyện

1. **Bổ sung Từ vựng (Vocabulary Expansion)**:
   - Tự động bổ sung các ký tự chữ cái đặc thù tiếng Ukraine (`і`, `ї`, `є`, `ґ`, `І`, `Ї`, `Є`, `Ґ`) vào `Tokenizer` và tự động điều chỉnh lại kích thước `Token Embeddings` của mô hình Decoder.
2. **Quản lý Đánh giá Chuyên biệt (Typed Decoding)**:
   - Mô hình huấn luyện đồng thời trên 3 loại vùng ảnh crop: `handwritten` (chữ viết tay), `printed` (chữ in), và `annotation` (ghi chú).
   - Thiết lập thông số decode riêng theo loại vùng:
     - `handwritten`: `num_beams=3`, `max_new_tokens=192`.
     - `printed`: `num_beams=3`, `max_new_tokens=128`.
     - `annotation`: `num_beams=1`, `max_new_tokens=16`.
3. **Tối ưu hóa GPU & Callback Tự động**:
   - Sử dụng `bfloat16` / `fp16` (Automatic Mixed Precision) kết hợp `Gradient Checkpointing`.
   - `TypedEvalCheckpointCallback`: Đánh giá nhanh (greedy `num_beams=1`) ở cuối mỗi Epoch trên tập Validation và lưu mô hình có chỉ số **CER (Character Error Rate)** tốt nhất.
   - **Early Stopping**: Tự động dừng huấn luyện nếu CER không cải thiện sau 3 Epoch liên tiếp.

---

## Hướng dẫn Chạy Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---

### Bước 2: Chuẩn bị dữ liệu và File Môi Trường (`.env`)

Cấu trúc thư mục dữ liệu yêu cầu:
```
DATA_ROOT/
├── labels.jsonl
└── [các file/thư mục chứa ảnh crop...]
```

Tạo file cấu hình môi trường `.env.hpa` (hoặc `.env`):

```bash
cat << 'EOF' > .env.hpa
# Token HuggingFace (nếu cần tải hoặc push mô hình)
HF_TOKEN=hf_your_huggingface_token_here

# Đường dẫn tới dữ liệu và thư mục đầu ra
HPA_DATA_ROOT=/home/ubuntu/dataset
HPA_OUTPUT_DIR=/home/ubuntu/second-try/outputs

# Đường dẫn Python (nếu dùng venv/conda)
PYTHON_BIN=python3
EOF
```

---

### Bước 3: Khởi chạy huấn luyện

Bạn có thể chạy huấn luyện theo các cách sau:

#### Cách 1: Chạy qua Script Bash với file `.env` (Khuyến nghị)

```bash
# Gán quyền thực thi cho script bash
chmod +x run_train_kansallisarkisto_hpa_cloud.sh

# Kích hoạt huấn luyện với file .env.hpa
ENV_FILE=.env.hpa ./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 2: Truyền trực tiếp biến môi trường khi chạy Bash Script

```bash
HPA_DATA_ROOT=/home/ubuntu/dataset \
HPA_OUTPUT_DIR=/home/ubuntu/second-try/outputs \
./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 3: Chạy trực tiếp bằng Python Script

```bash
python train_kansallisarkisto_hpa_with_augmented.py
```

---

## Các Biến Môi Trường Hỗ Trợ

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `/home/ubuntu/dataset` | Đường dẫn thư mục gốc chứa `labels.jsonl` và các tệp ảnh. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `/home/ubuntu/second-try/outputs` | Thư mục lưu kết quả đánh giá, checkpoint và mô hình thành phẩm. |
| `HPA_LABELS_JSONL` | `$DATA_ROOT/labels.jsonl` | Đường dẫn chi tiết tới file nhãn JSONL (nếu không nằm mặc định ở `$DATA_ROOT/labels.jsonl`). |
| `HF_TOKEN` | `None` | Token HuggingFace API. |
| `PYTHON_BIN` | `python` | Đường dẫn tới trình thông dịch Python. |
