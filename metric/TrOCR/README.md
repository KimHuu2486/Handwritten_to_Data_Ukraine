# TrOCR 3-Phase Fine-Tuning Pipeline for Ukraine OCR

Thư mục này chứa toàn bộ mã nguồn và kịch bản huấn luyện tinh chỉnh (Fine-tuning) mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) qua **3 giai đoạn (3-Phase Strategy)** cho bài toán nhận diện chữ viết tay tiếng Ukraine, dựa trên tập dữ liệu đã phân tách theo tỷ lệ **70/15/15** trong thư mục [`data/`](../data/).

---

## Cấu trúc thư mục

| File / Thư mục | Mô tả |
| --- | --- |
| [`train_kansallisarkisto_hpa_3phase.py`](./train_kansallisarkisto_hpa_3phase.py) | Python script chính điều khiển quy trình huấn luyện 3 phase, tự động bổ sung ký tự tiếng Ukraine, đánh giá phân loại (Typed Eval) và lưu checkpoint tối ưu theo chỉ số CER. |
| [`run_train_kansallisarkisto_hpa_cloud.sh`](./run_train_kansallisarkisto_hpa_cloud.sh) | Script Bash giúp nạp biến môi trường từ file `.env` và kích hoạt quá trình huấn luyện trên máy chủ/Cloud. |
| [`requirements.txt`](./requirements.txt) | Danh sách các thư viện Python cần thiết (`torch`, `transformers`, `jiwer`, `accelerate`, ...). |
| [`README.md`](./README.md) | Tài liệu hướng dẫn chi tiết về cấu trúc và kịch bản huấn luyện. |

---

## Mối liên hệ với Tập dữ liệu (Split 70/15/15)

Dữ liệu OCR được phân tách ở cấp độ trang quét (Page-level) trong thư mục [`data/`](../data/) với seed `1089` nhằm đảm bảo tính cân bằng giữa các nguồn (`archive`, `dictation`, `school`, `university`):

- **`train.jsonl` (70% - 931 trang)**: Sử dụng để huấn luyện mô hình (Phase 2 & Phase 3).
- **`val.jsonl` (15% - 199 trang)**: Sử dụng để đánh giá kiểm định (Validation), chọn checkpoint tối ưu và tune hyperparameter.
- **`test.jsonl` (15% - 200 trang)**: Giữ độc lập hoàn toàn để đánh giá và báo cáo chỉ số cuối cùng (CER/WER) trong bài báo khoa học.

### Quy chuẩn Dữ liệu Đầu vào cho TrOCR

Trước khi đưa vào mô hình TrOCR, các trang ảnh quét trong tệp `jsonl` cấp trang được trích xuất (crop) theo bounding box (`bbox`) của từng vùng dòng chữ (`handwritten`, `printed`, `annotation`) để tạo thành cấu trúc dữ liệu dòng:

```
DATA_ROOT/
├── train/
│   ├── metadata.jsonl
│   └── images/
├── val/
│   ├── metadata.jsonl
│   └── images/
└── silver/ (tùy chọn)
    ├── metadata.jsonl
    └── images/
```

---

## Kiến trúc Huấn luyện 3 Phase (3-Phase Strategy)

Mô hình được huấn luyện đồng thời trên 3 loại vùng dòng chữ: `handwritten` (chữ viết tay), `printed` (chữ in), và `annotation` (ghi chú).

1. **Phase 1: Silver Warm-up**
   - **Tập dữ liệu**: Tập Silver (dữ liệu giả lập / gán nhãn tự động).
   - **Cấu hình**: `1.0 epoch`, Learning Rate = `3e-5`.
   - **Mục tiêu**: Làm quen với đặc trưng dòng ảnh crop và bảng ký tự tiếng Ukraine.
2. **Phase 2: Gold Fine-tune**
   - **Tập dữ liệu**: Tập Train (trích xuất từ `train.jsonl`).
   - **Cấu hình**: `6.0 epochs`, Learning Rate = `1e-5`.
   - **Mục tiêu**: Tinh chỉnh sâu trên nhãn chất lượng cao từ tập Train chuẩn.
3. **Phase 3: Gold Recovery**
   - **Tập dữ liệu**: Tiếp tục tinh chỉnh trên tập Train.
   - **Cấu hình**: `2.0 epochs`, Learning Rate = `5e-6`.
   - **Mục tiêu**: Tối ưu hóa các mẫu khó (hard samples) với tốc độ học nhỏ hơn.

---

## Cấu hình Generation cho từng loại vùng (Typed Decoding)

Để tối ưu thời gian suy luận và độ chính xác cho từng dạng vùng chọn, mô hình áp dụng chiến lược decode riêng biệt:

- **`handwritten`**: `num_beams=3`, `max_new_tokens=192`.
- **`printed`**: `num_beams=3`, `max_new_tokens=128`.
- **`annotation`**: `num_beams=1` (greedy search), `max_new_tokens=16`.

---

## Hướng dẫn Chạy Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---

### Bước 2: Chuẩn bị File Môi Trường (`.env.hpa`)

Tạo file `.env.hpa` để khai báo đường dẫn dữ liệu và cấu hình chạy:

```bash
cat << 'EOF' > .env.hpa
# Token HuggingFace (nếu tải hoặc push mô hình private)
HF_TOKEN=hf_your_huggingface_token_here

# Đường dẫn tới thư mục gốc dữ liệu và nơi lưu outputs
HPA_DATA_ROOT=/home/ubuntu/dataset
HPA_OUTPUT_DIR=/home/ubuntu/second-try/outputs

# Phase bắt đầu chạy (1: Silver Warmup, 2: Gold Finetune, 3: Gold Recovery)
START_PHASE=1

# Trình thông dịch Python
PYTHON_BIN=python3
EOF
```

---

### Bước 3: Khởi chạy huấn luyện

Bạn có thể chạy huấn luyện theo các cách sau:

#### Cách 1: Chạy qua Script Bash với file `.env.hpa` (Khuyến nghị)

```bash
# Gán quyền thực thi cho script bash
chmod +x run_train_kansallisarkisto_hpa_cloud.sh

# Kích hoạt huấn luyện với file .env.hpa
ENV_FILE=.env.hpa ./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 2: Truyền trực tiếp biến môi trường trên dòng lệnh

```bash
HPA_DATA_ROOT=/home/ubuntu/dataset \
HPA_OUTPUT_DIR=/home/ubuntu/second-try/outputs \
START_PHASE=1 \
./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 3: Chạy trực tiếp Python Script

```bash
python train_kansallisarkisto_hpa_3phase.py
```

---

## Các Biến Môi Trường Hỗ Trợ

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `../../dataset` | Đường dẫn gốc chứa thư mục `train/`, `val/`, và `silver/`. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `./outputs_kansallisarkisto_hpa` | Thư mục lưu kết quả đánh giá và các checkpoint của từng phase. |
| `HPA_START_PHASE` / `START_PHASE` | `1` | Phase bắt đầu chạy (`1`, `2`, hoặc `3`). |
| `HPA_RESUME_MODEL_DIR` | `None` | Đường dẫn đến checkpoint cũ để tiếp tục huấn luyện (Resume). |
| `HF_TOKEN` | `None` | Token xác thực HuggingFace. |
| `PYTHON_BIN` | `python` | Đường dẫn tới trình thông dịch Python. |
