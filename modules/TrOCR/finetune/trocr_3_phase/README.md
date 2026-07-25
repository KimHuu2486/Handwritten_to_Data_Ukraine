# TrOCR 3-Phase Fine-Tuning Pipeline

Thư mục này chứa toàn bộ mã nguồn và kịch bản huấn luyện tinh chỉnh (Fine-tuning) mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) qua **3 giai đoạn (3-Phase Training)** cho bài toán nhận diện chữ viết tay tiếng Ukraine trên tập dữ liệu **Cropped RUKOPYS Dataset**.

---

## Cấu trúc thư mục

| File / Thư mục | Mô tả |
| --- | --- |
| [`train_kansallisarkisto_hpa_3_phase.py`](./train_kansallisarkisto_hpa_3_phase.py) | Python script chính thực hiện toàn bộ pipeline huấn luyện 3 phase, đánh giá phân loại (Typed Eval) và lưu checkpoint. |
| [`run_train_kansallisarkisto_hpa_cloud.sh`](./run_train_kansallisarkisto_hpa_cloud.sh) | Script Bash dùng để load biến môi trường từ file `.env` và kích hoạt quá trình huấn luyện trên máy chủ/cloud. |
| [`requirements.txt`](./requirements.txt) | Danh sách các thư viện Python cần thiết (`transformers`, `torch`, `jiwer`, `accelerate`, ...). |
| [`dataset.md`](./dataset.md) | Liên kết tham chiếu tới tập dữ liệu đã crop trên Kaggle ([Cropped RUKOPYS Dataset](https://www.kaggle.com/datasets/quii29/cropped-rukopys-dataset)). |

---

## Kiến trúc Huấn luyện 3 Phase (3-Phase Strategy)

Mô hình được huấn luyện đồng thời trên 3 loại vùng ảnh crop: `handwritten` (chữ viết tay), `printed` (chữ in), và `annotation` (ghi chú).

1. **Phase 1: Silver Warm-up**
   - **Tập dữ liệu**: Tập dữ liệu Silver (dữ liệu gắn nhãn tự động / giả lập).
   - **Cấu hình**: `1.0 epoch`, Learning Rate = `3e-5`.
   - **Mục tiêu**: Khởi động mô hình, làm quen với đặc trưng ảnh crop và định dạng văn bản tiếng Ukraine.
2. **Phase 2: Gold Fine-tune**
   - **Tập dữ liệu**: Tập dữ liệu Gold (Train split chuẩn).
   - **Cấu hình**: `6.0 epochs`, Learning Rate = `1e-5`.
   - **Mục tiêu**: Tinh chỉnh sâu trên nhãn chất lượng cao thu được từ tập train chuẩn.
3. **Phase 3: Gold Recovery**
   - **Tập dữ liệu**: Tiếp tục trên tập Gold.
   - **Cấu hình**: `2.0 epochs`, Learning Rate = `5e-6`.
   - **Mục tiêu**: Phục hồi và tối ưu hóa các trường hợp khó (hard samples) với tốc độ học nhỏ hơn.

> [!NOTE]
> Trong suốt cả 3 phase, tập **Validation Split** (10% trích ra từ tập `train`) được giữ cố định để đánh giá độ đo CER (Character Error Rate) và WER (Word Error Rate) một cách nhất quán.

---

## Hướng dẫn Chạy Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

Trước tiên, hãy cài đặt các thư viện Python được khai báo trong [`requirements.txt`](./requirements.txt):

```bash
pip install -r requirements.txt
```

---

### Bước 2: Chuẩn bị dữ liệu và File Môi Trường (`.env`)

Tạo một file `.env` (ví dụ `.env.hpa`) để khai báo đường dẫn dữ liệu và cấu hình chạy. 

Dữ liệu đầu vào cần có cấu trúc thư mục như sau:
```
DATA_ROOT/
├── train/
│   ├── metadata.jsonl
│   └── images/
└── silver/
    ├── metadata.jsonl
    └── images/
```

Tạo file `.env.hpa` bằng lệnh terminal:

```bash
cat << 'EOF' > .env.hpa
# Token HuggingFace (nếu muốn tải/push mô hình private)
HF_TOKEN=hf_your_huggingface_token_here

# Đường dẫn tới dữ liệu và nơi lưu checkpoint
HPA_DATA_ROOT=/home/ubuntu/dataset
HPA_OUTPUT_DIR=/home/ubuntu/outputs

# Phase bắt đầu (1: Silver Warmup, 2: Gold Finetune, 3: Gold Recovery)
START_PHASE=1

# Tùy chọn đường dẫn Python (nếu dùng venv hoặc conda)
PYTHON_BIN=python3
EOF
```

---

### Bước 3: Chạy Huấn Luyện

Bạn có thể chạy huấn luyện theo một trong các cách sau:

#### Cách 1: Chạy qua Script Bash với file `.env` (Khuyến nghị)

Gán quyền thực thi cho file script bash và truyền đường dẫn file `ENV_FILE`:

```bash
# Gán quyền thực thi cho script bash
chmod +x run_train_kansallisarkisto_hpa_cloud.sh

# Kích hoạt huấn luyện với file cấu hình .env.hpa
ENV_FILE=.env.hpa ./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 2: Chạy trực tiếp Script Bash với biến môi trường trên dòng lệnh (Inline)

Nếu không tạo file `.env.hpa`, bạn có thể truyền trực tiếp biến môi trường khi gọi script:

```bash
HPA_DATA_ROOT=/home/ubuntu/dataset \
HPA_OUTPUT_DIR=/home/ubuntu/outputs \
START_PHASE=1 \
./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 3: Chạy trực tiếp bằng Python Script

```bash
python train_kansallisarkisto_hpa_3_phase.py
```

---

## Danh sách Biến Môi Trường Hỗ Trợ

Bạn có thể tùy chỉnh kịch bản huấn luyện bằng cách thiết lập các biến môi trường sau:

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `../../dataset` | Đường dẫn gốc chứa thư mục `train/` và `silver/`. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `./outputs_kansallisarkisto_hpa` | Thư mục lưu kết quả và các checkpoint của từng phase. |
| `HPA_START_PHASE` / `START_PHASE` | `1` | Phase bắt đầu chạy (`1`, `2`, hoặc `3`). Hữu ích khi muốn tiếp tục huấn luyện từ Phase 2 hoặc 3. |
| `HPA_RESUME_MODEL_DIR` | `None` | Đường dẫn đến checkpoint cũ để tiếp tục huấn luyện (Resume). |
| `HF_TOKEN` | `None` | Token xác thực HuggingFace để tải trọng số hoặc push model. |
| `PYTHON_BIN` | `python` | Đường dẫn tới trình thông dịch Python cần dùng. |

---

## Cấu Hình Generation Cho Từng Loại Văn Bản (Typed Decoding)

Để tối ưu thời gian suy luận và độ chính xác cho từng dạng vùng chọn, mô hình áp dụng chiến lược sinh chuỗi (decoding) riêng biệt:

- **`handwritten`**: `num_beams=3`, `max_new_tokens=192` (cho phép sinh chuỗi dài).
- **`printed`**: `num_beams=3`, `max_new_tokens=128`.
- **`annotation`**: `num_beams=1` (greedy search), `max_new_tokens=16` (thích hợp cho các nhãn chú thích ngắn).
