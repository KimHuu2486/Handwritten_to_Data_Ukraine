# TrOCR 3-Phase Fine-Tuning

Pipeline này thực hiện huấn luyện tinh chỉnh mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) theo chiến lược **3 giai đoạn (3-Phase Training Strategy)** trên tập dữ liệu chữ viết tay tiếng Ukraine (**Cropped RUKOPYS Dataset**).

## Luồng chính

```text
Silver region crops
        ↓
Phase 1 — Silver Warm-up (1.0 epoch, LR 3e-5)
        ↓ Checkpoint Phase 1
Gold region crops (Train split)
        ↓
Phase 2 — Gold Fine-tune (6.0 epochs, LR 1e-5)
        ↓ Checkpoint Phase 2
Gold region crops (Recovery)
        ↓
Phase 3 — Gold Recovery (2.0 epochs, LR 5e-6)
        ↓
Typed Decoding Evaluation (handwritten / printed / annotation)
        ↓
Final Model Artifacts (final_cyrillic_htr_model)
```

## Thứ tự đọc

1. [`train_kansallisarkisto_hpa_3_phase.py`](./train_kansallisarkisto_hpa_3_phase.py)
2. [`run_train_kansallisarkisto_hpa_cloud.sh`](./run_train_kansallisarkisto_hpa_cloud.sh)
3. [`requirements.txt`](./requirements.txt)
4. [`dataset.md`](./dataset.md)

## Vai trò từng file

### `train_kansallisarkisto_hpa_3_phase.py`

Python script chính điều khiển toàn bộ pipeline 3 phase. Script tự động mở rộng tokenizer với các ký tự tiếng Ukraine đặc trưng, áp dụng custom evaluation callback (`TypedEvalCheckpointCallback`), lưu checkpoint tối ưu theo chỉ số CER và thực hiện Typed Generation cho kết quả cuối cùng.

Đầu ra chính:
```text
OUTPUT_DIR/
├── epoch_evaluations/                # Lưu dự đoán predictions.jsonl & metrics.json mỗi epoch
├── best_model/                       # Checkpoint có chỉ số CER tốt nhất
├── final_cyrillic_htr_model/         # Trọng số mô hình và processor thành phẩm sau 3 phase
├── final_full_val_predictions.jsonl  # Dự đoán chi tiết trên 100% mẫu validation
└── final_full_val_metrics.json       # Metrics tổng hợp và phân loại theo nhãn vùng
```

### `run_train_kansallisarkisto_hpa_cloud.sh`

Script Bash Wrapper hỗ trợ tự động nạp các biến môi trường từ tệp cấu hình `.env.hpa`, kiểm tra token HuggingFace và kích hoạt script Python chạy trên máy chủ Cloud/GPU.

### `requirements.txt`

Khai báo các phụ thuộc thư viện Python (`transformers`, `torch`, `jiwer`, `accelerate`, ...).

### `dataset.md`

Tài liệu tham chiếu liên kết nguồn dữ liệu Cropped RUKOPYS Dataset trên Kaggle.

---

## Kiến trúc Huấn luyện 3 Phase (3-Phase Strategy)

Mô hình được huấn luyện đồng thời trên 3 loại vùng ảnh crop: `handwritten` (chữ viết tay), `printed` (chữ in), và `annotation` (ghi chú).

1. **Phase 1: Silver Warm-up**
   - **Tập dữ liệu**: Tập dữ liệu Silver (gán nhãn tự động / giả lập).
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

## Cấu hình Generation Cho Từng Loại Văn Bản (Typed Decoding)

Để tối ưu thời gian suy luận và độ chính xác cho từng dạng vùng chọn, mô hình áp dụng chiến lược sinh chuỗi (decoding) riêng biệt:

- **`handwritten`**: `num_beams=3`, `max_new_tokens=192` (cho phép sinh chuỗi dài).
- **`printed`**: `num_beams=3`, `max_new_tokens=128`.
- **`annotation`**: `num_beams=1` (greedy search), `max_new_tokens=16` (thích hợp cho các nhãn chú thích ngắn).

---

## Hướng dẫn Chạy Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu và File Môi Trường (`.env`)

Dữ liệu đầu vào cần có cấu trúc thư mục như sau:
```text
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

### Bước 3: Chạy Huấn Luyện

Bạn có thể chạy huấn luyện theo một trong các cách sau:

#### Cách 1: Chạy qua Script Bash với file `.env` (Khuyến nghị)

```bash
# Gán quyền thực thi cho script bash
chmod +x run_train_kansallisarkisto_hpa_cloud.sh

# Kích hoạt huấn luyện với file cấu hình .env.hpa
ENV_FILE=.env.hpa ./run_train_kansallisarkisto_hpa_cloud.sh
```

#### Cách 2: Chạy trực tiếp Script Bash với biến môi trường trên dòng lệnh (Inline)

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

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `../../dataset` | Đường dẫn gốc chứa thư mục `train/` và `silver/`. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `./outputs_kansallisarkisto_hpa` | Thư mục lưu kết quả và các checkpoint của từng phase. |
| `HPA_START_PHASE` / `START_PHASE` | `1` | Phase bắt đầu chạy (`1`, `2`, hoặc `3`). Hữu ích khi muốn tiếp tục huấn luyện từ Phase 2 hoặc 3. |
| `HPA_RESUME_MODEL_DIR` | `None` | Đường dẫn đến checkpoint cũ để tiếp tục huấn luyện (Resume). |
| `HF_TOKEN` | `None` | Token xác thực HuggingFace để tải trọng số hoặc push model. |
| `PYTHON_BIN` | `python` | Đường dẫn tới trình thông dịch Python cần dùng. |
