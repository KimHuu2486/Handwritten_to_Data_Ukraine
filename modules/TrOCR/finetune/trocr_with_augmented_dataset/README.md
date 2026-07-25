# TrOCR Fine-Tuning with Augmented Dataset

Pipeline này thực hiện huấn luyện tinh chỉnh đơn tầng (Single-stage Fine-tuning) mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) trên tập dữ liệu đã qua tăng cường dữ liệu: **[kuropys-augmented-cropped-train](https://www.kaggle.com/datasets/habao2603/kuropys-augmented-cropped-train)**.

## Luồng chính

```text
Augmented Region Crops (kuropys-augmented-cropped-train / labels.jsonl)
                              ↓
              Tokenizer Expansion (Ukraine tokens)
                              ↓
          Single-Stage Fine-tuning (Seq2SeqTrainer)
                              ↓
Typed Eval Callback (Fast greedy evaluation on validation split every epoch)
                              ↓
           Final Model & Full Evaluation Metrics
```

## Thứ tự đọc

1. [`train_kansallisarkisto_hpa_with_augmented.py`](./train_kansallisarkisto_hpa_with_augmented.py)
2. [`run_train_kansallisarkisto_hpa_cloud.sh`](./run_train_kansallisarkisto_hpa_cloud.sh)
3. [`requirements.txt`](./requirements.txt)
4. [`dataset.md`](./dataset.md)

## Vai trò từng file

### `train_kansallisarkisto_hpa_with_augmented.py`

Python script chính thực hiện quá trình tinh chỉnh đơn tầng. Script đọc dữ liệu tăng cường từ tệp `labels.jsonl`, phân tách stratified split 90/10, tự động mở rộng vocab tokenizer cho tiếng Ukraine và sử dụng Custom Collator (`OcrDataCollator`) cùng `TypedEvalCheckpointCallback` để theo dõi chỉ số CER theo từng epoch.

Đầu ra chính:
```text
OUTPUT_DIR/
├── epoch_evaluations/                # Dự đoán predictions.jsonl & metrics.json mỗi epoch
├── best_model/                       # Checkpoint có CER validation thấp nhất
├── final_cyrillic_htr_model/         # Mô hình thành phẩm xuất cuối cùng
├── final_full_val_predictions.jsonl  # Dự đoán full beam search trên tập validation
└── final_full_val_metrics.json       # Báo cáo metrics chi tiết theo từng nhãn vùng
```

### `run_train_kansallisarkisto_hpa_cloud.sh`

Script Bash dùng để nạp file cấu hình môi trường `.env.hpa`, kiểm tra token xác thực HuggingFace và tự động khởi chạy script Python trên máy chủ GPU.

### `requirements.txt`

Khai báo các phụ thuộc thư viện Python (`transformers`, `torch`, `jiwer`, `accelerate`, ...).

### `dataset.md`

Tài liệu tham chiếu liên kết nguồn dữ liệu tăng cường trên Kaggle.

---

## Đặc điểm & Kiến trúc Huấn luyện

1. **Mở rộng Từ vựng (Vocabulary Expansion)**:
   - Tự động bổ sung các ký tự chữ cái đặc thù tiếng Ukraine (`і`, `ї`, `є`, `ґ`, `І`, `Ї`, `Є`, `Ґ`) vào Tokenizer và căn chỉnh lại `Token Embeddings` của Decoder.
2. **Custom Collator & Mixed Precision**:
   - Sử dụng `OcrDataCollator` thực hiện đệm động (Dynamic Padding) với giá trị pad token ID là `-100` để loại bỏ khỏi hàm loss.
   - Tự động kích hoạt `bfloat16` / `fp16` dựa trên nhân lõi GPU (A6000 / T4) kết hợp `Gradient Checkpointing`.
3. **Custom Callback & Early Stopping**:
   - `TypedEvalCheckpointCallback`: Thực hiện đánh giá nhanh (`num_beams=1`) ở cuối mỗi Epoch trên tập Validation và lưu mô hình tối ưu theo chỉ số **CER**.
   - Dừng sớm (Early Stopping) nếu CER không cải thiện sau 3 Epoch liên tiếp.

---

## Cấu hình Generation Cho Từng Loại Văn Bản (Typed Decoding)

Để tối ưu thời gian suy luận và độ chính xác cho từng dạng vùng chọn, mô hình áp dụng chiến lược sinh chuỗi (decoding) riêng biệt:

- **`handwritten`**: `num_beams=3`, `max_new_tokens=192`.
- **`printed`**: `num_beams=3`, `max_new_tokens=128`.
- **`annotation`**: `num_beams=1` (greedy search), `max_new_tokens=16`.

---

## Hướng dẫn Chạy Huấn Luyện (Run Guide)

### Bước 1: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu và File Môi Trường (`.env`)

Cấu trúc thư mục dữ liệu yêu cầu:
```text
DATA_ROOT/
├── labels.jsonl
└── [các file/thư mục chứa ảnh crop...]
```

Tạo file cấu hình môi trường `.env.hpa`:

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

## Biến môi trường hỗ trợ

| Biến Môi Trường | Mặc định | Mô tả |
| --- | --- | --- |
| `HPA_DATA_ROOT` / `DATA_ROOT` | `/home/ubuntu/dataset` | Đường dẫn thư mục gốc chứa `labels.jsonl` và các tệp ảnh crop. |
| `HPA_OUTPUT_DIR` / `OUTPUT_DIR` | `/home/ubuntu/second-try/outputs` | Thư mục lưu kết quả đánh giá, checkpoint và mô hình thành phẩm. |
| `HPA_LABELS_JSONL` | `$DATA_ROOT/labels.jsonl` | Đường dẫn chi tiết tới file nhãn JSONL (nếu không nằm mặc định ở `$DATA_ROOT/labels.jsonl`). |
| `HF_TOKEN` | `None` | Token HuggingFace API. |
| `PYTHON_BIN` | `python` | Đường dẫn tới trình thông dịch Python. |
