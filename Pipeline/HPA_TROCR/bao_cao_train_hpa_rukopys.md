# Báo cáo tổng hợp các thử nghiệm train HPA OCR cho RUKOPYS

## 1. Mục tiêu

Trong các thử nghiệm này, mục tiêu là fine-tune mô hình `Kansallisarkisto/cyrillic-htr-model` cho nhánh HPA của bài toán RUKOPYS OCR. Nhánh HPA phụ trách nhận dạng các vùng văn bản thuộc 3 loại chính:

- `handwritten`: chữ viết tay.
- `printed`: chữ in.
- `annotation`: chú thích/ngắn nhãn.

Mô hình được train theo hướng dùng chung một OCR model cho cả 3 loại vùng HPA, nhưng khi validation/inference thì dùng cấu hình sinh chuỗi khác nhau theo từng loại vùng. Cách này giúp vùng `annotation` không bị sinh chuỗi quá dài giống `handwritten`, đồng thời vẫn giữ được khả năng nhận dạng dòng chữ viết tay dài.

Các chỉ số chính được dùng để đánh giá gồm:

- **CER**: Character Error Rate, càng thấp càng tốt.
- **WER**: Word Error Rate, càng thấp càng tốt.
- **Exact Match**: tỷ lệ dự đoán khớp hoàn toàn với nhãn, càng cao càng tốt.

## 2. Train

**File chính:** `train_kansallisarkisto_hpa_load(3).py`

Đây là hướng train theo curriculum/multi-phase. Pipeline gồm 3 giai đoạn chính:

| Phase | Tên phase | Dữ liệu dùng | Epoch | Learning rate | Mục đích |
|---|---|---:|---:|---:|---|
| 1 | `silver_warmup` | Silver HPA data | 1.0 | `3e-5` | Warm-up mô hình trên tập silver lớn hơn để học đặc trưng tổng quát. |
| 2 | `gold_finetune` | Train/gold HPA data | 6.0 | `1e-5` | Fine-tune trên dữ liệu train chất lượng hơn. |
| 3 | `gold_recovery` | Train/gold HPA data | 2.0 | `5e-6` | Recovery/ổn định mô hình bằng learning rate thấp hơn. |

Các điểm kỹ thuật chính:

- Tạo hoặc tái sử dụng fixed validation split từ `dataset/train` với `VAL_RATIO = 0.10`.
- Dùng cả `DATA_ROOT/train` và `DATA_ROOT/silver`.
- Chỉ giữ các label HPA: `handwritten`, `printed`, `annotation`.
- Có kiểm tra token length, missing image, image open error và lưu report cho từng tập dữ liệu.
- Bổ sung các ký tự tiếng Ukraina vào tokenizer: `і`, `ї`, `є`, `ґ` và bản viết hoa tương ứng.
- Dùng typed generation config theo từng loại vùng:
  - `handwritten`: beam search 3, tối đa 192 token mới.
  - `printed`: beam search 3, tối đa 128 token mới.
  - `annotation`: beam 1, tối đa 16 token mới.
- Có callback đánh giá typed validation theo epoch và chọn checkpoint tốt nhất theo **typed validation CER**.
- Sau mỗi phase, model tốt nhất được lưu lại và dùng làm đầu vào cho phase tiếp theo.

Kết quả định lượng từ file `final_typed_val_metrics.json`:

| Nhóm | Số mẫu | CER ↓ | WER ↓ | Exact Match ↑ |
|---|---:|---:|---:|---:|
| Overall | 2222 | **0.1005** | 0.2788 | 0.3992 |
| Handwritten | 2147 | **0.0989** | 0.2753 | 0.4010 |
| Printed | 28 | 0.1499 | 0.3431 | 0.3214 |
| Annotation | 47 | 0.3594 | 0.7460 | 0.3617 |

Nhận xét:

- Đây là hướng đang cho kết quả tổng thể tốt nhất trong các thử nghiệm hiện tại, đặc biệt xét theo CER overall.
- Việc warm-up bằng silver data rồi fine-tune/recovery trên gold data giúp mô hình học được đặc trưng tổng quát trước, sau đó thích nghi dần với tập train chất lượng cao hơn.
- Phase 3 với learning rate thấp đóng vai trò ổn định lại mô hình, giảm rủi ro mô hình bị lệch sau phase fine-tune chính.

