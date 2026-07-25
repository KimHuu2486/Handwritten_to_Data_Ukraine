# TrOCR Zero-Shot Baseline

Thư mục này chứa mã nguồn baseline zero-shot sử dụng kiến trúc **TrOCR** (Transformer-based Optical Character Recognition) để nhận diện văn bản (OCR) trên tập dữ liệu chữ viết tay tiếng Ukraine (**RUKOPYS dataset**).

## Cấu trúc thư mục

| File / Thư mục | Mô tả |
| --- | --- |
| [`kaggle_zero_shot_baseline.ipynb`](./kaggle_zero_shot_baseline.ipynb) | Notebook chạy trên Kaggle thực hiện pipeline inference zero-shot, tính toán độ đo và xuất kết quả. |
| [`dataset.md`](./dataset.md) | Chứa liên kết tham chiếu tới tập dữ liệu Kaggle ([RUKOPYS Dataset](https://www.kaggle.com/datasets/quii29/rukopys-dataset)). |

---

## Mô hình & Processor

- **Model Weight**: [`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model) (được tinh chỉnh cho chữ viết tay Cyrillic từ Kansallisarkisto).
- **Processor**: [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) (bộ xử lý ảnh và mã hóa văn bản gốc của Microsoft TrOCR).
- **Mở rộng Vocab**: Do tiếng Ukraine có một số ký tự đặc trưng không thuộc bộ ký tự Cyrillic tiêu chuẩn, tokenizer được bổ sung các token:
  ```python
  ukrainian_tokens = ['Ґ', 'ґ', 'Є', 'є', 'І', 'і', 'Ї', 'ї', '’']
  ```

---

## Quy trình hoạt động (Pipeline Workflow)

### 1. Cấu hình & Mở rộng Vocab
Nạp mô hình `VisionEncoderDecoderModel` và `TrOCRProcessor`. Mở rộng tokenizer bằng các ký tự Ukraine đặc trưng và điều chỉnh kích thước `decoder_start_token_id` nếu cần.

### 2. Crop ảnh theo Bounding Box
Đọc siêu dữ liệu từ `test.jsonl`. Với mỗi dòng trong file:
- Duyệt qua từng vùng (`region`) có loại văn bản thuộc nhóm `HPA_TYPES` = `{"handwritten", "printed", "annotation"}`.
- Kiểm tra tọa độ `bbox` `[x1, y1, x2, y2]`, tiến hành crop vùng ảnh tương ứng từ ảnh gốc.

### 3. Suy luận (Inference)
- Chuyển ảnh đã crop qua `processor` để lấy `pixel_values`.
- Đưa qua `model.generate(pixel_values, max_length=128)` để nhận diện chuỗi ký tự.
- Lưu chuỗi kết quả dự đoán và ground-truth vào danh sách tương ứng.

### 4. Đánh giá chỉ số (Metrics Evaluation)
Sử dụng thư viện `jiwer` để tính toán các độ đo OCR cho từng nhóm (`handwritten`, `printed`, `annotation`) cũng như tổng thể (`overall`):
- **CER (Character Error Rate)**: Tỉ lệ lỗi mức ký tự.
- **WER (Word Error Rate)**: Tỉ lệ lỗi mức từ.
- **Exact Match (EM)**: Tỉ lệ khớp hoàn toàn 100%.

### 5. Xuất kết quả
- **`report.json`**: Báo cáo tổng hợp các chỉ số CER, WER, Exact Match và tổng số mẫu theo từng phân loại.
- **`submission.csv`**: File nộp bài định dạng CSV chuẩn gồm `image` và danh sách `regions` kèm văn bản nhận diện được.

---

## Môi trường & Đầu ra (Outputs)

> [!NOTE]
> Notebook được thiết kế để khởi chạy trực tiếp trên môi trường **Kaggle Notebook** có kích hoạt GPU (CUDA).

### Đường dẫn đầu ra trong Kaggle Working:
- Báo cáo kết quả: `/kaggle/working/report.json`
- File Submission: `/kaggle/working/submission.csv`
