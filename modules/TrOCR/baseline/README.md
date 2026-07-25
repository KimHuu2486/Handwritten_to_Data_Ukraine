# TrOCR Baseline

Pipeline này thực hiện suy luận zero-shot bằng mô hình **TrOCR** ([`Kansallisarkisto/cyrillic-htr-model`](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model)) để nhận diện văn bản (OCR) trên tập dữ liệu chữ viết tay tiếng Ukraine (**RUKOPYS dataset**). Model nhận diện nội dung từ các crop vùng văn bản được cắt dựa trên bounding box (`bbox`).

## Luồng chính

```text
RUKOPYS dataset (test.jsonl + images)
              ↓
Cắt crop theo bbox (handwritten, printed, annotation)
              ↓
Processor (microsoft/trocr-base-handwritten)
              ↓
TrOCR Model (Kansallisarkisto/cyrillic-htr-model)
              ↓
Tính độ đo CER / WER (jiwer) & Xuất submission.csv / report.json
```

## Thứ tự đọc

1. [`kaggle_zero_shot_baseline.ipynb`](./kaggle_zero_shot_baseline.ipynb)
2. [`dataset.md`](./dataset.md)

## Vai trò từng file

### `kaggle_zero_shot_baseline.ipynb`

Notebook chạy trên Kaggle GPU nạp mô hình TrOCR pre-trained, tự động mở rộng tokenizer với các ký tự tiếng Ukraine đặc trưng, cắt các region crop theo bbox và chạy suy luận (inference) không qua huấn luyện lại (zero-shot).

Đầu ra chính:
```text
/kaggle/working/
├── report.json        # Báo cáo CER, WER, Exact Match chi tiết theo từng nhãn vùng
└── submission.csv     # File kết quả dự đoán định dạng nộp bài
```

### `dataset.md`

Tài liệu ghi chú liên kết tham chiếu tới tập dữ liệu RUKOPYS gốc trên Kaggle.

---

## Chi tiết Pipeline suy luận

### 1. Cấu hình & Mở rộng Vocab Tokenizer
Nạp mô hình `VisionEncoderDecoderModel` và `TrOCRProcessor`. Mở rộng `tokenizer` bổ sung các ký tự chữ cái đặc thù tiếng Ukraine:
```python
ukrainian_tokens = ['Ґ', 'ґ', 'Є', 'є', 'І', 'і', 'Ї', 'ї', '’']
```

### 2. Crop ảnh theo Bounding Box
Đọc siêu dữ liệu từ `test.jsonl`. Với mỗi trang ảnh:
- Lọc các vùng chọn (`regions`) thuộc danh sách nhãn HPA: `handwritten`, `printed`, `annotation`.
- Trích xuất ảnh crop theo tọa độ `bbox`: `[xmin, ymin, xmax, ymax]`.

### 3. Suy luận Zero-Shot (Inference)
- Chuẩn hóa ảnh crop qua `processor` thành `pixel_values`.
- Chạy `model.generate(pixel_values, max_length=128)` để nhận diện chuỗi văn bản.

### 4. Đánh giá Chỉ số & Xuất kết quả
- Sử dụng `jiwer` tính độ đo **CER** (Character Error Rate), **WER** (Word Error Rate), và **Exact Match (EM)**.
- Ghi báo cáo chỉ số ra `report.json` và file nộp bài `submission.csv`.

---

## Hướng dẫn chạy notebook

1. **Môi trường**: Khởi chạy trên **Kaggle Notebook** (kích hoạt GPU T4 hoặc P100).
2. **Mount Dataset**: Đảm bảo dataset RUKOPYS đã được thêm vào Kaggle input (đường dẫn mặc định `/kaggle/input/rukopys-dataset`).
3. **Thực thi**:
   - Mở [`kaggle_zero_shot_baseline.ipynb`](./kaggle_zero_shot_baseline.ipynb).
   - Chọn **Run All** để thực thi toàn bộ pipeline từ nạp model, crop ảnh, suy luận đến xuất báo cáo.
4. **Kiểm tra kết quả**: Tải hai tệp thành phẩm `/kaggle/working/report.json` và `/kaggle/working/submission.csv` từ thư mục làm việc của Kaggle.

---

## Khác biệt với Fine-tuning

| Đặc điểm | Zero-Shot Baseline | Fine-Tuned TrOCR |
|---|---|---|
| Trọng số mô hình | Trọng số gốc `Kansallisarkisto` | Trọng số đã tinh chỉnh trên RUKOPYS |
| Huấn luyện lại | Không (Inference direct) | Có (Single-stage / 3-Phase Strategy) |
| Tokenizer | Bổ sung token tiếng Ukraine runtime | Bổ sung token + Re-embed Decoder |
| Điểm mạnh | Nhanh, không tốn tài nguyên train | Độ chính xác CER/WER cao hơn rõ rệt |

> [!NOTE]
> Notebook baseline này sử dụng bounding box có sẵn trong `test.jsonl` để đánh giá năng lực OCR đơn thuần của TrOCR zero-shot.
