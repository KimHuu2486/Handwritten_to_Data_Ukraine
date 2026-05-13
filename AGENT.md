# 📝 Báo cáo ngữ cảnh dự án (Project Context)

## 1. 🎯 Tóm tắt dự án (Project Overview)

- **Mục tiêu:** Xây dựng pipeline nhận dạng chữ viết tay (Handwritten Text Recognition - HTR) và trích xuất thông tin tài liệu cho cuộc thi "Handwritten to Data" (Kaggle). Dự án bao gồm việc phân loại vùng dữ liệu (handwritten, printed, formula...), nhận diện chữ bên trong và tối ưu các chỉ số theo luật của giải đấu (Detection F1, Classification Accuracy, Region CER, Page CER).
- **Trạng thái hiện tại:** Đang phát triển. Đã hoàn thiện baseline, tối ưu hóa pipeline huấn luyện VLM (Qwen) trên multi-GPU, và đang ghép nối hệ thống Inference (YOLO + VLM) để tạo file nộp (submission).

## 2. 🛠️ Công nghệ đã sử dụng (Tech Stack)

- **Ngôn ngữ chủ đạo:** Python.
- **Mô hình / AI:** Vision Language Models (Qwen3-VL-8B), LoRA (fine-tune), mô hình phát hiện vật thể (YOLOv10 cho region extraction).
- **Xử lý dữ liệu:** Pandas, JSON, CSV, Pillow.
- **Metric Đánh giá:** Xử lý Levenshtein distance, IoU Matching và Text Normalization (Regex, LaTeX parsing).

## 3. ✅ Công việc đã hoàn thành (Completed Tasks)

- [x] Phân tích kỹ rule chấm điểm của BTC (chi tiết về IoU >= 0.5, xử lý text normalization, không cần tự sort bbox).
- [x] Phát triển các baseline dự đoán (Zero-shot baseline) và pipeline tiền xử lý dataset (crop ảnh, sinh file JSONL metadata).
- [x] Viết kịch bản huấn luyện (Fine-tune VLM qua LoRA) và đánh giá các phiên bản (v2 với LoRA rank tăng cường).
- [x] Xây dựng script parse kết quả JSONL sang `submission_format.csv` và chuyển đổi tọa độ Bounding Box về [x1, y1, x2, y2].
- [x] Tối ưu hóa pipeline Inference kết hợp YOLO (trích xuất vùng) và Qwen-VL (nhận diện).
- [x] Ổn định kịch bản huấn luyện trên môi trường đa GPU Kaggle (T4), xử lý OOM và cấu hình phân bổ bộ nhớ (`PYTORCH_ALLOC_CONF`, `max_memory`).

## 4. 🐛 Những lỗi đã khắc phục (Fixed Bugs)

- **Bug 1: Định dạng và tọa độ Bounding Box** -> **Cách giải quyết:** Các model trả về theo [x1, y1, w, h], đã được quy chuẩn về số nguyên [x1, y1, x2, y2] sử dụng round() chuẩn của Python (quy tắc "Banker's rounding").
- **Bug 2: Lo lắng về việc sắp xếp Box khi submit** -> **Cách giải quyết:** Đã check kỹ source code chấm điểm của Kaggle và xác nhận thư viện chấm điểm sẽ tự động sort bbox (y1 trước, x1 sau), do đó ta chỉ cần focus vào output list box chuẩn xác.
- **Bug 3: CUDA OOM do phân mảnh VRAM khi train multi-GPU** -> **Cách giải quyết:** Áp dụng `PYTORCH_ALLOC_CONF` và giới hạn `max_memory` để cân bằng tải giữa 2 GPU (T4).
- **Bug 4: Crash `NotImplementedError` với Qwen-VL trên T4 GPU** -> **Cách giải quyết:** Ép sử dụng kiểu dữ liệu `Float16` hoặc `Float32` trên toàn bộ model và pipeline thay vì `BFloat16` (do T4 không hỗ trợ native `BFloat16`).
- **Bug 5: Xung đột thư viện Pillow trên Kaggle & lỗi crop ảnh** -> **Cách giải quyết:** Fix phiên bản Pillow cố định và tối ưu hóa logic RAM-based cropping + dynamic BBox scaling cho inference.

## 5. 🚀 Công việc tiếp theo (Current / Pending Tasks)

- [ ] Hoàn thiện và chạy pipeline Inference toàn diện (End-to-End) để xuất file `submission.csv` cuối cùng trên tập test.
- [ ] Đánh giá chi tiết hiệu suất fine-tune (v2) và tối ưu hóa thêm LoRA rank/hyperparameters để cải thiện accuracy.
- [ ] Chạy thử bộ chấm điểm local score() với kết quả từ pipeline để xem chỉ số thành phần nào bị thấp (Det-F1, CER...).
- [ ] Tích hợp Spell Checking cải thiện chữ tiếng Ukraina (finetune/spell_check_submission.py).

## 6. 📂 Cấu trúc thư mục cốt lõi (Core Structure)

- `baseline/` : Các đoạn script chạy mẫu zero-shot không cần (hoặc ít) qua fine-tuning.
- `dataset/` : Dữ liệu nguồn, metadata gốc của cuộc thi.
- `finetune/` : Các mã nguồn để huấn luyện VLM (train_qwen_lora, prepare dataset).
- `VLM_OCR/` : Pipeline chia tách và nhận diện, code cho model VLM Inference và các file chuyển đổi định dạng (VD: get CSV từ JSONL).
- `official-evaluation-metric....ipynb` : Notebook mô tả luật và hàm đánh giá chuẩn xác của Kaggle.
