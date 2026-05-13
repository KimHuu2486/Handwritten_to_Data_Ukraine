# 📝 Báo cáo ngữ cảnh dự án (Project Context)

## 1. 🎯 Tóm tắt dự án (Project Overview)

- **Mục tiêu:** Xây dựng pipeline nhận dạng chữ viết tay (Handwritten Text Recognition - HTR) và trích xuất thông tin tài liệu cho cuộc thi "Handwritten to Data" (Kaggle). Dự án bao gồm việc phân loại vùng dữ liệu (handwritten, printed, formula...), nhận diện chữ bên trong và tối ưu các chỉ số theo luật của giải đấu (Detection F1, Classification Accuracy, Region CER, Page CER).
- **Trạng thái hiện tại:** Đang phát triển. Đã có các baseline, đang xây dựng pipeline huấn luyện VLM (Vision Language Model)/OCR và tích hợp xuất file nộp (submission).

## 2. 🛠️ Công nghệ đã sử dụng (Tech Stack)

- **Ngôn ngữ chủ đạo:** Python.
- **Mô hình / AI:** Vision Language Models (Qwen), LoRA (để fine-tune), mô hình phát hiện vật thể (YOLOv10 cho detection).
- **Xử lý dữ liệu:** Pandas, JSON, CSV.
- **Metric Đánh giá:** Xử lý Levenshtein distance, IoU Matching và Text Normalization (Regex, LaTeX parsing).

## 3. ✅ Công việc đã hoàn thành (Completed Tasks)

- [x] Phân tích kỹ rule chấm điểm của BTC (chi tiết về IoU >= 0.5, xử lý text normalization, không cần tự sort bbox).
- [x] Phát triển các baseline dự đoán (Zero-shot baseline).
- [x] Tạo pipeline pipeline tiền xử lý dataset (crop ảnh, sinh file JSONL metadata).
- [x] Viết kịch bản huấn luyện (Fine-tune VLM qua LoRA).
- [x] Xây dựng script VLM_OCR/convert_jsonl_to_csv.py để parse kết quả từ dạng .jsonl sang submission_format.csv, chuyển đổi tọa độ Bounding Box từ dạng [x1, y1, w, h] sang [x1, y1, x2, y2].

## 4. 🐛 Những lỗi đã khắc phục (Fixed Bugs)

- **Bug 1: Định dạng và tọa độ Bounding Box** -> **Cách giải quyết:** Các model trả về theo [x1, y1, w, h], đã được quy chuẩn về số nguyên [x1, y1, x2, y2] sử dụng
  ound() chuẩn của Python (quy tắc "Banker's rounding").
- **Bug 2: Lo lắng về việc sắp xếp Box khi submit** -> **Cách giải quyết:** Đã check kỹ source code chấm điểm của Kaggle và xác nhận thư viện chấm điểm sẽ tự động sort bbox (y1 trước, x1 sau), do đó ta chỉ cần focus vào output list box chuẩn xác.

## 5. 🚀 Công việc tiếp theo (Current / Pending Tasks)

- [ ] Ghép toàn bộ pipe Inference (từ YOLO cắt ảnh, OCR lấy Text đưa vào file jsonl, rồi chạy tool Convert ra CSV cuối cùng).
- [ ] Chạy thử bộ chấm điểm local score() với kết quả từ pipeline để xem chỉ số thành phần nào bị thấp (Det-F1, CER...).
- [ ] Tích hợp Spell Checking cải thiện chữ tiếng Ukraina (inetune/spell_check_submission.py).

## 6. 📂 Cấu trúc thư mục cốt lõi (Core Structure)

- baseline/ : Các đoạn script chạy mẫu zero-shot không cần (hoặc ít) qua fine-tuning.
- dataset/ : Dữ liệu nguồn, metadata gốc của cuộc thi.
- finetune/ : Các mã nguồn để huấn luyện VLM (train_qwen_lora, prepare dataset).
- VLM_OCR/ : Pipeline chia tách và nhận diện, code cho model VLM Inference và các file chuyển đổi định dạng (VD: get CSV từ JSONL).
- official-evaluation-metric....ipynb : Notebook mô tả luật và hàm đánh giá chuẩn xác của Kaggle.
