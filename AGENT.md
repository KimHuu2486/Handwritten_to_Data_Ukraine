# 📝 Báo cáo ngữ cảnh dự án (Project Context)

## 1. 🎯 Tóm tắt dự án (Project Overview)

- **Mục tiêu:** Xây dựng pipeline nhận dạng chữ viết tay (Handwritten Text Recognition - HTR) và trích xuất thông tin tài liệu cho cuộc thi Kaggle "Handwritten to Data" và **chuẩn bị bài báo nghiên cứu khoa học (ACCV/CSONet 2026)**. Giải pháp tập trung vào thiết kế End-to-end với Qwen3-VL 8B và phương pháp **Uncertainty-Aware Chain-of-Thought (CoT)**.
- **Trạng thái hiện tại:** Đang phát triển mạnh mẽ. Đã hoàn thiện kiến trúc chiến lược tổng thể, thiết kế xong pipeline tạo Data CoT chuẩn sản xuất, và đang trong quá trình thu thập/loại nhiễu dữ liệu.

## 2. 🛠️ Công nghệ đã sử dụng (Tech Stack)

- **Ngôn ngữ chủ đạo:** Python.
- **Mô hình / AI:** Vision Language Models (Qwen3-VL-8B, GPT-4o cho sinh Data CoT), LoRA (fine-tune), mô hình phát hiện vật thể (YOLOv10 cho region extraction).
- **Xử lý dữ liệu:** Pandas, JSON, CSV, Pillow.
- **Metric Đánh giá:** Levenshtein distance, IoU Matching, Text Normalization.

## 3. ✅ Công việc đã hoàn thành (Completed Tasks)

- [x] Phân tích kỹ rule chấm điểm của BTC (chi tiết về IoU >= 0.5, xử lý text normalization, không cần tự sort bbox).
- [x] Phát triển các baseline dự đoán (Zero-shot baseline) và pipeline tiền xử lý dataset.
- [x] Kịch bản huấn luyện (Fine-tune VLM qua LoRA) trên multi-GPU (xử lý OOM PyTorch allocator trên T4, fallback Float16/Float32).
- [x] Tối ưu hóa pipeline Inference kết hợp YOLO (trích xuất vùng) và Qwen-VL (nhận diện).
- [x] Lập kế hoạch chiến lược kiến trúc tổng quan (`plan_VLM.md`) với pipeline Curriculum Learning (Silver -> Gold -> CoT -> Post-processing).
- [x] Xây dựng chi tiết hệ thống quy trình tạo dữ liệu CoT chuẩn sản xuất (`Data_CoT.md`): Lọc tập Gold Train, sinh lập luận từ API Vision Model (GPT-4o), kèm theo quy tắc QC khắt khe chống hallucination và lặp N-gram.

## 4. 🐛 Những lỗi đã khắc phục (Fixed Bugs)

- **Bug 1: Định dạng và tọa độ Bounding Box** -> **Cách giải quyết:** Các model trả về theo [x1, y1, w, h], đã được quy chuẩn về số nguyên [x1, y1, x2, y2] sử dụng round() chuẩn (Banker's rounding).
- **Bug 2: Lo lắng về việc sắp xếp Box khi submit** -> **Cách giải quyết:** Xác nhận Kaggle metric tự động sort bbox (y1 trước, x1 sau), chỉ cần tập trung độ chính xác của predictions.
- **Bug 3: CUDA OOM do phân mảnh VRAM khi train multi-GPU** -> **Cách giải quyết:** Áp dụng `PYTORCH_ALLOC_CONF` và giới hạn `max_memory`.
- **Bug 4: Crash `NotImplementedError` với Qwen-VL trên T4 GPU** -> **Cách giải quyết:** Ép sử dụng kiểu dữ liệu `Float16` hoặc `Float32` trên toàn bộ model thay vì `BFloat16`.

## 5. 🚀 Công việc tiếp theo (Current / Pending Tasks)

- [ ] Thực thi pipeline Data CoT: Crawl API lấy ~5,000 mẫu reasoning tốt nhất từ dữ liệu chữ viết tay tiếng Ukraina khó đọc (partially_legible / illegible).
- [ ] Chạy huấn luyện các giai đoạn (Train Phases): Pre-train Silver (dữ liệu bạc), Fine-tune Gold, và đặc biệt là fine-tune bộ dữ liệu CoT để mô hình biết tự đưa ra chain-of-thought trước khi chốt conclusion.
- [ ] Viết Inference pipeline End-To-End với logic tìm Entropies và Multimodal Chain-of-Thought Fallback.
- [ ] Tích hợp Post-processing tự sửa lỗi chính tả bằng AI (LLM Spell-check) và hoàn thiện `submission.csv`.

## 6. 📂 Cấu trúc thư mục cốt lõi (Core Structure)

- `plan_VLM.md` & `Data_CoT.md`: Tài liệu thiết kế kiến trúc toàn cục và chi tiết pipeline tạo data CoT.
- `dataset/` : Metadata thư mục chứa tập `train`, `test`, `sliver` của cuộc thi Kaggle.
- `baseline/` : Các đoạn script chạy mẫu zero-shot.
- `finetune/` : Các mã nguồn để huấn luyện VLM (train_qwen_lora, spell_check_submission).
- `VLM_OCR/` : Pipeline chia tách và nhận diện, inference pipeline và công cụ định dạng (VD: chuyển JSONL sang CSV).
