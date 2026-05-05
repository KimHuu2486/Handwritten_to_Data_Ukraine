# 📅 Kế Hoạch 5 Tuần: Dự Án Kaggle RUKOPYS (Handwritten to Data)

**Mục tiêu:** Lọt top 10 Leaderboard bằng hệ thống End-to-End Document Understanding.

## Strict Roles:
*   **M1 - Data Engine:** Làm việc với Dữ liệu (Dataloader, Augmentation, Synthetic Data, Pseudo-labeling).
*   **M2 - Vision & Layout:** Làm việc với Tầm nhìn không gian (YOLO, RT-DETR, Classifier 7 class, cắt Box, nhận diện Bảng).
*   **M3 - OCR & Metric:** Làm việc với Text từ Box đã cắt (TrOCR, CRNN, Text Normalization, Toán học LaTeX).
*   **M4 - Generative & VLM:** Làm việc với Mô hình Lớn (Qwen3-VL, Gemma, LLM Spell Check).
*   **M5 - MLOps & Architecture:** Làm việc với Kỹ thuật phần mềm (Định nghĩa Interface, Framework Cross-Validation, Ensembling WBF/ROVER, Tối ưu RAM Kaggle).

---

## 🏃 Tuần 1: Khởi Tạo, Nghiên Cứu & Baseline (Ngày 1 - Ngày 7)
**Mục tiêu:** Đọc hiểu phương pháp (Paper/Docs), chốt kiến trúc lõi.

| Thành viên | Đọc Paper / Nghiên Cứu (Nửa đầu tuần) | Nhiệm vụ Code (Nửa cuối tuần) |
| :--- | :--- | :--- |
| **M1 (Data)** | Đọc paper về *Curriculum Learning* và *Data-centric AI cho HTR*. | Viết Pytorch Dataloader chuẩn, thống kê EDA tập `train`/`silver`. |
| **M2 (Vision)** | Nghiên cứu *RT-DETR* và kiến trúc *Document Layout Analysis*. | Train baseline YOLO/Faster R-CNN cắt box. |
| **M3 (OCR)** | Đọc paper *TrOCR* và kỹ thuật nhận dạng chữ Cyrillic. | Viết Unit Test cho `score()` (TDD). Train baseline OCR cho ảnh crop. |
| **M4 (VLM)** | Đọc Docs của *HuggingFace* về LoRA/QoRA cho Qwen3-VL. | Thiết lập môi trường train VLM, chạy thử luồng infer rỗng. |
| **M5 (MLOps)** | Đọc Docs Kaggle và RAM optimization. | Code `interfaces.py` định nghĩa đầu vào/ra. Viết script ghép file submit. |

---

## 🚀 Tuần 2: Nâng Cấp Chuyên Môn & Độc Lập Phát Triển (Ngày 8 - Ngày 14)

| Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done) |
| :--- | :--- | :--- |
| **M1 (Data)** | Xây dựng pipeline Augmentation (Albumentations) chống lóa, nhiễu. | Pipeline sinh ảnh đã augment để M2/M3 gọi hàm trực tiếp. |
| **M2 (Vision)** | Train **RT-DETR** và Classifier ResNet phân biệt 7 class. | Model Detection (F1 > 0.85) và Classifier (Acc > 90%). |
| **M3 (OCR)** | Nâng cấp model OCR. Tích hợp luật Text Normalization của Kaggle. | Model OCR xử lý tốt chữ viết tay tiếng Ukraina. |
| **M4 (VLM)** | Fine-tune **Qwen3-VL-8B**. Bật `Gradient Checkpointing` chống OOM. | Train thành công 1 epoch VLM không crash RAM. |
| **M5 (MLOps)** | Xây dựng Cross-Validation framework (chia K-Fold chống rò rỉ data). | Module `cv_splitter.py` chuẩn bị sẵn để chấm điểm Local. |

---

## ⚖️ Tuần 3: Mở Rộng Dữ Liệu & Rèn Giũa (Ngày 15 - Ngày 21)
**Mục tiêu:** Vượt qua giới hạn của tập `train` 1330 ảnh bằng các kỹ thuật xử lý dữ liệu phức tạp.

| Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done) |
| :--- | :--- | :--- |
| **M1 (Data)** | Tạo **Pseudo-labels** từ tập `silver` bằng cách gióng hàng text dictation. | Sinh ra dataset mới chất lượng cao để M2/M3 train lại. |
| **M2 (Vision)** | Cải thiện độ nhạy bắt Box cho `table` và `formula`. | Các bảng và công thức nhỏ không bị bỏ sót. |
| **M3 (OCR)** | Chuyên biệt hóa model OCR để đọc mã LaTeX cho `formula`. | Model thứ hai chuyên giải mã LaTeX từ Box do M2 cắt. |
| **M4 (VLM)** | Kỹ thuật Prompt Engineering để VLM sinh đúng định dạng Bảng (`a\|b\|c`). | VLM tự xuất ra chuỗi table chuẩn Kaggle. |
| **M5 (MLOps)** | Viết thuật toán ghép Box (WBF) và ghép Text (ROVER) độc lập. | Các hàm Ensembling sẵn sàng nhận mảng JSON. |

---

## 🔍 Tuần 4: Xử Lý Ca Khó & Hậu Xử Lý (Ngày 22 - Ngày 28)
**Mục tiêu:** Phân tích lỗi (Error Analysis) và dùng thuật toán phụ trợ để vớt điểm CER.

| Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done) |
| :--- | :--- | :--- |
| **M1 (Data)** | Error Analysis phía Data: Lọc ra các ảnh có điểm CER/F1 tệ nhất. | Báo cáo: "Data đang thiếu ảnh chụp nghiêng góc". |
| **M2 (Vision)** | Fix lỗi Box: Xử lý các Box bị cắt lẹm mất mép chữ. | Script Post-process mở rộng Box +2 pixels. |
| **M3 (OCR)** | Fix lỗi OCR: Tinh chỉnh hàm dự đoán cho các ký tự dễ nhầm lẫn. | Tăng độ chính xác cho ký tự Cyrillic đặc thù (`ї`, `і`, `є`). |
| **M4 (VLM)** | **Spell Check:** Huấn luyện SymSpell hoặc LLM nhỏ chuyên sửa typo. | Text đầu ra cuối cùng được sửa lỗi chính tả siêu tốc (< 0.1s/ảnh). |
| **M5 (MLOps)** | Chạy thử toàn bộ Pipeline trên phần cứng Kaggle. | Khắc phục mọi lỗi Time Out (vượt 9h) hoặc Out of Memory. |

---

## 🏆 Tuần 5: Lắp Ráp, Tối Ưu & Nộp Bài (Ngày 29 - Ngày 35)
**Mục tiêu:** Bơm mọi cấu hình mạnh nhất vào module do M5 thiết kế và chốt sổ.

| Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done) |
| :--- | :--- | :--- |
| **M1, M2, M3, M4** | Đóng gói Model weights. Xuất dự đoán (Predictions) đẩy vào module của M5. | Hoàn thành việc tối ưu thông số mô hình cá nhân. |
| **M5 (MLOps)** | Bật Ensembling. Xóa rác, ép RAM (`gc.collect()`). Chọn 2 file submit. | 1 Submission cực an toàn (Local CV cao) + 1 Đột phá. |
| **Cả Team** | Review lại code lần cuối. Nghỉ ngơi chờ kết quả Private Leaderboard. | Hoàn thành chiến dịch. |
