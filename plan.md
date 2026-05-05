# 📅 Kế Hoạch 5 Tuần: Dự Án Kaggle RUKOPYS (Handwritten to Data)

**Mục tiêu :** Lọt top 10 Leaderboard bằng hệ thống End-to-End Document Understanding.
*   **M1:** Data Engineering (Dataloader, Augmentation, Pseudo-labeling, Curriculum).
*   **M2:** Detection Modeling (Chuyên huấn luyện model cắt box: YOLO, RT-DETR).
*   **M3, M4:** Recognition & VLM (Chuyên OCR, Metric, và Fine-tune VLM).
*   **M5:** MLOps & Architecture (Thiết kế Interface, Tối ưu RAM/Speed, Ensembling).

---

## 🏃 Tuần 1: Sprint 1 - Kiến Trúc Lõi & Baseline (Ngày 1 - Ngày 7)
**Mục tiêu Sprint:** Có 1 luồng code chạy tự động từ ảnh đầu vào -> file `submission.csv` dựa trên các Giao diện (Interfaces) chuẩn mực. Đảm bảo Metric được test kỹ lưỡng.

| Giai đoạn | Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done - DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 1** | M5 | **Architecture (Seams):** Định nghĩa Interface `AbstractPredictor` chuẩn. Yêu cầu M2, M4 viết class kế thừa (Adapter) trả về đúng định dạng JSON thống nhất. | Code base có `interfaces.py`. Luồng Integration không bị phụ thuộc cứng vào model nào. |
| **Ngày 1-2** | M3 | **TDD cho Metric:** Viết Unit Test cho file `official-evaluation-metric-text-normalization.ipynb`. Code lại hàm `score()` để dùng cục bộ. | Hệ thống Unit Test báo `Pass 100%` cho các case normalization (LaTeX, ký tự...). |
| **Ngày 2-3** | M1 | **Data Pipeline:** Viết Pytorch Dataloader. Phân tích phân phối của `train` và `silver`. | Thư mục `notebooks/01_EDA.ipynb` hoàn thiện. Có hàm `get_dataloader()`. |
| **Ngày 4-6** | M2, M4 | **Train Baseline:** M2 train YOLOv8-nano (Detection). M4 train TrOCR (Recognition). Cả hai bọc code bằng Adapter do M5 thiết kế. | Model weights: `yolo.pt`, `trocr.bin`. |
| **Ngày 7** | M5 | **Inference:** Chạy pipeline tích hợp và submit lên Kaggle. | Script xuất ra `solution.csv` hợp lệ, đạt Score > 0.1 trên Leaderboard. |

---

## 🚀 Tuần 2: Sprint 2 - Nâng Cao Model & Quản Trị Rủi Ro VLM (Ngày 8 - Ngày 14)
**Mục tiêu Sprint:** Tách làm 2 nhánh để tìm xem phương pháp nào (Agentic chia nhỏ hay VLM End-to-End) cho kết quả tốt hơn.

| Giai đoạn | Nhánh | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done - DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 8-11** | Nhánh A (M2) | **Agentic:** Nâng cấp lên **RT-DETR** để xử lý ảnh văn bản dày đặc. Build model Classifier (ResNet) phân biệt đủ 7 class (`handwritten`, `printed`, `formula`, `table`, `annotation`, `image`, `graph`). | Model Detection mới (F1-score > 0.85). Model Classifier (Acc > 90%). |
| **Ngày 8-11** | Nhánh B (M3, M4) | **VLM & OOM Safety:** Fine-tune **Qwen3-VL-8B-Instruct**. Bắt buộc bật `Gradient Checkpointing`, dùng `LoRA rank < 32`. Code script giảm dynamic batch size để tránh lỗi Out Of Memory. | Train thành công 1 epoch không crash RAM/VRAM. |
| **Ngày 12-14** | M5 | Tối ưu hóa Inference Pipeline. Cấu hình để Kaggle có thể chạy 2 luồng này độc lập (thông qua config file). | Chạy 386 ảnh test < 10 giây/ảnh. |
| **Ngày 14** | Cả Team | **Review Sprint:** Đánh giá điểm số giữa Nhánh A (Agentic) và Nhánh B (VLM). | Đưa ra quyết định chọn kiến trúc lõi cho tuần sau. |

---

## ⚖️ Tuần 3: Sprint 3 - Data Scaling & Text Normalization (Ngày 15 - Ngày 21)
**Mục tiêu Sprint:** Tận dụng 8,207 ảnh `silver` và dữ liệu ngoài (Pseudo-labeling) mà không làm rò rỉ dữ liệu.

| Giai đoạn | Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done - DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 15-18** | M1 | **Data Scaling:** Áp dụng Curriculum Learning (train trên `silver` trước, fine-tune trên `train` sau). Viết module Pseudo-labeling gióng hàng tập Dictation. | Pipeline train tự động chuyển đổi dataset. Có thêm ~500 ảnh Pseudo-label chất lượng. |
| **Ngày 15-18** | M3, M4 | Tích hợp bộ luật Text Normalization của Kaggle vào Loss function hoặc bước hậu xử lý ngay trong quá trình Validation. | Lỗi sai do khác biệt định dạng LaTeX được loại bỏ. |
| **Ngày 19-21** | M2 | Khám phá thư viện Data Augmentation (Albumentations) chuyên biệt cho ảnh chụp điện thoại (chống lóa, mờ, nhiễu). | Tập Validation cải thiện điểm F1 nhờ Augmentation hợp lý. |

---

## 🔍 Tuần 4: Sprint 4 - Edge Cases (Ca Khó) & Hiệu Năng (Ngày 22 - Ngày 28)
**Mục tiêu Sprint:** Sửa các lỗi vụn vặt (bảng, công thức) và áp dụng Spell Check siêu nhẹ, đảm bảo tốc độ.

| Giai đoạn | Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done - DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 22-24** | M2, M3 | Tập trung xử lý vùng `formula` và `table`. Gộp các box nhỏ thành chuỗi Pipe-separated (`a\*b\*c`). Phân tích top 100 ảnh sai nhiều nhất. | Nâng điểm riêng cho Label `table` và `formula`. |
| **Ngày 25-28** | M5, M4 | **Fast Spell Checking:** Tích hợp từ điển `Hunspell` (hoặc `SymSpell`) tiếng Ukraina để bắt lỗi typo do OCR gây ra (e.g., `Привітт` -> `Привіт`). | Module Spell Check chạy cực nhanh (+0.1s/ảnh). Khắc phục được các lỗi chính tả cơ bản. |

---

## 🏆 Tuần 5: Sprint 5 - Ensembling & Nộp Bài (Ngày 29 - Ngày 35)
**Mục tiêu Sprint:** Tách biệt module Ensembling, gộp các kết quả tốt nhất và chốt Submission.

| Giai đoạn | Thành viên | Nhiệm vụ cụ thể (Actionable Tasks) | Kết quả cần đạt (Definition of Done - DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 29-31** | M5 | **Tách Module Ensembling:** Viết 1 module độc lập chỉ nhận mảng `PredictionList` (JSON). Tích hợp thuật toán Weighted Boxes Fusion (WBF) cho box và Character-level Voting (ROVER) cho text. | Code Ensembling gọn gàng, không gọi đan chéo vào các hàm của model. |
| **Ngày 1-3 (T5)**| M1, M2, M3, M4 | Xuất các bản predictions của từng model mạnh nhất (Agentic, VLM, Model train thêm pseudo-labels) và đẩy qua module Ensembling của M5. | Chọn được bộ thông số Ensemble cho điểm Public LB cao nhất. |
| **Ngày 34-35** | Cả Team | **Dọn dẹp code (Code Cleanup):** Xóa các thư viện dư thừa, áp dụng `gc.collect()` ép RAM. Chọn 2 file submission cuối cùng (1 an toàn, 1 đột phá rủi ro). | Sẵn sàng cho Phase Private Leaderboard. |
