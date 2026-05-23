# AGENT CONTEXT: RUKOPYS Kaggle Competition

## 1. Tổng Quan Bài Toán
Cuộc thi **Handwritten to Data** yêu cầu xây dựng hệ thống AI để nhận dạng tài liệu tiếng Ukraina viết tay (Handwritten Text Recognition - HTR). Dữ liệu thuộc tập RUKOPYS, bao gồm nhiều loại tài liệu từ kho lưu trữ lịch sử, bài thi đại học, bài tập về nhà của học sinh và bài thi chính tả quốc gia.

**Yêu cầu và Ràng buộc cốt lõi:**
* **Open Models Only:** Chỉ được phép sử dụng các mô hình mã nguồn mở (open-weight models). Nghiêm cấm dùng API độc quyền lúc inference.
* **Phần cứng:** Toàn bộ pipeline inference phải chạy vừa trên một GPU NVIDIA H100 80GB.
* **Đầu ra (Submission):** File CSV chứa `bbox`, `type` và `text` (nội dung văn bản/LaTeX).
* **Phân loại vùng (Region Types):** `handwritten`, `printed`, `formula`, `table`, `annotation`, `image`, `graph`.

## 2. Kiến Trúc Pipeline (Routing Pipeline 3 Nhánh)
Hệ thống sử dụng cơ chế **Routing Pipeline** dựa trên `type` của từng vùng văn bản để điều hướng tới nhánh mô hình chuyên biệt.

### Nhánh 1: HPA (Handwritten / Printed / Annotation)
* **Mô hình chính:** TrOCR (`Kansallisarkisto/cyrillic-htr-model`).
* **Chiến lược:** Xử lý văn bản dòng đơn qua Batch Inference.

### Nhánh 2: Formula (Công Thức Toán Học)
* **Mô hình chính:** `UniMERNet` (phiên bản `unimernet_base`).
* **Đầu ra:** Định dạng LaTeX (sẽ được chuẩn hóa về Unicode bởi metric của cuộc thi).

### Nhánh 3: Table (Bảng Biểu) - Cập nhật Micro-Routing
Vùng bảng biểu được xử lý qua quy trình đa tầng để bảo toàn cấu trúc và độ chính xác của nội dung hỗn hợp:
1.  **Cell Detection:** `DocLayoutYOLO` phát hiện và định vị từng ô (cells).
2.  **Micro-Routing (Classification Layer):** Mỗi ô sau khi detect được đưa qua một lớp phân loại để xác định là `HPA` (Văn bản) hay `Formula` (Công thức).
3.  **Inference chuyên biệt:** - Các ô văn bản được gom lại và xử lý bởi mô hình **TrOCR**.
    - Các ô chứa công thức được gom lại và xử lý bởi mô hình **UniMERNet**.
4.  **Tái tạo cấu trúc (Heuristic Grouping):** Sắp xếp và ghép kết quả từ hai mô hình theo hàng và cột dựa trên tọa độ trung tâm.
5.  **Định dạng đầu ra:** Xuất dữ liệu dưới dạng **Pipe-Separated Values (PSV)** (ví dụ: `Cột 1 | Cột 2 \n Cột 1 | Cột 2`).

## 3. Nhiệm Vụ Của Agent
Khi làm việc trên project, Agent cần tập trung vào:
1.  **Tối ưu Micro-Routing:** Đảm bảo lớp phân loại trong nhánh Table hoạt động chính xác để tránh gửi nhầm dữ liệu sang mô hình không phù hợp (ví dụ: gửi công thức sang TrOCR).
2.  **Quản lý Batching:** Thiết kế cơ chế gom nhóm (accumulation) hiệu quả cho các ô trong bảng để tận dụng sức mạnh tính toán của H100 mà không gây tràn bộ nhớ.
3.  **Cải thiện Heuristic Grouping:** Tinh chỉnh thuật toán ghép hàng/cột để xử lý tốt các bảng biểu viết tay không thẳng hàng hoặc có cấu trúc phức tạp.
4.  **Đảm bảo tính tuân thủ:** Duy trì tính offline và giới hạn tài nguyên của cuộc thi.
