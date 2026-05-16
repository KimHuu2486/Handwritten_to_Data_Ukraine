Bài báo **HALP (Hallucination Prediction via Pre-Generation Probing)** cung cấp một nền tảng phương pháp luận hoàn hảo để tối ưu hóa **Stage 1 & 2 (Dự báo điểm mù)** trong pipeline Explainable & Uncertainty-Aware OCR của bạn. 

Đóng góp lớn nhất của họ là cơ chế phát hiện rủi ro ảo giác *ngay trước khi sinh văn bản* (Pre-Generation Prediction) thông qua một lần chạy xuôi (single forward pass), loại bỏ việc phải đợi mô hình sinh ra từ ngữ rồi mới phát hiện lỗi. Dưới đây là chi tiết các đóng góp và quy trình thực hiện thực tế để bạn tích hợp thẳng vào hệ thống:

--------------------------------------------------------------------------------

#### 1. Quy trình thực hiện Stage 1 & 2: Dự báo điểm mù và Từ chối sớm (Early Deferral Pipeline)
Thay vì đợi mô hình sinh ra từng token để tính toán Entropy $H(X)$ gây tốn kém thời gian và tài nguyên sinh văn bản, HALP thiết kế một cơ chế chặn và đánh giá rủi ro lập tức ngay bên trong các lớp kiến trúc của VLM. Để áp dụng, hệ thống của bạn sẽ chạy theo quy trình sau:
*   **Bước 1: Nạp đầu vào và Trích xuất đặc trưng (Feature Extraction):** Ngay sau khi truyền ảnh tài liệu và prompt qua mô hình, hệ thống can thiệp vào các lớp giải mã sâu (ví dụ: Layer 3L/4 hoặc Layer L) để trích xuất **Đại diện Token truy vấn (Query Token - QT)**. Đây là vị trí chứa ngữ cảnh đa phương thức đã được hợp nhất đầy đủ nhất.
*   **Bước 2: Đánh giá bằng Bộ dò tìm (Probing):** Truyền đặc trưng QT này qua một mạng phân loại siêu nhẹ (ví dụ: mạng MLP 3 lớp). Bộ dò tìm này đóng vai trò như một "trực giác", lập tức đánh giá và trả về một điểm số rủi ro ảo giác $s$.
*   **Bước 3: So sánh Ngưỡng và Rẽ nhánh (Thresholding & Routing):** Hệ thống lấy điểm số $s$ so sánh với ngưỡng $\tau$ được thiết lập trước.
    *   Nếu $s \le \tau$: Mô hình tự tin văn bản rõ ràng, tiếp tục quá trình giải mã và sinh text bình thường.
    *   Nếu $s > \tau$: Nguy cơ đọc sai/ảo giác rất cao, hệ thống lập tức kích hoạt cơ chế **"từ chối sớm" (early deferral)**, bật ngay cờ `[UNCERTAIN]` và tạm dừng quá trình sinh văn bản để đẩy sang Stage 3 (Visual Zoom-in).
*   **Bước 4: Đánh giá dự phòng bằng Đặc trưng thị giác (Visual Features - VF):** Đối với các vùng chữ mờ nhòe do chất lượng ảnh, bạn có thể trích xuất thẳng đặc trưng VF từ bộ mã hóa hình ảnh (Vision Encoder) ngay cả trước khi đi qua lớp kết hợp đa phương thức. Nếu bộ dò VF báo điểm rủi ro cao, hệ thống có thể kết luận ảnh quá mờ và yêu cầu phóng to lập tức mà không cần qua LLM.

--------------------------------------------------------------------------------

#### 2. Đóng góp cho Quá trình Huấn luyện: Tích hợp Bộ dò tìm (Probe Training)
Để hệ thống có khả năng đưa ra điểm số dự báo rủi ro $s$ chuẩn xác, HALP đề xuất phương pháp huấn luyện thêm các bộ phân loại MLP cực nhẹ đi kèm với mô hình chính:
*   **Dữ liệu Huấn luyện (Training Data):** Quá trình này không đòi hỏi bộ nhãn mới. Bạn tận dụng luôn các nhãn đúng/sai (kết quả đọc OCR đúng hay sai) có sẵn từ quá trình đánh giá bằng thuật toán hoặc bằng mô hình LLM làm giám khảo (LLM-as-a-judge) để tạo nhãn nhị phân cho bài toán phân loại của mạng MLP.
*   **Huấn luyện Đồng thời (Training Configuration):** Khi tiến hành Fine-tune (LoRA/QLoRA) cho VLM lõi, các mạng MLP 3 lớp này cũng được cấu hình để huấn luyện song song. Hàm Loss của chúng sẽ cập nhật trọng số để mạng học cách phân biệt ranh giới giữa một trạng thái Hidden States an toàn và một trạng thái dễ gây ra ảo giác.
*   **Chi phí Tính toán Cực thấp (Lightweight Overhead):** Điểm mấu chốt là việc nhúng thêm mạng MLP 3 lớp này cực kỳ nhẹ nhàng. Quá trình nội suy qua mạng MLP này chỉ mất từ 10-15ms trên GPU tiêu chuẩn, tức là làm tăng **dưới 1% tổng thời gian tính toán** so với quá trình sinh văn bản (decoding) thông thường. Điều này cho phép hệ thống triển khai tốt ở các ứng dụng thời gian thực.