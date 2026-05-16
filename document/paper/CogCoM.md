Bài báo **CogCoM (Chain-of-Manipulations Reasoning)** cung cấp một nền tảng phương pháp luận hoàn hảo cho **Stage 3 (Multimodal Chain-of-Thought)** và chiến lược tạo dữ liệu huấn luyện trong pipeline của bạn. 

Đóng góp lớn nhất của họ là cơ chế suy luận tương tác từng bước dựa trên bằng chứng (evidential reasoning), cho phép mô hình chủ động sinh ra các thao tác (manipulations) như *định vị (grounding)*, *cắt/phóng to (crop & zoom-in)*, và *đọc chữ (ocr)* thay vì phụ thuộc vào một chuỗi code cứng (rule-based) bên ngoài. Dưới đây là chi tiết các đóng góp và quy trình thực hiện thực tế để bạn tích hợp thẳng vào hệ thống:

--------------------------------------------------------------------------------

#### 1. Quy trình thực hiện Stage 3: Suy luận tương tác qua Chuỗi thao tác (Interactive Inference Pipeline)
Thay vì sinh toàn bộ văn bản một mạch và nhận ảnh độ phân giải cao một cách thụ động, CogCoM biến VLM thành một tác tử (agent) tự ra quyết định điều khiển hình ảnh. Để áp dụng cho vùng chữ viết tay bị mờ, hệ thống của bạn sẽ chạy theo quy trình sau:
*   **Bước 1: Gọi lệnh Định vị (Grounding Invocation):** Khi gặp từ khó đọc, VLM tự phân tích ngữ cảnh và xuất ra dòng lệnh nội bộ, ví dụ `GROUNDING(từ bị mờ)`. Lệnh này yêu cầu mô hình xác định tọa độ vùng cần quan tâm và trả về bounding box dạng `[x1, y1, x2, y2]`.
*   **Bước 2: Cắt và Phóng to (Crop & Zoom-in):** Dựa trên tọa độ thu được, VLM tiếp tục gọi lệnh `CropZoomIn(bbox, tỷ_lệ)`. Hệ thống sẽ cắt vùng bounding box bị mờ trên ảnh gốc và phóng to ảnh theo tỷ lệ (scale) để tạo ra một bức ảnh mới sắc nét hơn.
*   **Bước 3: Dệt ảnh qua Bộ nhớ Đa ảnh (Multi-Image KV-Cache):** Bức ảnh cận cảnh được sinh ra sẽ được mã hóa để đưa ngược lại vào VLM. Để mô hình vừa thấy được nét chữ vi mô, vừa không quên ngữ cảnh vĩ mô (Context_A, Context_B) từ ảnh gốc, CogCoM ghép nối chuỗi bộ nhớ KV-cache theo công thức:
$$K'_t = trunc(concat(K_0, K_1, ..., K_t))$$
$$V'_t = trunc(concat(V_0, V_1, ..., V_t))$$
*   **Bước 4: Tổng hợp kết quả (Final Aggregation):** Nhờ cơ chế bộ nhớ đa lượt (multi-turn), VLM tận dụng ảnh nét chữ đã được dệt vào ngữ cảnh để tiếp tục sinh ra chuỗi lý luận (CoT) hoặc gọi lệnh `OCR(ảnh_mới)` để chốt đáp án chính xác cuối cùng dựa trên các bằng chứng trực quan thực tế.

--------------------------------------------------------------------------------

#### 2. Đóng góp cho Quá trình Huấn luyện: Chế tạo Dữ liệu Xếp tầng (Cascading Data Generation)
Để dạy VLM biết cách tự sinh ra tọa độ, lệnh cắt ảnh và chuỗi lý luận chính xác, CogCoM đề xuất đường ống tạo dữ liệu tự động (Cascading Data Generation Pipeline) giúp sinh hàng ngàn Dataset CoT chuẩn xác:
*   **Người lập logic (Linguistic Annotators - LLMs):** Sử dụng các mô hình ngôn ngữ mạnh như GPT-4 đóng vai trò "người lên kế hoạch". GPT-4 sẽ nhận một prompt cấu trúc yêu cầu nó sinh ra các bước giải quyết vấn đề (ví dụ: Bước 1 gọi hàm nào, Bước 2 làm gì tiếp theo) theo định dạng thao tác chuẩn mà không cần tự tính toán tọa độ giả.
*   **Người thực thi thị giác (Visual Annotators - VFMs):** Hệ thống gọi các mô hình thị giác nền tảng (Foundation Models) như GroundingDINO (để tìm tọa độ) hoặc PaddleOCR (để đọc text) để thực thi chính xác các hàm mà LLM vừa yêu cầu. Kết quả thực tế từ VFM (ví dụ tọa độ chính xác của từ) sẽ được điền vào các biến giữ chỗ trong chuỗi của LLM.
*   **Thuật toán Lọc dữ liệu bằng Cây tìm kiếm (Depth-First Search - DFS):** Việc một tham số đầu vào có thể cho ra nhiều kết quả tạo thành một cấu trúc Cây suy luận (Tree $\mathcal{T}$). Hệ thống sẽ chạy thuật toán Duyệt theo chiều sâu (DFS) trên cây này để đánh giá và tìm ra các nhánh (Positive Chains) kết thúc bằng một đáp án đúng (Golden Answer $\mathcal{A}$). Quá trình này được biểu diễn bằng công thức:
$$[\varsigma_1, \varsigma_2, ...] = DFS(\mathcal{T} | \mathcal{A})$$
*   Chỉ những chuỗi dẫn tới kết quả nhận dạng chuẩn cuối cùng mới được lưu lại, loại bỏ những dữ liệu bị nhiễu hoặc do mô hình đoán mò sinh ra ảo giác, đảm bảo dữ liệu huấn luyện CoT hoàn toàn minh bạch.