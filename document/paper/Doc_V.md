Bài báo: *Coarse-to-Fine Interactive Visual Reasoning for Multi-Page Document VQA** mang đến một triết lý thiết kế (Agentic framework) rất phù hợp với hệ thống OCR dựa trên VLM, đặc biệt là trong việc quản lý chi phí tính toán và xây dựng quy trình suy luận vòng lặp (ReAct). Dù Doc-V* giải quyết bài toán đa trang (multi-page), tư tưởng của nó có thể ánh xạ hoàn hảo sang bài toán "đa vùng" (multi-region/crop) trong tài liệu viết tay.

## I. Những đóng góp cốt lõi của Doc-V* cho Pipeline

### 1. Đóng góp cho Stage 1 & 2: Chiến lược "Global Thumbnail Overview" (Nhìn thô - Đánh giá nhanh)
* Thay vì đẩy toàn bộ ảnh độ phân giải cực cao vào VLM ngay từ đầu (rất tốn kém token và dễ bị nhiễu), Doc-V* đề xuất bắt đầu bằng một **"Global Thumbnail Overview"** (Ảnh thu nhỏ toàn cục).
* Ảnh ban đầu được resize về kích thước nhỏ (ví dụ 256x256) và đưa cho VLM xem xét cấu trúc tổng thể. Trong pipeline OCR, Stage 1 có thể dùng một ảnh độ phân giải thấp/vừa phải để VLM "đọc lướt". Nếu phát hiện nét chữ mờ không thể nhận diện ở độ phân giải này, hệ thống sẽ bật cờ `[UNCERTAIN]` để chuyển sang Stage 3. Việc này tiết kiệm đáng kể chi phí Attention của VLM.

### 2. Đóng góp cho Stage 3: Quy trình "Structured Visual Reasoning" (Định dạng suy luận cấu trúc)
* Doc-V* ép buộc VLM phải tuân thủ một giao thức tương tác cố định theo chuẩn ReAct (Reasoning + Acting). Mô hình bắt buộc phải xuất ra chuỗi định dạng: `<think> ... </think> <action> ... </action>`.
* Bên trong thẻ `<think>`, bài báo yêu cầu chia nhỏ thành 3 block quan trọng:
  * `<analysis>`: Phân tích vùng ảnh hiện tại xem có chứa manh mối không.
  * `<plan>`: Lập kế hoạch các bước tiếp theo (ví dụ: "Chữ này quá mờ, tôi cần zoom vào tọa độ X, Y").
  * `<summary>`: Tóm tắt lại thông tin đã thu thập được ở bước này.
* **Bộ nhớ làm việc (Working Memory):** Để tránh việc mô hình bị quên (forgetting) hoặc lặp lại một thao tác zoom (drift) khi phải zoom nhiều lần, Doc-V* nối (concatenate) tất cả các `<summary>` ở các bước trước lại thành một bộ nhớ làm việc ($W_t$) và đưa vào ngữ cảnh ở bước tiếp theo.

### 3. Đóng góp cho quá trình Huấn luyện (RL với GRPO & Reward Function)
* Doc-V* sử dụng Group Relative Policy Optimization (GRPO) với một hàm Reward tổng hợp (Composite Reward) không cần dữ liệu trung gian.
* Hàm phần thưởng $R(T)$ của họ được chia làm 3 thành phần chính:
  * $R_a$ (Answer Correctness): Thưởng nếu kết quả cuối cùng đúng.
  * $R_e$ (Evidence Recall): Thưởng nếu mô hình ra quyết định truy xuất/fetch đúng vùng chứa bằng chứng.
  * $R_f$ (Format Validity): Phạt/Thưởng dựa trên việc mô hình có tuân thủ đúng định dạng `<think>...<action>` hay không.

---

## II. Quy trình thực hiện cụ thể áp dụng vào Pipeline của bạn

Dựa trên Doc-V\*, dưới đây là quy trình thực thi (Implementation Process) ánh xạ vào hệ thống 4-stage:

**Bước 1: Khởi tạo quan sát (Initial Observation - Stage 1 & 2)**

* Hệ thống nhận ảnh tài liệu gốc $I_0$. Khởi tạo `Working_Memory = []`.
* Downscale $I_0$ thành ảnh thumbnail $I_{coarse}$ và đưa vào VLM cùng với prompt: *"Hãy đọc tài liệu này. Nếu có từ nào bị mờ, hãy bật cờ [UNCERTAIN]"*.
* Nếu mô hình nhận diện được hết, chuyển thẳng Stage 4. Nếu phát hiện vùng mù, kích hoạt Stage 3.

**Bước 2: Kích hoạt vòng lặp Tương tác (Interactive Loop - Stage 3)**

* Khi cờ kích hoạt, bắt đầu vòng lặp `for t = 0 to T` (giới hạn số bước $T$, ví dụ $T=3$ để tránh lặp vô hạn).
* **Prompt hệ thống bắt buộc:** *"Bạn phải suy luận theo định dạng: `<think> <analysis>...</analysis> <plan>...</plan> <summary>...</summary> </think> <action> CropZoomIn([x1,y1,x2,y2]) </action>`"*.
* **Lượt sinh của VLM:**
  * VLM nhìn ảnh $I_{coarse}$, sinh ra phân tích: *"Chữ ở góc trái bị mờ do nét đứt"*.
  * Lên kế hoạch: *"Tôi sẽ crop và zoom vào vùng ..."*.
  * Đưa ra Action: `<action> CropZoomIn( [14] ... ) </action>`.
  * Tóm tắt `<summary>` được hệ thống lưu lại vào `Working_Memory`.

**Bước 3: Thực thi Action từ Môi trường (Environment Feedback)**

* Pipeline (code Python của bạn) sẽ tự động bắt (parse) cú pháp `<action>`.
* Tiến hành cắt ảnh gốc (crop) tại tọa độ ... và giữ ở độ phân giải gốc cao nhất (High-resolution caching).
* Chèn ảnh đã zoom vào chuỗi token (visual tokens) kèm theo text tiền tố, ví dụ: *"Vùng ảnh đã zoom:"*.

**Bước 4: Suy luận tiếp hoặc Chốt kết quả (Stage 3 -> Stage 4)**

* Ở lượt $t+1$, VLM nhận được input gồm: Câu hỏi gốc + Ảnh đã zoom + `Working_Memory` (các tóm tắt trước đó).
* Bên trong thẻ `<think>` mới, VLM sẽ có thêm block `<relevant_pages>` (hoặc `<relevant_regions>`) để tự đánh giá xem vùng ảnh vừa zoom có ích không.
* Nếu chữ đã rõ, VLM sinh `<action> answer </action>` và xuất kết quả JSON.
* Nếu vẫn chưa rõ nét, nó có thể `<action> CropZoomIn(...) </action>` sang một vùng lân cận để lấy thêm ngữ cảnh (context xung quanh nét chữ).