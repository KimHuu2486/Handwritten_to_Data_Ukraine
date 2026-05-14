Bài báo **VLM-R³ (Visual Language Model with Region Recognition and Reasoning)** cung cấp một nền tảng phương pháp luận hoàn hảo cho **Stage 3 (Multimodal Chain-of-Thought)** và chiến lược huấn luyện trong pipeline 

Đóng góp lớn nhất của họ là cơ chế suy luận tương tác (Interactive Inference) cho phép mô hình tự động quyết định *khi nào* cần thêm thông tin, *ở đâu* cần cắt ảnh, và *làm thế nào* để dệt (weave) các đặc trưng ảnh cục bộ vào chuỗi lý luận văn bản. Dưới đây là chi tiết các đóng góp và quy trình thực hiện thực tế để bạn tích hợp thẳng vào hệ thống:

---

## 1. Quy trình thực hiện Stage 3: Suy luận tương tác và Dệt ngữ cảnh (Interactive Inference Pipeline)

Thay vì sinh toàn bộ văn bản một mạch, VLM-R³ thiết kế một vòng lặp bị ngắt quãng chủ động. Để áp dụng cho vùng chữ viết tay bị mờ, hệ thống của bạn sẽ chạy theo quy trình sau:

* **Bước 1: Kích hoạt bằng Prompt Hệ thống:** Bạn cần thiết lập prompt hướng dẫn VLM phân tích trong các cặp thẻ quy định. Khi muốn nhìn kỹ vùng bị mờ, VLM phải gọi lệnh theo định dạng chuẩn: `{"bbox_2d": [x1, y1, x2, y2]}`.
* **Bước 2: Ngắt quãng (Interception):** Ngay khi VLM sinh ra chuỗi JSON chứa tọa độ `bbox_2d`, pipeline suy luận của bạn lập tức tạm dừng quá trình sinh văn bản (intercept the output).
* **Bước 3: Cắt và Phóng to động (Dynamic Crop & Zoom):** Hệ thống lấy tọa độ để crop ảnh gốc. Đặc biệt, VLM-R³ sử dụng một **quy tắc Zoom Scaling** rất hiệu quả dựa trên tỷ lệ diện tích $r = \frac{A_{bbox}}{A_{orig}}$:
    * Nếu vùng chữ quá nhỏ ($r < 0.125$): Phóng to với tỷ lệ $scale = 2.0$.
    * Nếu vùng chữ đã đủ lớn ($r \ge 0.5$): Giữ nguyên $scale = 1.0$.
    * Ở khoảng giữa: Nội suy tuyến tính theo công thức $scale = \frac{2.0 - r - 0.125}{0.375}$.
* **Bước 4: Dệt ảnh vào ngữ cảnh (Context Weaving):** Ảnh crop sau khi zoom sẽ được mã hóa thành các visual tokens và nối trực tiếp (appended) vào chuỗi đầu vào hiện tại của VLM. Sau đó, VLM được kích hoạt chạy tiếp để sinh văn bản nhận dạng nét chữ dựa trên hình ảnh sắc nét vừa được chèn vào. Vòng lặp này lặp lại cho đến khi VLM chốt được đáp án và đặt vào thẻ kết quả.

---

## 2. Đóng góp cho Quá trình Huấn luyện: Thuật toán R-GRPO và Hàm Reward

Để dạy VLM biết cách tự sinh ra tọa độ `bbox_2d` và lý luận chính xác, VLM-R³ đề xuất thuật toán **Region-Conditioned Reinforcement Policy Optimization (R-GRPO)**. Bạn có thể áp dụng cơ chế tính Loss và Reward này vào quá trình RL (Reinforcement Learning) cho mô hình OCR của bạn:

* **Masking Gradient trong Loss Function:** Trong chuỗi dữ liệu huấn luyện, có cả văn bản do VLM tự sinh (action) và các token hình ảnh do hệ thống tự động dệt vào (state). Khi tính toán đạo hàm (gradient) cho Policy Model, bạn **bắt buộc phải dùng Mask để chỉ tính gradient trên các token văn bản và lệnh tạo tọa độ do mô hình sinh ra**, bỏ qua hoàn toàn gradient của các token hình ảnh được chèn vào.
* **Thiết kế Hàm Phần thưởng (Reward Function):** Tổng phần thưởng $r_{overall}$ cho một trajectory được tính bằng công thức [7]:

$$r_{overall} = r_{acc} + r_{format} + r_{length} + r_{region} \cdot \mathbb{I}(r_{acc}=1.0)$$

* $r_{acc}$: Thưởng 1 nếu kết quả OCR cuối cùng đúng, ngược lại là 0.
* $r_{format}$: Thưởng 1 nếu mô hình dùng đúng thẻ định dạng, ngược lại là 0.
* $r_{length}$: Thưởng 0.001 cho mỗi ký tự lý luận để khuyến khích giải thích, nhưng giới hạn (capped) tối đa ở 0.25 để tránh mô hình nói dài dòng vô ích.
* **$r_{region} \cdot \mathbb{I}(r_{acc}=1.0)$:** Đây là điểm mấu chốt cho bài toán Uncertainty. Mô hình được thưởng 0.5 cho mỗi tọa độ crop hợp lệ. Tuy nhiên, phần thưởng này bị nhân với hàm chỉ báo $\mathbb{I}(r_{acc}=1.0)$. Tức là, **mô hình chỉ được nhận điểm thưởng cho việc tự tin crop/zoom vùng chữ mờ nếu và chỉ nếu kết quả nhận dạng cuối cùng là chính xác**. Điều này sẽ phạt cực nặng các trường hợp mô hình hallucinate, tự sinh tọa độ lung tung hoặc tự tin thái quá nhưng chốt đáp án sai.

---