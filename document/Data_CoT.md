# Tạo dữ liệu CoT

## Tóm tắt
Tạo bộ dữ liệu Chain-of-Thought (CoT) cho OCR từ tập train, tập trung vào vùng chữ viết tay khó. Dữ liệu gốc nằm trong thư mục [dataset/](dataset/). Pipeline tạo JSONL sạch, có version, có QC, có retry, và đầy đủ metadata phục vụ huấn luyện và phân tích.

## Mục tiêu
- Tạo ~5,000 mẫu vùng chữ khó nhất (region crops) CoT từ tập hợp 25.523 regions của tập gold train.
- Ưu tiên `legibility` = partially_legible hoặc illegible.
- Đảm bảo output parse được, nhất quán, truy vết được về input.
- Có log và thống kê để theo dõi độ phủ và tỉ lệ QC pass.

## Không làm
- Không huấn luyện model trong pipeline này.
- Không có UI annotate thủ công hoặc vòng sửa tay.
- Không đóng gói release dataset (xử lý ở bước sau).

## Đầu vào
Nguồn dữ liệu theo cấu trúc thư mục:
- dataset/
  - train/metadata.jsonl
  - test/metadata.jsonl
  - sliver/metadata.jsonl
  - sample_submission.csv
  - source.md

Mỗi candidate region (lấy từ dataset/train/metadata.jsonl) cần:
- image_path (hoặc image_id để map ra đường dẫn ảnh)
- image_id
- region_bbox (x1, y1, x2, y2)
- region_type (handwritten, printed, formula, table, annotation)
- ground_truth
- legibility
- source

## Đầu ra (Kết quả mong đợi)
Bao gồm 3 file chính để quản lý dữ liệu và theo dõi quá trình chạy.

### 1. File `cot_samples.jsonl` (File dữ liệu chính)
- **Mục đích:** Chứa các mẫu (samples) AI tạo thành công và vượt qua vòng kiểm duyệt chất lượng (QC). File này tóm gọn quá trình "động não" của AI, sau này sẽ dùng trực tiếp để dạy cho mô hình chính (Qwen3-VL 8B) học cách suy luận.
- **Giải thích tham số:**
  - `cot_reasoning`: Toàn bộ lý luận từng bước của mô hình (vì sao lại đọc ra được chữ đó, nét móc kia giống chữ gì).
  - `uncertainty_words` / `uncertainty_scores`: Các từ mà AI cảm thấy khó đoán nhất và độ tự tin tương ứng.
  - `ground_truth`: Đáp án text chuẩn gốc (rất quan trọng, để đảm bảo AI suy luận dựa trên đáp án đúng).
- **Format mẫu:**
```json
{
  "image_id": "uuid_xxx",
  "region_bbox": [x1, y1, x2, y2],
  "region_type": "handwritten",
  "ground_truth": "Сьогодні гарна погода",
  "cot_reasoning": "<ambiguous_chars>...</ambiguous_chars>...",
  "uncertainty_words": ["Сьогодні"],
  "uncertainty_scores": [0.73],
  "model_name": "gpt-4o",
  "prompt_version": "cot_v1",
  "created_at": "2026-05-14T00:00:00Z"
}
```

### 2. File `cot_failed.jsonl` (File lưu các mẫu bị lỗi)
- **Mục đích:** Lưu trữ lại những lần gọi API bị thất bại (lỗi mạng, model trả về thiếu thẻ, hoặc AI suy luận ra một văn bản sai lệch với `ground_truth`). File này giúp DE kiểm tra (debug) lỗi mà không cần tốn tiền chạy lại API.
- **Format mẫu:** Chú ý có trường `raw_response` để xem thực sự AI đã trả về dòng text rác nào.
```json
{
  "image_id": "uuid_xxx",
  "region_bbox": [x1, y1, x2, y2],
  "error_type": "parse_error|api_error|qc_fail",
  "error_detail": "Missing <conclusion> tag",
  "raw_response": "...nội dung lỗi trả về từ AI...",
  "model_name": "gpt-4o",
  "prompt_version": "cot_v1",
  "created_at": "2026-05-14T00:00:00Z"
}
```

### 3. File `cot_stats.json` (File thống kê)
- **Mục đích:** Tóm tắt tình trạng pipeline: tổng số lượng mẫu đưa vào, phần trăm thành công (pass rate), phân bổ lỗi. Rất tiện để team OCR nắm tiến độ tiến độ.

---

## Pipeline end-to-end (Quy trình thực thi chi tiết)
Dưới đây là các bước tự động theo luồng mà Data Engineer (DE) cần viết code, áp dụng tuần tự cho danh sách vùng chữ `region` tạo ra:

**Bước 1: Lọc dữ liệu đầu vào (Filtering)**
- Đọc data từ `dataset/train/metadata.jsonl`.
- Loại bỏ các ảnh dễ đọc, chỉ giữ lại những vùng chữ có đánh nhãn `legibility` là `partially_legible` (hơi nhòe/khó đọc) hoặc `illegible` (rất khó đọc).
- Cố gắng nhặt ra khoảng ~5,000 vùng chữ phù hợp.

**Bước 2: Cắt ảnh (Image Cropping)**
- Dùng ảnh gốc được chỉ định trong thông tin `image_id` (hoặc `image_path`).
- Dựa vào tọa độ box (`region_bbox`), viết hàm cắt trực tiếp vùng chữ đó ra thành một hình con nhỏ hơn (region crop). Mục đích để Vision Model tập trung phân tích đúng dòng chữ mờ này thay vì nhìn cả tờ giấy.

**Bước 3: Chuẩn bị Prompt & Gửi API (Generation)**
- Gửi cho Model Vision (GPT-4o/Gemini/model đủ lớn, đủ đáng tin cậy)  2 thứ: **Ảnh vừa cắt** + **Text thực tế của ảnh đó** (`ground_truth`).
- Prompt (tự custom lại cho hiệu quả hơn): *"This is a handwritten image, and the answer extracted from the image is [ground_truth]. You need to act like an expert and explain why you deciphered that answer. Are there any ink strokes or letters that are easily confused with other words? Analyze it step-by-step for me."*

**Bước 4: Bóc tách text trả về (Parsing)**
- Lấy kết quả AI trả ra, dùng Regex lập trình sẵn để tách dữ liệu nhét vào 5 thẻ sau:
  - `<ambiguous_chars>`: chữ cái nào khó đọc / dễ nhầm.
  - `<visual_analysis>`: phân tích đặc điểm thị giác (ví dụ: nét móc quá dài).
  - `<context_clues>`: manh mối ngữ nghĩa từ đoạn văn giúp dịch được chữ khó.
  - `<reasoning>`: chuỗi suy luận chi tiết.
  - `<conclusion>`: kết luận chốt văn bản.

**Bước 5: Kiểm duyệt tự động (QC - Quality Control)**
- Code kiểm tra khắt khe để tránh đưa rác vào dữ liệu training:
  1. Thẻ `conclusion` CÓ KHỚP 100% với `ground_truth` ban đầu không? (Vì nếu AI lái sang kết quả khác tức là hallucination/sai lệch).
  2. Bố cục trả về có đủ 5 thẻ không?
  3. Chuỗi `reasoning` có quá ngắn không?
  4. Có bị dính vòng lặp n-gram (bệnh nói nhảm liên tục lặp lại 1 câu của LLM) không?

**Bước 6: Trích xuất Uncertainty**
- Dùng script cào qua vùng lý luận để lọc thêm 1 mảng các từ được AI nhấn mạnh là khó đoán (`uncertainty_words`).

**Bước 7: Phân luồng lưu trữ & Retry**
- Nếu **Vượt qua bước QC (Bước 5):** Đóng gói thành Json và nối dòng vào `cot_samples.jsonl`.
- Nếu **Rớt QC** (ví dụ thiếu thẻ) hoặc **Lỗi mạng API**:
  - Code sẽ tự động chạy lại Bước 3 (thử lại tối đa 3 lần).
  - Thử cạn lần mà vẫn thất bại -> Ghi lịch sử lỗi và raw response vào `cot_failed.jsonl` để DE có thể kiểm tra sau. Bỏ qua cái đoạn chữ này đi tiếp.

**Bước 8: Cập nhật**
- Tăng biến đếm để cập nhật file thống kê (`cot_stats.json`).

## Prompt và parsing
- Prompt template có version (`prompt_version`).
- Model phải trả đủ tag, mỗi tag đúng một lần.
- Parse fail nếu thiếu tag hoặc sai định dạng.

## QC Rules
Mục đích của bộ Rules này là đảm bảo dữ liệu đưa vào fine-tune phải "sạch", triệt tiêu hoàn toàn sự "ảo giác" (hallucination) thường gặp của AI. Một mẫu (sample) sẽ bị ĐÁNH TRƯỢT nếu vi phạm 1 trong 4 quy tắc sau:

**1. Rule 1: Khớp đáp án (Conclusion Match)**
- So sánh chuỗi text nằm trong thẻ `<conclusion>` với nhãn gốc (`ground_truth`).
- **Yêu cầu:** Bắt buộc phải giống nhau **100% (exact match)**.
- **Tại sao:** Nếu AI phân tích một hồi nhưng lại chốt ra một đáp án khác với thực tế, có nghĩa là tư duy (reasoning) của nó bị sai lệch, dòng tư duy đó không thể dùng để dạy mô hình khác được.

**2. Rule 2: Cấu trúc nguyên vẹn (Format Completion)**
- Chuỗi trả về từ AI bắt buộc phải bóc tách (parse) thành công đủ cả 5 thẻ xml.
- **Yêu cầu:** Không thiếu thẻ nào, không sai chính tả thẻ (ví dụ thẻ quên viết `</reasoning>`).
- **Tại sao:** Pipeline cần sự đồng nhất. Khi train, mô hình cần học đúng chuẩn format đã định trước. Một kết quả trả thiếu thẻ sẽ làm gãy schema JSON khi lưu data.

**3. Rule 3: Độ dài lý luận (Reasoning Length)**
- Đếm tổng số ký tự bên trong thẻ `<reasoning>`.
- **Yêu cầu:** Chiều dài chuỗi lý luận $\ge 20$ ký tự.
- **Tại sao:** Loại bỏ những lần AI "lười biếng", đưa ra luận điểm quá ngắn gọn (Ví dụ: "Chữ này mờ nên tôi đoán thế"). Chúng ta muốn thu thập các lập luận từng nét vẽ kỹ càng.

**4. Rule 4: Chống lặp vòng (N-gram Loop Detection)**
- Quét qua toàn bộ nội dung mà LLM sinh ra.
- **Yêu cầu:** Không được có một chuỗi ký tự (hoặc từ) bị lặp lại liên tiếp quá giới hạn (threshold = 5).
- **Tại sao:** Hiện tượng sinh lặp (repetition/degeneration) là lỗi cực kỳ phổ biến ở các Large Language Model khi chúng bị lạc hướng. Code QC cần phát hiện và chặn lại để không đẩy đoạn text rác "nét móc này nét móc này nét móc này..." vào data.

**Xử lý khi trượt QC:** Tích hợp với luồng Retry (Thử lại). Bất cứ rule nào Fail đều sẽ gán cờ "qc_fail", ghi rõ lỗi vào biến `error_detail` (để log lại vào `cot_failed.jsonl`), sau đó bắt đầu gọi lại API.

## Xử lý lỗi và retry
- Retry khi lỗi API với exponential backoff (tối đa 3 lần).
- Timeout mỗi call (cấu hình được, mặc định 60s).
- Lưu raw_response cho mọi lỗi.

## Cấu trúc lưu trữ
- data/cot/
  - cot_samples.jsonl
  - cot_failed.jsonl
  - cot_stats.json
  - prompts/
    - cot_v1.txt

## Logging và metrics
- Log latency từng mẫu, model name, token usage.
- cot_stats.json gồm:
  - total_candidates
  - total_success
  - total_failed
  - pass_rate
  - distribution_by_source
  - distribution_by_legibility

## Tiêu chí nghiệm thu
- 5,000+ mẫu trong cot_samples.jsonl.
- QC pass rate >= 85%.
- Tất cả mẫu parse đúng schema.
- Stats file khớp với số lượng thực tế.
