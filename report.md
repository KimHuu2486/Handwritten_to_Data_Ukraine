# 📝 Báo Cáo Dự Án: Kaggle RUKOPYS — Handwritten to Data

**Cuộc thi:** [Handwritten to Data](https://www.kaggle.com/competitions/handwritten-to-data)
**Dataset:** [RUKOPYS](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys) — Ukrainian Catholic University
**Thời gian:** 16/04/2026 — 15/06/2026 | **Giải thưởng:** $7,000
**License:** CC BY-NC-SA 4.0

---

## 1. Tổng Quan Bài Toán

### 1.1. Mô tả

Bài toán yêu cầu xây dựng hệ thống **End-to-End Document Understanding** cho tài liệu viết tay tiếng Ukraina, bao gồm 3 nhiệm vụ con:

1. **Region Detection** — Phát hiện vùng nội dung trong ảnh tài liệu (bounding box)
2. **Region Classification** — Phân loại vùng thành 1 trong 7 loại
3. **Text Transcription** — Nhận dạng và phiên âm nội dung văn bản

> **Điểm đặc biệt:** RUKOPYS là dataset HTR (Handwritten Text Recognition) mở đầu tiên quy mô lớn cho tiếng Ukraina — ngôn ngữ Slavic với 45M+ người nói bản ngữ nhưng trước đây chưa có dataset HTR chuyên dụng.

### 1.2. Tại sao bài toán này khó?

| Chiều biến thiên | Phạm vi trong RUKOPYS |
|---|---|
| **Thời kỳ** | 1919–1935 (mực & bút lông) → 2020–2025 (bút bi, bút chì) |
| **Người viết** | Học sinh (lớp 5–11), sinh viên đại học, công dân trưởng thành |
| **Loại tài liệu** | Văn bản lưu trữ nhà nước, bài chính tả, bài thi, bài tập về nhà |
| **Phương pháp chụp** | Máy scan phẳng vs camera điện thoại |
| **Chính tả** | Chính tả cổ trước cải cách (1920s) → tiếng Ukraina hiện đại |
| **Nội dung** | Văn xuôi, công thức toán, hóa học, bảng biểu, ghi chú giáo viên |

---

## 2. Phân Tích Dataset RUKOPYS

### 2.1. Tổng quan các split

| Split | Số ảnh | Số regions | Nguồn annotation | Mô tả |
|---|---|---|---|---|
| **train** | 1,330 | 25,523 | `annotator` / `volunteer` | Gán nhãn bởi người — bbox + transcription đã xác minh |
| **silver** | 8,207 | 161,065 | `auto` | Gán nhãn tự động bởi Qwen3-VL 8B + Gemini — dùng cho self-training |
| **test** | 386 | — (ẩn) | — | Chỉ có ảnh — submit dự đoán lên Kaggle |
| **private benchmark** | 21 | — (ẩn) | — | Tập đánh giá cuối cùng, công bố sau 15/06 |

### 2.2. Thành phần tập Train theo nguồn dữ liệu

| Nguồn (`source`) | ID | Giai đoạn | Mô tả |
|---|---|---|---|
| **National Dictation** | `dictation` | 2020–2025 | Ảnh chụp điện thoại bài chính tả quốc gia. Cùng một văn bản mỗi năm, hàng nghìn phong cách viết tay. |
| **State Archive** | `archive` | 1919–1935 | Tài liệu scan từ 12 quỹ lưu trữ ЦДАВО. Mực & bút lông, chính tả cổ. |
| **University (KNUTE)** | `university` | 2024–2025 | Bài thi sinh viên scan từ 5 khoa: văn bản, công thức toán, hóa học, bảng. |
| **School Homework** | `school` | 2024–2025 | Ảnh chụp điện thoại bài tập về nhà (lớp 5–11, 20+ môn). |

**Phân bổ chi tiết:**

| Nguồn | Professional | Volunteer | Tổng |
|---|---|---|---|
| `dictation` | 221 | 138 | 359 |
| `archive` | 90 | 37 | 127 |
| `university` | 136 | 26 | 162 |
| `school` | 398 | 284 | 682 |
| **Tổng** | **845** | **485** | **1,330** |

### 2.3. Schema annotation

Mỗi record chứa trường `regions` — danh sách các vùng nội dung:

```json
{
  "file_name": "images/abc123.jpg",
  "image_width": 3024,
  "image_height": 4032,
  "source": "dictation",
  "annotation_source": "annotator",
  "regions": [
    {
      "bbox": [134, 766, 3754, 1197],
      "type": "handwritten",
      "language": "uk",
      "legibility": "legible",
      "text": "Спочатку був брехунець."
    }
  ]
}
```

- **`bbox`**: `[x1, y1, x2, y2]` — tọa độ pixel, gốc top-left
- **`type`**: 1 trong 7 loại (xem bảng dưới)
- **`language`**: `uk` hoặc `other`
- **`legibility`**: `legible` hoặc `illegible`

### 2.4. Bảy loại Region

| Type | Mô tả | Transcription |
|---|---|---|
| `handwritten` | Dòng chữ viết tay | Văn bản chính xác, 1 bbox = 1 dòng |
| `printed` | Dòng chữ in/đánh máy | Văn bản chính xác |
| `formula` | Biểu thức toán/hóa học | LaTeX |
| `table` | Bảng đầy đủ | Pipe-separated values |
| `annotation` | Điểm số, ghi chú giáo viên | Văn bản ngắn |
| `image` | Con dấu, hình vẽ | Rỗng |
| `graph` | Biểu đồ, đồ thị | Rỗng |

### 2.5. Ký hiệu đặc biệt trong text

| Ký hiệu | Ý nghĩa |
|---|---|
| `~~word~~` | Chữ bị gạch ngang |
| `~~old~~{new}` | Gạch ngang kèm sửa chữa |
| `[illegible]` | Từ không đọc được trong dòng legible |

### 2.6. Thiết kế chống rò rỉ dữ liệu (Anti-Leakage)

| Nguồn | Train | Test | Đảm bảo |
|---|---|---|---|
| **Dictation** | Năm 2024 | Năm 2020, 2022, 2025 | Các văn bản canonical khác nhau |
| **Archive** | Bộ hồ sơ A | Bộ hồ sơ B | Không trùng lặp tài liệu |
| **University** | Nhóm PDF A | Nhóm PDF B | Bài thi khác sinh viên |
| **School** | Lớp 5, 6, 7, 9, 11 | Lớp 8, 10 | Khác nhóm lớp |

### 2.7. Silver Split — Pipeline tự động gán nhãn

```
Stage 1: Qwen3-VL 8B → phát hiện block
Stage 2: Gemini Flash → phân loại block
Stage 3: Qwen3-VL 8B → phân đoạn dòng trong text block
Stage 4: Gemini Flash → phiên âm
```

**Hạn chế đã biết:**
- Bbox có thể trôi trên văn bản dày đặc
- Box axis-aligned có thể cắt dòng nghiêng
- ~440 file archive chứa văn bản Ukraina/Nga hỗn hợp (1919–1935)

---

## 3. Hệ Thống Đánh Giá (Evaluation Metric)

### 3.1. Công thức Composite Score

$$
\text{Score} = 0.15 \cdot \text{Det-F1} + 0.05 \cdot \text{ClassAcc} + 0.30 \cdot (1 - \text{CER}) + 0.50 \cdot (1 - \text{PageCER})
$$

| Thành phần | Trọng số | Đo lường |
|---|---|---|
| **Detection F1** | 0.15 | Khớp bbox tại IoU ≥ 0.5 (không phân biệt type) |
| **Classification Accuracy** | 0.05 | Đúng `type` giữa các cặp đã khớp |
| **Region CER** | 0.30 | CER trung bình per-region trên các vùng scorable đã khớp |
| **Page CER** | 0.50 | CER toàn trang, sắp xếp top-to-bottom rồi left-to-right |

> **Insight quan trọng:** PageCER chiếm **50%** tổng điểm → hệ thống End-to-End VLM có lợi thế lớn vì đọc toàn bộ trang theo đúng thứ tự, giảm lỗi ghép nối giữa Detection và OCR.

### 3.2. Greedy IoU Matching

- Tính IoU giữa mọi cặp (GT, Pred)
- Chỉ giữ các cặp có IoU ≥ 0.5
- Sắp xếp theo IoU giảm dần, greedy matching (1-1)
- Các GT không khớp → False Negative; Pred không khớp → False Positive

### 3.3. Vùng không tính CER (Scorable check)

Các vùng sau được **loại khỏi CER** (chỉ dựa trên thuộc tính GT):
- `type` = `image` hoặc `graph`
- `language` = `other`
- `legibility` = `illegible`

### 3.4. Text Normalization

Normalizer được áp dụng **đồng nhất** cho cả GT và Prediction trước khi tính CER:

**Chung (mọi region type):**
- Ký tự Cyrillic/Latin giống nhau → thống nhất sang Cyrillic (`c` → `с`, `o` → `о`, ...)
- Các loại dấu gạch ngang → hyphen-minus
- Dấu ngoặc kép thống nhất
- Khoảng trắng thu gọn
- Strikethrough: `~~old~~{new}` → `new`, `~~x~~` → `x`

**Riêng cho formula/table:**
- Ký hiệu LaTeX → Unicode (`\pi` → `π`, `\cdot` → `·`, ...)
- Phân số: `\frac{a}{b}` ↔ `a/b`
- Căn: `\sqrt{169}` ↔ `√169`
- Vector: `\overrightarrow{AB}` ↔ `→AB`
- Sub/superscript: `x²` ↔ `x^2`, `y₃` ↔ `y_3`
- Nhân: `*`, `∗`, `⋅` → `·`
- Ma trận, định thức → PSV format
- Table: bỏ khoảng trắng quanh pipe

### 3.5. Định dạng Submission

```csv
image,regions
test_0001.jpg,"[{""bbox"":[50,100,850,130],""type"":""handwritten"",""text"":""Доброго ранку""}]"
test_0002.jpg,[]
```

- Cần có đủ 386 ảnh test; ảnh không có detection dùng `[]`
- Mỗi region: `bbox`, `type`, `text` (bắt buộc)
- Không cần dự đoán `language` hay `legibility`

---

## 4. Hướng Tiếp Cận Kỹ Thuật

### 4.1. Hướng 1: Pipeline Modular (M2 + M3)

**Kiến trúc:**
```
Ảnh → Detection (RT-DETR/YOLO) → Crop regions → Classification (ResNet)
                                              → OCR (TrOCR) → Text
```

**Ưu điểm:**
- Mỗi thành phần có thể tối ưu độc lập.
- Dễ debug từng bước.
- Tận dụng các model pretrained mạnh cho Detection (YOLOv8, RT-DETR) và OCR (TrOCR).

**Nhược điểm:**
- Lỗi cascade: sai ở bước detection sẽ dẫn đến OCR sai theo.
- Khó đạt điểm cao ở PageCER (trọng số 50%) vì cần ghép nối chính xác thứ tự.

### 4.2. Hướng 2: VLM End-to-End (M4)

**Kiến trúc:**
```
Ảnh → VLM (Qwen3-VL-8B + LoRA) → JSON [{bbox, type, text}, ...]
```

**Ưu điểm:**
- End-to-End, xuất trực tiếp dữ liệu dạng JSON.
- Tối ưu tốt cho PageCER vì VLM đọc văn bản từ trên xuống dưới một cách tự nhiên.
- Tận dụng khả năng suy luận logic sẵn có của mô hình LLM lớn.

**Chiến lược Fine-tune:**
- Sử dụng Qwen3-VL-8B (ưu tiên 1) do tối ưu tốt VRAM (~6GB).
- Fine-tune bằng kỹ thuật LoRA và QLoRA 4-bit, bật `Gradient Checkpointing` chống OOM.
- Batch = 1, Accumulation = 8.

### 4.3. Hướng 3: Ensemble (M5)

Kết hợp cả 2 phương pháp trên ở mức Pipeline bằng cách sử dụng các kỹ thuật ghép nối:
- **WBF (Weighted Box Fusion):** Để kết hợp và tinh chỉnh dự đoán Bounding Box từ nhiều nguồn.
- **ROVER (Recognizer Output Voting Error Reduction):** Bầu chọn và ghép từ giữa các mô hình OCR khác nhau để giảm thiểu CER.

### 4.4. Kỹ thuật nâng cao dữ liệu (M1)
1. **Self-training:** Áp dụng Curriculum Learning bằng cách sử dụng nhãn từ tập `silver` sau đó fine-tune bằng tập `train` gán nhãn tay (gold).
2. **Pseudo-labeling:** Dùng văn bản có sẵn (từ National Dictation) gióng hàng với ảnh để sinh dữ liệu.
3. **Synthetic Data:** Render bằng TextRecognitionDataGenerator (TRDG) với font hỗ trợ ký tự Cyrillic tiếng Ukraina (Ґ, Є, І, Ї).

### 4.5. Hậu Xử Lý & Spell Check
- Sử dụng SymSpell (`max_edit_distance=2`) và từ điển 50k từ tiếng Ukraina cho các `handwritten` và `printed`.
- Tinh chỉnh Box cắt lẹm bằng cách padding (+2 pixels).
- Test cẩn thận Text Normalization của Kaggle trên local trước khi Submit.

---

## 5. Lộ Trình 5 Tuần

* **Tuần 1:** Nghiên cứu bài toán, EDA dữ liệu, đọc hiểu Pipeline. Dựng baseline cho Detection, OCR và VLM Zero-shot.
* **Tuần 2:** Nâng cấp Dataset bằng Augmentations, bắt đầu huấn luyện độc lập RT-DETR/YOLO, và Fine-tune Qwen3-VL bằng LoRA.
* **Tuần 3:** Pseudo-labeling từ tập Silver. Tạo chuyên biệt mô hình đọc LaTeX và Bảng. Tuning kỹ thuật prompt cho VLM.
* **Tuần 4:** Rèn giũa Error Analysis (Phân tích lỗi mô hình) và tích hợp Spell Checker siêu tốc. Tối ưu thời gian chạy Kaggle (< 9h).
* **Tuần 5:** Đóng gói weights, bật Ensembling WBF/ROVER với 2 luồng: 1 file an toàn (CV Local cao) + 1 file đột phá.

---

*Báo cáo được tổng hợp để trình bày lộ trình và kiến trúc giải pháp cho giáo viên hướng dẫn.*
