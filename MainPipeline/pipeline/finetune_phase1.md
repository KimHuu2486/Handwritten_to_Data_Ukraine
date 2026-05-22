# Phase 1 — Baseline an toàn

Phase 1 tạo baseline end-to-end sạch, ổn định và dễ debug cho pipeline OCR RUKOPYS. Mục tiêu của phase này không phải tối đa điểm ngay, mà là tạo một nền tảng vững để phân tích lỗi theo `source`, `type`, bbox, OCR text và reading order.

Phase này bám theo nguyên tắc:

```text
Silver dùng để dạy rộng và dạy format.
GT dùng để chỉnh chuẩn và học hành vi chính xác.
```

---

## 1. Phạm vi Phase 1

Phase 1 bật các thành phần tối thiểu:

| Stage | Trạng thái trong Phase 1 | Vai trò |
|---|---|---|
| Stage A | `page_layout_only` | Detect bbox/type trên full page |
| Stage B | Crop OCR theo `type` | Đọc text chính xác từng region |
| Stage C | **OFF — chưa bật** | Không risk gate, không routing/refine |
| Stage D | **OFF — chưa bật** | Không reread, không refinement |
| Stage E | Assembly + schema guardrail | Sort, enforce schema, xuất CSV |

Chưa bật trong Phase 1:

```text
text_draft
layout_with_text_draft
contextual OCR bằng ocr_context_light
weighted risk gate
rule risk
multi-view refinement
reread/refinement
VLM/LLM selector
draft-aware reranking
```

---

## 2. Input và Output tổng quát

### 2.1. Input của Phase 1

| Nhóm input | Nội dung | Mục đích |
|---|---|---|
| Silver metadata | `silver/metadata.jsonl` với bbox/type/text auto | Stage 1 warm-up task format, layout, OCR |
| Silver images | `silver/images/**` | Full-page layout và crop OCR |
| GT metadata | `train/metadata.jsonl` human-annotated | Stage 2 chỉnh chuẩn theo nhãn thật |
| GT images | `train/images/**` | Fine-tune layout/OCR chất lượng cao |
| Test/validation manifest | Frozen validation manifest v1 | Đánh giá ổn định và so sánh ablation |
| Source metadata | `source`: `dictation`, `archive`, `school`, `university` | Source hint cho Stage A/B |

### 2.2. Output của Phase 1

| Output | Dạng | Mục đích |
|---|---|---|
| `ckpt_phase1_silver` | Model/checkpoint sau silver warm-up | Nền model đã học task và format |
| `ckpt_phase1_gt_baseline` | Model/checkpoint sau GT fine-tune | Baseline chính để inference/so sánh |
| Validation predictions | CSV/JSONL | Phân tích bbox/type/text trên validation |
| Score summary | JSON/CSV/log | Detection F1, ClassAcc, RegionCER, PageCER, total score |
| Error report | bảng lỗi theo source/type | Chọn hard cases cho Phase 2 |
| Final submission candidate | CSV theo Kaggle contract | Submission baseline nếu cần |

---

## 3. Task A1 — Full-page Layout Proposal

Task A1 dạy Stage A detect vùng và phân loại `type`. Trong Phase 1, Stage A chỉ dùng `page_layout_only`.

### Input

```text
full page image
+ source metadata
+ source-aware layout prompt
```

### Output

```json
[
  {"bbox":[x1,y1,x2,y2],"type":"handwritten"},
  {"bbox":[x1,y1,x2,y2],"type":"printed"},
  {"bbox":[x1,y1,x2,y2],"type":"formula"}
]
```

Contract output:

| Field | Yêu cầu |
|---|---|
| `bbox` | `[x1, y1, x2, y2]`, 0-1000 grid trong prompt/model output |
| `type` | Một trong 7 class: `handwritten`, `printed`, `formula`, `table`, `annotation`, `image`, `graph` |
| `text` | Không có trong output Stage A Phase 1 |
| `text_draft` | Không dùng |

### Rule lọc silver cho A1

Silver layout nên được giữ nếu:

```text
- bbox hợp lệ, không inverted, không degenerate
- bbox nằm trong ảnh hoặc clamp được về ảnh
- type thuộc schema 7 class
- handwritten/printed/formula/annotation ở mức line hoặc standalone item
- table là full table, không split cell
- image/graph không bị ép thành text region
- reading order tương đối hợp lý
```

Silver layout nên loại hoặc hạ trọng số nếu:

```text
- merge nhiều dòng thành một bbox lớn
- hallucinate cột bbox nhỏ hàng loạt
- bbox trùng lặp nhiều với cùng text/type
- class `image`/`graph` bị gán nhầm sang `handwritten`/`formula`
```

---

## 4. Task B1 — Crop OCR cơ bản

Task B1 dạy Stage B đọc chính xác từng region crop theo `type`.

### Input

```text
crop image
+ type
+ short source hint
+ type-specific OCR instruction
```

### Output

```text
exact transcription
```

Với structural types:

```text
type=image -> output=""
type=graph -> output=""
```

### OCR contract theo type

| Type | Output mong muốn | Không được làm |
|---|---|---|
| `handwritten` | Exact visible handwriting | Không hoàn thành từ ngữ cảnh/canonical text |
| `printed` | Exact printed/typewritten text | Không hiện đại hóa/chỉnh chính tả |
| `formula` | Công thức như ảnh, LaTeX nếu phù hợp | Không solve, simplify, normalize |
| `table` | Pipe-separated rows/cells | Không suy diễn cell thiếu, không rebalance |
| `annotation` | Teacher mark/note ngắn | Không diễn giải ý nghĩa |
| `image` | Chuỗi rỗng | Không OCR label/diễn giải hình |
| `graph` | Chuỗi rỗng | Không OCR axis/label trong graph |

Prompt assembly gợi ý:

```text
{short_source_hint}
{type_specific_instruction}
Use [illegible] only for unreadable words inside an otherwise legible text region.
Use ~~word~~ for visible strikethrough and ~~old~~{new} for visible correction.
Do not explain.
```

Stage B Phase 1 không được:

```text
- detect layout lại
- sửa bbox
- suy diễn nội dung từ full page
- hoàn thành text từ canonical dictation
- hiện đại hóa chính tả archive
- giải công thức
```

---

## 5. Stage 1 Training trên Silver

Phase 1 nên train trên silver theo curriculum:

```text
Step 1: train B1 crop OCR trước
Step 2: train A1 layout-only
Step 3: mix nhẹ A1 + B1 để giữ cả OCR và format layout
```

Tỉ lệ task gợi ý:

| Task | Tỉ lệ | Lý do |
|---|---:|---|
| B1 crop OCR cơ bản | 60-70% | RegionCER/PageCER phụ thuộc mạnh vào OCR |
| A1 layout-only | 30-40% | Bbox/type/reading order vẫn quyết định PageCER |

### Sampling silver

Nên oversample:

```text
formula
table
annotation
archive
crop nhỏ
crop mờ
crop dài
image/graph hard negative
```

Đặc biệt giữ đủ mẫu `image` và `graph` để model học trả chuỗi rỗng, thay vì cố OCR mọi thứ.

---

## 6. Stage 2 Fine-tune trên GT

Sau silver warm-up, fine-tune trên GT với learning rate thấp hơn. GT là nơi chỉnh hành vi theo nhãn thật và metric thật.

Tỉ lệ task gợi ý:

| Task | Tỉ lệ | Mục tiêu |
|---|---:|---|
| B1 crop OCR cơ bản | 65% | Exact transcription, marker, hard type text |
| A1 layout-only | 35% | Bbox/type chuẩn hơn, giảm merge/split sai |

GT fine-tune tập trung vào:

```text
- bbox/type chuẩn cho Stage A
- exact transcription cho Stage B
- marker đặc biệt: ~~word~~, ~~old~~{new}, [illegible]
- policy image/graph = ""
- archive spelling: preserve old spelling, không modernize
- formula/table: giữ format, không solve/summarize
```

Không đưa synthetic context, risk gate, reread hoặc refinement vào Phase 1, để baseline đủ sạch và dễ quy lỗi.

---

## 7. Inference Flow Phase 1

```text
Input page
  ↓
Stage A layout-only
  ↓
crop từng region theo bbox Stage A
  ↓
Stage B crop OCR
  ↓
Stage E assembly + schema guardrail
  ↓
final CSV
```

### Stage E Phase 1

Stage E cần enforce:

```text
if type in {"image", "graph"}:
    text = ""
```

Stage E cần preserve:

```text
~~word~~
~~old~~{new}
[illegible]
```

Stage E không được:

```text
- tự sửa chính tả
- tự hoàn thành câu
- hiện đại hóa archive
- giải công thức
- thêm text không được crop hỗ trợ
```

---

## 8. Validation Gate

Phase 1 chỉ được coi là đạt nếu:

| Gate | Kỳ vọng |
|---|---|
| Schema | CSV/JSON parse ổn định, mỗi row có `regions` hợp lệ |
| Bbox | Không có degenerate/out-of-image nghiêm trọng sau normalize/clamp |
| Type | Chỉ sinh 7 class hợp lệ |
| Structural types | `image`/`graph` luôn text rỗng |
| Runtime | Chạy được full validation hoặc manifest v1 với log rõ |
| Metric | Có score summary đủ Detection F1, ClassAcc, RegionCER, PageCER |
| Debuggability | Có lỗi phân tích được theo source/type/page/region |

Output mong muốn:

```text
- Detection F1 ổn
- RegionCER chấp nhận được
- PageCER chưa tối ưu nhưng không vỡ reading order
- JSON/schema ổn định
- lỗi đủ rõ để hard-mine cho Phase 2
```

---

## 9. Artifact cần lưu

| Artifact | Tên gợi ý | Ghi chú |
|---|---|---|
| Silver checkpoint | `ckpt_phase1_silver` | Không commit model nặng vào repo |
| GT checkpoint | `ckpt_phase1_gt_baseline` | Baseline chính cho Phase 2 |
| Validation predictions | `phase1_validation_predictions.csv` | Theo Kaggle-like contract |
| Raw outputs | `phase1_validation_raw_outputs.jsonl` | Debug JSON/parser/runaway |
| Score summary | `phase1_score_summary.json` | Lock metric |
| Error analysis | `phase1_error_breakdown.csv` | Theo source/type/failure mode |

---

## 10. Quyết định mặc định của Phase 1

```text
Default:
  Stage A = page_layout_only
  Stage B = crop OCR theo type
  Stage C = OFF, chưa bật risk gate
  Stage D = OFF, chưa bật reread/refinement
  Stage E = schema + reading-order assembly

Không dùng:
  text_draft
  ocr_context_light trong first-pass
  weighted risk gate
  rule risk đơn giản
  reread/refinement
  multi-view full refinement
```

Phase 1 hoàn thành khi có baseline sạch đủ tin cậy để Phase 2 bắt đầu thêm contextual OCR, risk gate và high-risk refinement.
