# Pipeline:

```text
Input Page
   ↓
Stage A — Coarse Layout Proposal
   ↓
Stage A+ — Deterministic Page Context Light
   ↓
Stage B — Region OCR Pass
   ↓
Stage C — Risk Estimation / Uncertainty Gate
   ↓
┌─────── Low-risk ───────────┐
│                            ↓
│                        Accept OCR text
│
└─ ───── High-risk ─────────┐
                            ↓
Stage D — Interactive Refinement: crop / zoom / context / reread
      ↓
Stage E — Page-level Assembly & Metric-aware Postprocess
      ↓
Final submission JSON
```

---

# 1. Stage A — **Coarse Layout Proposal**, tách detection khỏi transcription

## Mục tiêu
Full-page model **không chịu trách nhiệm đọc text chính xác nữa**.  
Nó nên tập trung vào 1 đầu ra chính:

1. `regions`: bbox/type ở mức vùng.

Sau đó code sẽ build `page_context_light` deterministic từ `regions`, thay vì bắt model sinh context phức tạp không có ground truth trực tiếp.

`regions` nên có dạng:

```json
[
  {"bbox":[...], "type":"handwritten"},
  {"bbox":[...], "type":"printed"},
  {"bbox":[...], "type":"formula"}
]
```

`text_draft` là feature optional cho ablation, **không phải mặc định bắt buộc** và không dùng làm text cuối cùng.

---

## Vì sao nên tách?

Trong RUKOPYS, dữ liệu có:
- handwriting,
- printed text,
- formula,
- table,
- annotation,
- image,
- graph,  
với domain rất khác nhau: archive, dictation, school, university.

Nếu full-page pass vừa detect vừa đọc, mô hình dễ:
- bỏ sót line nhỏ,
- merge hai dòng thành một,
- bịa text dài để “lấp chỗ trống”,
- lệch reading order.

Trong khi metric có:
- **Detection F1**
- **Page CER** rất nặng  
nên **bbox thiếu hoặc reading order sai sẽ gây hại trực tiếp tới PageCER**, dù text từng crop có tốt.

---

## Prompt Stage A theo `source`

Stage A là nơi dùng `source`, vì đây là tầng nhìn toàn trang để hiểu domain, phát hiện vùng và phân loại `type`.  
`source` **không nên là prompt chính của crop OCR**, vì crop đã có bbox/type và chỉ cần đọc đúng nội dung vùng.

Stage A nên có **rule chung cho mọi source** trước, sau đó mới thêm context riêng theo `source`.

### Rule chung cho Stage A

```text
Extract layout regions from the full page.
Return only a compact JSON array.
For page_layout_only, each item must have keys bbox,type.
For page_layout_with_text_draft, each item must have keys bbox,type,text_draft.
bbox is [x1,y1,x2,y2] on a 0-1000 grid.
type is one of handwritten, printed, formula, table, annotation, image, graph.

Prioritize bbox and type accuracy over transcription.
text_draft is optional and only used in page_layout_with_text_draft ablation. If included, keep it short and visually grounded. Use "" if unsure.

Granularity rules:
- handwritten, printed, formula, annotation: one bbox per visual line or standalone item.
- table: one bbox for the full table, not one bbox per cell.
- image: one bbox for drawings, stamps, seals, illustrations, or non-text figures.
- graph: one bbox for charts, coordinate plots, axes-based plots, or data visualizations.

Do not merge separate text lines.
Do not split a single visual table into cells.
Do not force drawings, diagrams, or graphs into handwritten/printed/formula.
Detect small but meaningful regions such as numbering, teacher marks, dates, grades, and short annotations.
Preserve reading order.
```

### Type decision rules

| Visual content | Type |
|---|---|
| Handwritten prose, answers, notes, sentences | `handwritten` |
| Printed/typewritten headers, textbook fragments, forms, labels | `printed` |
| Standalone math, logic, vector, matrix, chemistry, symbolic expression | `formula` |
| Full tabular structure with rows/columns | `table` |
| Teacher marks, grades, corrections, short labels, dates, numbering when standalone | `annotation` |
| Drawing, stamp, seal, illustration, diagram, scientific figure without axes | `image` |
| Coordinate plane, chart, plotted graph, axes-based visualization | `graph` |

README cho thấy 4 domain khác nhau:
- `dictation`: phone photo, handwriting prose
- `archive`: 1919–1935, nét mực cũ, có chính tả cổ
- `university`: exam, formula, chemistry, tables
- `school`: notebook, bài tập, teacher marks

Vì test metadata có `source`, Stage A nên dùng prompt theo source để phát hiện vùng và phân loại tốt hơn.

### Dictation
> "This is a Ukrainian national dictation page, usually prose handwriting captured by phone. Detect all visible document regions in reading order. Return bbox and type for each region. Do not infer missing lines from the canonical dictation text. Keep text_draft short and visually grounded if included."

### Archive
> "This is an archival document from 1919-1935. It may contain handwriting, typewritten or printed headers, stamps, old Cyrillic/Ukrainian orthography, and dense administrative layout. Detect all meaningful text and non-text regions. Classify handwritten vs printed carefully. Do not modernize or complete text_draft from context."

### School
> "This is a school homework page. It may contain handwritten answers, printed fragments, formulas, tables, teacher annotations, drawings, diagrams, coordinate plots, and charts. Detect all meaningful regions. Classify drawings/illustrations as image, and axes-based plots/charts as graph. Do not force non-text visuals into handwritten, printed, or formula."

### University
> "This is a university exam or coursework page. It may contain handwritten text, printed text, mathematical or chemical formulas, tables, diagrams, plotted graphs, coordinate charts, and scientific figures. Detect all meaningful regions. Classify standalone equations or chemistry notation as formula, tabular structures as table, plotted axes/charts as graph, and diagrams/figures as image."

Stage A có 2 chế độ output để train/infer:

### `page_layout_only`

Output chỉ có bbox/type:

```json
[
  {"bbox":[x1,y1,x2,y2],"type":"handwritten"},
  {"bbox":[x1,y1,x2,y2],"type":"formula"}
]
```

Đây là chế độ nên ưu tiên ở inference nếu mục tiêu là bbox/type ổn định nhất.

Đây cũng là **default competition trước ablation**, vì nó giảm nguy cơ transcription làm nhiễu layout detection.

### `page_layout_with_text_draft`

Output có thêm `text_draft` ngắn:

```json
[
  {"bbox":[x1,y1,x2,y2],"type":"handwritten","text_draft":"optional short draft"},
  {"bbox":[x1,y1,x2,y2],"type":"formula","text_draft":"optional short draft"}
]
```

`text_draft` chỉ dùng làm tín hiệu phụ cho risk/rerank, **không phải text cuối cùng**. Nếu model không chắc, `text_draft` nên là chuỗi rỗng thay vì đoán.

Chế độ này **không nên coi là default** cho đến khi validation chứng minh:
- Detection F1 không giảm đáng kể so với `page_layout_only`.
- Final score tăng nhờ risk/rerank dùng `text_draft`.
- Runtime và JSON stability vẫn chấp nhận được.

### Ablation bắt buộc cho Stage A

Phải so sánh ít nhất 2 pipeline:

| Ablation | Stage A output | Stage C/D dùng `text_draft`? | Mục tiêu |
|---|---|---:|---|
| A — layout-only | `bbox,type` | Không | Baseline an toàn cho bbox/type |
| B — layout+draft | `bbox,type,text_draft` | Có, nếu draft hợp lệ | Kiểm tra draft có cải thiện risk/rerank không |

Nếu B làm Detection F1 giảm hoặc JSON/page layout kém ổn định, giữ A làm default.

---

## Stage A+ — `page_context_light` deterministic

Không nên train model sinh `page_context` đầy đủ, vì:
- không có ground truth trực tiếp để supervised ổn định;
- context do model hallucinate có thể kéo Stage B/D OCR sai;
- complexity tăng mạnh và khó debug.

Thay vào đó, sau Stage A, code build `page_context_light` deterministic từ output `regions`.

### Mục tiêu của `page_context_light`

`page_context_light` là metadata rẻ, ổn định, dễ debug:
- source/domain của trang;
- region id, bbox, type;
- vị trí trong reading order;
- type của vùng trước/sau;
- nearby region ids theo geometry;
- quan hệ gần table/graph/formula theo bbox;
- optional `text_draft` nếu Stage A có sinh, nhưng chỉ dùng cho risk/rerank, không đưa vào Stage B OCR mặc định.

Nó không phải OCR toàn trang và không chứa kết luận semantic phức tạp.

### Schema đề xuất

```json
{
  "source": "school",
  "region_id": "r012",
  "type": "formula",
  "bbox": [120, 430, 780, 510],
  "reading_order_position": 12,
  "num_regions": 38,
  "previous_region_id": "r011",
  "previous_region_type": "handwritten",
  "next_region_id": "r013",
  "next_region_type": "handwritten",
  "nearby_region_ids": ["r011", "r013"],
  "same_line_region_ids": [],
  "parent_table_id": null,
  "near_table_id": "t001",
  "near_graph_id": null,
  "near_formula_ids": ["r010"],
  "bbox_size_percentile": 0.22,
  "line_height_ratio": 0.73,
  "table_overlap": 0.0,
  "graph_proximity": 0.0,
  "text_draft": "optional short visual draft, only if ablation B enables it"
}
```

### Cách build bằng rule geometry

- Sort regions theo `(y1, x1)` để gán `reading_order_position`.
- Gán `region_id` ổn định theo thứ tự đọc: `r001`, `r002`, ...
- `previous/next`: lấy region liền trước/sau trong reading order.
- `nearby_region_ids`: các bbox có khoảng cách tâm gần hoặc overlap theo trục y/x.
- `same_line_region_ids`: các bbox có y-center gần nhau và height tương đồng.
- `near_table_id`: table bbox gần hoặc overlap vùng hiện tại.
- `parent_table_id`: chỉ dùng nếu region nằm trong bbox table; mặc định nhiều pipeline sẽ không split cell nên thường là null.
- `near_graph_id`: graph bbox gần hoặc overlap.
- `bbox_size_percentile`, `line_height_ratio`: tính theo thống kê page.
- `text_draft`: copy từ Stage A nếu ablation B có bật, nhưng chỉ dùng cho risk/rerank sau OCR; không đưa vào Stage B first-pass prompt mặc định.

### Hai loại excerpt

Để tránh anchor OCR vào draft sai, tách `page_context_light` thành 2 excerpt:

| Excerpt | Dùng ở đâu | Có `text_draft`? | Mục tiêu |
|---|---|---:|---|
| `ocr_context_light` | Stage B first-pass OCR | Không | Cung cấp geometry/context mà không làm model bị neo vào draft sai |
| `risk_rerank_context_light` | Stage C risk score, Stage D rerank/selector | Có, nếu ablation B bật | Dùng draft như tín hiệu phụ sau khi đã có OCR candidate |

### Cách dùng

```text
Stage B default:
  crop image
  + source hint
  + type prompt
  + ocr_context_light excerpt
  → OCR text

Stage D high-risk:
  crop/zoom/expanded crop
  + risk_rerank_context_light excerpt
  + optional page thumbnail only if visual fallback is needed
  → refined OCR text
```

### Guardrail

`page_context_light` phải luôn đi kèm quy tắc:

```text
Use ocr_context_light only for layout, neighboring region types, table/graph proximity, and local orientation during first-pass OCR.
The final OCR text must be supported by the crop or expanded crop.
Do not complete missing words from page_context_light, optional text_draft, language prior, or canonical dictation text.
```

Như vậy context được code tạo ra sau Stage A: ổn định, kiểm soát được, dễ ablation, còn full-page thumbnail chỉ là fallback thị giác cho vùng thật sự khó.

---

# 2. Stage B — **Region OCR Reader**, tiếp tục dùng crop OCR nhưng mạnh hơn

Stage B kế thừa phần tốt nhất từ baseline hiện tại:  
**crop từng vùng → OCR text chính xác.**

Nhưng tôi đề xuất nâng cấp Stage B theo 2 hướng chính:
- prompt crop OCR theo `type`, chỉ kèm `source hint` ngắn;
- train thêm `contextual crop OCR`.

---

## 2.1. Prompt crop OCR theo `type` + source hint ngắn

Sau Stage A, mỗi crop đã có `type`, nên prompt chính phải theo `type`. `source` chỉ nên là hint ngắn để tránh model chuẩn hóa sai domain.

### Source hint ngắn

| Source | Hint dùng trong crop OCR |
|---|---|
| `dictation` | "Ukrainian dictation handwriting. Do not complete from canonical text; read only visible characters." |
| `archive` | "Historical Ukrainian/Cyrillic document. Preserve old spelling; do not modernize." |
| `school` | "School homework. It may contain corrections, teacher marks, formulas, and mixed handwriting/print." |
| `university` | "University exam/coursework. It may contain formulas, tables, chemistry notation, and technical symbols." |

### Prompt theo type

| Type | Crop OCR instruction |
|---|---|
| `handwritten` / `printed` | "Transcribe the visible text exactly. Preserve punctuation, line content, corrections, and strikethrough markers. Return only text." |
| `formula` | "Read this standalone math, logic, vector, matrix, determinant, set/relation, statistics, physics, or chemistry expression exactly as written. Return only formula text, using LaTeX when it is the clearest representation and plain Unicode when it better matches the handwriting. Preserve visible symbols, indices, superscripts, subscripts, arrows, fractions, matrix/determinant structure, punctuation, numbering, and strikethrough/correction markers. Do not solve, simplify, normalize, explain, or convert old notation into a different style." |
| `table` | "Read this table region exactly. Return only pipe-separated table text. Use one output line per visual row and `\|` between cells. Preserve empty cells with empty fields, e.g. `A\|\|C`. Preserve row order, column order, multi-word cell text, wrapped cell text, numbers, units, punctuation, dashes, and visible spelling mistakes. Do not infer missing cells, do not rebalance columns, do not summarize, and do not explain." |
| `annotation` | "Read this short annotation or teacher mark. Return only the exact visible text." |
| `image` / `graph` | "Return an empty string." |

Prompt cuối nên được ghép theo mẫu:

```text
{short_source_hint}
{type_specific_instruction}
Use [illegible] only for unreadable words inside an otherwise legible text region.
Use ~~word~~ for visible strikethrough and ~~old~~{new} for visible correction.
Do not explain.
```

Điều này có khả năng cải thiện đáng kể các vùng dễ bị model “chuẩn hóa ngôn ngữ” sai, đồng thời tránh việc crop OCR bị phân tâm bởi nhiệm vụ layout.

Với `formula` và `table`, nên xem đây là hard types mặc định:
- oversample trong Stage 1/2 training vì số lượng ít hơn handwriting nhưng ảnh hưởng CER lớn;
- đưa vào Risk Gate high-risk nếu crop nhỏ, dài, nhiều ký hiệu, nhiều hàng/cột, hoặc OCR pass 1 sinh text quá ngắn;
- ưu tiên Stage D multi-view refinement thay vì chỉ đọc một crop duy nhất.

---

## 2.2. Train thêm `contextual crop OCR`

### Task B2 — `crop_with_page_context`
Input:
1. `ocr_context_light` excerpt built by code after Stage A
2. crop region
3. optional expanded crop or page thumbnail for high-risk examples

Output:
```text
exact transcription
```

Lợi ích:
- Một dòng đơn lẻ đôi khi khó đọc nếu tách khỏi câu trước/sau.
- `ocr_context_light` giúp định vị dòng trước/sau, type lân cận, table/graph proximity, ký hiệu trong bảng, đoạn archive.
- Page thumbnail chỉ dùng như visual fallback trong hard examples, không phải input mặc định cho mọi crop.
- Không đưa `text_draft` vào Stage B first-pass OCR prompt mặc định để tránh anchor vào draft sai.

Ý tưởng này tương ứng với:
- **Doc-V\***: giữ global overview + fine detail.
- **CogCoM**: multi-image, multi-turn reasoning giúp tận dụng ảnh gốc và ảnh crop cùng lúc.

---

## 2.3. Cách dùng `ocr_context_light` trong Stage B

Stage B nên dùng `ocr_context_light` deterministic làm context mặc định. Full-page thumbnail không nên truyền lại cho mọi crop, và `text_draft` cũng không nên đưa vào first-pass OCR prompt mặc định.

### Stage B first pass

Mặc định nên chạy:

```text
crop image
+ source hint
+ type-specific prompt
+ ocr_context_light excerpt
→ first-pass OCR text
```

`ocr_context_light excerpt` chỉ nên chứa thông tin geometry/context liên quan đến region hiện tại:
- source;
- region id, bbox, type;
- previous/next region ids;
- previous/next region types;
- nearby region ids;
- same-line region ids;
- table overlap / near_table_id;
- graph proximity / near_graph_id;
- nearby formula ids nếu công thức nối tiếp dòng trước/sau;
- bbox size percentile và line height ratio.

Với các vùng có nguy cơ cao ngay từ đầu, có thể dùng thêm visual context:

```text
crop image
+ expanded crop
+ optional page thumbnail
+ bbox/type/source metadata
+ ocr_context_light excerpt
+ type-specific prompt
→ contextual OCR text
```

Các vùng nên ưu tiên contextual OCR:
- `source=archive`
- `type=formula` hoặc `type=table`
- bbox quá nhỏ, quá dài, hoặc crop bị mờ/thiếu contrast
- dòng có khả năng bị cắt mất chữ đầu/cuối
- vùng trong bảng cần header/row label để hiểu đúng
- vùng công thức nối tiếp dòng trước/sau

### Guardrail bắt buộc

Prompt contextual OCR phải nói rõ:

```text
Use ocr_context_light only for layout, local geometry, neighboring region types, table/graph proximity, and surrounding region ids.
Use the page thumbnail only as visual fallback when provided.
The final transcription must be supported by the crop or expanded crop.
Do not complete missing words from ocr_context_light, text_draft, full page, language prior, or canonical dictation text.
If a character or word is not visible in the crop/context views, do not invent it.
```

Như vậy page context giúp giải mã vùng khó, nhưng không biến OCR thành bài toán suy diễn nội dung.

---

# 3. Stage C — **Uncertainty Gate / Risk Estimator**

## Mục tiêu
Không xử lý mọi crop giống nhau.

Sau lần OCR đầu, mỗi region được gán `risk_score` liên tục thay vì chỉ dùng rule nhị phân:

```text
LOW-RISK  → giữ nguyên text
HIGH-RISK → đưa vào Stage D để đọc lại có chủ đích
```

---

## Heuristic risk score

Trước khi huấn luyện HALP probe thật, nên làm **Risk Gate** bằng weighted heuristic score.

Không nên dùng OR-rule kiểu “chỉ cần dính một điều kiện là high-risk”, vì dễ refine quá nhiều vùng và làm inference nặng mà gain không tương xứng.

### Công thức gợi ý

```text
risk_score =
  w1  * small_bbox
+ w2  * low_line_height
+ w3  * blur
+ w4  * low_contrast
+ w5  * hard_type
+ w6  * archive_source
+ w7  * text_too_short
+ w8  * empty_text_for_scorable
+ w9  * suspicious_chars
+ w10 * draft_crop_disagreement   # only if ablation B enables text_draft
+ w11 * type_visual_mismatch
+ w12 * table_or_formula_long_region
```

Feature có thể chuẩn hóa về `[0, 1]`:
- `small_bbox`: bbox có diện tích nhỏ hoặc width/height bất thường.
- `low_line_height`: chiều cao dòng thấp so với median page line height.
- `blur`: crop có sharpness thấp.
- `low_contrast`: crop có contrast thấp.
- `hard_type`: `formula`, `table`, hoặc `annotation` khó.
- `archive_source`: tài liệu archive có orthography cũ, form dày, mixed handwriting/printed.
- `text_too_short`: OCR pass 1 quá ngắn so với bbox, hoặc so với `text_draft` nếu ablation B bật.
- `empty_text_for_scorable`: OCR rỗng nhưng type là scorable.
- `suspicious_chars`: nhiều `?`, replacement char, ký tự lạ, lặp ký tự/ngram.
- `draft_crop_disagreement`: chỉ dùng khi ablation B bật; `text_draft` và crop OCR khác nhau lớn sau normalization.
- `type_visual_mismatch`: type dự đoán mâu thuẫn với visual appearance hoặc geometry trong `page_context_light`.
- `table_or_formula_long_region`: công thức/bảng dài, nhiều hàng/cột/ký hiệu.

### Quyết định refine

Refine region nếu:

```text
risk_score >= threshold
OR region nằm trong top-K% risk_score cao nhất của page
OR hard override được kích hoạt
```

Hard override nên ít nhưng chắc:
- OCR rỗng cho scorable region.
- `formula`/`table` sinh output quá ngắn hoặc sai format rõ ràng.
- bbox hợp lệ nhưng crop OCR trả JSON/explanation thay vì text.
- vùng có `text_draft` đáng tin nhưng crop OCR rỗng hoặc lệch hoàn toàn; chỉ áp dụng khi ablation B bật.

### Budget cap

Để kiểm soát compute, mỗi page nên có cap:

```text
max_refine_regions_per_page = min(ceil(num_regions * top_k_percent), hard_cap)
```

Gợi ý ban đầu để ablation:
- `threshold = 0.55`
- `top_k_percent = 0.20`
- `hard_cap = 12 regions/page`

Các giá trị này phải tune trên validation bằng trade-off:

```text
score gain vs runtime cost
PageCER gain vs number of refined regions
formula/table CER gain vs over-refinement
```

---

# 4. Stage D — **Interactive Refinement cho vùng high-risk**

Đây là phần hấp thụ trực tiếp từ:
- **VLM-R³**
- **CogCoM**
- một phần **Doc-V\***

## Mục tiêu
Khi crop khó, không chỉ đọc lại y hệt.  
Hệ thống phải **nhìn lại có chiến lược**.

---

## 4.1. Stage D v1 — Multi-view + page-context refinement, dễ triển khai

Với mỗi region high-risk, dùng `risk_rerank_context_light` excerpt mặc định và tạo tối đa 4 visual view:

### View 1 — Original crop
Crop bbox với padding hiện tại.

### View 2 — Zoomed crop
Phóng to theo area ratio.  
VLM-R³ cho thấy crop/zoom động giúp xử lý vùng chi tiết khó; quan trọng hơn, paper chứng minh interleaved visual evidence giúp model giữ attention vào vùng cần đọc thay vì “suy đoán bằng ngôn ngữ”.

### View 3 — Expanded context crop
Mở bbox ra:
- +10–20% ngang
- +10–15% dọc

để model thấy từ trước/sau, đặc biệt với:
- dictation,
- archive prose,
- handwritten continuous strokes.

### View 4 — Page thumbnail
Chỉ thêm ảnh toàn trang ở độ phân giải thấp khi `risk_rerank_context_light` + crop/expanded crop vẫn chưa đủ. Page thumbnail giúp giữ visual context:
- vị trí vùng trên trang,
- dòng trước/sau,
- header/cột/hàng trong bảng,
- công thức hoặc bài giải nối tiếp,
- nguồn ngữ cảnh để phân biệt text thật với annotation/figure/graph.

Page thumbnail là **visual fallback**, không dùng để thay thế crop và không nên là input mặc định cho mọi high-risk region.

---

## 4.2. Cách infer

Cho model 2-4 lượt OCR, tùy budget inference:

```text
Candidate 1: original crop
Candidate 2: zoom crop
Candidate 3: context crop
Candidate 4: risk_rerank_context_light excerpt + optional page thumbnail + crop/context crop
```

Sau đó chọn text cuối bằng:

### Rule-based reranking
- ưu tiên output hợp lệ theo type,
- ít ký tự lạ hơn,
- gần `text_draft` hơn nếu ablation B bật và draft có vẻ ổn,
- gần canonical/retrieved lexicon hơn nếu thuộc dictation,
- tránh output quá ngắn/ quá dài bất thường.

### Hoặc LLM/VLM selector nhẹ
Input:
- 3 candidate text
- crop image
- expanded crop
- risk_rerank_context_light excerpt
- optional page thumbnail
- yêu cầu: “Select the exact transcription best grounded in the image.”

---

## 4.3. Prompt refinement theo `type`

Stage D không quay lại prompt theo `source` đầy đủ. Nó dùng lại logic của Stage B:
- prompt chính theo `type`;
- source hint ngắn nếu domain có rủi ro đặc biệt;
- thêm yêu cầu so sánh nhiều view/candidate và chọn output được ảnh hỗ trợ tốt nhất.

Mẫu prompt refinement:

```text
{short_source_hint}
This region was marked high-risk.
You are given one or more views of the same region, optionally including a full-page thumbnail.
Use risk_rerank_context_light for layout, local geometry, neighboring region types, table/graph proximity, and local orientation.
Use the page thumbnail only as visual fallback when provided.
The final transcription must be supported by the crop or expanded crop.
Do not complete missing words from risk_rerank_context_light, optional text_draft, full page, language prior, or canonical dictation text.
{type_specific_instruction}
Choose the transcription best supported by the image views.
Return only the final text.
```

Với `image` và `graph`, Stage D không cần OCR lại; text cuối vẫn là chuỗi rỗng.

---

# 5. Stage E — **Page-level Assembly & Metric-aware Postprocess**

Notebook metric của bạn cho thấy score phụ thuộc rất mạnh vào:

- `PageCER`: **0.50**
- `Region CER`: **0.30**
- `Detection F1`: **0.15**
- `ClassAcc`: **0.05**

Vì vậy Stage E phải được xem là một **module chiến lược**, không chỉ là serialization.

---

### E1. Reading order chuẩn
Sort:
```python
(y1, x1)
```
nhưng cần thêm heuristic:
- dòng cùng hàng thì x-order,
- table giữ cấu trúc riêng,
- annotation không nên chen sai vị trí vào page text nếu dễ phá PageCER.

### E2. Deduplicate regions
Nếu Stage A sinh overlap:
- IoU lớn,
- text gần giống,
- cùng type,  
giữ box tốt nhất.

### E3. Không cho `image`, `graph` có text
README schema quy định structural types này phải có text rỗng.

### E4. Preserve special markers
Dataset dùng:
- `~~word~~`
- `~~old~~{new}`
- `[illegible]`

Nếu không train/không postprocess kỹ, model sẽ hay làm mất marker, gây CER.

---
