# Kế hoạch triển khai train và finetune pipeline OCR

## 0. Mục tiêu tổng quát

Kế hoạch này đề xuất quá trình triển khai train và finetune cho pipeline OCR gồm 5 stage:

```text
Input Page
   ↓
Stage A — Layout Proposal + Deterministic Page Context Light
   ↓
Stage B — Region OCR Pass
   ↓
Stage C — Risk Estimation / Uncertainty Gate
   ↓
Stage D — Interactive Refinement: crop / zoom / context / reread
   ↓
Stage E — Page-level Assembly & Metric-aware Postprocess
   ↓
Final submission CSV
```

Chiến lược chính:

- **Stage 1 trên silver data**: dùng để dạy model biết task, format output, layout/OCR cơ bản, type-specific behavior.
- **Stage 2 trên GT data**: dùng để chỉnh chuẩn theo metric thật, sửa lỗi bbox/type/text, giảm hallucination, tối ưu PageCER/RegionCER.
- **3 phase triển khai**:
  1. Phase 1 — Baseline an toàn.
  2. Phase 2 — Contextual OCR + Risk Gate.
  3. Phase 3 — Final với draft ablation, multi-view refinement và metric-aware tuning.

Nguyên tắc quan trọng:

```text
Silver dùng để dạy rộng và dạy format.
GT dùng để chỉnh chuẩn và học hard cases.
```

---

# 1. Tổng quan chiến lược train

Không nên train thẳng theo kiểu:

```text
silver full pipeline → GT full pipeline
```

Cách này dễ làm model học nhiễu từ silver, đặc biệt ở các lỗi:

- bbox lệch;
- type sai;
- merge nhiều dòng;
- hallucinate text;
- sai reading order;
- sinh text cho `image` hoặc `graph`.

Thay vào đó, nên tách thành các task nhỏ:

```text
Stage 1 silver:
  học format
  học layout proposal
  học OCR crop
  học type-specific behavior
  học hard type policy

Stage 2 GT:
  sửa chuẩn bbox/type/text
  học hard cases
  học risk/refinement
  tune theo metric thật
```

Pipeline hiện tại đã tách rõ:

| Stage | Vai trò |
|---|---|
| Stage A | Detect layout, bbox, type |
| Stage B | OCR chính xác từng crop |
| Stage C | Ước lượng risk |
| Stage D | Refine vùng high-risk |
| Stage E | Assembly và postprocess theo metric |

Do đó quá trình train cũng nên bám theo cách tách nhiệm vụ này.

---

# 2. Phase 1 — Baseline an toàn

## 2.1. Mục tiêu

Phase 1 nhằm tạo một baseline end-to-end sạch, ít hallucination, dễ debug.

Ở phase này:

```text
Stage A: page_layout_only
Stage B: crop OCR theo type
Stage C: rule risk đơn giản hoặc chưa bật
Stage D: chưa bật hoặc chỉ reread 1 lần
Stage E: assembly + schema guardrail
```

Chưa nên bật:

```text
text_draft
multi-view refinement phức tạp
VLM selector
draft-aware reranking
```

Mục tiêu không phải tối đa điểm ngay, mà là có một hệ thống ổn định để phân tích lỗi.

---

## 2.2. Chuẩn bị silver data cho Stage 1

Silver nên chia thành hai nhóm task chính:

```text
Task A1 — full-page layout proposal
Task B1 — crop OCR cơ bản
```

---

## 2.3. Task A1 — full-page layout proposal

### Input

```text
full page image + source metadata
```

### Output

```json
[
  {"bbox":[x1,y1,x2,y2],"type":"handwritten"},
  {"bbox":[x1,y1,x2,y2],"type":"printed"},
  {"bbox":[x1,y1,x2,y2],"type":"formula"}
]
```

Ở Phase 1 chỉ dùng:

```text
page_layout_only
```

Không dùng:

```text
text_draft
```

Lý do:

- Stage A chỉ cần học bbox/type ổn định.
- Nếu ép model vừa detect vừa draft text quá sớm, nó có thể giảm tập trung vào layout.
- `text_draft` là tín hiệu optional cho risk/rerank, không phải text cuối cùng.

### Rule lọc silver layout

Silver layout cần lọc theo các tiêu chí:

```text
- bbox hợp lệ
- type thuộc một trong 7 class:
  handwritten, printed, formula, table, annotation, image, graph
- không merge nhiều dòng text thành một bbox
- không split table thành cell nhỏ
- image/graph không bị ép thành handwritten/printed/formula
- reading order tương đối hợp lý
```

### Chính sách với structural type

Với:

```text
image
graph
```

text cuối phải là:

```text
""
```

Do đó trong train layout, cần giữ các class này như region hợp lệ, nhưng trong OCR crop thì target text là chuỗi rỗng.

---

## 2.4. Task B1 — crop OCR cơ bản

### Input

```text
crop image + type + source hint
```

### Output

```text
exact transcription
```

Với `image` và `graph`:

```text
output = ""
```

Stage B không làm các việc sau:

```text
- không detect layout lại
- không sửa bbox
- không suy diễn nội dung từ full page
- không hoàn thành text từ ngữ cảnh
- không giải công thức
- không hiện đại hóa chính tả archive
```

### Prompt input gợi ý

```text
{short_source_hint}
{type_specific_instruction}
Use [illegible] only for unreadable words inside an otherwise legible text region.
Use ~~word~~ for visible strikethrough and ~~old~~{new} for visible correction.
Do not explain.
```

### Type-specific behavior

| Type | Hành vi OCR |
|---|---|
| `handwritten` | Transcribe chính xác chữ viết tay |
| `printed` | Transcribe chính xác chữ in/đánh máy |
| `formula` | Đọc công thức, không giải, không simplify |
| `table` | Xuất pipe-separated table text |
| `annotation` | Đọc chính xác teacher mark / note ngắn |
| `image` | Trả chuỗi rỗng |
| `graph` | Trả chuỗi rỗng |

---

## 2.5. Train Stage 1 trên silver trong Phase 1

Nên train theo curriculum:

```text
Step 1: train Stage B crop OCR trước
Step 2: train Stage A layout-only
Step 3: mix nhẹ A + B để model không quên format
```

Tỉ lệ gợi ý:

```text
B1 crop OCR cơ bản: 60–70%
A1 layout-only: 30–40%
```

Lý do:

- Region CER và PageCER chịu ảnh hưởng mạnh từ chất lượng OCR.
- Stage A vẫn quan trọng vì bbox thiếu hoặc reading order sai sẽ làm PageCER tăng.
- Crop OCR cần được học sớm để model nắm type-specific transcription.

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

Cần đặc biệt giữ nhiều mẫu `image` và `graph` để model học trả text rỗng.

---

## 2.6. Finetune Stage 2 trên GT trong Phase 1

Sau Stage 1, chuyển sang GT với learning rate thấp hơn.

GT finetune tập trung vào:

```text
- sửa bbox/type của Stage A
- sửa exact transcription của Stage B
- học marker đặc biệt:
  ~~word~~
  ~~old~~{new}
  [illegible]
- học structural policy:
  image/graph = ""
```

Tỉ lệ train GT gợi ý:

```text
B1 crop OCR cơ bản: 65%
A1 layout-only: 35%
```

Ở phase này chưa nên cho model học quá nhiều synthetic context hoặc refinement. Mục tiêu là baseline sạch.

---

## 2.7. Inference Phase 1

```text
Input page
  ↓
Stage A layout-only
  ↓
build page_context_light bằng code
  ↓
crop từng region
  ↓
Stage B crop OCR
  ↓
Stage E assembly
  ↓
final CSV
```

Stage E cần enforce:

```python
if type in {"image", "graph"}:
    text = ""
```

Cần preserve marker:

```text
~~word~~
~~old~~{new}
[illegible]
```

---

## 2.8. Output mong muốn của Phase 1

Sau Phase 1 cần có:

```text
- Detection F1 ổn
- Region CER chấp nhận được
- PageCER chưa tối ưu nhưng không vỡ reading order
- schema JSON ổn định
- lỗi dễ phân tích theo source/type
```

Checkpoint nên lưu:

```text
ckpt_phase1_silver
ckpt_phase1_gt_baseline
```

---

# 3. Phase 2 — Contextual OCR + Risk Gate

## 3.1. Mục tiêu

Phase 2 bắt đầu dùng đúng sức mạnh của pipeline:

```text
Stage A: layout-only vẫn là default
Stage B: thêm ocr_context_light
Stage C: risk gate
Stage D: reread/refine vùng high-risk
Stage E: metric-aware hơn
```

Ở phase này vẫn chưa nên bật `text_draft` làm default.

Có thể chuẩn bị ablation riêng, nhưng default vẫn là:

```text
page_layout_only
```

---

## 3.2. Bổ sung page_context_light

Sau Stage A, code build `page_context_light` deterministic.

Không train model sinh page context.

Nên tách context thành hai excerpt:

| Excerpt | Dùng ở đâu | Có text_draft? |
|---|---|---|
| `ocr_context_light` | Stage B first-pass OCR | Không |
| `risk_rerank_context_light` | Stage C/D risk/rerank/refine | Có nếu ablation bật |

Trong Phase 2, `ocr_context_light` là phần chính.

Nội dung có thể gồm:

```text
region id
bbox
type
reading order position
previous/next region type
nearby region ids
same-line region ids
near_table_id
near_graph_id
near_formula_ids
bbox_size_percentile
line_height_ratio
```

Không đưa `text_draft` vào Stage B first-pass.

---

## 3.3. Task B2 — contextual crop OCR

### Input

```text
crop image
+ type
+ source hint
+ ocr_context_light
```

### Output

```text
exact transcription
```

Mục tiêu:

- giúp model đọc crop khó bằng thông tin vị trí và vùng lân cận;
- không để model suy diễn nội dung từ context;
- hỗ trợ formula/table/annotation/archive.

Guardrail trong prompt:

```text
Use ocr_context_light only for layout, local geometry, neighboring region types, table/graph proximity, and surrounding region ids.
The final transcription must be supported by the crop.
Do not complete missing words from context, language prior, or canonical text.
```

---

## 3.4. Task C1 — risk labeling

Có thể train risk model riêng hoặc dùng rule-based risk score.

Nếu có GT/pseudo-GT, tạo risk label bằng cách so sánh OCR first-pass với target:

```text
risk = high nếu:
- CER cao
- OCR rỗng nhưng type scorable
- formula/table output quá ngắn
- text có nhiều ký tự lạ
- crop quá nhỏ/mờ/low contrast
- source=archive và OCR không ổn
```

Feature risk gợi ý:

```text
small_bbox
low_line_height
blur
low_contrast
hard_type
archive_source
text_too_short
empty_text_for_scorable
suspicious_chars
type_visual_mismatch
table_or_formula_long_region
```

Risk score có thể tính theo công thức:

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
+ w10 * type_visual_mismatch
+ w11 * table_or_formula_long_region
```

---

## 3.5. Task D1 — high-risk reread

### Input

```text
original crop
+ zoomed crop hoặc expanded crop
+ type
+ source hint
+ risk_rerank_context_light không có text_draft
```

### Output

```text
correct transcription
```

Ở Phase 2, Stage D có thể đơn giản:

```text
Candidate 1: original crop OCR
Candidate 2: zoom crop OCR
Candidate 3: expanded crop OCR
Rule-based selector chọn output tốt nhất
```

Chưa cần VLM selector phức tạp.

---

## 3.6. Train Stage 1 trên silver trong Phase 2

Train từ checkpoint Phase 1, không train lại từ đầu.

Tỉ lệ task gợi ý:

```text
B1 crop OCR cơ bản: 35%
B2 contextual crop OCR: 35%
A1 layout-only: 20%
D1 high-risk reread: 10%
```

Oversample:

```text
formula
table
annotation
archive
crop nhỏ
crop mờ
OCR first-pass rỗng/sai
image/graph hard negative
```

Mục tiêu của Stage 1 Phase 2 là cho model quen với context nhưng không phụ thuộc vào context để bịa text.

---

## 3.7. Finetune Stage 2 trên GT trong Phase 2

GT finetune nên tập trung mạnh vào hard cases.

Tỉ lệ gợi ý:

```text
B2 contextual crop OCR: 45%
B1 crop OCR cơ bản: 25%
A1 layout-only: 20%
D1 high-risk reread: 10%
```

Vẫn giữ B1 để model không quên baseline OCR.

Hard mining lấy từ lỗi của Phase 1:

```text
- pages có PageCER cao
- regions có RegionCER cao
- formula/table sai format
- annotation bị bỏ sót
- image/graph bị OCR nhầm thành text
- archive bị modernize spelling
- school/university có graph/image bị phân loại sai
```

---

## 3.8. Inference Phase 2

```text
Input page
  ↓
Stage A layout-only
  ↓
build page_context_light
  ↓
Stage B contextual crop OCR
  ↓
Stage C rule risk gate
  ↓
Stage D reread high-risk bằng multi-view nhẹ
  ↓
Stage E assembly/postprocess
  ↓
final CSV
```

Risk gate ban đầu:

```text
threshold = 0.55
top_k_percent = 0.20
hard_cap = 12 regions/page
```

Sau đó tune theo validation:

```text
PageCER gain vs số region refine
runtime vs score gain
formula/table CER gain
```

---

## 3.9. Output mong muốn của Phase 2

Phase 2 cần cải thiện:

```text
- Region CER trên crop khó
- formula/table CER
- archive OCR
- số lỗi OCR rỗng ở scorable region
- PageCER nhờ reading order/context ổn hơn
```

Checkpoint nên lưu:

```text
ckpt_phase2_silver_context
ckpt_phase2_gt_risk_refine
```

---

# 4. Phase 3 — Final

## 4.1. Mục tiêu

Phase 3 là phase cuối để đẩy điểm competition.

Lúc này mới thử các thành phần rủi ro hơn:

```text
- Stage A layout_with_text_draft ablation
- dùng text_draft cho risk/rerank, không dùng cho Stage B first-pass
- Stage D multi-view đầy đủ
- selector/reranker
- metric-aware postprocess
- final ensemble nếu budget cho phép
```

Quan trọng:

```text
layout+draft chỉ được giữ nếu validation chứng minh tốt hơn layout-only
```

Nếu draft làm Detection F1 giảm hoặc JSON kém ổn định, giữ layout-only làm default.

---

## 4.2. Task A2 — layout with text draft

### Input

```text
full page image + source metadata
```

### Output

```json
[
  {
    "bbox":[x1,y1,x2,y2],
    "type":"handwritten",
    "text_draft":"short visual draft"
  }
]
```

Rule bắt buộc:

```text
text_draft ngắn
visually grounded
không chắc thì ""
không dùng làm text cuối
không dùng trong Stage B first-pass
```

Silver cho text_draft cần lọc rất mạnh. Nếu draft nhiễu, nó sẽ làm risk/rerank sai.

---

## 4.3. Train Stage 1 trên silver trong Phase 3

Thêm các task:

```text
A2 layout-with-draft
D full multi-view refinement
selector/reranker
```

Dữ liệu silver nên gồm:

```text
- layout-only examples
- layout-with-draft examples
- contextual OCR examples
- high-risk multi-view examples
- candidate selection examples
```

Không bỏ A1 layout-only, vì vẫn cần một nhánh an toàn.

---

## 4.4. Finetune Stage 2 trên GT trong Phase 3

GT final finetune gồm 5 nhóm task:

```text
A1 layout-only
A2 layout-with-draft
B2 contextual crop OCR
D multi-view refinement
selector/reranker
```

Tỉ lệ gợi ý:

```text
B2 contextual crop OCR: 35%
D multi-view refinement: 20%
A1 layout-only: 15%
A2 layout-with-draft: 15%
selector/reranker: 15%
```

Rule quan trọng:

```text
A1 layout-only vẫn phải được train giữ lại
```

Nếu A2 làm model quen sinh text_draft quá nhiều, nó có thể làm layout JSON kém ổn định.

---

## 4.5. Stage D final

Stage D final nên chạy đủ multi-view cho high-risk:

```text
Candidate 1: original crop
Candidate 2: zoomed crop
Candidate 3: expanded context crop
Candidate 4: crop + expanded crop + optional page thumbnail + risk_rerank_context_light
```

Sau đó chọn bằng:

```text
rule-based reranker + lightweight selector
```

Selector input:

```text
candidate texts
crop
expanded crop
type
source
risk_rerank_context_light
optional text_draft nếu A2 bật
```

Selector output:

```text
final text only
```

Guardrail bắt buộc:

```text
text cuối phải được crop hoặc expanded crop hỗ trợ
không complete từ text_draft
không complete từ canonical dictation
không modernize archive
không solve formula
không OCR image/graph
```

---

## 4.6. Stage E final tuning

Stage E nên có validation-driven postprocess, nhưng không được sửa nội dung quá mạnh.

Nên tune:

```text
reading order heuristic
deduplicate overlap
table placement
annotation placement
image/graph empty text
marker preservation
```

Vì PageCER có ảnh hưởng rất lớn, Stage E cần ưu tiên không phá page-level text assembly.

---

## 4.7. Final validation configs

Cần so sánh ít nhất 4 cấu hình:

```text
Config 1:
A layout-only
B contextual OCR
rule risk
D light

Config 2:
A layout-only
B contextual OCR
tuned risk
D full multi-view

Config 3:
A layout-with-draft
B contextual OCR
draft-aware risk/rerank
D full

Config 4:
Ensemble A layout-only / A layout-with-draft
selector final
```

Chọn config theo validation score, không chọn theo cảm giác.

---

# 5. Lộ trình triển khai cụ thể

## 5.1. Phase 1 — Baseline

```text
Train silver:
  A1 layout-only
  B1 crop OCR

Finetune GT:
  A1 layout-only
  B1 crop OCR

Inference:
  A layout-only
  B crop OCR
  E assembly
```

Mục tiêu:

```text
có baseline sạch, ít hallucination, schema ổn
```

Checkpoint:

```text
ckpt_phase1_silver
ckpt_phase1_gt_baseline
```

---

## 5.2. Phase 2 — Context + risk/refine

```text
Train silver:
  A1 layout-only
  B1 crop OCR
  B2 contextual crop OCR
  D1 high-risk reread

Finetune GT:
  hard-mined B2
  hard-mined D1
  giữ A1/B1 để không quên

Inference:
  A layout-only
  B contextual OCR
  C risk gate
  D multi-view nhẹ
  E assembly
```

Mục tiêu:

```text
giảm Region CER
cải thiện formula/table/archive
giảm OCR rỗng
```

Checkpoint:

```text
ckpt_phase2_silver_context
ckpt_phase2_gt_risk_refine
```

---

## 5.3. Phase 3 — Final

```text
Train silver:
  A2 layout-with-draft
  D multi-view
  selector/reranker

Finetune GT:
  A1 + A2
  B2 contextual OCR
  D full refinement
  selector final

Inference:
  ablation layout-only vs layout+draft
  draft chỉ dùng cho risk/rerank
  D full multi-view cho high-risk
  E metric-aware postprocess
```

Mục tiêu:

```text
đẩy final score
tối ưu PageCER và RegionCER
```

Checkpoint:

```text
ckpt_phase3_silver_final
ckpt_phase3_gt_final
```

---

# 6. Khuyến nghị final default

Hướng final an toàn nhất:

```text
Stage A = layout-only
Stage B = contextual crop OCR
Stage C = rule-based tuned risk gate
Stage D = multi-view refinement cho top high-risk
Stage E = metric-aware assembly
```

Hướng experimental:

```text
Stage A = layout-with-draft
text_draft chỉ dùng cho risk/rerank
giữ lại nếu validation score tăng rõ ràng
```

Không nên để `text_draft` đi vào Stage B first-pass OCR.

---

# 7. Các nguyên tắc bắt buộc

## 7.1. Không train model sinh page_context_light

`page_context_light` nên được build deterministic bằng code.

Lý do:

```text
- không có GT trực tiếp cho context phức tạp
- model dễ hallucinate context
- context sai có thể kéo OCR sai
- khó debug hơn rule geometry
```

---

## 7.2. Không dùng text_draft trong Stage B first-pass

`text_draft` chỉ được dùng cho:

```text
risk scoring
reranking
selector
ablation
```

Không dùng cho:

```text
first-pass crop OCR
```

Lý do:

```text
text_draft sai có thể anchor model vào text sai
```

---

## 7.3. image/graph luôn text rỗng

Rule cuối cùng:

```python
if type in {"image", "graph"}:
    text = ""
```

Không OCR lại `image` và `graph` ở Stage D.

---

## 7.4. Hard type phải được oversample

Các nhóm cần oversample trong cả silver và GT:

```text
formula
table
annotation
archive
image
graph
crop nhỏ/mờ/dài
OCR first-pass rỗng
OCR first-pass sai nặng
```

---

## 7.5. Không sửa text bằng suy diễn ngôn ngữ ở Stage E

Stage E được phép:

```text
- sort reading order
- deduplicate nhẹ
- enforce schema
- giữ marker
- enforce image/graph empty text
```

Stage E không nên:

```text
- tự sửa chính tả
- tự hoàn thành câu
- hiện đại hóa archive
- giải công thức
- thêm text không được ảnh hỗ trợ
```

---

# 8. Kết luận

Quá trình triển khai nên đi theo hướng:

```text
Phase 1:
  tạo baseline sạch

Phase 2:
  thêm contextual OCR, risk gate, high-risk refinement

Phase 3:
  bật layout+draft ablation, multi-view full, selector, metric-aware final tuning
```

Tư duy chính:

```text
silver học rộng
GT chỉnh chuẩn
layout-only làm default an toàn
draft chỉ là ablation
context build bằng code
Stage B/D phải đọc từ ảnh
Stage E không được suy diễn text
```

Final recommendation:

```text
Default final:
  A layout-only
  B contextual crop OCR
  C tuned risk gate
  D multi-view high-risk refinement
  E metric-aware assembly

Experimental final:
  A layout-with-draft
  draft-aware risk/rerank
  chỉ giữ nếu validation score tăng rõ ràng
```
