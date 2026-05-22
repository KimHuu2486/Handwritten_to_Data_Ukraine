# Phase 3 — Final ablation, multi-view refinement và metric-aware tuning

Phase 3 là phase cuối để đẩy điểm competition sau khi đã có baseline sạch ở Phase 1 và contextual OCR/risk gate ở Phase 2. Đây là phase thử các thành phần rủi ro hơn: `layout_with_text_draft`, draft-aware risk/rerank, multi-view refinement đầy đủ, selector/reranker và Stage E tuning theo metric.

Nguyên tắc quan trọng:

```text
layout_with_text_draft chỉ được giữ nếu validation chứng minh tốt hơn layout-only.
text_draft không bao giờ đi vào Stage B first-pass OCR.
text cuối phải được crop hoặc expanded crop hỗ trợ.
```

---

## 1. Phạm vi Phase 3

| Stage | Trạng thái trong Phase 3 | Vai trò |
|---|---|---|
| Stage A | A1 `layout-only` + A2 `layout-with-draft` ablation | So sánh default an toàn vs draft signal |
| Stage B | Contextual crop OCR | Vẫn dùng `ocr_context_light`, không dùng draft |
| Stage C | Tuned risk + optional draft-aware features | Chọn high-risk chính xác hơn |
| Stage D | Full multi-view refinement | Original/zoom/expanded/context/page thumbnail optional |
| Stage E | Metric-aware final tuning | Reading order, dedupe, marker/schema guardrail |
| Selector | Rule-based + lightweight selector nếu validation có gain | Chọn candidate cuối cho high-risk |

Phase 3 có hai nhánh:

```text
Default candidate:
  A layout-only
  B contextual OCR
  C tuned risk
  D full multi-view
  E metric-aware assembly

Experimental candidate:
  A layout-with-draft
  draft-aware risk/rerank
  D full multi-view
  E metric-aware assembly
```

---

## 2. Input và Output tổng quát

### 2.1. Input của Phase 3

| Nhóm input | Nội dung | Mục đích |
|---|---|---|
| Checkpoint Phase 2 | `ckpt_phase2_gt_risk_refine` | Điểm khởi đầu |
| Silver filtered data | layout/OCR/context/refine examples | Warm-up các task final |
| GT data | human-annotated train | Fine-tune final theo metric thật |
| Phase 2 predictions | validation outputs/scores/errors | Hard mining và ablation target |
| Risk labels/cache | từ Phase 2 | Tune risk và selector |
| Refine candidates | từ Phase 2 hoặc build mới | Train/tune reranker/selector |
| Validation manifest | frozen validation v1 | Chọn final config |

### 2.2. Output của Phase 3

| Output | Dạng | Mục đích |
|---|---|---|
| `ckpt_phase3_silver_final` | checkpoint | Warm-up final task mix |
| `ckpt_phase3_gt_final` | checkpoint | Final candidate checkpoint |
| A1/A2 ablation predictions | CSV/JSONL | So sánh layout-only vs layout-with-draft |
| Full refinement candidates | JSONL | Debug D full multi-view |
| Selector decisions | JSONL/CSV | Audit candidate selection |
| Final validation score table | CSV | Chọn config cuối |
| Final submission candidate | CSV | Candidate submit Kaggle |

---

## 3. Task A2 — Layout with Text Draft

A2 là ablation cho Stage A. Nó thêm `text_draft` ngắn vào output layout, nhưng draft chỉ dùng cho risk/rerank sau OCR.

### Input

```text
full page image
+ source metadata
+ source-aware layout-with-draft prompt
```

### Output

```json
[
  {
    "bbox": [x1, y1, x2, y2],
    "type": "handwritten",
    "text_draft": "short visual draft"
  },
  {
    "bbox": [x1, y1, x2, y2],
    "type": "formula",
    "text_draft": ""
  }
]
```

### Contract của `text_draft`

| Rule | Yêu cầu |
|---|---|
| Ngắn | Không biến Stage A thành full OCR |
| Visually grounded | Chỉ ghi nếu nhìn thấy trong ảnh |
| Optional | Không chắc thì `""` |
| Không phải final text | Không ghi vào submission trực tiếp |
| Không vào Stage B | Không dùng trong first-pass crop OCR |
| Chỉ dùng sau OCR | Risk scoring, rerank, selector |

Silver cho A2 phải lọc mạnh. Nếu draft nhiễu, nó có thể làm risk/rerank tệ hơn và làm Stage A mất tập trung khỏi bbox/type.

### A2 chỉ được giữ nếu

```text
- Detection F1 không giảm đáng kể so với A1
- JSON stability không xấu đi
- final validation score tăng rõ
- risk/rerank dùng draft tạo gain thực sự
- runtime vẫn chấp nhận được
```

Nếu không đạt, final default quay về A1 `layout-only`.

---

## 4. Stage B Final — Contextual OCR vẫn draft-free

Stage B Phase 3 giữ nguyên nguyên tắc Phase 2:

### Input

```text
crop image
+ type
+ short source hint
+ type-specific OCR instruction
+ ocr_context_light
```

### Output

```text
exact transcription
```

Không thêm `text_draft` vào prompt Stage B, kể cả khi Stage A chạy A2.

Lý do:

```text
text_draft sai có thể anchor model vào text sai.
Stage B phải đọc từ crop/expanded crop, không đoán từ draft.
```

---

## 5. Stage C Final — Tuned Risk + Draft-aware Ablation

### Input

```text
Stage A region
+ page_context_light
+ Stage B OCR text
+ crop quality signals
+ risk_rerank_context_light
+ optional text_draft nếu A2 bật
```

### Output

```json
{
  "region_id": "r012",
  "risk_score": 0.81,
  "risk_label": "high",
  "risk_reasons": ["draft_crop_disagreement", "hard_type", "low_contrast"]
}
```

### Risk features final

Phase 3 kế thừa Phase 2 và có thể thêm:

```text
draft_crop_disagreement
candidate_disagreement
selector_uncertainty
layout_mode_risk
region_count_suspicious
raw_json_incomplete
```

Draft-aware feature chỉ bật trong nhánh A2:

```text
draft_crop_disagreement =
  distance(normalize(text_draft), normalize(stage_b_text))
```

Không dùng draft để tự sửa text. Draft chỉ là tín hiệu risk/rerank.

---

## 6. Stage D Final — Full Multi-view Refinement

Stage D final xử lý high-risk bằng nhiều view và chọn text được ảnh hỗ trợ tốt nhất.

### Input

```text
high-risk region
+ original crop
+ zoomed crop
+ expanded context crop
+ optional page thumbnail
+ type
+ short source hint
+ risk_rerank_context_light
+ optional text_draft nếu A2 bật
```

### Output

```text
final refined transcription
```

Với structural types:

```text
type=image -> output=""
type=graph -> output=""
```

### Candidate generation

```text
Candidate 1: original crop
Candidate 2: zoomed crop
Candidate 3: expanded context crop
Candidate 4: crop + expanded crop + optional page thumbnail + risk_rerank_context_light
```

Candidate 4 chỉ bật khi:

```text
- vùng thật sự high-risk
- crop/expanded crop chưa đủ visual context
- table/formula/annotation cần surrounding structure
- validation cho thấy page thumbnail giúp tăng score
```

### Guardrail bắt buộc

```text
The final transcription must be supported by the crop or expanded crop.
Use page thumbnail only as visual fallback.
Use text_draft only as weak rerank signal.
Do not complete from text_draft.
Do not complete from canonical dictation.
Do not modernize archive spelling.
Do not solve/simplify formula.
Do not OCR image/graph.
Return only final text.
```

---

## 7. Selector / Reranker Final

Selector chọn candidate cuối cho Stage D high-risk.

### Input

```text
candidate texts
+ original crop
+ expanded crop
+ type
+ source
+ risk_rerank_context_light
+ optional page thumbnail
+ optional text_draft nếu A2 bật
```

### Output

```json
{
  "region_id": "r012",
  "selected_candidate": 3,
  "final_text": "...",
  "selection_reasons": ["valid_formula_format", "less_repetition", "best_supported_by_expanded_crop"]
}
```

Nếu selector output được dùng trực tiếp cho submission, phần `final_text` phải được trích ra thành text thuần. `selection_reasons` chỉ dùng để debug/logging.

### Rule-based reranker

Ưu tiên:

```text
- output hợp lệ theo type
- không explanation/JSON thừa
- ít ký tự lạ hoặc repetition
- không quá ngắn/quá dài bất thường
- table giữ pipe-separated rows/cells
- formula giữ symbol/LaTeX hợp lý, không solve
- gần text_draft hơn nếu A2 bật và draft đáng tin
- image/graph luôn rỗng
```

### Lightweight selector

Chỉ dùng nếu validation cho thấy rule-based selector chưa đủ. Selector phải được audit bằng `selector_decisions.jsonl` để tránh chọn theo ngôn ngữ prior thay vì ảnh.

---

## 8. Stage 1 Training trên Silver

Train từ checkpoint Phase 2.

Task mix Phase 3 silver:

| Task | Vai trò |
|---|---|
| A1 layout-only | Giữ nhánh an toàn |
| A2 layout-with-draft | Học draft ngắn, grounded |
| B2 contextual OCR | Giữ OCR chính |
| D full multi-view refinement | Học candidate refinement |
| Selector/reranker | Học chọn candidate |

Không bỏ A1, vì A1 là fallback nếu A2 làm giảm Detection F1 hoặc JSON stability.

Silver cần lọc chặt hơn Phase 2:

```text
- loại draft dài/hallucinated
- loại layout merge/split sai nặng
- giữ hard examples có GT/pseudo-GT đáng tin
- giữ image/graph hard negatives
- ưu tiên examples có Phase 2 error rõ
```

---

## 9. Stage 2 Fine-tune trên GT

GT final fine-tune gồm 5 nhóm task:

| Task | Tỉ lệ gợi ý | Mục tiêu |
|---|---:|---|
| B2 contextual crop OCR | 35% | Giữ OCR chính xác |
| D multi-view refinement | 20% | Sửa high-risk hard cases |
| A1 layout-only | 15% | Giữ layout fallback ổn |
| A2 layout-with-draft | 15% | Kiểm tra draft ablation |
| Selector/reranker | 15% | Chọn candidate final |

Rule quan trọng:

```text
A1 layout-only vẫn phải được train giữ lại.
```

Nếu A2 làm model quen sinh text quá nhiều hoặc JSON kém ổn định, giảm tỉ lệ A2 hoặc loại khỏi final default.

---

## 10. Inference Flow Phase 3

### Config default candidate

```text
Input page
  ↓
Stage A layout-only
  ↓
build page_context_light
  ↓
Stage B contextual crop OCR
  ↓
Stage C tuned risk gate
  ↓
Stage D full multi-view cho high-risk
  ↓
Stage E metric-aware assembly
  ↓
final CSV
```

### Config experimental candidate

```text
Input page
  ↓
Stage A layout-with-draft
  ↓
build page_context_light
  ↓
Stage B contextual crop OCR, không dùng text_draft
  ↓
Stage C draft-aware risk/rerank
  ↓
Stage D full multi-view + optional draft-aware selector
  ↓
Stage E metric-aware assembly
  ↓
final CSV
```

---

## 11. Stage E Final Tuning

Stage E là module chiến lược vì `PageCER` có trọng số lớn.

### Input

```text
Stage A regions
+ Stage B OCR text
+ Stage D refined text for high-risk regions
+ risk/selector logs
```

### Output

```text
submission CSV with regions JSON
```

### Tuning được phép

```text
reading order heuristic
same-line ordering
table placement
annotation placement
deduplicate overlap nhẹ
image/graph empty text
marker preservation
bad output flags
```

### Không được làm

```text
tự sửa chính tả
tự hoàn thành câu
hiện đại hóa archive
giải công thức
thêm text không được ảnh hỗ trợ
dedupe quá mạnh làm mất Detection F1
```

---

## 12. Final Validation Configs

Cần so sánh ít nhất 4 cấu hình:

| Config | Stage A | Stage C/D | Mục tiêu |
|---|---|---|---|
| Config 1 | layout-only | contextual OCR + rule risk + D light | Baseline Phase 2 reference |
| Config 2 | layout-only | tuned risk + D full multi-view | Final safe candidate |
| Config 3 | layout-with-draft | draft-aware risk/rerank + D full | Experimental draft candidate |
| Config 4 | ensemble A1/A2 | selector final | Chỉ dùng nếu score/runtime đáng giá |

Chọn final config theo validation score, không chọn theo cảm giác.

Metric cần theo dõi:

```text
total_score
detection_f1
class_acc
region_cer
page_cer
runtime_per_page
refine_regions_per_page
json_parse_failure_rate
raw_json_incomplete_rate
region_count_suspicious_rate
```

---

## 13. Validation Gate

Phase 3 chỉ được chọn làm final nếu:

| Gate | Kỳ vọng |
|---|---|
| Total score | Tăng rõ so với Phase 2 |
| PageCER | Không bị Stage D/E làm xấu |
| RegionCER | Giảm ở hard regions |
| Detection F1 | A2 không làm giảm nếu dùng draft |
| JSON stability | Output parse ổn định |
| Runtime | Full D/selector nằm trong budget |
| Guardrail | Không OCR `image/graph`, không solve formula, không complete text |
| Auditability | Có logs candidate/selector đủ truy vết |

Nếu A2 không vượt A1:

```text
Final default = A1 layout-only + Phase 3 D/E tuning.
```

Nếu D full không vượt D light sau tính runtime:

```text
Final default = tuned D light hoặc refine ít hơn theo cap.
```

---

## 14. Artifact cần lưu

| Artifact | Tên gợi ý | Ghi chú |
|---|---|---|
| Silver checkpoint | `ckpt_phase3_silver_final` | Final task warm-up |
| GT checkpoint | `ckpt_phase3_gt_final` | Main final checkpoint |
| A1 predictions | `phase3_a1_validation_predictions.csv` | Layout-only candidate |
| A2 predictions | `phase3_a2_validation_predictions.csv` | Layout-with-draft candidate |
| Refine candidates | `phase3_refine_candidates.jsonl` | Multi-view outputs |
| Selector decisions | `phase3_selector_decisions.jsonl` | Audit final selection |
| Score table | `phase3_validation_ablation_results.csv` | Chọn final |
| Runtime summary | `phase3_runtime_summary.json` | Cost vs gain |
| Final submission | `phase3_final_submission.csv` | Candidate submit |

---

## 15. Quyết định final khuyến nghị

Hướng final an toàn nhất:

```text
Stage A = layout-only
Stage B = contextual crop OCR
Stage C = rule-based/tuned weighted risk gate
Stage D = full hoặc capped multi-view refinement cho top high-risk
Stage E = metric-aware assembly
```

Hướng experimental:

```text
Stage A = layout-with-draft
text_draft chỉ dùng cho risk/rerank/selector
giữ lại chỉ nếu validation score tăng rõ và Detection F1 không giảm
```

Không nên:

```text
- để text_draft đi vào Stage B first-pass OCR
- train model sinh page_context_light
- để Stage E sửa text bằng suy diễn ngôn ngữ
- OCR image/graph
- chọn config final nếu chưa có validation ablation rõ ràng
```

Phase 3 hoàn thành khi có bảng ablation đủ rõ để chọn final submission config theo metric, runtime và guardrail.
