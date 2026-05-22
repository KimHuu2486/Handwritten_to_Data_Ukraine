# Phase 2 — Contextual OCR + Risk Gate

Phase 2 bắt đầu dùng đúng sức mạnh của pipeline tách layout/OCR: Stage A vẫn giữ `layout-only` làm default, nhưng Stage B được bổ sung `ocr_context_light`, Stage C bắt đầu ước lượng risk, và Stage D đọc lại có chọn lọc các vùng high-risk.

Phase này không train lại từ đầu. Nó tiếp tục từ checkpoint Phase 1 để cải thiện các lỗi hard case đã quan sát được trên validation.

---

## 1. Phạm vi Phase 2

| Stage | Trạng thái trong Phase 2 | Vai trò |
|---|---|---|
| Stage A | `page_layout_only` | Vẫn là default an toàn cho bbox/type |
| Stage B | Contextual crop OCR | Dùng `ocr_context_light` để đọc crop khó hơn |
| Stage C | Rule-based hoặc lightweight weighted risk gate | Chọn region cần refine |
| Stage D | Multi-view nhẹ | Reread high-risk bằng crop/zoom/expanded crop |
| Stage E | Metric-aware hơn | Assembly, schema, dedupe nhẹ, marker preservation |

Vẫn chưa bật làm default:

```text
layout_with_text_draft
text_draft trong Stage B first-pass
draft-aware risk/rerank
VLM selector phức tạp
ensemble final
```

---

## 2. Input và Output tổng quát

### 2.1. Input của Phase 2

| Nhóm input | Nội dung | Mục đích |
|---|---|---|
| Checkpoint Phase 1 | `ckpt_phase1_gt_baseline` | Điểm khởi đầu, tránh train lại từ đầu |
| Silver metadata/images | Silver đã lọc + hard examples | Dạy contextual OCR và reread rộng hơn |
| GT metadata/images | Train split human-annotated | Fine-tune hard cases chuẩn |
| Phase 1 predictions | validation predictions/raw outputs | Hard mining và risk label |
| Phase 1 error report | lỗi theo source/type/CER | Chọn region/page khó |
| Validation manifest | frozen validation v1 | Tune threshold/top-K/cap |

### 2.2. Output của Phase 2

| Output | Dạng | Mục đích |
|---|---|---|
| `ckpt_phase2_silver_context` | checkpoint | Model quen với contextual OCR |
| `ckpt_phase2_gt_risk_refine` | checkpoint | Model chính cho inference Phase 2 |
| `page_context_light` cache | JSONL/Parquet | Debug context deterministic |
| `risk_labels_cache` | Parquet/JSONL | Train/tune risk gate |
| `refine_candidates` | JSONL | Debug multi-view/refinement |
| Validation predictions | CSV/JSONL | So sánh với Phase 1 |
| Score/runtime summary | JSON/CSV/log | Tune score gain vs compute |

---

## 3. `page_context_light` deterministic

Sau Stage A, code build `page_context_light` từ `regions`. Không train model sinh context.

### Input

```text
Stage A regions
+ source metadata
+ page geometry
```

### Output

Một record context cho mỗi region:

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
  "near_table_id": "t001",
  "near_graph_id": null,
  "near_formula_ids": ["r010"],
  "bbox_size_percentile": 0.22,
  "line_height_ratio": 0.73
}
```

### Hai excerpt sử dụng trong Phase 2

| Excerpt | Dùng ở đâu | Có `text_draft`? | Ghi chú |
|---|---|---:|---|
| `ocr_context_light` | Stage B first-pass OCR | Không | Context mặc định Phase 2 |
| `risk_rerank_context_light` | Stage C/D | Không trong Phase 2 default | Dùng metadata để risk/refine, chưa draft-aware |

Nội dung tối thiểu của `ocr_context_light`:

```text
region_id
bbox
type
source
reading_order_position
previous/next region type
nearby/same-line region ids
near_table_id
near_graph_id
near_formula_ids
bbox_size_percentile
line_height_ratio
```

---

## 4. Task B2 — Contextual Crop OCR

Task B2 mở rộng B1 bằng cách thêm context deterministic nhưng vẫn bắt OCR phải grounded vào crop.

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

Với structural types:

```text
type=image -> output=""
type=graph -> output=""
```

### Guardrail prompt

```text
Use ocr_context_light only for layout, local geometry, neighboring region types, table/graph proximity, and surrounding region ids.
The final transcription must be supported by the crop.
Do not complete missing words from context, language prior, text_draft, full page, or canonical dictation text.
If a character or word is not visible in the crop/context views, do not invent it.
Do not explain.
```

### Khi B2 hữu ích nhất

```text
archive source
formula/table/annotation
crop nhỏ hoặc dài
low contrast/blur
dòng bị cắt đầu/cuối
vùng gần table/graph/formula
OCR first-pass Phase 1 rỗng hoặc quá ngắn
```

---

## 5. Task C1 — Risk Labeling / Risk Score

Stage C quyết định region nào giữ nguyên Stage B OCR và region nào đưa sang Stage D.

### Input

```text
Stage A region
+ page_context_light
+ Stage B OCR text
+ crop quality signals
+ optional Phase 1/2 error labels
```

### Output

```json
{
  "region_id": "r012",
  "risk_score": 0.72,
  "risk_label": "high",
  "risk_reasons": ["hard_type", "text_too_short", "low_line_height"]
}
```

### Risk features Phase 2

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

Gợi ý công thức:

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

Trong Phase 2 không dùng `draft_crop_disagreement`, vì `text_draft` chưa bật default.

### Decision policy ban đầu

```text
refine nếu:
  risk_score >= threshold
  OR region nằm trong top-K% risk cao nhất của page
  OR hard override được kích hoạt
```

Gợi ý initial config:

| Parameter | Initial value |
|---|---:|
| `threshold` | `0.55` |
| `top_k_percent` | `0.20` |
| `hard_cap` | `12 regions/page` |

Hard override:

```text
- OCR rỗng cho scorable region
- formula/table output quá ngắn hoặc sai format rõ
- crop OCR trả JSON/explanation thay vì text
- bbox hợp lệ nhưng output có nhiều ký tự lạ/repetition
```

---

## 6. Task D1 — High-risk Reread nhẹ

Stage D Phase 2 đọc lại vùng high-risk bằng multi-view nhẹ, chưa cần selector phức tạp.

### Input

```text
high-risk region
+ original crop
+ zoomed crop hoặc expanded crop
+ type
+ short source hint
+ risk_rerank_context_light không có text_draft
```

### Output

```text
corrected/refined transcription
```

Với structural types:

```text
type=image -> output=""
type=graph -> output=""
```

### Candidate generation Phase 2

```text
Candidate 1: original crop OCR
Candidate 2: zoomed crop OCR
Candidate 3: expanded crop OCR
```

### Selector Phase 2

Ưu tiên rule-based selector:

```text
- output hợp lệ theo type
- không rỗng nếu scorable
- ít ký tự lạ/repetition
- không quá ngắn/quá dài so với bbox/type
- table có pipe format hợp lý
- formula không bị diễn giải bằng văn xuôi
- image/graph luôn rỗng
```

Chưa cần VLM selector full ở Phase 2 để tránh tăng runtime và complexity quá sớm.

---

## 7. Stage 1 Training trên Silver

Train từ `ckpt_phase1_gt_baseline`, không train lại từ đầu.

Tỉ lệ task gợi ý:

| Task | Tỉ lệ | Mục tiêu |
|---|---:|---|
| B1 crop OCR cơ bản | 35% | Giữ năng lực OCR không context |
| B2 contextual crop OCR | 35% | Học dùng context nhẹ đúng cách |
| A1 layout-only | 20% | Không quên bbox/type layout |
| D1 high-risk reread | 10% | Học reread vùng khó |

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

Mục tiêu của silver Phase 2 là giúp model quen với context nhưng không phụ thuộc vào context để bịa text.

---

## 8. Stage 2 Fine-tune trên GT

GT Phase 2 tập trung mạnh vào hard cases từ Phase 1.

Tỉ lệ task gợi ý:

| Task | Tỉ lệ | Mục tiêu |
|---|---:|---|
| B2 contextual crop OCR | 45% | Sửa OCR hard cases với context |
| B1 crop OCR cơ bản | 25% | Giữ baseline OCR ổn định |
| A1 layout-only | 20% | Giữ Stage A ổn định |
| D1 high-risk reread | 10% | Refine vùng high-risk |

Hard mining lấy từ:

```text
- pages có PageCER cao
- regions có RegionCER cao
- formula/table sai format
- annotation bị bỏ sót hoặc OCR sai
- image/graph bị OCR nhầm thành text
- archive bị modernize spelling
- school/university có graph/image bị phân loại sai
- output rỗng ở scorable region
- raw JSON incomplete hoặc có explanation
```

---

## 9. Inference Flow Phase 2

```text
Input page
  ↓
Stage A layout-only
  ↓
build page_context_light
  ↓
Stage B contextual crop OCR với ocr_context_light
  ↓
Stage C rule/weighted risk gate
  ↓
Stage D reread high-risk bằng multi-view nhẹ
  ↓
Stage E assembly/postprocess
  ↓
final CSV
```

### Stage E Phase 2

Stage E nên bổ sung nhẹ:

```text
- reading-order heuristic ổn hơn
- deduplicate overlap nhẹ
- enforce image/graph empty text
- preserve special markers
- flag region count suspicious
- flag raw JSON incomplete
```

Không sửa text bằng suy diễn ngôn ngữ.

---

## 10. Validation Gate

Phase 2 chỉ được coi là tốt hơn Phase 1 nếu validation chứng minh:

| Gate | Kỳ vọng |
|---|---|
| Total score | Tăng so với Phase 1 baseline |
| RegionCER | Giảm rõ ở hard crop |
| PageCER | Không xấu đi do refine/reading order |
| Formula/table | CER hoặc format error giảm |
| Archive | Ít modernize/sai chính tả cổ hơn |
| Structural types | `image`/`graph` không bị sinh text |
| Runtime | Refine budget chấp nhận được |
| Risk curve | Score gain hợp lý theo số region refine |

Cần tune:

```text
PageCER gain vs number of refined regions
runtime vs score gain
formula/table CER gain
threshold/top_k_percent/hard_cap
```

---

## 11. Artifact cần lưu

| Artifact | Tên gợi ý | Ghi chú |
|---|---|---|
| Silver checkpoint | `ckpt_phase2_silver_context` | Contextual OCR warm-up |
| GT checkpoint | `ckpt_phase2_gt_risk_refine` | Main Phase 2 checkpoint |
| Context cache | `phase2_page_context_light.jsonl` | Deterministic, rebuildable |
| Risk labels | `phase2_risk_labels_cache.parquet` | Rule/train/tune risk |
| Refine candidates | `phase2_refine_candidates.jsonl` | Debug candidate/selector |
| Validation predictions | `phase2_validation_predictions.csv` | Compare Phase 1 |
| Raw outputs | `phase2_validation_raw_outputs.jsonl` | Parser/runaway/debug |
| Score summary | `phase2_score_summary.json` | Lock metrics |
| Runtime summary | `phase2_runtime_summary.json` | Cost/per-page/refine count |

---

## 12. Quyết định mặc định của Phase 2

```text
Default:
  Stage A = page_layout_only
  Stage B = contextual crop OCR với ocr_context_light
  Stage C = tuned rule/weighted risk gate
  Stage D = light multi-view high-risk reread
  Stage E = metric-aware assembly nhẹ

Không dùng default:
  layout_with_text_draft
  text_draft trong Stage B first-pass
  draft-aware reranking
  full selector/ensemble
```

Phase 2 hoàn thành khi contextual OCR và risk/refine cải thiện validation score với runtime chấp nhận được, đồng thời tạo đủ hard-case evidence cho Phase 3.
