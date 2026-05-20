# Đề xuất pipeline mới: **Uncertainty-Aware Coarse-to-Fine Agentic HTR**

Sau khi đối chiếu:

- **Baseline hiện tại** của bạn:  
  `Silver SFT → Gold SFT → Inference 2-pass: page_json → crop_ocr all_text`
- **4 paper**: Doc-V\*, HALP, VLM-R³, CogCoM
- **4 file .md** tóm tắt paper
- **README dataset** và **notebook metric**

tôi đề xuất **không thay toàn bộ baseline**, mà **nâng cấp baseline thành một pipeline 4 tầng có chọn lọc, biết phát hiện vùng khó, chỉ “suy luận/crop thêm” khi cần**.

---

# 1. Nhận xét ngắn về baseline hiện tại

Baseline hiện tại của bạn **đúng hướng** và đã có nền tốt:

### Điểm mạnh
1. **Curriculum Silver → Gold** là hợp lý, đúng với khuyến nghị của dataset RUKOPYS: dùng `silver` để pretrain/noisy training, sau đó fine-tune trên `train` human-verified.  
2. Bạn đã tách:
   - **page-level extraction**: full page → JSON regions
   - **crop-level OCR**: crop từng vùng → text tốt hơn  
3. Inference `CROP_OCR_MODE="all_text"` là quyết định hợp lý vì metric đánh rất nặng vào chất lượng text.

### Vấn đề còn tồn tại
Từ các notebook baseline, tôi thấy pipeline hiện tại có 4 điểm nghẽn lớn:

1. **Page pass đang làm quá nhiều việc cùng lúc**  
   Nó phải vừa:
   - phát hiện bbox,
   - phân loại type,
   - sinh text đầy đủ.  
   Điều này khiến layout detection dễ bị nhiễu bởi transcription.

2. **Crop OCR chạy đồng đều cho mọi vùng**, chưa phân biệt:
   - vùng dễ,
   - vùng mờ,
   - vùng nhỏ,
   - vùng dễ hallucinate,
   - vùng formula/table/archive khó.

3. **Không có cơ chế “nhìn lại”**  
   Một crop đọc sai là kết thúc. Trong khi VLM-R³ và CogCoM cho thấy crop/zoom nhiều bước có lợi rõ rệt khi cần chi tiết thị giác.

4. **Không có uncertainty/risk routing**  
   HALP chỉ ra rằng có thể phát hiện nguy cơ hallucination **trước khi generate**, và dùng nó để **selective routing** sang pipeline mạnh hơn.

---

# 2. Pipeline mới tôi đề xuất

Tôi gọi pipeline này là:

> **UA-CF-HTR**  
> **U**ncertainty-**A**ware **C**oarse-to-**F**ine Handwritten Text Recognition

Luồng tổng thể:

```text
Input Page
   ↓
Stage A — Coarse Layout Proposal
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

# 3. Stage A — **Coarse Layout Proposal**, tách detection khỏi transcription

## Mục tiêu
Full-page model **không nên chịu trách nhiệm đọc text chính xác nữa**.  
Nó nên tập trung vào:

```json
[
  {"bbox":[...], "type":"handwritten"},
  {"bbox":[...], "type":"printed"},
  {"bbox":[...], "type":"formula"}
]
```

Có thể vẫn giữ `text_draft` như tín hiệu phụ, nhưng **không dùng làm text cuối cùng**.

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

## Cách huấn luyện Stage A

Từ dữ liệu hiện có, tạo thêm task mới:

### Task A1 — `page_layout_only`
Input: full page  
Output:

```json
[
  {"bbox":[x1,y1,x2,y2],"type":"handwritten"},
  {"bbox":[x1,y1,x2,y2],"type":"formula"}
]
```

### Task A2 — `page_layout_with_text_draft`
Giữ giống baseline hiện tại, nhưng **weight thấp hơn** để model vẫn học quan hệ giữa region và text.

Tỷ lệ gợi ý trong SFT:
- `page_layout_only`: **35%**
- `page_json_full`: **15%**
- `crop_ocr`: **50%**

---

# 4. Stage B — **Region OCR Reader**, tiếp tục dùng crop OCR nhưng mạnh hơn

Stage B kế thừa phần tốt nhất từ baseline hiện tại:  
**crop từng vùng → OCR text chính xác.**

Nhưng tôi đề xuất nâng cấp 3 điểm.

---

## 4.1. Prompt theo `type`

Bạn đã làm:
- formula prompt
- table prompt
- annotation prompt
- default prompt (handwritten/printed)

Điểm này nên giữ.

---

## 4.2. Prompt theo `source`

README cho thấy 4 domain khác nhau:
- `dictation`: phone photo, handwriting prose
- `archive`: 1919–1935, nét mực cũ, có chính tả cổ
- `university`: exam, formula, chemistry, tables
- `school`: notebook, bài tập, teacher marks

Vì test metadata có `source`, tôi đề xuất prompt OCR nên thêm context:

### Dictation
> "This is a handwritten Ukrainian dictation line. Transcribe only the characters and punctuation that are visually supported by the image. Do not infer the canonical dictation text, do not fill uncertain words from language context, and do not rewrite the sentence. Preserve visible corrections and strikethroughs using the dataset conventions: ~~word~~ for struck-through text, ~~old~~{new} for struck-through text with a visible correction, and [illegible] only for an unreadable word inside an otherwise legible line."

### Archive
> “Transcribe the region exactly. It may contain historical Ukrainian orthography or mixed old Cyrillic. Do not modernize spelling. Preserve visible corrections and strikethroughs using ~~word~~ and ~~old~~{new} when clearly present. Use [illegible] only for an unreadable word within an otherwise legible line; do not guess the word from context.”

### School
> “This is a school homework page. It may contain handwritten answers, printed textbook fragments, formulas, tables, teacher annotations, drawings, diagrams, coordinate plots, charts, or other non-text visual regions. Detect all meaningful regions carefully. Classify drawings, illustrations, sketches, and other pictorial content as image; classify coordinate plots, charts, axes-based plots, and plotted diagrams as graph. For image and graph, return an empty text string. For textual regions, preserve visible corrections and strikethroughs exactly: use ~~word~~ for struck-through text, ~~old~~{new} for struck-through text with a visible correction, and [illegible] only for an unreadable word inside an otherwise legible line. Do not force non-text visual regions into handwritten, printed, or formula.”

### University
> “This is a university exam or coursework page. It may contain handwritten text, printed text, mathematical or chemical formulas, tables, diagrams, plotted graphs, coordinate charts, scientific figures, and other non-text visual regions. Detect all meaningful regions carefully. Classify standalone equations or chemistry notation as formula; classify tabular structures as table; classify charts, plots, axes-based graphs, and data visualizations as graph; classify diagrams, illustrations, sketches, scientific figures, or other pictorial regions as image. For image and graph, return an empty text string. For textual regions, preserve visible corrections and strikethroughs exactly: use ~~word~~ for struck-through text, ~~old~~{new} for struck-through text with a visible correction, and [illegible] only for an unreadable word inside an otherwise legible line. Do not misclassify visual figures or graphs as text, formulas, or tables.”

Điều này có khả năng cải thiện đáng kể các vùng dễ bị model “chuẩn hóa ngôn ngữ” sai.

---

## 4.3. Train thêm `contextual crop OCR`

Baseline crop OCR chỉ đưa **crop**.  
Tôi đề xuất thêm một task mới:

### Task B2 — `crop_with_page_context`
Input:
1. thumbnail/page view
2. crop region

Output:
```text
exact transcription
```

Lợi ích:
- Một dòng đơn lẻ đôi khi khó đọc nếu tách khỏi câu trước/sau.
- Page context giúp đọc tên riêng, mẫu câu dictation, ký hiệu trong bảng, đoạn archive.

Ý tưởng này tương ứng với:
- **Doc-V\***: giữ global overview + fine detail.
- **CogCoM**: multi-image, multi-turn reasoning giúp tận dụng ảnh gốc và ảnh crop cùng lúc.

---

# 5. Stage C — **Uncertainty Gate / Risk Estimator**

Đây là phần quan trọng nhất để biến baseline thành pipeline mới.

## Mục tiêu
Không xử lý mọi crop giống nhau.

Sau lần OCR đầu, mỗi region được phân thành:

```text
LOW-RISK  → giữ nguyên text
HIGH-RISK → đưa vào Stage D để đọc lại có chủ đích
```

---

## 5.1. Heuristic risk

Trước khi huấn luyện HALP probe thật, có thể làm **Risk Gate v1** bằng các tín hiệu rẻ:

### Region bị đánh dấu high-risk nếu:
- bbox quá nhỏ / chiều cao dòng thấp,
- crop quá mờ hoặc contrast thấp,
- source = `archive`,
- type = `formula` hoặc `table`,
- text OCR lần 1 quá ngắn bất thường,
- text chứa chuỗi lạ, nhiều `?`, ký tự Latin bất thường,
- page-level text draft và crop-level OCR khác nhau nhiều,
- crop OCR sinh output trống nhưng region là scorable type,
- predicted type và visual appearance mâu thuẫn.

Đây là bản nên triển khai **ngay**, vì không cần thay model.

---

## 5.2. Mạnh hơn, có khả năng publish: HALP-style probe (Làm sau)

HALP cho thấy:
- hallucination risk có thể phát hiện **trước generation**,
- query-token hidden states thường hiệu quả nhất,
- có thể dùng risk score để **selective routing** sang nhánh xử lý mạnh hơn.

### Áp dụng cho OCR của bạn

Huấn luyện một **MLP probe** dự đoán:

```text
region_will_be_wrong = 0/1
```

### Cách tạo nhãn
Trên gold validation:
1. chạy crop OCR baseline,
2. tính CER giữa prediction và GT text sau chuẩn hóa theo metric,
3. gán:
   - `risk=1` nếu CER > ngưỡng, ví dụ 0.15
   - `risk=0` nếu CER ≤ ngưỡng.

### Feature probe
Trích xuất hidden states từ Qwen3-VL:
- visual encoder pooled feature,
- last vision token hidden state,
- final query token hidden state.

### Dùng risk score
- Nếu `risk < τ`: chấp nhận OCR
- Nếu `risk ≥ τ`: gọi Stage D

HALP gợi ý đúng triết lý này: dùng probe như tín hiệu điều khiển, không phải chỉ để đánh giá.

### Lưu ý
HALP báo cáo trên Qwen2.5-VL chứ không phải Qwen3-VL, nên với Qwen3-VL bạn **phải ablate**:
- VF probe,
- VT probe,
- QT probe,
- layer L/2, 3L/4, L.

---

# 6. Stage D — **Interactive Refinement cho vùng high-risk**

Đây là phần hấp thụ trực tiếp từ:
- **VLM-R³**
- **CogCoM**
- một phần **Doc-V\***

## Mục tiêu
Khi crop khó, không chỉ đọc lại y hệt.  
Hệ thống phải **nhìn lại có chiến lược**.

---

## 6.1. Stage D v1 — Multi-view refinement, dễ triển khai

Với mỗi region high-risk, tạo 3 view:

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

---

## 6.2. Cách infer

Cho model 2 hoặc 3 lượt OCR:

```text
Candidate 1: original crop
Candidate 2: zoom crop
Candidate 3: context crop
```

Sau đó chọn text cuối bằng:

### Rule-based reranking
- ưu tiên output hợp lệ theo type,
- ít ký tự lạ hơn,
- gần page draft hơn nếu page draft có vẻ ổn,
- gần canonical/retrieved lexicon hơn nếu thuộc dictation,
- tránh output quá ngắn/ quá dài bất thường.

### Hoặc LLM/VLM selector nhẹ
Input:
- 3 candidate text
- crop image
- yêu cầu: “Select the exact transcription best grounded in the image.”

---

## 6.3. Stage D v2 — Agent action policy

Nếu muốn đi theo hướng paper/publication, thay multi-view cứng bằng **policy model**:

Action space:

```text
ANSWER(text)
ZOOM_IN(bbox, scale)
EXPAND_CONTEXT(bbox)
RECHECK_WITH_PAGE_CONTEXT(bbox)
```

### Inference protocol
Ví dụ:

```xml
<think>
  <analysis>Current crop is too small; several characters are merged.</analysis>
  <plan>Zoom in before final transcription.</plan>
  <summary>Need a sharper view of the central word.</summary>
</think>
<action>ZOOM_IN([x1,y1,x2,y2], 2.0)</action>
```

Doc-V\* chứng minh cấu trúc:
- overview,
- reasoning,
- action,
- working memory  
giúp hệ thống chủ động thu thập evidence thay vì xử lý thụ động.

CogCoM cho thấy chain of manipulations như:
- Grounding
- CropZoomIn
- OCR  
có thể được tổ chức thành chuỗi thao tác minh bạch và chuyển thành dữ liệu multi-turn VQA để fine-tune.

---

# 7. Stage E — **Page-level Assembly & Metric-aware Postprocess** (Thực hiện sau khi đã chạy được pipeline hoàn chỉnh)

Notebook metric của bạn cho thấy score phụ thuộc rất mạnh vào:

- `PageCER`: **0.50**
- `Region CER`: **0.30**
- `Detection F1`: **0.15**
- `ClassAcc`: **0.05**

Vì vậy Stage E phải được xem là một **module chiến lược**, không chỉ là serialization.

---

## 7.1. Các bước bắt buộc

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

## 7.2. Dictation-specific retrieval/alignment

README nói rõ:
- ground-truth canonical text cho từng năm dictation là công khai,
- có thể dùng cho text-line alignment.

Tôi đề xuất một nhánh riêng cho `source=dictation`:

```text
Predicted page transcription
   ↓
Retrieve closest canonical dictation text/year
   ↓
Align predicted lines to canonical sequence
   ↓
Use alignment only to rerank uncertain candidates
```

Không nên auto-replace toàn bộ bằng canonical text vì:
- bbox/line segmentation khác nhau,
- test dictation có handwriting noise,
- có thể có thiếu dòng, cắt trang.

Nhưng dùng canonical text để:
- chọn giữa Candidate 1/2/3,
- sửa lỗi rất nhỏ ở high-risk lines,
- kiểm tra page ordering,  
thì rất đáng thử.

---

# 8. Pipeline training mới

Tôi đề xuất chuyển từ **2-stage baseline** thành **4-stage training**.

---

## Stage 1 — Silver pretraining, giữ baseline nhưng làm sạch hơn

Dùng silver như hiện tại, nhưng:
- giảm weight archive silver nếu mixed Ukrainian/Russian và bbox drift,
- filter box quá méo, text rỗng bất thường,
- tách task:
  - layout-only,
  - page-json-full,
  - crop-ocr.

README cũng nói silver có:
- bbox sequence drift trên dense text,
- box axis-aligned có thể clip skewed lines,
- archive silver có mixed Ukrainian/Russian.

### Mục tiêu
Học:
- layout schema,
- bbox pattern,
- type prediction,
- rough OCR.

---

## Stage 2 — Gold multi-task SFT

Trên train gold, train theo mixture:

| Task | Tỷ lệ gợi ý |
|---|---:|
| crop OCR | 40% |
| layout-only page | 25% |
| full page JSON | 15% |
| crop + page context OCR | 10% |
| hard-region reread | 10% |

### Sampling nên có trọng số
Oversample:
- archive,
- formula,
- table,
- annotation,
- small bbox,
- illegible / near-illegible lines,
- volunteer data nếu validation chỉ ra mismatch.

---

## Stage 3 — Risk probe training

Huấn luyện HALP-style MLP probe:

```text
Input hidden state → output probability OCR will be wrong
```

Training labels lấy từ:
- normalized CER trên gold validation,
- exact string mismatch,
- hoặc page-draft vs crop-OCR disagreement.

Kết quả của probe không dùng để thay text, chỉ dùng để **route**.

---

## Stage 4 — Interactive refinement SFT / optional RL

### 4.1. SFT trajectories trước
Tạo trajectory kiểu:

```text
page_thumbnail
→ select region
→ crop
→ zoom if hard
→ final text
```

Nguồn tạo trajectory:
- GT bbox có sẵn,
- hard region mining từ baseline lỗi,
- synthetic crop perturbation.

CogCoM chứng minh chain data có thể tự động sinh bằng pipeline thao tác + lọc positive path; VLM-R³ dùng rationale liên ảnh để dạy model “khi nào cần nhìn lại”.

### 4.2. RL/R-GRPO chỉ nên làm sau
VLM-R³ dùng R-GRPO để thưởng:
- answer correct,
- format valid,
- region selection hợp lý,
- reasoning length vừa phải.

Với project của bạn, reward nên biến thành:

```text
R = 
  + α * improvement_in_normalized_CER
  + β * correct_region_format
  + γ * bbox_validity
  - δ * unnecessary_refinement_steps
```

Nhưng tôi **không khuyến nghị làm RL ngay** cho bản Kaggle đầu tiên.  
Trình tự đúng nên là:

1. Multi-view heuristic refinement  
2. SFT action trajectories  
3. Sau khi có validation gain ổn định mới thử RL

---

# 9. Inference pipeline cuối cùng tôi đề xuất

```text
Input test page
   ↓
[1] Layout-only page detector
    → regions: bbox, type, reading order
   ↓
[2] Crop OCR first pass for all scorable regions
   ↓
[3] Risk gate
    - low risk: keep OCR
    - high risk: send to refinement
   ↓
[4] Refinement for high-risk only
    - original crop
    - zoom crop
    - expanded context crop
    - optional page-thumbnail + crop
    - select best candidate
   ↓
[5] Page assembly
    - dedup regions
    - type sanitation
    - reading-order fix
    - final JSON output
   ↓
submission.csv
```

---

# 10. Vì sao pipeline này tốt hơn baseline hiện tại?

| Vấn đề baseline | Pipeline mới xử lý |
|---|---|
| Page model vừa detect vừa OCR | Tách **layout** và **reading** |
| Crop OCR mọi vùng như nhau | Có **risk routing** |
| Vùng mờ đọc sai là hết | Có **multi-view / zoom / reread** |
| Không tận dụng context toàn trang | Có **thumbnail + crop context** |
| Không phát hiện hallucination trước generation | Có **HALP-style probe** |
| Không ưu tiên điểm PageCER | Có **page assembly + ordering + candidate rerank** |

---

# 11. Bản triển khai nên ưu tiên theo thứ tự

## Phase 1 — Nên làm ngay, khả năng tăng Kaggle score cao nhất
1. **Tách layout-only page pass**
2. **Giữ crop OCR all regions**
3. **Thêm heuristic risk gate**
4. **Thêm high-risk multi-view refinement**
5. **Tối ưu page assembly / reading order**

Đây là phiên bản **competition-first**, ít rủi ro nhất.

---

## Phase 2 — Tăng điểm tiếp
6. Train thêm:
   - `crop_with_page_context`
   - `hard_region_reread`
7. Làm source-specific prompts
8. Dictation retrieval/alignment reranking

---

## Phase 3 — Có màu sắc paper
9. HALP-style OCR risk probe
10. Interactive action SFT
11. Optional R-GRPO cho refinement policy

Đây là phần giúp project chuyển từ:
> “pipeline tối ưu Kaggle”

sang:
> “một hệ thống uncertainty-aware interactive HTR có đóng góp nghiên cứu rõ ràng”.

---

# 12. Kết luận đề xuất

Tôi khuyến nghị pipeline chính thức của bạn nên là:

> **Qwen3-VL-8B + Silver/Gold Curriculum + Layout/OCR Decoupling + Uncertainty Routing + Interactive Crop Refinement + Metric-aware Assembly**

Đây là hướng cân bằng nhất giữa:
- **khả năng tăng leaderboard Kaggle**,
- **khả năng triển khai được trên compute hiện tại**,
- **khả năng mở rộng thành paper**.

Trong 4 paper, vai trò áp dụng nên là:

| Paper | Vai trò trong pipeline mới |
|---|---|
| **Doc-V\*** | Coarse-to-fine, thumbnail overview, working memory, active evidence |
| **HALP** | Pre-generation risk estimation, selective routing |
| **VLM-R³** | Dynamic crop/zoom, interleaved visual refinement, optional R-GRPO |
| **CogCoM** | Chain of manipulations, synthetic trajectory data, multi-turn multi-image training |

Tôi đánh giá đây là hướng **tốt hơn rõ rệt** so với việc chỉ tiếp tục “train thêm Stage 3 gold” trên baseline hiện tại.
