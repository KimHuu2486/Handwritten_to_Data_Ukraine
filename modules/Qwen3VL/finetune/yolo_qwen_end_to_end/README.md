# YOLO + Qwen3-VL End-to-End

Thư mục này tập hợp pipeline hybrid dùng DocLayout-YOLO cho `bbox/type` và
Qwen3-VL LoRA cho crop OCR. Ngoài luồng huấn luyện chính, thư mục còn có các
nhánh tăng cường hard types, formula/table và nhiều mức inference khác nhau.

> [!NOTE]
> Không cần chạy toàn bộ notebook theo một chuỗi duy nhất. Hãy chọn nhánh phù hợp
> với mục tiêu huấn luyện hoặc inference.

## Kiến trúc chung

```text
Input page
    ↓
DocLayout-YOLO
    ↓ bbox + type
Crop từng region văn bản
    ↓
Qwen3-VL + LoRA
    ↓ transcription
Post-process + checkpoint
    ↓
submission.csv
```

## Bản đồ notebook

### 1. Huấn luyện chính

```text
Silver data
    ↓
Stage 1 Hybrid Prompt v2
    ↓ LoRA Stage 1
Gold data
    ↓
Stage 2 Hybrid Prompt v2
    ↓ LoRA Stage 2
```

| Thứ tự | Notebook | Vai trò |
|---:|---|---|
| 1 | [`rukopys_qwen3vl_stage1_silver_finetune_hybrid_prompt_v2.ipynb`](./rukopys_qwen3vl_stage1_silver_finetune_hybrid_prompt_v2.ipynb) | Warm-up LoRA trên silver data với prompt dùng cho pipeline hybrid |
| 2 | [`rukopys_qwen3vl_stage2_gold_finetune_hybrid_prompt_v2.ipynb`](./rukopys_qwen3vl_stage2_gold_finetune_hybrid_prompt_v2.ipynb) | Tiếp tục từ Stage 1 trên gold data |

Đây là nhánh nên đọc đầu tiên để hiểu adapter tổng quát được dùng trong các
notebook inference.

### 2. Stage 2B — Hard-type augmentation

```text
Gold regions khó
    ↓
Build augmented crop dataset
    ↓ cached crops + JSONL
Fine-tune từ cached dataset
    ↓
Stage 2B adapter
```

| Thứ tự | Notebook | Vai trò |
|---:|---|---|
| 1 | [`rukopys_qwen3vl_stage2b_build_hardtype_aug_dataset.ipynb`](./rukopys_qwen3vl_stage2b_build_hardtype_aug_dataset.ipynb) | Tạo crop tăng cường, manifest và validation records |
| 2 | [`rukopys_qwen3vl_stage2b_finetune_from_aug_dataset.ipynb`](./rukopys_qwen3vl_stage2b_finetune_from_aug_dataset.ipynb) | Fine-tune từ dataset đã materialize |

Build notebook tạo các artifact chính:

```text
stage2b_hardtype_aug_samples.jsonl
gold_validation_records.jsonl
prompt_config.json
crops/*.jpg
```

Hai bước được tách riêng để augmentation không phải chạy lại mỗi lần fine-tune.
Build notebook dùng đường dẫn local; fine-tune notebook được thiết kế để nhận
artifact đã cache như một Kaggle input.

### 3. Stage 2C — Formula/Table

[`rukopys_qwen3vl_stage2c_finetune_formula_table_aug.ipynb`](./rukopys_qwen3vl_stage2c_finetune_formula_table_aug.ipynb)
fine-tune LoRA hiện có chỉ trên crop `formula` và `table`.

Input crop mong đợi:

```text
ocr_region_crops_wrong_aug/
├── labels.jsonl
├── formula/*.jpg
├── table/*.jpg
└── stats.json
```

Đây là nhánh specialist tùy chọn. Nó không bắt buộc để chạy các notebook hybrid
submit đang trỏ tới adapter Stage 2 Gold tổng quát.

## Các lựa chọn inference

### Hybrid baseline

[`rukopys_yolo_qwen3vl_hybrid_submit.ipynb`](./rukopys_yolo_qwen3vl_hybrid_submit.ipynb)
dùng một checkpoint DocLayout-YOLO:

```text
YOLO bbox/type → Qwen crop OCR → submission.csv
```

Nên đọc notebook này trước để hiểu pipeline hybrid cơ bản.

### Hybrid ensemble V2

[`rukopys_yolo_qwen3vl_hybrid_submit_V2.ipynb`](./rukopys_yolo_qwen3vl_hybrid_submit_V2.ipynb)
thay detector đơn bằng ensemble năm checkpoint YOLO. Các bbox chồng lấn được gom
cluster, tính bbox trung bình có trọng số và chọn class theo tổng confidence.

V2 có thể cho layout ổn định hơn nhưng tốn thêm bộ nhớ và thời gian detector.

### Lite A/B/C/D/E pipeline

[`lite-pipeline-yolo-qwen3vl.ipynb`](./lite-pipeline-yolo-qwen3vl.ipynb) bổ sung:

```text
A — YOLO layout
B — Qwen crop OCR
C — Lite risk gate
D — Reread crop rủi ro cao
E — Metric-aware post-process
```

Lite pipeline không dùng `text_draft`. Nó reread bằng zoom/expanded crop và chỉ
giữ kết quả mới khi chất lượng được cải thiện.

### Full A/B/C/D/E pipeline

[`full-pipeline-yolo-qwen3vl.ipynb`](./full-pipeline-yolo-qwen3vl.ipynb) mở rộng
Lite pipeline bằng:

- `page_context_light`: geometry, reading order và quan hệ region lân cận.
- Weighted risk gate với nhiều tín hiệu hơn.
- Multi-view refinement gồm original, zoom và expanded crop.
- Rule-based rerank và schema guardrail chặt hơn.

Đây là pipeline inference đầy đủ nhất trong thư mục, đồng thời có nhiều
hyperparameter và chi phí xử lý cao hơn Lite.

### Refine một submission có sẵn

[`rukopys_qwen3vl_submit.ipynb`](./rukopys_qwen3vl_submit.ipynb) không tự tạo
layout mới. Notebook đọc `BASELINE_SUBMISSION_CSV`, giữ nguyên region
`image/graph`, OCR lại các region văn bản rồi ghi `submission.csv`.

Dùng notebook này khi đã có bbox/type từ một pipeline khác và chỉ muốn cải thiện
transcription.

## Chọn notebook theo mục tiêu

| Mục tiêu | Notebook nên đọc/chạy |
|---|---|
| Hiểu quá trình huấn luyện LoRA tổng quát | Stage 1 Hybrid Prompt → Stage 2 Gold |
| Tăng cường các region khó | Stage 2B build → Stage 2B fine-tune |
| Chuyên biệt formula/table | Stage 2C formula/table |
| Hiểu hybrid inference cơ bản | Hybrid submit |
| Thử ensemble detector | Hybrid submit V2 |
| Muốn risk gate nhẹ hơn | Lite pipeline |
| Muốn context và refinement đầy đủ | Full pipeline |
| Đã có submission và chỉ muốn OCR lại text | Qwen text-region refinement |

## So sánh các pipeline submit

| Pipeline | Detector | Context/risk gate | Reread | Chi phí tương đối |
|---|---|---|---|---|
| Hybrid | Một YOLO | Không | Không | Thấp nhất |
| Hybrid V2 | Ensemble YOLO | Không | Không | Detector cao hơn |
| Lite | Một YOLO | Lite | Zoom + expanded | Trung bình |
| Full | Một YOLO | Context + weighted risk | Original + zoom + expanded | Cao nhất |
| Text refinement | Bbox có sẵn | Không | OCR lại text regions | Phụ thuộc submission đầu vào |

## Thứ tự đọc khuyến nghị

1. Hybrid submit để hiểu luồng YOLO → crop OCR.
2. Stage 1 và Stage 2 để hiểu adapter được huấn luyện như thế nào.
3. Hybrid V2 để xem ensemble bbox.
4. Lite rồi Full để theo dõi cách risk gate và refinement được mở rộng.
5. Stage 2B, Stage 2C và text refinement khi cần nghiên cứu từng nhánh riêng.

## Artifact và phụ thuộc chính

Các notebook inference thường cần:

- Qwen3-VL base model.
- LoRA adapter phù hợp, mặc định là Stage 2 Gold.
- RUKOPYS dataset.
- Một hoặc nhiều DocLayout-YOLO weights.
- Partial CSV hoặc validation records nếu cần resume/đánh giá.

> [!IMPORTANT]
> Kiểm tra các biến đường dẫn ở cell config trước khi chạy. Một số notebook dùng
> đường dẫn Kaggle cố định, trong khi Stage 2B build dùng đường dẫn local Windows.
