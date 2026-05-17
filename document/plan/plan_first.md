# Plan First - Bàn Giao Phase A
## Data Engineer Làm Trước, Model Engineer Chuẩn Bị Song Song

Tài liệu này là kế hoạch vận hành đầu tiên cho phase tiếp theo của RUKOPYS HTR. Mục tiêu là để Data Engineer và Model Engineer làm đúng thứ tự, không train/chạy ablation khi validation artifact chưa được khóa.

Nguồn tham chiếu:

- `document/plan/Data_Plan.md`
- `document/plan/Model_Plan.md`
- B2 hiện tại: silver -> gold Qwen3-VL LoRA pipeline

---

# 1. Kết Luận Thứ Tự Làm Việc

**Data Engineer làm trước trên critical path.**

Model Engineer được phép chuẩn bị song song, nhưng **không được chạy B2 baseline, crop ablation, risk label, candidate, crop/zoom, hay trajectory** cho đến khi Data Engineer khóa xong frozen validation manifest và pass gate.

Thứ tự bắt buộc:

1. Data Engineer khóa `frozen_validation_manifest.v1`.
2. Model Engineer nhận manifest đã pass gate.
3. Model Engineer chạy B2 baseline đầu tiên với `crop_mode=none`.
4. Data + Model cùng lock B2 baseline score.
5. Sau đó mới chạy crop mode ablation, risk labels, candidates, crop/zoom.

---

# 2. Contract Chung Đã Chốt

## 2.1 Artifact root

Tất cả artifact phase này đặt dưới:

```text
artifacts/
```

## 2.2 Manifest contract

Manifest version đầu tiên:

```text
frozen_validation_manifest.v1
```

Đường dẫn bắt buộc:

```text
artifacts/manifests/frozen_validation_manifest.v1.jsonl
artifacts/manifests/frozen_validation_manifest.v1.config.json
artifacts/manifests/frozen_validation_manifest.v1.gate_report.json
```

Nguồn validation split được chốt:

1. Ưu tiên dùng validation split đã được B2 Stage 2 hiện tại sử dụng.
2. Nếu đã có `gold_validation_records.jsonl` từ notebook Stage 2 hiện tại, promote file đó thành source of truth cho manifest v1.
3. Nếu chưa có file validation nào được lock, tạo đúng một lần từ `train/metadata.jsonl` bằng split deterministic:
   - stratify theo `source`
   - `seed = 42`
   - `val_ratio = 0.12`
   - giữ thứ tự output ổn định sau khi split
4. Sau khi tạo `frozen_validation_manifest.v1`, không regenerate split nữa. Nếu bắt buộc thay đổi, tạo `frozen_validation_manifest.v2`.

## 2.3 Manifest schema

Mỗi dòng JSONL là một image record.

Field bắt buộc:

```text
image_id
file_name
submission_image
source
image_width
image_height
regions
```

Quy ước:

- `image_id`: `file_name` normalized bằng forward slash, không kèm dataset root.
- `submission_image`: basename của `file_name`, dùng để map với submission.
- `file_name`: path từ metadata gốc, normalized bằng forward slash.
- `source`: một trong `archive`, `school`, `dictation`, `university`, `unknown`; giá trị lạ thì map về `unknown` và log trong gate report.
- `regions`: list region gốc nếu validation có annotation.
- `bbox`: format nội bộ bắt buộc là `xyxy` theo pixel trên ảnh gốc.
- `region_id`: nếu metadata gốc thiếu, tạo theo format `r0000`, `r0001`, ... dựa trên thứ tự region gốc trong image.

Không được silently drop image. Region bbox lỗi có thể bị đánh dấu trong gate report, nhưng image record vẫn phải được xử lý rõ ràng.

## 2.4 B2 baseline contract

Run B2 baseline đầu tiên:

```text
run_id = b2_stage2_gold__crop-none__prompt-v1__val-v1
checkpoint_id = b2_stage2_gold
prompt_version = prompt-v1
artifact_version = frozen_validation_manifest.v1
crop_mode = none
metric_version = official-compatible-v1
```

Baseline cache path:

```text
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/
```

Output bắt buộc:

```text
config.json
validation_predictions.csv
validation_raw_outputs.jsonl
validation_score.json
failed_rows.jsonl
runtime_summary.json
```

Generation params chốt cho baseline:

```text
do_sample = false
num_beams = 1
crop_mode = none
max_new_tokens_page = 4096
max_new_tokens_crop = 0
max_pixels_page = value from B2 prompt config if available, otherwise 650000
max_pixels_crop = not used for baseline crop-none
```

Prompt contract:

- Dùng `PAGE_PROMPT` đã lưu trong `rukopys_prompt_config.json` của B2 LoRA nếu có.
- Nếu file prompt config không tồn tại, dùng `prompt-v1` hiện có trong notebook inference.
- Prompt text phải được copy vào `config.json`, không chỉ ghi tên version.

## 2.5 Predictions schema

`validation_predictions.csv` bắt buộc có các cột:

```text
image_id
file_name
submission_image
regions
parse_ok
error_type
raw_output_id
checkpoint_id
prompt_version
runtime_sec
```

Trong đó:

- `regions`: JSON string của parsed region list.
- `parse_ok`: true/false.
- `error_type`: rỗng nếu pass; nếu fail thì ghi lý do ngắn gọn.
- `raw_output_id`: khóa join sang `validation_raw_outputs.jsonl`.

## 2.6 Raw output schema

`validation_raw_outputs.jsonl` mỗi dòng gồm:

```text
raw_output_id
image_id
file_name
prompt_version
generation_params
raw_text
created_at
```

Raw output phải được lưu trước khi parse/clean.

## 2.7 Score contract

`validation_score.json` bắt buộc có:

```text
total_score
detection_f1
class_acc
region_cer
page_cer
row_count
parse_fail_count
runtime_total_sec
runtime_per_page_sec
metric_version
```

Metric direction:

- `total_score`, `detection_f1`, `class_acc`: càng cao càng tốt.
- `region_cer`, `page_cer`, `runtime_per_page_sec`: càng thấp càng tốt nếu score không giảm.

## 2.8 Ablation log contract

Đường dẫn:

```text
artifacts/ablations/ablation_results.csv
```

Dòng đầu tiên phải là B2 baseline:

```text
stage = B2_baseline
run_id = b2_stage2_gold__crop-none__prompt-v1__val-v1
checkpoint_id = b2_stage2_gold
artifact_version = frozen_validation_manifest.v1
prompt_version = prompt-v1
crop_mode = none
routing_rule = none
decision = keep
decision_reason = baseline anchor locked
```

---

# 3. Kế Hoạch Chi Tiết Phương Án 2

## Phase 0 - Chốt contract chung

Owner: Data Engineer + Model Engineer

Trạng thái: đã chốt trong tài liệu này.

Cần đảm bảo cả hai bên đồng ý:

- artifact root là `artifacts/`
- manifest version đầu tiên là `frozen_validation_manifest.v1`
- B2 run đầu tiên là `b2_stage2_gold__crop-none__prompt-v1__val-v1`
- baseline crop mode bắt buộc là `none`
- chỉ chạy `smart` hoặc `all_text` sau khi B2 crop-none đã lock score

Pass khi:

- Data Engineer hiểu manifest schema.
- Model Engineer hiểu cache schema.
- Không còn notebook nào tự ý tạo validation split riêng để score baseline.

## Phase 1A - Data Engineer khóa frozen validation manifest

Owner: Data Engineer

Việc cần làm:

1. Xác định validation split B2 hiện tại đang dùng.
2. Nếu có `gold_validation_records.jsonl` từ Stage 2 hiện tại, dùng nó làm input cho manifest v1.
3. Nếu chưa có, tạo split deterministic một lần theo `seed=42`, `val_ratio=0.12`, stratify theo `source`.
4. Resolve image path để kiểm tra file tồn tại.
5. Chuẩn hóa `image_id`, `submission_image`, `file_name`.
6. Chuẩn hóa bbox về `xyxy` pixel.
7. Tạo `region_id` ổn định nếu thiếu.
8. Ghi manifest, config, gate report.

Output:

```text
artifacts/manifests/frozen_validation_manifest.v1.jsonl
artifacts/manifests/frozen_validation_manifest.v1.config.json
artifacts/manifests/frozen_validation_manifest.v1.gate_report.json
```

Pass gate:

- số image đúng validation split;
- mọi `file_name` resolve được về image tồn tại;
- mọi `image_id` unique;
- nếu có region, mọi `(image_id, region_id)` unique;
- bbox hợp lệ hoặc lỗi được log rõ;
- config ghi rõ source split, seed, val_ratio, code/notebook version, created_at.

Fail gate:

- thiếu image;
- duplicate `image_id`;
- không biết validation split đến từ đâu;
- bbox không thể chuẩn hóa mà không có error report;
- output order không ổn định.

## Phase 1B - Model Engineer chuẩn bị song song

Owner: Model Engineer

Được làm song song với Phase 1A, nhưng chưa chạy inference.

Việc cần làm:

1. Xác định B2 LoRA adapter path và base model path.
2. Xác định `checkpoint_id = b2_stage2_gold`.
3. Xác định `prompt_version = prompt-v1`.
4. Xác định parser/normalizer output region.
5. Xác định metric runner official-compatible.
6. Chuẩn bị `config.json` template cho B2 baseline.
7. Chuẩn bị cache writer theo schema trong tài liệu này.

Không được làm:

- không chạy validation inference khi manifest chưa pass;
- không chạy crop mode `smart` hoặc `all_text`;
- không tune threshold bằng validation;
- không tạo risk label/candidate.

## Phase 2 - Gate bàn giao Data sang Model

Owner: Data Engineer bàn giao, Model Engineer xác nhận nhận.

Điều kiện để bàn giao:

- `frozen_validation_manifest.v1.jsonl` tồn tại.
- config và gate report tồn tại.
- gate report pass.
- row count và image order đã được Data Engineer xác nhận.

Sau gate:

- Model Engineer chỉ được dùng manifest v1 làm validation input.
- Không dùng split từ notebook Stage 2 nếu nó không map 1-1 với manifest v1.
- Mọi score baseline phải ghi `source_manifest = frozen_validation_manifest.v1`.

## Phase 3 - Model Engineer chạy B2 baseline cache

Owner: Model Engineer

Run đầu tiên:

```text
b2_stage2_gold__crop-none__prompt-v1__val-v1
```

Việc cần làm:

1. Load frozen validation manifest v1.
2. Chạy B2 first-pass page inference với `crop_mode=none`.
3. Lưu raw output trước khi parse.
4. Parse raw output thành regions.
5. Ghi failed rows riêng.
6. Score bằng metric official-compatible.
7. Ghi runtime summary.
8. Ghi full config để tái lập.

Output:

```text
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/config.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_predictions.csv
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_raw_outputs.jsonl
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/validation_score.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/failed_rows.jsonl
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/runtime_summary.json
```

Pass gate:

- prediction row count bằng manifest row count;
- `regions` parse được thành JSON list;
- parse fail `<= 0.5%` hoặc `<= 3 rows`, lấy ngưỡng nào lớn hơn;
- score có đủ `total_score`, `detection_f1`, `class_acc`, `region_cer`, `page_cer`;
- runtime/page được ghi;
- config có checkpoint, prompt, generation params, metric version, source manifest.

Fail gate:

- thiếu prediction row;
- nhiều raw output không parse được;
- metric không chạy được;
- thiếu raw output;
- không truy vết được checkpoint/prompt/config.

## Phase 4 - Lock B2 baseline

Owner: Data Engineer + Model Engineer

Data Engineer kiểm tra:

- cache join ngược được về manifest theo `image_id`;
- failed rows có lý do;
- row count/image order đúng;
- output artifact không ghi đè version cũ.

Model Engineer kiểm tra:

- score breakdown hợp lệ;
- metric direction được đọc đúng;
- runtime/page được tính;
- B2 baseline config tái lập được.

Output:

```text
artifacts/ablations/ablation_results.csv
```

Dòng B2 baseline có `decision=keep`.

Chỉ sau khi Phase 4 pass mới được sang cheap gains.

---

# 4. Việc Làm Sau Khi B2 Baseline Đã Lock

Thứ tự tiếp theo:

1. Crop mode ablation trên cùng manifest:
   - `none`
   - `smart`
   - `all_text`
2. Mỗi mode có cache riêng, score riêng, runtime riêng.
3. Chọn `B2_locked` hoặc `B2_plus_best`.
4. Tạo risk labels từ cache tốt nhất.
5. Tạo refine candidates.
6. Tạo executed crop/zoom cache.
7. Chỉ tạo trajectory/refinement data nếu zoom-only có gain.

Không nhảy thẳng sang B3/B4/B5 nếu B2 baseline và cheap gains chưa lock.

---

# 5. Checklist Gửi Cho Data Engineer

Data Engineer cần làm đầu tiên:

- [ ] Tìm validation split B2 hiện tại.
- [ ] Promote hoặc tạo `frozen_validation_manifest.v1.jsonl`.
- [ ] Tạo `frozen_validation_manifest.v1.config.json`.
- [ ] Tạo `frozen_validation_manifest.v1.gate_report.json`.
- [ ] Kiểm row count.
- [ ] Kiểm image file tồn tại.
- [ ] Kiểm `image_id` unique.
- [ ] Kiểm `(image_id, region_id)` unique nếu có region.
- [ ] Kiểm bbox `xyxy` pixel hợp lệ.
- [ ] Xác nhận manifest pass gate.
- [ ] Bàn giao manifest v1 cho Model Engineer.

Data Engineer không cần chạy model ở bước đầu.

---

# 6. Checklist Gửi Cho Model Engineer

Model Engineer làm song song nhưng chưa execute:

- [ ] Xác định B2 adapter/base model.
- [ ] Chốt `checkpoint_id = b2_stage2_gold`.
- [ ] Chốt `prompt_version = prompt-v1`.
- [ ] Chốt generation params baseline crop-none.
- [ ] Chuẩn bị parser/normalizer.
- [ ] Chuẩn bị metric runner official-compatible.
- [ ] Chuẩn bị cache schema writer.
- [ ] Chuẩn bị ablation row schema.

Model Engineer chỉ execute sau khi Data Engineer bàn giao manifest pass gate.

---

# 7. Definition of Done Cho Bước Đầu

Bước đầu được coi là xong khi có đủ:

```text
artifacts/manifests/frozen_validation_manifest.v1.jsonl
artifacts/manifests/frozen_validation_manifest.v1.config.json
artifacts/manifests/frozen_validation_manifest.v1.gate_report.json
artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1/
artifacts/ablations/ablation_results.csv
```

Và B2 baseline pass:

- manifest locked;
- B2 crop-none cache pass;
- score official-compatible đã ghi;
- runtime/page đã ghi;
- failed rows đã ghi;
- ablation row đầu tiên đã ghi;
- Data + Model đồng ý đây là baseline anchor.

Nếu thiếu một trong các artifact trên, chưa được chạy risk labels, candidates, crop/zoom, trajectory, B3, B4, hoặc B5.
