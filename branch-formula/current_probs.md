# Current Problems - UniMERNet Formula Branch

## Kết Luận Chính

Mô hình hiện tại **không tệ với công thức ngắn**, nhưng suy giảm khá rõ với các nhóm khó hơn:

- công thức dài,
- ma trận / determinant,
- bảng nhỏ / truth table,
- phép chia dọc,
- hóa học,
- biểu thức có chữ tự nhiên hoặc đơn vị.

Chỉ số `val_cer = 0.3566` đang là **corpus CER**, tức:

```text
tổng edit distance / tổng số ký tự reference
```

Khi tính thêm **mean per-sample CER**, kết quả khoảng `0.2668`. Hai chỉ số này khác nhau vì một số mẫu rất dài hoặc sai nặng đang kéo corpus CER lên.

## Thống Kê Nhanh

| Chỉ số | Giá trị |
|---|---:|
| Số mẫu validation | 295 |
| Exact match | 75 / 295 = 25.42% |
| Corpus CER | 35.66% |
| Mean per-sample CER | 26.68% |
| Median per-sample CER | 14.29% |
| Số mẫu CER <= 10%, gồm exact | 124 / 295 = 42.03% |
| Số mẫu CER > 50% | 46 / 295 = 15.59% |

Diễn giải: model **không fail toàn bộ**. Gần một nửa mẫu là đúng hoặc gần đúng. Tuy nhiên khoảng `15-16%` mẫu lỗi rất nặng, và chính nhóm này làm điểm tổng xấu.

## Phân Bố Lỗi Theo Độ Dài Công Thức

| Độ dài reference | Số mẫu | Exact | Corpus CER |
|---|---:|---:|---:|
| <= 10 ký tự | 58 | 27 | 22.74% |
| 11-20 | 105 | 38 | 16.14% |
| 21-40 | 76 | 9 | 31.25% |
| 41-80 | 49 | 1 | 40.65% |
| > 80 | 7 | 0 | 72.50% |

Đây là dấu hiệu rất rõ: model đang làm tốt nhất ở công thức ngắn/trung bình, nhưng không giữ được cấu trúc dài.

Ví dụ lỗi:

- Mẫu ma trận/vector bị lặp chuỗi số rất dài thay vì giữ format `|` và newline.
- Reference dạng `В) (→а, →b, →с) = 2|1|3...` bị prediction thành một chuỗi số lặp trong ngoặc dài bất thường.
- Một mẫu dài khác bắt đầu bằng `28. Umv...` bị decode gần như hallucination, có ký tự lỗi `�` và mất hoàn toàn cấu trúc toán.

## Các Nhóm Lỗi Chính

### 1. Lỗi Format / Cấu Trúc Nhiều Dòng

Đây là nhóm nghiêm trọng nhất. Các mẫu có newline, dấu `|`, ma trận, bảng nhỏ hoặc phép chia dọc thường có CER rất cao.

Các dạng thường gặp:

- truth table dạng `х|у|F...`,
- ma trận/vector mất dấu `|` và xuống dòng,
- phép chia dọc có `\cline`, `\hline`,
- determinant hoặc matrix bị decode thành chuỗi số lặp,
- bảng nhỏ bị model xử lý như một công thức inline.

Thống kê đã tính được:

| Nhóm | Số mẫu | Corpus CER |
|---|---:|---:|
| Có newline trong reference | 18 | 68.08% |
| Có ký tự `|` trong reference | 13 | cần log lại/tách riêng |

Kết luận cho nhóm này: UniMERNet hiện tại **không nên gánh toàn bộ table/matrix/vertical arithmetic như formula inline** nếu chưa có xử lý riêng.

Trong pipeline, các mẫu dạng này nên được cân nhắc route sang:

- nhánh table/cell,
- hoặc nhánh `structured formula` riêng,
- hoặc một phase fine-tune riêng với structured gold và learning rate thấp.

### 2. Lỗi Hóa Học

Công thức hóa học bị nhầm chữ và chỉ số khá nhiều.

Ví dụ:

- `2. С2Н5Вr → С2Н5CN` bị dự đoán thành `2. С2Н5Вr → С2Н5Сl`.
- `Al(OH3) + NaOH → Na[Al(OH)4]` bị biến dạng khá nặng.

Đây là lỗi kiểu:

- ký tự gần giống,
- token hiếm,
- chữ Latin/Cyrillic trộn lẫn,
- nhầm `Br`, `CN`, `Cl/Сl`, `Al/Аl`, `Н/OH`.

Với hóa học, sai một ký tự có thể làm sai nghĩa hoàn toàn.

### 3. Lỗi Đơn Vị Và Chữ Tự Nhiên

Model thường đọc được phần số/toán, nhưng sai phần đơn vị hoặc chữ đi kèm.

Ví dụ:

- `1) 3610 : 100 = 36,1 (ц/га)` bị thành `(км)`.
- `1) 1 · 20 = 20 (м)` bị thành `(км)`.
- `m1 = 200г = 0,2кг` bị thành `m_1 = 1802 = 0,18 м`.

Điều này cho thấy model ưu tiên pattern toán phổ biến, nhưng yếu ở text ngắn kèm đơn vị. Nếu metric chấm nguyên chuỗi, các lỗi đơn vị vẫn bị phạt đầy đủ.

### 4. Lỗi Ký Tự Gần Giống

Có nhiều lỗi nhận nhầm ký tự gần giống:

- `х → у / n`,
- `ОС → С`,
- `АВС^2 → ВС2`,
- `Н2 → у_2`,
- `Qd → С2d`,
- `TR/AR → ВП/ВСД`.

Ví dụ đầu file:

```text
Reference:  х^2 + 4х - х^2 = 2х - 8
Prediction: х^2 + 4х - у = 2х - 8
```

Phần toán tổng thể gần đúng, nhưng một biến bị nhận sai. Nhóm lỗi này có thể cải thiện bằng synthetic data, augmentation và postprocess nhẹ, nhưng không nên postprocess quá mạnh vì dễ sửa sai thành sai hơn.

### 5. Hallucination / Lặp Chuỗi

Một số prediction dài hơn reference rất nhiều, đặc biệt ở ma trận và biểu thức dài.

Thống kê:

```text
20 / 295 mẫu có pred_len / ref_len > 1.5
```

Ví dụ:

- Reference ma trận/vector ngắn bị dự đoán thành chuỗi số lặp kéo dài.
- Một số bài logic/limit sinh `∞ ∞ ∞` hoặc lặp pattern.

Nguyên nhân thường gặp:

- crop chứa quá nhiều nội dung hoặc không sạch,
- công thức dài vượt khả năng decoder,
- generation không có ràng buộc chống lặp,
- model chưa học tốt cấu trúc nhiều dòng.

## Điểm Tích Cực

Model đã nhận khá tốt nhiều công thức ngắn:

- `z = -5` đúng hoàn toàn.
- `1) 8 = 2 · 2 · 2` đúng.
- `R(Q) = р · Q` đúng.
- `(-2/3) · (-2) + (-1/2) · 0 = 4/3` đúng.
- `х = 12,9 - 8,5` đúng.
- `R(Q) = 200Q - 2Q^2` đúng.
- Nhiều phương trình tuyến tính ngắn như `2х = 10`, `х = 5`, `7х = 28` đúng.

Vì vậy model đã học được domain. Đây không phải là train hỏng.

## Đánh Giá Thực Tế

Mức hiện tại có thể xem là:

> Usable baseline cho formula ngắn/in-line, nhưng chưa đủ tốt cho formula phức tạp, multi-line, ma trận, bảng nhỏ và hóa học.

Nếu đưa vào pipeline Kaggle:

- Có thể giúp ở các vùng công thức đơn giản.
- Có thể làm Page CER xấu đi ở công thức dài, bảng/ma trận hoặc prediction hallucination dài.

Trong metric RUKOPYS, Page CER có trọng số lớn, nên hallucination dài là rủi ro đáng chú ý.

## Việc Nên Làm Tiếp Theo

### Ưu Tiên 1: Tách Validation Theo Nhóm Độ Khó

Trong script eval, log thêm:

- `ref_len`,
- `pred_len`,
- `has_newline`,
- `has_pipe`,
- `has_chem`,
- `cer`.

Sau đó báo riêng CER cho các nhóm:

- `inline_simple`,
- `long_formula`,
- `matrix_or_table_like`,
- `chemistry`,
- `natural_text_with_units`.

### Ưu Tiên 2: Route Lại Dữ Liệu Có Cấu Trúc

Nếu reference/crop có dấu hiệu nhiều dòng, dấu `|`, phép chia dọc, bảng logic, determinant/matrix, không nên để UniMERNet inline xử lý đơn thuần.

Nên cân nhắc route sang:

- nhánh table/cell,
- structured formula branch,
- hoặc xử lý OCR từng cell rồi ghép PSV.

Nhánh table trong pipeline đã hướng tới cell detection + OCR + PSV, phù hợp hơn cho nhóm này.

### Ưu Tiên 3: Dùng Checkpoint Epoch 5 / Best

Log trước đó cho thấy best là epoch 5 với:

```text
val_cer ≈ 0.3566
```

File prediction hiện tại cũng khớp epoch 5:

```text
exact match = 25.42%
```

Không nên mặc định dùng epoch 8 nếu validation đã xấu hơn.

### Ưu Tiên 4: Kiểm Tra Crop Của Top Lỗi Nặng

Nên xuất debug crop cho top 30 mẫu có CER cao nhất.

Các lỗi như:

- hallucination dài,
- sai hoàn toàn `с ⟂ b → 0 16`,
- `1 ≠ 2 → 472`,

có thể đến từ crop quá nhỏ, lệch bbox hoặc ảnh vùng công thức thiếu thông tin.

### Ưu Tiên 5: Fine-Tune Có Kiểm Soát

Không nên tăng epoch mù quáng. Train loss giảm nhưng val CER dao động.

Nên thử:

- early stopping,
- train 3-5 epoch gold thay vì 8,
- một run `SKIP_SILVER=True` để kiểm tra silver có gây nhiễu không,
- structured phase riêng với LR thấp.

## Kết Luận Cuối

Model hiện tại đi đúng hướng, nhưng bottleneck chính không còn là “train thêm epoch”. Bottleneck nằm ở:

- dữ liệu formula quá đa dạng,
- đang trộn inline math, matrix, table-like, chemistry, text-with-unit,
- model yếu với cấu trúc nhiều dòng và ký hiệu hiếm,
- một số prediction hallucination dài gây hại lớn cho CER,
- cần routing/cleaning/eval theo nhóm thay vì nhìn một số `val_cer` chung.

Bước tiếp theo tốt nhất:

1. Phân nhóm lỗi.
2. Debug crop top lỗi nặng.
3. Quyết định nhóm nào dùng UniMERNet.
4. Quyết định nhóm nào chuyển sang table/HPA/postprocess.