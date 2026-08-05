# Qwen3-VL OOF M1

Thư mục này chứa workflow **out-of-fold (OOF)** cho LoRA M1 của Qwen3-VL.
Mục tiêu là đo chất lượng OCR trên các phần dữ liệu mà adapter tương ứng chưa
từng thấy khi huấn luyện, sau đó dùng adapter M1 trong pipeline YOLO + Qwen3-VL
để tạo submission full-system.

## Luồng tổng quát

```text
metadata_part1/2/3
    ↓ chia thành 3 fold
Train hai part bằng HTD_OOF_M1
    ↓ LoRA M1 của từng fold
Predict part còn lại bằng HTD_OOF_M1_predict_heldout
    ↓ OOF crop predictions / submission CSV
YOLO bbox + type → Qwen3-VL M1 OCR
    ↓
full-system submission.csv
```

## Bản đồ file

| File | Tác dụng |
|---|---|
| [`HTD_OOF_M1.ipynb`](./HTD_OOF_M1.ipynb) | Huấn luyện LoRA M1 theo một fold trên Google Colab A100. Chọn `FOLD_ID`, dùng hai part để train và lấy 15% của dữ liệu train làm validation có stratify. |
| [`HTD_OOF_M1_predict_heldout.ipynb`](./HTD_OOF_M1_predict_heldout.ipynb) | Load LoRA của fold tương ứng trên Kaggle, OCR **part bị giữ lại**, rồi tạo crop-prediction CSV và `submission.csv` có schema `image,regions`. |
| [`rukopys-csv-bbox-qwen3-vl-hybrid-submit-m1.ipynb`](./rukopys-csv-bbox-qwen3-vl-hybrid-submit-m1.ipynb) | Inference full-system: DocLayout-YOLO tự dự đoán `bbox/type`, Qwen3-VL + LoRA M1 OCR crop, sau đó ghi submission. Không train và không tự tạo fold. |
| [`metadata_part1.jsonl`](./metadata_part1.jsonl) | Metadata/GT của part 1, dùng để tạo một phần của fold. |
| [`metadata_part2.jsonl`](./metadata_part2.jsonl) | Metadata/GT của part 2. |
| [`metadata_part3.jsonl`](./metadata_part3.jsonl) | Metadata/GT của part 3. |
| [`dataset.md`](./dataset.md) | Liên kết đến dataset nguồn. |

## Cách chia fold

| `FOLD_ID` | Part train | Part held-out để predict |
|---:|---|---|
| 1 | 1 + 2 | 3 |
| 2 | 1 + 3 | 2 |
| 3 | 2 + 3 | 1 |

Chạy đủ ba fold sẽ tạo OOF prediction cho cả ba part: mỗi mẫu được OCR bởi
adapter không được train trên chính mẫu đó. Validation 15% trong notebook train
chỉ để theo dõi/điều chỉnh training; nó khác với held-out part dùng cho OOF.

## Thứ tự đọc/chạy khuyến nghị

1. Đọc `metadata_part1/2/3.jsonl` và `dataset.md` để hiểu ba part dữ liệu.
2. Chạy `HTD_OOF_M1.ipynb` với `FOLD_ID=1`, rồi lặp lại cho 2 và 3 để có ba LoRA.
3. Với mỗi LoRA, chạy `HTD_OOF_M1_predict_heldout.ipynb` ở `FOLD_ID` tương ứng để lấy prediction trên held-out part.
4. Dùng `rukopys-csv-bbox-qwen3-vl-hybrid-submit-m1.ipynb` khi cần kiểm tra hoặc tạo submission full-system từ YOLO + adapter M1.

## Khác biệt quan trọng giữa hai notebook predict

| Nội dung | `HTD_OOF_M1_predict_heldout` | `rukopys-csv-bbox-qwen3-vl-hybrid-submit-m1` |
|---|---|---|
| Bbox/type | Dùng metadata/GT của held-out part | Do DocLayout-YOLO tự dự đoán |
| Mục tiêu | Đánh giá riêng nhánh OCR theo OOF | Đánh giá hoặc submit toàn hệ thống |
| Phạm vi OCR | Crop đã có region | Crop sinh từ layout prediction |
| Metric detector/class | Không phản ánh detector thực tế | Có thể chấm full-system nếu input có GT |

> [!IMPORTANT]
> Notebook hybrid hiện nhận một `QWEN_LORA_DIR` cho mỗi lần chạy. Khi so sánh ba
> adapter OOF, hãy chạy với đúng LoRA/fold hoặc thực hiện ensemble ở bước ngoài
> notebook; notebook không tự train hay tự ghép ba fold.
