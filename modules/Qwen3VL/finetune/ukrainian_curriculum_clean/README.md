# Qwen3-VL Ukrainian Curriculum Clean

Thư mục này là nhánh huấn luyện theo **curriculum** cho OCR tiếng Ukraina:
làm quen với glyph/chữ viết, học OCR trên GT crop sạch, rồi fine-tune theo đúng
prompt và format output của pipeline cuối. Notebook submission ghép adapter
Stage 2 với DocLayout-YOLO để chạy trên toàn trang.

## Luồng tổng quát

```text
Stage 0: synthetic Ukrainian glyph/script warm-up (tùy chọn)
    ↓
Stage 1: clean GT region OCR
    ↓
Stage 2: final prompt + output format
    ↓
YOLO bbox/type → Qwen3-VL Stage 2 crop OCR → submission.csv
```

## Bản đồ file

| Thứ tự | File | Tác dụng |
|---:|---|---|
| 0 | [`00_stage0_ukrainian_glyph_pretrain.ipynb`](./00_stage0_ukrainian_glyph_pretrain.ipynb) | Warm-up tùy chọn bằng glyph/script tiếng Ukraina và mẫu tổng hợp; giúp model thích nghi ký tự trước OCR dữ liệu thật. |
| 1 | [`01_stage1_clean_region_finetune.ipynb`](./01_stage1_clean_region_finetune.ipynb) | Fine-tune LoRA trên clean crop của region GT. Đây là bước OCR nền tảng của curriculum. |
| 2 | [`02_stage2_final_format_finetune.ipynb`](./02_stage2_final_format_finetune.ipynb) | Fine-tune tiếp từ Stage 1 với prompt, schema và format transcription cuối cùng. Adapter Stage 2 là đầu vào cho submission. |
| 3 | [`03_submission_yolo_qwen3vl_stage2.ipynb`](./03_submission_yolo_qwen3vl_stage2.ipynb) | Load YOLO và adapter Stage 2 để dự đoán `bbox/type`, OCR crop, ghi `submission.csv`. Không huấn luyện thêm. |
| — | [`dataset.md`](./dataset.md) | Liên kết dataset RUKOPYS được dùng bởi workflow. |

## Thứ tự đọc/chạy khuyến nghị

1. Đọc `dataset.md` và cell config của Stage 1 để kiểm tra dataset/model path.
2. Nếu cần tăng khả năng nhận dạng chữ Ukraina, chạy Stage 0; nếu đã có adapter
   warm-up phù hợp, có thể bỏ qua và bắt đầu Stage 1.
3. Chạy Stage 1 để tạo adapter OCR clean-region.
4. Chạy Stage 2 với adapter Stage 1 làm checkpoint đầu vào.
5. Chạy notebook submission với adapter Stage 2 và YOLO weights.

## Khác biệt giữa các stage

| Stage | Dữ liệu/chức năng chính | Khác với stage trước |
|---|---|---|
| 0 | Glyph/script tổng hợp tiếng Ukraina | Không phụ thuộc region GT; chỉ là bước thích nghi ký tự. |
| 1 | GT region crop sạch | Bắt đầu học OCR thực tế theo từng region. |
| 2 | Dữ liệu OCR theo prompt/format cuối | Tối ưu đầu ra để khớp contract của inference/submission, không thay Stage 1. |
| 3 | Inference YOLO + Qwen | Không dùng GT bbox/type; đo/submit full pipeline thay vì train OCR module. |

## Khi chọn workflow này

Chọn nhánh này khi trọng tâm là OCR tiếng Ukraina có curriculum rõ ràng và cần
adapter Stage 2 dùng cho inference. Nếu cần workflow page-level/risk-aware với
VLM-R3 và HALP, xem folder `yolo_qwen_end_to_end_VLM_R^3_HALP` thay vì trộn
notebook giữa hai nhánh.

> [!IMPORTANT]
> Trước khi chạy, kiểm tra cell config trong mỗi notebook: đường dẫn dataset,
> Qwen3-VL base, adapter đầu vào và YOLO weights. Các artifact LoRA/checkpoint
> nên được lưu ngoài Git vì có dung lượng lớn.
