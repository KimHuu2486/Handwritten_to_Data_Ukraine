# Báo cáo Kỹ thuật: Pipeline Phát hiện Vùng Văn bản Viết Tay (RUKOPYS Dataset)

> **Dự án:** Đồ án Tư duy Tính toán — Handwritten Text Detection trên bộ dữ liệu chữ viết tay tiếng Ukraina (RUKOPYS)  
> **Nền tảng:** Kaggle | **Framework:** DocLayout-YOLO (YOLOv10)  
> **Ngày thực hiện:** 31/05/2026

---

## 1. Tổng quan Pipeline

Toàn bộ hệ thống được chia thành **3 giai đoạn** tương ứng với 3 notebook:

| Notebook | Vai trò |
|---|---|
| `Bbox-Class-Preprocess-2.ipynb` | Tiền xử lý dữ liệu & Data Augmentation |
| `Bbox-Class-Funetunning.ipynb` | Chuyển đổi format & Fine-tuning Model |
| `Bbox-Class-Submit-3.ipynb` | Inference & Sinh file nộp |

---

## 2. Giai đoạn 1: Tiền xử lý & Data Augmentation

### 2.1 Nguồn dữ liệu đầu vào

Dữ liệu được lấy từ **2 nguồn** và gộp lại thành bộ `Merged_Rukopys_V1`:

- **GOLD dataset:** `quii29/rukopys-dataset` — 1.330 ảnh huấn luyện có annotation chất lượng cao
- **SILVER dataset:** `quii29/rukopys-silver-rare-classes` — 679 ảnh bổ trợ tập trung vào các class hiếm

### 2.2 Phân loại Class

Hệ thống định nghĩa **7 class** phát hiện và phân chúng thành 3 nhóm để phục vụ chiến lược cân bằng dữ liệu:

| Nhóm | Classes | Class IDs (COCO) |
|---|---|---|
| **HEAD** (dominant) | `handwritten`, `formula` | 1, 3 |
| **MID** (trung bình) | `printed`, `annotation`, `table` | 2, 5, 4 |
| **TAIL** (hiếm) | `image`, `graph` | 6, 7 |

### 2.3 Đọc & Chuẩn hóa ảnh

Toàn bộ ảnh được đọc bằng **OpenCV** (thay vì `PIL.Image`) để tránh lỗi EXIF permission và bùng nổ bộ nhớ với ảnh cực lớn:

- Ảnh được **resize về `MAX_IMAGE_SIZE = 4000 px`** nếu vượt ngưỡng này
- Tọa độ bounding box được scale đồng tỷ lệ sau khi resize
- Format bbox nội bộ: **COCO `[x, y, w, h]`**

Hàm `clamp_coco_bbox()` ép tọa độ box luôn nằm trong phạm vi ảnh, đảm bảo `w > 1` và `h > 1`:

```python
def clamp_coco_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0.0, min(float(x), float(img_w) - 2.0))
    y = max(0.0, min(float(y), float(img_h) - 2.0))
    w = max(1.0, min(float(w), float(img_w) - x))
    h = max(1.0, min(float(h), float(img_h) - y))
    return [x, y, w, h]
```

### 2.4 Chiến lược Tái cân bằng Class (Class Rebalancing)

Hàm `compute_image_priority()` tính điểm ưu tiên cho từng ảnh để quyết định số lần augment:

**Công thức tính điểm:**

```
score = (tail_boxes × 3.0 + mid_boxes × 1.2 + rare_ratio × 8.0
       + counts['image'] × 2.0 + counts['graph'] × 3.0)
       - (dominant_ratio × 6.0 + max(0, counts['handwritten'] - 20) × 0.15
       + max(0, counts['formula'] - 10) × 0.20)
```

**Quy tắc gán số lần lặp (`repeats`) và pipeline:**

| Điều kiện | `repeats` | `pipeline` |
|---|---|---|
| `tail_boxes > 0` và `rare_ratio ≥ 0.1` | 3 hoặc 4 | `aug_rare` |
| `tail_boxes > 0` hoặc `mid_boxes > 5` | 1 hoặc 2 | `aug_rare` / `aug_dense` |
| `counts['handwritten'] ≥ 15` | 1 | `aug_dense` |
| Ảnh head-class thuần (`dominant_ratio ≥ 0.8` và `tail ≤ 1`) | 0 | Bỏ qua |
| `MAX_REPEATS` = 4 | Giới hạn tối đa | — |

### 2.5 Ba Pipeline Augmentation (Albumentations)

Ba pipeline được thiết kế với mức độ biến đổi tăng dần.

**`aug_light`** — cho ảnh thường:

```python
A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2)
], bbox_params=bbox_params)
```

**`aug_dense`** — cho ảnh có nhiều chữ:

```python
A.Compose([
    A.Affine(rotate=(-3, 3), translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.ISONoise(p=0.3)
], bbox_params=bbox_params)
```

**`aug_rare`** — cho ảnh chứa class hiếm:

```python
A.Compose([
    A.Affine(rotate=(-3, 3), scale=(0.97, 1.03), p=0.8),
    A.RandomBrightnessContrast(p=0.6)
], bbox_params=bbox_params)
```

**Tham số BboxParams chung:**

```python
bbox_params = A.BboxParams(
    format='coco',
    label_fields=['region_idx'],
    min_visibility=0.5   # Box bị che khuất > 50% sẽ bị loại
)
```

### 2.6 Kết quả sau Augmentation

Sau khi chạy toàn bộ pipeline trên 2.009 ảnh gốc:

| Class | Trước | Tăng thêm | Tổng |
|---|---|---|---|
| `handwritten` | 27.651 | +5.592 | 33.243 |
| `formula` | 4.040 | +899 | 4.939 |
| `printed` | 3.710 | +5.858 | 9.568 |
| `annotation` | 1.057 | +856 | 1.913 |
| `table` | 687 | +400 | 1.087 |
| `image` | 556 | +983 | 1.539 |
| `graph` | 98 | +166 | 264 |

- Tổng ảnh gốc: **2.009** | Ảnh augment thêm: **753** | Box bị loại: **0**

### 2.7 Xuất dữ liệu

Pipeline xuất đồng thời ba format:

1. **`metadata.jsonl`** — metadata gốc theo schema RUKOPYS
2. **`annotations_detection.json`** — định dạng COCO Detection
3. **`annotations_relations.json`** — đồ thị thứ tự đọc (reading order) giữa các vùng

**Thuật toán sinh Reading Order:**

Gom các region có centroid-Y gần nhau trong cùng 1 dòng với ngưỡng `threshold = 0.5 × max(height_prev, height_curr)`, sau đó sắp xếp theo centroid-X. Các cạnh `"next"` kết nối region theo thứ tự này.

---

## 3. Giai đoạn 2: Chuyển Format & Fine-tuning

### 3.1 Chuyển đổi sang YOLO Format

Dữ liệu từ bước 1 (format COCO `[x, y, w, h]`) được chuyển sang **YOLO normalized `[cx, cy, nw, nh]`**:

```
cx = (x1 + w/2) / img_w
cy = (y1 + h/2) / img_h
nw = w / img_w
nh = h / img_h
```

**Class mapping cho YOLO:**

| Class | YOLO ID |
|---|---|
| `handwritten` | 0 |
| `printed` | 1 |
| `formula` | 2 |
| `table` | 3 |
| `annotation` | 4 |
| `image` | 5 |
| `graph` | 6 |

### 3.2 Chia tập Train / Val

Việc chia tập được thực hiện ở cấp độ **base image** (không phải augmented image) để tránh data leakage:

- **`VAL_RATIO = 0.15`** → 15% ảnh gốc dành cho validation
- **`SEED = 42`** để đảm bảo tái lập
- Ảnh augment (`_aug` suffix) **không được phép vào tập val**

**Kết quả split:**

- Train: **2.364 ảnh** | Val: **301 ảnh**
- Gold records: **2.784** | Silver records: **0** (USE_SILVER = False)

**Phân bố class:**

| Class | Train | Val |
|---|---|---|
| `handwritten` | 26.187 | 4.124 |
| `printed` | 6.044 | 503 |
| `formula` | 3.827 | 619 |
| `annotation` | 1.141 | 141 |
| `table` | 802 | 88 |
| `image` | 772 | 82 |
| `graph` | 150 | 24 |

### 3.3 YAML Config Dataset

```yaml
path: /kaggle/working/layout_data/rukopys_v3
train: train.txt
val: val.txt
nc: 7
names: [handwritten, printed, formula, table, annotation, image, graph]
```

### 3.4 Pretrained Backbone

Model pretrain được tải từ HuggingFace Hub:

- **Repo:** `juliozhao/DocLayout-YOLO-DocStructBench`
- **Checkpoint:** `doclayout_yolo_docstructbench_imgsz1024.pt`
- **Kiến trúc:** `YOLOv10m-doclayout` — 636 layers, **19.970.122 parameters**, 68.0 GFLOPs
- **Transferred weights:** 1039/1051 items từ pretrained

**Kiến trúc đặc biệt** của DocLayout-YOLO bao gồm:
- `G2L_CRM` (Global-to-Local Cross-Representation Module) với Deformable Convolution và GLU activation
- `SCDown` (Stride-aware Conv Downsampling)
- `C2fCIB` (C2f with Compact Inverted Block)
- `PSA` (Partial Self-Attention)
- `v10Detect` head

**3 Patches source code áp dụng trước khi train:**

| Patch | File | Mục đích |
|---|---|---|
| `check_amp()` → `return True` | `utils/checks.py` | Kích hoạt AMP |
| `torch.load(..., weights_only=False)` | `utils/torch_utils.py` | Tương thích PyTorch 2.6+ |
| `getattr(self.dcv, 'bn', None)` | `nn/modules/g2l_crm.py` | Fix G2L_CRM fuse bug |

### 3.5 Hyperparameters Huấn luyện

| Tham số | Giá trị |
|---|---|
| `epochs` | **40** |
| `batch` | **8** |
| `imgsz` | **1280** |
| `optimizer` | **Adam** |
| `lr0` | **0.001** |
| `lrf` | **0.01** |
| `momentum` | **0.9** |
| `weight_decay` | **0.0005** |
| `warmup_epochs` | **1.0** |
| `warmup_momentum` | **0.8** |
| `warmup_bias_lr` | **0.1** |
| `patience` (early stopping) | **8** |
| `save_period` | **10** epochs |
| `device` | **0,1** (Dual Tesla T4 — DDP) |
| `workers` | **8** |
| `amp` | **True** |
| `close_mosaic` | **10** |
| `mosaic` | **0.1** |
| `iou` (NMS training) | **0.7** |
| `max_det` | **300** |
| `seed` | **0** |
| `pretrained` | **True** |
| `fraction` | **1.0** |

**Loss weights:**

| Loss component | Hệ số |
|---|---|
| `box` (CIoU) | 7.5 |
| `cls` | 0.5 |
| `dfl` (Distribution Focal Loss) | 1.5 |

**Augmentation nội bộ YOLO:**

| Tham số | Giá trị |
|---|---|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `translate` | 0.1 |
| `scale` | 0.5 |
| `fliplr` | 0.5 |
| `erasing` | 0.4 |
| `auto_augment` | `randaugment` |

### 3.6 Kết quả Huấn luyện

Model được huấn luyện **40 epochs trong ~3.98 giờ** trên dual Tesla T4 (14.9 GB VRAM/GPU) với DDP:

| Epoch | mAP50 | mAP50-95 |
|---|---|---|
| 1 | 0.313 | 0.189 |
| 7 | 0.546 | 0.341 |
| 18 | 0.630 | 0.393 |
| 32 | 0.680 | 0.432 |
| **40** | **0.682** | **0.441** |

Model xuất ra hai checkpoint: `best.pt` và `last.pt`.

---

## 4. Giai đoạn 3: Inference & Submission

### 4.1 Thiết lập Inference

Pipeline inference sử dụng **2 model (ensemble)** chạy song song để tăng độ bao phủ:

- `MODEL_PATH_1`: `DoclayoutYoloV4.1.pt`
- `MODEL_PATH_2`: `DoclayoutYoloV4.2.pt`

**Tham số inference cho mỗi model:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `imgsz` | **1280** | Nhất quán với lúc train |
| `conf` | **0.1** | Ngưỡng thấp, không mất box |
| `max_det` | **200** | Giới hạn số box tối đa/ảnh |
| `verbose` | `False` | Tắt log để tăng tốc |

### 4.2 Hậu xử lý: Class-Agnostic NMS + Padding

Sau khi thu kết quả từ 2 model, hàm `postprocess_boxes()` thực hiện 2 bước:

**Bước 1 — Class-Agnostic NMS:**

- Gộp tất cả box từ cả 2 model vào 1 danh sách (sử dụng `torchvision.ops.nms`)
- Áp dụng NMS **không phân biệt class** (`agnostic_nms=True`)
- **`IOU_NMS = 0.6`** — loại bỏ box có IoU > 0.6, giữ box confidence cao nhất

**Bước 2 — Padding theo chiều cao dòng:**

```python
pad_x = pad_scale_x × h   # h = chiều cao của box
pad_y = pad_scale_y × h

x1 = max(0.0, x1 - pad_x)
y1 = max(0.0, y1 - pad_y)
x2 = min(img_w, x2 + pad_x)
y2 = min(img_h, y2 + pad_y)
```

> **Lưu ý:** Trong version submit cuối cùng, `PAD_SCALE_X = 0.0` và `PAD_SCALE_Y = 0.0` (padding bị tắt hoàn toàn). Phiên bản thử nghiệm trước dùng `PAD_SCALE_X = 0.3`, `PAD_SCALE_Y = 0.1`.

**Signature đầy đủ của hàm:**

```python
def postprocess_boxes(
    results_list,          # List kết quả từ nhiều model
    img_w: int,
    img_h: int,
    names,                 # Dict class names
    iou_thr: float = 0.6,
    pad_scale_x: float = 0.3,
    pad_scale_y: float = 0.1,
    agnostic_nms: bool = True
) -> list[dict]
```

### 4.3 Fix Module Hub

Một `dummy module` được inject vào `sys.modules` để fix lỗi import:

```python
dummy_hub = ModuleType("doclayout_yolo.utils.callbacks.hub")
dummy_hub.callbacks = {}
sys.modules["doclayout_yolo.utils.callbacks.hub"] = dummy_hub
```

### 4.4 Output Format

Mỗi ảnh sinh ra một record theo schema RUKOPYS:

```json
{
  "file_name": "images/{image_name}",
  "image_width": 1200,
  "image_height": 1600,
  "annotation_source": "yolov10_prediction",
  "regions": [
    {
      "bbox": [x1, y1, x2, y2],
      "type": "handwritten",
      "text": ""
    }
  ]
}
```

Hai file được xuất ra:

| File | Mục đích | Format |
|---|---|---|
| `metadata.jsonl` | Debug, schema RUKOPYS | JSONL |
| `submission.csv` | Nộp Kaggle | CSV (cột: `image`, `regions`) |

---

## 5. Tóm lược Kỹ thuật Chính

| Kỹ thuật | Mô tả | Tham số quan trọng |
|---|---|---|
| **Class-aware Augmentation** | 3 pipeline riêng cho HEAD/MID/TAIL class | `max_repeats=4`, `min_visibility=0.5` |
| **Pretrained Transfer Learning** | DocLayout-YOLO DocStructBench → RUKOPYS | 1039/1051 layers transferred |
| **Dual-GPU DDP Training** | Phân tán 2 × Tesla T4 | `device=0,1`, `batch=8` |
| **Ensemble Inference** | 2 model kết hợp kết quả | `conf=0.1`, `max_det=200` |
| **Class-Agnostic NMS** | Lọc trùng lặp không phân biệt class | `iou_thr=0.6` |
| **No-leak Train/Val Split** | Tách theo base image, aug không vào val | `VAL_RATIO=0.15`, `SEED=42` |
| **AMP Training** | Automatic Mixed Precision (fp16) | `amp=True` |
| **Reading Order Graph** | Sinh đồ thị thứ tự đọc tự động | `threshold = 0.5 × box_height` |
| **Source Code Patching** | 3 patches tương thích môi trường Kaggle | PyTorch 2.6+, AMP, G2L_CRM |

---

*Báo cáo được tổng hợp từ 3 notebooks: Bbox-Class-Preprocess-2.ipynb, Bbox-Class-Funetunning.ipynb, Bbox-Class-Submit-3.ipynb*
