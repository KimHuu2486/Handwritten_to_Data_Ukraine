# YOLO Metrics

Thu muc nay chua cac notebook danh gia module YOLO tren split noi bo va bang
COCOeval. Day la phan dung de lay so lieu dua vao bao cao, khong phai pipeline
submit full test.

## Thu tu doc

1. `htd-bbox-merge-dataset.ipynb`
2. `htd-final-finetune-yolo-bbox V2.2.ipynb`
3. `htd-final-finetune-yolo-bbox V2.1.ipynb`
4. `htd-final-get-score-bbox-v2.ipynb`

## File trong thu muc

- `htd-bbox-merge-dataset.ipynb`: tao dataset merged cho metric. Notebook nay
  gop gold/silver va augment rare classes. Output co 1809 anh goc va 747 anh
  augment.
- `htd-final-finetune-yolo-bbox V2.2.ipynb`: train checkpoint local V2.2, trong
  bao cao map sang V4.1. Config chinh: `doclayout_yolo_small`, `imgsz=1024`,
  `batch=8`, `Adam`, `lr0=0.001`, `mosaic=0.1`, early stop o 40 epochs, best
  epoch 30.
- `htd-final-finetune-yolo-bbox V2.1.ipynb`: train checkpoint local V2.1, trong
  bao cao map sang V4.2. Config chinh: `doclayout_yolo_small`, `imgsz=1024`,
  `batch=8`, `Adam`, `lr0=0.001`, `mosaic=0.25`, early stop o 68 epochs, best
  epoch 58.
- `htd-final-get-score-bbox-v2.ipynb`: notebook scoring chinh. Dung
  `pycocotools COCOeval`, 200 anh, 3821 GT boxes, `imgsz=1280`,
  `max_det=300`, `NMS IoU=0.55`, `agnostic_nms=True`.

## So lieu final trong bao cao

| Model | AP@50 | AP@50:95 | AP@75 | AR@100 |
|---|---:|---:|---:|---:|
| V4.1 only / local V2.2 | 0.6163 | 0.3918 | 0.4245 | 0.5642 |
| V4.2 only / local V2.1 | 0.6961 | 0.4337 | 0.4453 | 0.5713 |
| Ensemble V4.1 + V4.2 | 0.7022 | 0.4424 | 0.4471 | 0.5877 |

## Luu y

Ten local va ten trong bao cao khac nhau:

- `htd-final-finetune-yolo-bbox V2.2.ipynb` tuong ung model paper V4.1.
- `htd-final-finetune-yolo-bbox V2.1.ipynb` tuong ung model paper V4.2.
