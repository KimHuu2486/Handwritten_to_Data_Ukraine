# 00 Baseline

Thu muc nay chua cac thu nghiem ban dau truoc khi chot DocLayout-YOLO lam huong
chinh.

## Thu tu doc

1. `htd-box-yolo-faster-r-cnn.ipynb`
2. `htd-box-test.ipynb`

## File trong thu muc

- `htd-box-yolo-faster-r-cnn.ipynb`: baseline object detection ban dau. Notebook
  thu YOLOv8 tren Rukopys dataset, chia validation 15%, `imgsz=1280`,
  `batch=8`, `epochs=20`. Muc dich la kiem tra nhanh kha nang phat hien bbox
  bang YOLO thong thuong truoc khi chuyen sang DocLayout-YOLO.
- `htd-box-test.ipynb`: notebook inference nhanh voi model DocLayout-YOLO dau
  tien (`DocLayoutYOLO first version.pt`), `imgsz=1280`, `conf=0.35`,
  `max_det=200`. Dung de kiem tra model co sinh bbox va format output dung
  khong.

## Khac biet chinh

`htd-box-yolo-faster-r-cnn.ipynb` la notebook train baseline. `htd-box-test.ipynb`
khong phai notebook train, ma la notebook test/inference nhanh cho checkpoint
DocLayout-YOLO dau tien.
