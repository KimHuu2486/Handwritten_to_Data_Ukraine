# 05 Submission

Thu muc nay chua notebook inference va tao file `submission.csv` cho Kaggle.

## Thu tu doc

1. `htd-box-submit.ipynb`
2. `htd-box-submit-3-fold.ipynb`

## File trong thu muc

- `htd-box-submit.ipynb`: notebook submit chinh o giai doan cuoi. Dung 2 model
  final (`Doclayout Yolo Final V 1.0.pt` va `Doclayout Yolo Final V 1.1.pt`),
  `imgsz=1280`, `conf=0.25`, `max_det=200`, `IOU_NMS=0.6`. Output gom
  `metadata.jsonl` de debug va `submission.csv` dung format Kaggle.
- `htd-box-submit-3-fold.ipynb`: notebook ensemble nhieu checkpoint/fold, dung
  `conf=0.15`, `imgsz=1280`, `max_det=200`, `IOU_NMS=0.6`. Output cung la
  `submission.csv`.

## Khac biet chinh

`htd-box-submit.ipynb` la duong submit gon va gan voi model final 2 checkpoint.
`htd-box-submit-3-fold.ipynb` thu ensemble tu nhieu fold, phu hop de tham khao
cach hop nhat du doan khi co nhieu checkpoint.
