# 02 DocLayout Training

Thu muc nay chua cac lan train/fine-tune DocLayout-YOLO chinh tren cac bien the
dataset truoc khi co final metric.

## Thu tu doc

1. `htd-box-doclayout-yolo.ipynb`
2. `htd-box-doclayout-yolo-v2.ipynb`
3. `htd-box-fine-tune-doclayout-yolo-v2.ipynb`
4. `htd-box-doclayout-yolo-v4.ipynb`

## File trong thu muc

- `htd-box-doclayout-yolo.ipynb`: ban DocLayout-YOLO dau tien. Dung
  `rukopys-dataset`, checkpoint DocStructBench, `epochs=80`, `batch=8`,
  `imgsz=1280`, `lr0=0.002`. Notebook co patch `check_amp` de tranh loi moi
  truong Kaggle.
- `htd-box-doclayout-yolo-v2.ipynb`: chuyen sang dataset
  `rukopys_augmented/train` va silver. Van dung `batch=8`, `lr0=0.002`; run nay
  hoan tat 45 epochs.
- `htd-box-fine-tune-doclayout-yolo-v2.ipynb`: fine-tune tiep tu checkpoint
  `Data Preprocessing DocLayout-Yolo.pt` tren gold train goc, `lr0=0.0002`,
  early stop voi best epoch 9 va tong 19 epochs.
- `htd-box-doclayout-yolo-v4.ipynb`: nhanh gan final tren
  `Merged_Rukopys_V1/train`, `USE_SILVER=False`, `lr0=0.001`, run dung o 40
  epochs. Notebook nay gan voi pipeline full train trong `Pipeline/bbox_YOLO`.

## Khac biet chinh

Ban dau uu tien thu backbone va format YOLO. Cac ban sau cai thien dataset va
learning rate. V4 la moc quan trong nhat trong nhom nay vi dung dataset merged va
cach train gan voi checkpoint final.
