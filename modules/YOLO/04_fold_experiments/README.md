# 04 Fold Experiments

Thu muc nay chua cac thu nghiem MRV2 va MRV2.1 theo fold. Muc tieu la thu cac
cach chia train/validation khac nhau va tao nhieu checkpoint de ensemble.

## Thu tu doc

1. `htd-box-doclayout-yolo-v5-2.ipynb`
2. `htd-box-doclayout-yolo-v5-3.ipynb`
3. `htd-box-doclayout-yolo-v5-1-1.ipynb`
4. `htd-box-doclayout-yolo-v5-1-2.ipynb`
5. `htd-box-doclayout-yolo-v5-1-3.ipynb`
6. `htd-box-doclayout-yolo-v5-1-4.ipynb`
7. `htd-box-doclayout-yolo-v5-1-5.ipynb`

## File trong thu muc

- `htd-box-doclayout-yolo-v5-2.ipynb`: MRV2 voi metadata 3-part,
  `VAL_FOLD=2`, `mosaic=0.30`, `epoch=100`, `imgsz=1024`, `patience=8`. Run
  dung o 64 epochs, best epoch 56.
- `htd-box-doclayout-yolo-v5-3.ipynb`: MRV2 voi metadata 3-part, dung fold khac
  voi v5-2. Run dung o 23 epochs, best epoch 15.
- `htd-box-doclayout-yolo-v5-1-1.ipynb`: MRV2.1 metadata, fold 1. Run dung o 44
  epochs, best epoch 36.
- `htd-box-doclayout-yolo-v5-1-2.ipynb`: MRV2.1 metadata, fold 2. Run dung o 19
  epochs, best epoch 11.
- `htd-box-doclayout-yolo-v5-1-3.ipynb`: MRV2.1 metadata, fold 3. Run dung o 41
  epochs, best epoch 33.
- `htd-box-doclayout-yolo-v5-1-4.ipynb`: MRV2.1 metadata, fold 4. Run dung o 21
  epochs, best epoch 13.
- `htd-box-doclayout-yolo-v5-1-5.ipynb`: MRV2.1 metadata, fold 5. Run dung o 22
  epochs, best epoch 14.

## Khac biet chinh

Nhom v5 dung `MRV2_metadata` 3-part. Nhom v5.1 dung `MRV2.1_Metadata` 5 fold.
Tat ca deu train DocLayout-YOLO tren `Merged_Rukopys_V2/train`, khong dung silver
trong lan train chinh.
