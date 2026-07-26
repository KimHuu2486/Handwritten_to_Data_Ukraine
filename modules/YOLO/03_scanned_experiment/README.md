# 03 Scanned Experiment

Thu muc nay chua thu nghiem rieng tren tap scanned, theo yeu cau cap nhat them
phan chay tren dataset scanned.

## File trong thu muc

- `htd-box-doclayout-yolo-v4-new-perspective.ipynb`: train DocLayout-YOLO tren
  `BetterGoldScanned V1/train`, `USE_SILVER=False`, `VAL_RATIO=0.15`,
  `batch=8`, `lr0=0.001`. Run nay chay 100 epochs tren GPU T4 x2.

## Khi nao doc file nay

Doc sau khi da doc `02_doclayout_training/htd-box-doclayout-yolo-v4.ipynb`. Hai
notebook co cau truc training gan nhau, nhung file nay thay dataset tu merged
normal sang scanned dataset de kiem tra kha nang tong quat tren anh scan.
