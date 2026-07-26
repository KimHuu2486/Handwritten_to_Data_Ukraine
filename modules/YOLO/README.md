# YOLO / DocLayout-YOLO Bounding Box Module

Thu muc nay tong hop toan bo qua trinh thu nghiem module phat hien vung tai lieu
bang YOLO/DocLayout-YOLO cho bai toan Handwritten to Data. Module nay chiu trach
nhiem lay bbox va gan nhan 7 class: `handwritten`, `printed`, `formula`, `table`,
`annotation`, `image`, `graph`.

## Thu tu doc de xay lai quy trinh

1. `00_baseline`: cac thu nghiem ban dau voi YOLOv8 va quick inference.
2. `01_dataset_preparation`: cac notebook lam sach, gop, augment va tao bien the
   dataset.
3. `02_doclayout_training`: tien trinh chinh tu DocLayout-YOLO ban dau den V4.
4. `03_scanned_experiment`: thu nghiem rieng tren tap scanned.
5. `04_fold_experiments`: cac thu nghiem MRV2/MRV2.1 theo fold.
6. `05_submission`: notebook sinh file `submission.csv` cho Kaggle.
7. `../metric/YOLO`: cac notebook danh gia theo split noi bo va COCOeval.

## Y tuong chinh

Qua trinh bat dau tu baseline YOLOv8, sau do chuyen sang DocLayout-YOLO vi day la
backbone phu hop hon cho document layout detection. Cac lan sau tap trung vao:

- Cai thien du lieu gold/silver va augment class hiem.
- Fine-tune DocLayout-YOLO tu checkpoint DocStructBench.
- Thu nghiem dataset scanned va cac bien the MRV2/MRV2.1.
- Ensemble nhieu checkpoint khi sinh submission.
- Danh gia lai bang COCOeval de lay so lieu bao cao.

## File phu tro

- `notebook.txt`: danh sach notebook Kaggle, version hien tai va direct download
  link theo `scriptcontent/<run_id>/download`.
