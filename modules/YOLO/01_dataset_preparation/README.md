# 01 Dataset Preparation

Thu muc nay gom cac notebook tao va lam sach dataset cho detector. Day la phan
nen de hieu vi sao cac model sau dung nhieu bien the dataset khac nhau.

## Thu tu doc

1. `htd-box-lam-dep-dataset.ipynb`
2. `htd-box-better-gold.ipynb`
3. `htd-final-better-gold.ipynb`
4. `htd-bbox-merge-dataset-for-dlformer.ipynb`

## File trong thu muc

- `htd-box-lam-dep-dataset.ipynb`: lam dep/augment dataset goc, tao
  `rukopys_augmented_original_format.zip`.
- `htd-box-better-gold.ipynb`: tao bien the BetterGoldDatasetV1 tu train goc,
  co debug visualization va augment them 157 anh.
- `htd-final-better-gold.ipynb`: tao ban final better gold de dung cho cac
  notebook finetune va scoring ve sau.
- `htd-bbox-merge-dataset-for-dlformer.ipynb`: gop gold/silver va augment rare
  classes thanh `Merged_Rukopys_V1`; day la notebook dataset prep duoc dung trong
  nhanh metric/bao cao.

## Khac biet chinh

Cac notebook dau tap trung lam dep/gold dataset. Notebook merge dataset them silver
rare classes va augment de giam lech class, dac biet cac class it nhu `graph`,
`table`, `image`, `annotation`.
