# Chia dữ liệu OCR 70/15/15

- File gốc: `metadata.jsonl`
- Tổng số mẫu page-level: **1330**
- Seed: **1089**
- Nguyên tắc: chia theo **page-level**, không tách các `regions` trong cùng một page sang nhiều split khác nhau.
- Cách chia: shuffle có seed và stratify theo trường `source` để train/val/test giữ phân phối nguồn dữ liệu tương đối cân bằng.
- Mục đích: `train` dùng để huấn luyện, `val` dùng để chọn checkpoint/tune hyperparameter, `test` giữ riêng để báo cáo metric cuối cùng trong paper.

## Số lượng mẫu

| Split | Số page | Tỉ lệ |
|---|---:|---:|
| Train | 931 | 70.00% |
| Validation | 199 | 14.96% |
| Test | 200 | 15.04% |

## Phân phối theo source

| Source | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| archive | 89 | 19 | 19 | 127 |
| dictation | 251 | 54 | 54 | 359 |
| school | 478 | 102 | 102 | 682 |
| university | 113 | 24 | 25 | 162 |

## Thống kê

| Thuộc tính | Train | Val | Test |
|---|---:|---:|---:|
| Tổng số regions | 18007 | 3823 | 3821 |
| Tổng số ký tự text | 538833 | 110607 | 116449 |
| Page có formula | 231 | 51 | 53 |
| Page có annotation | 201 | 44 | 40 |
| Page có table | 50 | 9 | 6 |
| Page có image | 38 | 16 | 10 |
| Page có graph | 29 | 5 | 5 |
| Page có language khác uk | 102 | 31 | 16 |
| Page có vùng không legible | 51 | 18 | 16 |

## Lưu ý sử dụng trong paper

Because official test annotations are hidden, the internal test split should be used only for final evaluation. Pseudo-labeled data, if used, should be reported as semi-supervised augmentation rather than ground-truth evaluation.
