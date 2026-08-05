# Evaluate Qwen3VL Metric Results

Hai notebook trong folder này có vai trò khác nhau:

| Notebook | Vai trò |
|---|---|
| [`official-evaluation-metric-text-normalization.ipynb`](./official-evaluation-metric-text-normalization.ipynb) | **Metric chính thức của BTC**: nguồn chuẩn cho normalisation, IoU matching, CER/PageCER và composite score. |
| [`evaluate_paper_submissions.ipynb`](./evaluate_paper_submissions.ipynb) | Batch evaluator của project: dùng logic official để chấm nhiều submission CSV và tạo bảng cho paper. |

## Metric BTC

```text
0.15 × Detection F1
+ 0.05 × Classification Accuracy
+ 0.30 × (1 − Region CER)
+ 0.50 × (1 − Page CER)
```

Metric chuẩn hóa text trước khi tính CER, gồm lookalike Cyrillic/Latin, dash, whitespace, apostrophe, format formula và table. Khi báo điểm chính thức, phải đối chiếu với notebook official thay vì dùng CER tự viết.

## Chuẩn bị để chạy `evaluate_paper_submissions.ipynb`

Bạn cần GT và một hoặc nhiều submission:

```text
<data-root>/test.jsonl
<submission-root>/full_system_results/*.csv
<submission-root>/routing_ablation/*.csv
<submission-root>/qwen_ablation/*.csv
<submission-root>/module_results/*.csv
```

Mỗi CSV phải có hai cột `image,regions`; `regions` là JSON list, mỗi region có `bbox: [x1,y1,x2,y2]`, `type` hợp lệ và `text`. Không có region thì ghi `[]`.

Notebook mặc định tìm root tên `fillpaper`, nhưng folder hiện tại là `metric/Qwen3VL`. Vì vậy sửa cell config đầu notebook thành đường dẫn rõ ràng:

```python
FILLPAPER_ROOT = Path("/kaggle/input/<dataset-chua-test-jsonl>")
GT_JSONL = FILLPAPER_ROOT / "test.jsonl"
OUTPUT_DIR = Path("/kaggle/working/paper_metric_outputs")

SUBMISSION_SEARCH_DIRS = [
    Path("/kaggle/input/<dataset-chua-submissions>/full_system_results"),
    Path("/kaggle/input/<dataset-chua-submissions>/routing_ablation"),
    Path("/kaggle/input/<dataset-chua-submissions>/qwen_ablation"),
    Path("/kaggle/input/<dataset-chua-submissions>/module_results"),
    Path("/kaggle/working"),
]
```

Nếu CSV ở chỗ khác, thêm path vào `EXTRA_SUBMISSION_FILES`. Trước khi chạy, kiểm tra `GT_JSONL.exists()` và schema của từng CSV.

## Thứ tự chạy

1. Chạy notebook experiment để sinh submission CSV.
2. Đặt CSV vào một folder trong `SUBMISSION_SEARCH_DIRS`.
3. Chạy toàn bộ cell của `evaluate_paper_submissions.ipynb`.
4. Đối chiếu điểm/công thức với notebook official của BTC.

Evaluator tạo `solution_from_test_jsonl.csv`, `all_submission_metrics.csv`, `per_type_metrics.csv`, `per_source_metrics.csv`, chi tiết invalid Qwen, `evaluation_errors.csv` và các bảng CSV/Markdown `paper_*` để đưa vào paper.

> [!IMPORTANT]
> Qwen/module ablation dùng GT boxes. Chỉ dùng Formula/Table CER, Structure Accuracy và Invalid Rate để kết luận về Qwen; không dùng full-page official score của chúng để thay thế kết quả full-system.
