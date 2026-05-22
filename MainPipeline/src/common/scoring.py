from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_metric_module(notebook_path: Path, metric_path: Path):
    """Load the official-compatible metric module, extracting it from the notebook if needed."""
    if not metric_path.exists():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metric_code: str | None = None
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            if "%%writefile kaggle_metric.py" in source and "def score_detailed" in source:
                lines = source.splitlines()
                if lines and lines[0].startswith("%%writefile"):
                    lines = lines[1:]
                metric_code = "\n".join(lines).rstrip() + "\n"
                break
        if metric_code is None:
            raise ValueError(f"Could not find kaggle_metric.py cell in {notebook_path}")
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(metric_code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("main_pipeline_kaggle_metric", metric_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import metric module from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def solution_dataframe(manifest_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert validation manifest rows into the official solution dataframe shape."""
    rows = []
    for row in manifest_rows:
        gt_regions = []
        for region in row.get("regions", []):
            gt_regions.append(
                {
                    "bbox": region.get("bbox", [0, 0, 0, 0]),
                    "type": region.get("type", "handwritten"),
                    "text": "" if region.get("text") is None else str(region.get("text")),
                }
            )
        rows.append({"image": row["submission_image"], "regions": json.dumps(gt_regions, ensure_ascii=False)})
    return pd.DataFrame(rows)


def score_validation(
    manifest_rows: list[dict[str, Any]],
    prediction_csv: Path,
    metric_notebook_path: Path,
    metric_path: Path,
) -> dict[str, Any]:
    """Score a validation predictions CSV against the frozen manifest."""
    metric = ensure_metric_module(metric_notebook_path, metric_path)
    predictions = pd.read_csv(prediction_csv, encoding="utf-8")
    submission = predictions[["image", "regions"]].copy()
    detailed = metric.score_detailed(solution_dataframe(manifest_rows), submission, "image")
    total_score = detailed.get("total_score", detailed.get("composite_score"))
    return {
        "total_score": float(total_score),
        "detection_f1": float(detailed["detection_f1"]),
        "class_acc": float(detailed["classification_accuracy"]),
        "region_cer": float(detailed["region_cer"]),
        "page_cer": float(detailed["page_cer"]),
        "metric_details": {
            "detection_precision": float(detailed.get("detection_precision", 0.0)),
            "detection_recall": float(detailed.get("detection_recall", 0.0)),
            "matched_regions": int(detailed.get("matched_regions", detailed.get("n_matched_regions", 0))),
            "false_positives": int(detailed.get("n_false_positives", 0)),
            "false_negatives": int(detailed.get("n_false_negatives", 0)),
            "n_images": int(detailed.get("n_images", len(predictions))),
        },
    }

