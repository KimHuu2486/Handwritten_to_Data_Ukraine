#!/usr/bin/env python
"""
Compare RUKOPYS validation predictions against gold records.

This script is intentionally separate from the inference notebooks. Pass one or
more prediction CSV files and it will:

- build a validation solution CSV from gold_validation_records.jsonl
- load the official metric implementation from the official metric notebook
- score every prediction file
- produce per-image, per-region, and cross-model comparison reports
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GOLD = "gold_validation_records.jsonl"
DEFAULT_METRIC = "official-evaluation-metric-text-normalization.ipynb"
DEFAULT_OUT_DIR = "artifacts/validation_compare"

IMAGE_REPORT_COLUMNS = [
    "model",
    "image",
    "source",
    "annotation_source",
    "image_width",
    "image_height",
    "gt_regions",
    "pred_regions",
    "matched_regions",
    "false_positives",
    "false_negatives",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "classification_accuracy",
    "class_wrong",
    "scorable_matches",
    "text_errors",
    "empty_text_errors",
    "region_cer",
    "page_cer",
    "image_score",
    "gt_page_chars",
    "pred_page_chars",
    "gt_page_text",
    "pred_page_text",
]

MATCH_REPORT_COLUMNS = [
    "model",
    "image",
    "source",
    "gt_index",
    "pred_index",
    "iou",
    "gt_type",
    "pred_type",
    "type_correct",
    "scorable",
    "cer",
    "flags",
    "gt_text",
    "pred_text",
    "gt_norm",
    "pred_norm",
]

ERROR_REPORT_COLUMNS = [
    "model",
    "image",
    "source",
    "error_type",
    "reason",
    "gt_index",
    "pred_index",
    "iou",
    "gt_type",
    "pred_type",
    "cer",
    "gt_text",
    "pred_text",
    "gt_norm",
    "pred_norm",
]

TYPE_LOSS_REPORT_COLUMNS = [
    "model",
    "rank_by_estimated_total_loss",
    "rank_by_error_rate",
    "rank_by_loss_per_gt_region",
    "region_type",
    "gt_regions",
    "pred_regions_as_type",
    "matched_regions",
    "scorable_matches",
    "missed_regions",
    "extra_regions",
    "wrong_type_count",
    "text_error_count",
    "empty_text_count",
    "bad_gt_regions",
    "error_rate",
    "missed_rate",
    "extra_rate",
    "wrong_type_rate",
    "avg_cer",
    "cer_loss_sum",
    "detection_loss",
    "classification_loss",
    "text_loss",
    "estimated_total_loss",
    "estimated_loss_per_gt_region",
    "loss_share",
    "dominant_issue",
]


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score and compare validation prediction CSVs against "
            "gold_validation_records.jsonl using the official metric notebook."
        )
    )
    parser.add_argument(
        "--gold",
        default=DEFAULT_GOLD,
        help=f"Path to gold validation JSONL. Default: {DEFAULT_GOLD}",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Official metric .ipynb or .py path. Default: {DEFAULT_METRIC}",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for reports. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help=(
            "Prediction CSV. Repeat for each model, for example "
            "--pred full=full_validation_pred.csv. If NAME= is omitted, "
            "the CSV stem is used."
        ),
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="Optional bare prediction CSV paths. Model name is inferred from the filename stem.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for debug matching. Default matches the official metric: 0.5.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of worst images/errors to show in the Markdown report.",
    )
    return parser.parse_args()


def parse_prediction_specs(named_specs: list[str], bare_csvs: list[str]) -> list[PredictionSpec]:
    specs: list[PredictionSpec] = []
    seen: set[str] = set()

    def add(raw: str, force_bare: bool = False) -> None:
        if "=" in raw and not force_bare:
            name, path_text = raw.split("=", 1)
            name = name.strip()
            path = Path(path_text.strip())
        else:
            path = Path(raw.strip())
            name = path.stem
        if not name:
            raise SystemExit(f"Invalid prediction spec: {raw!r}. Use NAME=path.csv.")
        if name in seen:
            raise SystemExit(f"Duplicate model name: {name!r}")
        if not path.exists():
            raise SystemExit(f"Prediction CSV not found for {name!r}: {path}")
        seen.add(name)
        specs.append(PredictionSpec(name=name, path=path))

    for item in named_specs:
        add(item)
    for item in bare_csvs:
        add(item, force_bare=True)
    if not specs:
        raise SystemExit("Pass at least one prediction CSV with --pred NAME=path.csv.")
    return specs


def load_metric_namespace(metric_path: Path) -> dict[str, Any]:
    if not metric_path.exists():
        raise SystemExit(f"Official metric file not found: {metric_path}")

    if metric_path.suffix.lower() == ".py":
        code = metric_path.read_text(encoding="utf-8")
        source_label = str(metric_path)
    elif metric_path.suffix.lower() == ".ipynb":
        notebook = json.loads(metric_path.read_text(encoding="utf-8"))
        code = ""
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source") or [])
            if "%%writefile kaggle_metric.py" in source and "def score(" in source:
                lines = source.splitlines()
                if lines and lines[0].lstrip().startswith("%%writefile"):
                    lines = lines[1:]
                code = "\n".join(lines)
                break
        if not code:
            code_cells = [
                "".join(cell.get("source") or [])
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            ]
            code = "\n\n".join(code_cells)
        source_label = f"{metric_path}:metric"
    else:
        raise SystemExit("Metric must be a .ipynb or .py file.")

    namespace: dict[str, Any] = {"__name__": "loaded_kaggle_metric"}
    exec(compile(code, source_label, "exec"), namespace)

    required = [
        "score_detailed",
        "_parse_regions",
        "_greedy_match",
        "_compute_iou",
        "_is_scorable",
        "_normalize_text",
        "_levenshtein",
        "_build_page_text",
    ]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise SystemExit(f"Official metric is missing required functions: {missing}")
    return namespace


def load_gold_records(gold_path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if not gold_path.exists():
        raise SystemExit(f"Gold validation JSONL not found: {gold_path}")

    rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {gold_path}:{line_no}: {exc}") from exc
            image = Path(rec["file_name"]).name
            regions = rec.get("regions") or []
            rows.append({"image": image, "regions": json.dumps(regions, ensure_ascii=False)})
            metadata[image] = {
                "file_name": rec.get("file_name", image),
                "source": rec.get("source", ""),
                "annotation_source": rec.get("annotation_source", ""),
                "image_width": rec.get("image_width"),
                "image_height": rec.get("image_height"),
                "gt_region_count": len(regions),
            }
    if not rows:
        raise SystemExit(f"No records found in {gold_path}")
    return pd.DataFrame(rows), metadata


def prepare_prediction_df(
    spec: PredictionSpec,
    solution_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_df = pd.read_csv(spec.path)
    if "image" not in raw_df.columns or "regions" not in raw_df.columns:
        raise SystemExit(f"{spec.path} must contain columns: image, regions")

    raw_len = len(raw_df)
    df = raw_df[["image", "regions"]].copy()
    df["image"] = df["image"].map(lambda x: Path(str(x)).name)
    duplicate_count = int(df.duplicated(subset=["image"]).sum())
    df = df.drop_duplicates(subset=["image"], keep="last")
    df["regions"] = df["regions"].fillna("[]")

    expected = solution_df[["image"]].copy()
    expected_set = set(expected["image"])
    pred_set = set(df["image"])
    missing = sorted(expected_set - pred_set)
    extra = sorted(pred_set - expected_set)

    scoring_df = expected.merge(df, on="image", how="left")
    scoring_df["regions"] = scoring_df["regions"].fillna("[]")

    meta = {
        "raw_rows": raw_len,
        "scored_rows": len(scoring_df),
        "duplicate_rows": duplicate_count,
        "missing_images": len(missing),
        "extra_images": len(extra),
        "missing_examples": "; ".join(missing[:10]),
        "extra_examples": "; ".join(extra[:10]),
    }
    return scoring_df, meta


def safe_mean(values: list[float], default: float = math.nan) -> float:
    clean = [v for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return default
    return float(sum(clean) / len(clean))


def truncate_text(value: Any, max_len: int = 220) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def classify_text_error(gt_norm: str, pred_norm: str, region_type: str, cer: float) -> str:
    if not str(pred_norm).strip():
        return "empty_text"
    gt_len = len(gt_norm)
    pred_len = len(pred_norm)
    if gt_len and pred_len < gt_len * 0.55:
        return "missing_or_truncated_text"
    if gt_len and pred_len > gt_len * 1.6:
        return "extra_or_hallucinated_text"
    if region_type in {"formula", "table"}:
        return f"{region_type}_text_mismatch"
    if cer < 0.15:
        return "minor_text_mismatch"
    if cer >= 0.50:
        return "severe_text_mismatch"
    return "text_mismatch"


def image_component_score(det_f1: float, class_acc: float, region_cer: float, page_cer: float) -> float:
    return (
        0.15 * det_f1
        + 0.05 * class_acc
        + 0.30 * max(0.0, 1.0 - region_cer)
        + 0.50 * max(0.0, 1.0 - page_cer)
    )


def analyze_prediction(
    model_name: str,
    solution_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    metric: dict[str, Any],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    parse_regions = metric["_parse_regions"]
    greedy_match = metric["_greedy_match"]
    compute_iou = metric["_compute_iou"]
    is_scorable = metric["_is_scorable"]
    normalize_text = metric["_normalize_text"]
    levenshtein = metric["_levenshtein"]
    build_page_text = metric["_build_page_text"]

    pred_lookup = dict(zip(pred_df["image"], pred_df["regions"]))
    image_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for _, sol_row in solution_df.iterrows():
        image = sol_row["image"]
        meta = metadata.get(image, {})
        source = meta.get("source", "")
        gt = parse_regions(sol_row["regions"], image, is_submission=False)
        pred = parse_regions(pred_lookup.get(image, "[]"), image, is_submission=True)

        matched, unmatched_gt, unmatched_pred = greedy_match(gt, pred, threshold=iou_threshold)
        matched_pairs = set(matched)
        matched_gt = {gi for gi, _ in matched_pairs}
        matched_pred = {pi for _, pi in matched_pairs}

        det_tp = len(matched)
        det_fp = len(unmatched_pred)
        det_fn = len(unmatched_gt)
        det_prec = det_tp / max(det_tp + det_fp, 1)
        det_rec = det_tp / max(det_tp + det_fn, 1)
        det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-9)

        class_total = len(matched)
        class_correct = sum(1 for gi, pi in matched if gt[gi]["type"] == pred[pi]["type"])
        class_acc = class_correct / max(class_total, 1)

        region_cers: list[float] = []
        text_error_count = 0
        empty_text_count = 0
        wrong_type_count = 0

        for gi, pi in matched:
            gt_region = gt[gi]
            pred_region = pred[pi]
            region_type = gt_region.get("type", "handwritten")
            iou = compute_iou(gt_region["bbox"], pred_region["bbox"])
            type_correct = gt_region.get("type") == pred_region.get("type")
            scorable = bool(is_scorable(gt_region))
            gt_norm = ""
            pred_norm = ""
            cer = math.nan
            flags: list[str] = []

            if not type_correct:
                wrong_type_count += 1
                flags.append("wrong_type")
                error_rows.append(
                    {
                        "model": model_name,
                        "image": image,
                        "source": source,
                        "error_type": "wrong_type",
                        "reason": "bbox_matched_but_type_differs",
                        "gt_index": gi,
                        "pred_index": pi,
                        "iou": iou,
                        "gt_type": gt_region.get("type"),
                        "pred_type": pred_region.get("type"),
                        "cer": "",
                        "gt_text": truncate_text(gt_region.get("text", "")),
                        "pred_text": truncate_text(pred_region.get("text", "")),
                    }
                )

            if scorable:
                gt_norm = normalize_text(gt_region.get("text", ""), region_type)
                pred_norm = normalize_text(pred_region.get("text", ""), region_type)
                cer = levenshtein(pred_norm, gt_norm) / max(len(gt_norm), 1)
                region_cers.append(float(cer))
                if not str(pred_region.get("text", "")).strip():
                    empty_text_count += 1
                if cer > 0:
                    text_error_count += 1
                    reason = classify_text_error(gt_norm, pred_norm, region_type, float(cer))
                    flags.append(reason)
                    error_rows.append(
                        {
                            "model": model_name,
                            "image": image,
                            "source": source,
                            "error_type": "text_error",
                            "reason": reason,
                            "gt_index": gi,
                            "pred_index": pi,
                            "iou": iou,
                            "gt_type": gt_region.get("type"),
                            "pred_type": pred_region.get("type"),
                            "cer": cer,
                            "gt_text": truncate_text(gt_region.get("text", "")),
                            "pred_text": truncate_text(pred_region.get("text", "")),
                            "gt_norm": truncate_text(gt_norm),
                            "pred_norm": truncate_text(pred_norm),
                        }
                    )

            match_rows.append(
                {
                    "model": model_name,
                    "image": image,
                    "source": source,
                    "gt_index": gi,
                    "pred_index": pi,
                    "iou": iou,
                    "gt_type": gt_region.get("type"),
                    "pred_type": pred_region.get("type"),
                    "type_correct": type_correct,
                    "scorable": scorable,
                    "cer": cer,
                    "flags": ";".join(flags),
                    "gt_text": truncate_text(gt_region.get("text", "")),
                    "pred_text": truncate_text(pred_region.get("text", "")),
                    "gt_norm": truncate_text(gt_norm),
                    "pred_norm": truncate_text(pred_norm),
                }
            )

        for gi in unmatched_gt:
            gt_region = gt[gi]
            nearest_iou = max(
                [compute_iou(gt_region["bbox"], pred_region["bbox"]) for pred_region in pred],
                default=0.0,
            )
            reason = "bbox_shift_low_iou" if nearest_iou >= 0.20 else "missing_detection"
            error_rows.append(
                {
                    "model": model_name,
                    "image": image,
                    "source": source,
                    "error_type": "missed_region",
                    "reason": reason,
                    "gt_index": gi,
                    "pred_index": "",
                    "iou": nearest_iou,
                    "gt_type": gt_region.get("type"),
                    "pred_type": "",
                    "cer": "",
                    "gt_text": truncate_text(gt_region.get("text", "")),
                    "pred_text": "",
                }
            )

        for pi in unmatched_pred:
            pred_region = pred[pi]
            nearest_iou = max(
                [compute_iou(gt_region["bbox"], pred_region["bbox"]) for gt_region in gt],
                default=0.0,
            )
            reason = "bbox_shift_low_iou" if nearest_iou >= 0.20 else "extra_detection"
            error_rows.append(
                {
                    "model": model_name,
                    "image": image,
                    "source": source,
                    "error_type": "extra_region",
                    "reason": reason,
                    "gt_index": "",
                    "pred_index": pi,
                    "iou": nearest_iou,
                    "gt_type": "",
                    "pred_type": pred_region.get("type"),
                    "cer": "",
                    "gt_text": "",
                    "pred_text": truncate_text(pred_region.get("text", "")),
                }
            )

        pred_drop = {pi for gi, pi in matched if not is_scorable(gt[gi])}
        gt_page = build_page_text(gt, normalize=True)
        pred_page = build_page_text(pred, normalize=True, drop_indices=pred_drop)
        page_cer = (
            levenshtein(pred_page, gt_page) / len(gt_page)
            if len(gt_page) > 0
            else math.nan
        )
        region_cer = safe_mean(region_cers, default=1.0)
        page_cer_for_score = float(page_cer) if not math.isnan(page_cer) else 1.0
        image_score = image_component_score(det_f1, class_acc, region_cer, page_cer_for_score)

        image_rows.append(
            {
                "model": model_name,
                "image": image,
                "source": source,
                "annotation_source": meta.get("annotation_source", ""),
                "image_width": meta.get("image_width", ""),
                "image_height": meta.get("image_height", ""),
                "gt_regions": len(gt),
                "pred_regions": len(pred),
                "matched_regions": det_tp,
                "false_positives": det_fp,
                "false_negatives": det_fn,
                "detection_precision": det_prec,
                "detection_recall": det_rec,
                "detection_f1": det_f1,
                "classification_accuracy": class_acc,
                "class_wrong": wrong_type_count,
                "scorable_matches": len(region_cers),
                "text_errors": text_error_count,
                "empty_text_errors": empty_text_count,
                "region_cer": region_cer,
                "page_cer": page_cer,
                "image_score": image_score,
                "gt_page_chars": len(gt_page),
                "pred_page_chars": len(pred_page),
                "gt_page_text": truncate_text(gt_page, 400),
                "pred_page_text": truncate_text(pred_page, 400),
            }
        )

    return image_rows, match_rows, error_rows


def build_slice_summary(
    image_report: pd.DataFrame,
    match_report: pd.DataFrame,
    error_report: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not image_report.empty:
        for (model, source), group in image_report.groupby(["model", "source"], dropna=False):
            rows.append(
                {
                    "scope": "source",
                    "value": source,
                    "model": model,
                    "n_images": len(group),
                    "n_regions": int(group["gt_regions"].sum()),
                    "avg_image_score": group["image_score"].mean(),
                    "avg_detection_f1": group["detection_f1"].mean(),
                    "avg_region_cer": group["region_cer"].mean(),
                    "avg_page_cer": group["page_cer"].mean(),
                    "errors": int(error_report[(error_report["model"] == model) & (error_report["source"] == source)].shape[0])
                    if not error_report.empty
                    else 0,
                }
            )

    if not match_report.empty:
        for (model, gt_type), group in match_report.groupby(["model", "gt_type"], dropna=False):
            scorable = group[group["scorable"] == True]  # noqa: E712
            type_errors = int((group["type_correct"] == False).sum())  # noqa: E712
            rows.append(
                {
                    "scope": "gt_type_matched",
                    "value": gt_type,
                    "model": model,
                    "n_images": group["image"].nunique(),
                    "n_regions": len(group),
                    "avg_image_score": math.nan,
                    "avg_detection_f1": math.nan,
                    "avg_region_cer": scorable["cer"].mean() if not scorable.empty else math.nan,
                    "avg_page_cer": math.nan,
                    "errors": type_errors,
                }
            )

    if error_report.empty:
        return pd.DataFrame(rows)

    missed = error_report[error_report["error_type"] == "missed_region"]
    if not missed.empty:
        for (model, gt_type), group in missed.groupby(["model", "gt_type"], dropna=False):
            rows.append(
                {
                    "scope": "gt_type_missed",
                    "value": gt_type,
                    "model": model,
                    "n_images": group["image"].nunique(),
                    "n_regions": len(group),
                    "avg_image_score": math.nan,
                    "avg_detection_f1": math.nan,
                    "avg_region_cer": math.nan,
                    "avg_page_cer": math.nan,
                    "errors": len(group),
                }
            )

    return pd.DataFrame(rows)


def build_type_loss_summary(
    match_report: pd.DataFrame,
    error_report: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if match_report.empty and error_report.empty:
        return pd.DataFrame(columns=TYPE_LOSS_REPORT_COLUMNS)

    models = set()
    if not match_report.empty:
        models.update(str(model) for model in match_report["model"].dropna().unique())
    if not error_report.empty:
        models.update(str(model) for model in error_report["model"].dropna().unique())

    for model in sorted(models):
        matches = (
            match_report[match_report["model"].astype(str) == model].copy()
            if not match_report.empty
            else pd.DataFrame()
        )
        errors = (
            error_report[error_report["model"].astype(str) == model].copy()
            if not error_report.empty
            else pd.DataFrame()
        )

        region_types: set[str] = set()
        if not matches.empty:
            region_types.update(str(value) for value in matches["gt_type"].dropna().unique() if str(value))
            region_types.update(str(value) for value in matches["pred_type"].dropna().unique() if str(value))
        if not errors.empty:
            region_types.update(str(value) for value in errors["gt_type"].dropna().unique() if str(value))
            region_types.update(str(value) for value in errors["pred_type"].dropna().unique() if str(value))

        for region_type in sorted(region_types):
            type_matches = (
                matches[matches["gt_type"].astype(str) == region_type].copy()
                if not matches.empty
                else pd.DataFrame()
            )
            pred_type_matches = (
                matches[matches["pred_type"].astype(str) == region_type].copy()
                if not matches.empty
                else pd.DataFrame()
            )
            missed = (
                errors[
                    (errors["error_type"] == "missed_region")
                    & (errors["gt_type"].astype(str) == region_type)
                ].copy()
                if not errors.empty
                else pd.DataFrame()
            )
            extra = (
                errors[
                    (errors["error_type"] == "extra_region")
                    & (errors["pred_type"].astype(str) == region_type)
                ].copy()
                if not errors.empty
                else pd.DataFrame()
            )

            gt_regions = len(type_matches) + len(missed)
            pred_regions_as_type = len(pred_type_matches) + len(extra)
            matched_regions = len(type_matches)

            if type_matches.empty:
                scorable = pd.DataFrame()
                cer_values = pd.Series(dtype="float64")
                wrong_type_count = 0
                text_error_mask = pd.Series(dtype="bool")
                bad_gt_keys: set[tuple[str, str]] = set()
            else:
                scorable = type_matches[type_matches["scorable"] == True].copy()  # noqa: E712
                cer_values = pd.to_numeric(scorable["cer"], errors="coerce").dropna()
                wrong_mask = type_matches["type_correct"] == False  # noqa: E712
                wrong_type_count = int(wrong_mask.sum())
                text_error_mask = pd.to_numeric(type_matches["cer"], errors="coerce").fillna(0) > 0
                bad_matches = type_matches[wrong_mask | text_error_mask]
                bad_gt_keys = {
                    (str(row["image"]), str(row["gt_index"]))
                    for _, row in bad_matches.iterrows()
                }

            missed_keys = {
                (str(row["image"]), str(row["gt_index"]))
                for _, row in missed.iterrows()
            }
            bad_gt_keys.update(missed_keys)

            text_error_count = int((cer_values > 0).sum())
            empty_text_count = 0
            if not errors.empty:
                empty_text_count = int(
                    errors[
                        (errors["error_type"] == "text_error")
                        & (errors["reason"] == "empty_text")
                        & (errors["gt_type"].astype(str) == region_type)
                    ].shape[0]
                )

            missed_regions = len(missed)
            extra_regions = len(extra)
            scorable_matches = len(scorable)
            bad_gt_regions = len(bad_gt_keys)
            cer_loss_sum = float(cer_values.clip(lower=0.0, upper=1.0).sum())
            avg_cer = float(cer_values.mean()) if not cer_values.empty else math.nan

            detection_loss = 0.15 * (missed_regions + extra_regions)
            classification_loss = 0.05 * wrong_type_count
            text_loss = 0.30 * cer_loss_sum
            estimated_total_loss = detection_loss + classification_loss + text_loss

            component_losses = {
                "detection_missed_or_extra": detection_loss,
                "classification_wrong_type": classification_loss,
                "text_cer": text_loss,
            }
            dominant_issue, dominant_loss = max(component_losses.items(), key=lambda kv: kv[1])
            if dominant_loss <= 0:
                dominant_issue = "none"

            rows.append(
                {
                    "model": model,
                    "region_type": region_type,
                    "gt_regions": gt_regions,
                    "pred_regions_as_type": pred_regions_as_type,
                    "matched_regions": matched_regions,
                    "scorable_matches": scorable_matches,
                    "missed_regions": missed_regions,
                    "extra_regions": extra_regions,
                    "wrong_type_count": wrong_type_count,
                    "text_error_count": text_error_count,
                    "empty_text_count": empty_text_count,
                    "bad_gt_regions": bad_gt_regions,
                    "error_rate": bad_gt_regions / gt_regions if gt_regions else math.nan,
                    "missed_rate": missed_regions / gt_regions if gt_regions else math.nan,
                    "extra_rate": extra_regions / pred_regions_as_type if pred_regions_as_type else math.nan,
                    "wrong_type_rate": wrong_type_count / matched_regions if matched_regions else math.nan,
                    "avg_cer": avg_cer,
                    "cer_loss_sum": cer_loss_sum,
                    "detection_loss": detection_loss,
                    "classification_loss": classification_loss,
                    "text_loss": text_loss,
                    "estimated_total_loss": estimated_total_loss,
                    "estimated_loss_per_gt_region": estimated_total_loss / gt_regions
                    if gt_regions
                    else math.nan,
                    "dominant_issue": dominant_issue,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=TYPE_LOSS_REPORT_COLUMNS)

    summary["loss_share"] = 0.0
    summary["rank_by_estimated_total_loss"] = 0
    summary["rank_by_error_rate"] = 0
    summary["rank_by_loss_per_gt_region"] = 0

    for model, group in summary.groupby("model", dropna=False):
        idx = group.index
        total_loss = float(group["estimated_total_loss"].sum())
        if total_loss > 0:
            summary.loc[idx, "loss_share"] = group["estimated_total_loss"] / total_loss
        summary.loc[idx, "rank_by_estimated_total_loss"] = (
            group["estimated_total_loss"].rank(method="dense", ascending=False).astype(int)
        )
        summary.loc[idx, "rank_by_error_rate"] = (
            group["error_rate"].fillna(-1).rank(method="dense", ascending=False).astype(int)
        )
        summary.loc[idx, "rank_by_loss_per_gt_region"] = (
            group["estimated_loss_per_gt_region"]
            .fillna(-1)
            .rank(method="dense", ascending=False)
            .astype(int)
        )

    return summary.sort_values(
        ["model", "rank_by_estimated_total_loss", "estimated_total_loss"],
        ascending=[True, True, False],
    )


def build_model_comparison(image_report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if image_report.empty:
        return pd.DataFrame()

    for image, group in image_report.groupby("image", sort=False):
        sorted_group = group.sort_values(
            ["image_score", "page_cer", "region_cer", "detection_f1"],
            ascending=[False, True, True, False],
        )
        best = sorted_group.iloc[0]
        runner = sorted_group.iloc[1] if len(sorted_group) > 1 else None
        row: dict[str, Any] = {
            "image": image,
            "source": best.get("source", ""),
            "winner": best["model"],
            "winner_score": best["image_score"],
            "runner_up": runner["model"] if runner is not None else "",
            "margin": best["image_score"] - runner["image_score"] if runner is not None else math.nan,
            "winner_reason": infer_winner_reason(best, runner),
        }
        for _, item in group.iterrows():
            prefix = str(item["model"])
            row[f"{prefix}_score"] = item["image_score"]
            row[f"{prefix}_det_f1"] = item["detection_f1"]
            row[f"{prefix}_region_cer"] = item["region_cer"]
            row[f"{prefix}_page_cer"] = item["page_cer"]
            row[f"{prefix}_fp"] = item["false_positives"]
            row[f"{prefix}_fn"] = item["false_negatives"]
        rows.append(row)
    return pd.DataFrame(rows)


def infer_winner_reason(best: pd.Series, runner: pd.Series | None) -> str:
    if runner is None:
        return "only_model"
    diffs = {
        "better_page_cer": runner["page_cer"] - best["page_cer"],
        "better_region_cer": runner["region_cer"] - best["region_cer"],
        "better_detection": best["detection_f1"] - runner["detection_f1"],
        "better_classification": best["classification_accuracy"] - runner["classification_accuracy"],
    }
    reason, value = max(diffs.items(), key=lambda kv: kv[1] if not math.isnan(float(kv[1])) else -999)
    if value <= 1e-9:
        return "small_or_mixed_margin"
    return reason


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if math.isnan(value):
                    value = ""
                else:
                    value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    extra_columns = [column for column in df.columns if column not in columns]
    return df[columns + extra_columns]


def write_markdown_report(
    out_path: Path,
    scores: pd.DataFrame,
    comparison: pd.DataFrame,
    image_report: pd.DataFrame,
    error_report: pd.DataFrame,
    slice_summary: pd.DataFrame,
    type_loss_summary: pd.DataFrame,
    output_files: list[Path],
    top_k: int,
) -> None:
    score_rows = scores.sort_values("composite_score", ascending=False).to_dict("records")

    best_component_rows = []
    for component, ascending in [
        ("composite_score", False),
        ("detection_f1", False),
        ("classification_accuracy", False),
        ("region_cer", True),
        ("page_cer", True),
    ]:
        ordered = scores.sort_values(component, ascending=ascending)
        if not ordered.empty:
            row = ordered.iloc[0]
            best_component_rows.append(
                {
                    "component": component,
                    "best_model": row["model"],
                    "value": row[component],
                }
            )

    worst_rows = []
    if not image_report.empty:
        worst_rows = (
            image_report.sort_values(["image_score", "page_cer"], ascending=[True, False])
            .head(top_k)
            .loc[:, ["model", "image", "source", "image_score", "detection_f1", "region_cer", "page_cer", "false_positives", "false_negatives", "text_errors"]]
            .to_dict("records")
        )

    error_counts = []
    if not error_report.empty:
        counts = error_report.groupby(["model", "error_type", "reason"]).size().reset_index(name="count")
        error_counts = counts.sort_values("count", ascending=False).head(top_k).to_dict("records")

    source_winners = []
    if not slice_summary.empty:
        source_rows = slice_summary[slice_summary["scope"] == "source"].copy()
        for source, group in source_rows.groupby("value", dropna=False):
            best = group.sort_values("avg_image_score", ascending=False).iloc[0]
            source_winners.append(
                {
                    "source": source,
                    "best_model": best["model"],
                    "avg_image_score": best["avg_image_score"],
                    "n_images": best["n_images"],
                }
            )

    model_wins = []
    if not comparison.empty:
        wins = comparison["winner"].value_counts().reset_index()
        wins.columns = ["model", "image_wins"]
        model_wins = wins.to_dict("records")

    type_loss_rows = []
    if not type_loss_summary.empty:
        type_loss_rows = (
            type_loss_summary.sort_values(
                ["model", "rank_by_estimated_total_loss", "estimated_total_loss"],
                ascending=[True, True, False],
            )
            .groupby("model", dropna=False)
            .head(top_k)
            .loc[
                :,
                [
                    "model",
                    "rank_by_estimated_total_loss",
                    "rank_by_error_rate",
                    "region_type",
                    "gt_regions",
                    "error_rate",
                    "estimated_total_loss",
                    "estimated_loss_per_gt_region",
                    "loss_share",
                    "missed_regions",
                    "extra_regions",
                    "wrong_type_count",
                    "text_error_count",
                    "avg_cer",
                    "dominant_issue",
                ],
            ]
            .to_dict("records")
        )

    lines = [
        "# Validation Comparison Report",
        "",
        "## Score summary",
        markdown_table(
            score_rows,
            [
                "model",
                "composite_score",
                "detection_f1",
                "detection_precision",
                "detection_recall",
                "classification_accuracy",
                "region_cer",
                "page_cer",
                "n_false_positives",
                "n_false_negatives",
                "missing_images",
                "extra_images",
            ],
        ),
        "",
        "## Best model by component",
        markdown_table(best_component_rows, ["component", "best_model", "value"]),
        "",
        "## Image wins",
        markdown_table(model_wins, ["model", "image_wins"]),
        "",
        "## Best model by source",
        markdown_table(source_winners, ["source", "best_model", "avg_image_score", "n_images"]),
        "",
        "## Most frequent errors",
        markdown_table(error_counts, ["model", "error_type", "reason", "count"]),
        "",
        "## Weakest types by estimated loss",
        markdown_table(
            type_loss_rows,
            [
                "model",
                "rank_by_estimated_total_loss",
                "rank_by_error_rate",
                "region_type",
                "gt_regions",
                "error_rate",
                "estimated_total_loss",
                "estimated_loss_per_gt_region",
                "loss_share",
                "missed_regions",
                "extra_regions",
                "wrong_type_count",
                "text_error_count",
                "avg_cer",
                "dominant_issue",
            ],
        ),
        "",
        f"## Worst {top_k} image/model rows",
        markdown_table(
            worst_rows,
            [
                "model",
                "image",
                "source",
                "image_score",
                "detection_f1",
                "region_cer",
                "page_cer",
                "false_positives",
                "false_negatives",
                "text_errors",
            ],
        ),
        "",
        "## Output files",
    ]
    lines.extend(f"- {path}" for path in output_files)
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    specs = parse_prediction_specs(args.pred, args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = load_metric_namespace(Path(args.metric))
    solution_df, metadata = load_gold_records(Path(args.gold))
    solution_path = out_dir / "validation_solution.csv"
    solution_df.to_csv(solution_path, index=False)

    score_rows: list[dict[str, Any]] = []
    all_image_rows: list[dict[str, Any]] = []
    all_match_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []
    prepared_prediction_paths: list[Path] = []

    for spec in specs:
        pred_df, pred_meta = prepare_prediction_df(spec, solution_df)
        prepared_path = out_dir / f"{spec.name}_validation_pred_scored.csv"
        pred_df.to_csv(prepared_path, index=False)
        prepared_prediction_paths.append(prepared_path)

        breakdown = metric["score_detailed"](solution_df, pred_df, "image")
        score_row = {"model": spec.name, "prediction_csv": str(spec.path)}
        score_row.update(breakdown)
        score_row.update(pred_meta)
        score_rows.append(score_row)

        image_rows, match_rows, error_rows = analyze_prediction(
            spec.name,
            solution_df,
            pred_df,
            metadata,
            metric,
            iou_threshold=args.iou_threshold,
        )
        all_image_rows.extend(image_rows)
        all_match_rows.extend(match_rows)
        all_error_rows.extend(error_rows)

    scores_df = pd.DataFrame(score_rows).sort_values("composite_score", ascending=False)
    image_report_df = with_columns(pd.DataFrame(all_image_rows), IMAGE_REPORT_COLUMNS)
    match_report_df = with_columns(pd.DataFrame(all_match_rows), MATCH_REPORT_COLUMNS)
    error_report_df = with_columns(pd.DataFrame(all_error_rows), ERROR_REPORT_COLUMNS)
    slice_summary_df = build_slice_summary(image_report_df, match_report_df, error_report_df)
    type_loss_summary_df = with_columns(
        build_type_loss_summary(match_report_df, error_report_df),
        TYPE_LOSS_REPORT_COLUMNS,
    )
    comparison_df = build_model_comparison(image_report_df)

    scores_path = out_dir / "validation_scores_summary.csv"
    image_path = out_dir / "validation_image_report.csv"
    match_path = out_dir / "validation_region_matches.csv"
    error_path = out_dir / "validation_region_errors.csv"
    slice_path = out_dir / "validation_slice_summary.csv"
    type_loss_path = out_dir / "validation_type_loss_summary.csv"
    comparison_path = out_dir / "validation_model_comparison.csv"
    report_path = out_dir / "validation_report.md"

    scores_df.to_csv(scores_path, index=False)
    image_report_df.to_csv(image_path, index=False)
    match_report_df.to_csv(match_path, index=False)
    error_report_df.to_csv(error_path, index=False)
    slice_summary_df.to_csv(slice_path, index=False)
    type_loss_summary_df.to_csv(type_loss_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    output_files = [
        solution_path,
        *prepared_prediction_paths,
        scores_path,
        image_path,
        match_path,
        error_path,
        slice_path,
        type_loss_path,
        comparison_path,
        report_path,
    ]
    write_markdown_report(
        report_path,
        scores_df,
        comparison_df,
        image_report_df,
        error_report_df,
        slice_summary_df,
        type_loss_summary_df,
        output_files,
        top_k=args.top_k,
    )

    print("Validation comparison complete.")
    print(f"Models: {', '.join(spec.name for spec in specs)}")
    print(f"Report: {report_path}")
    print()
    print(scores_df[["model", "composite_score", "detection_f1", "region_cer", "page_cer"]].to_string(index=False))
    if not type_loss_summary_df.empty:
        top_types = (
            type_loss_summary_df.sort_values(
                ["model", "rank_by_estimated_total_loss", "estimated_total_loss"],
                ascending=[True, True, False],
            )
            .groupby("model", dropna=False)
            .head(5)
        )
        print()
        print("Weakest types by estimated loss:")
        print(
            top_types[
                [
                    "model",
                    "rank_by_estimated_total_loss",
                    "region_type",
                    "gt_regions",
                    "error_rate",
                    "estimated_total_loss",
                    "loss_share",
                    "dominant_issue",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
