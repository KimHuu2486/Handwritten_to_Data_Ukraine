#!/usr/bin/env python
"""Create the AdditionalQualitativeResults supplementary TeX and figures."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


VALID_TYPES = {
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
}

TYPE_COLORS = {
    "handwritten": "#2563eb",
    "printed": "#0f766e",
    "formula": "#dc2626",
    "table": "#7c3aed",
    "annotation": "#d97706",
    "image": "#16a34a",
    "graph": "#0891b2",
    "missed": "#6b7280",
    "false positive": "#be123c",
}

SOURCE_ORDER = ["school", "dictation", "university", "archive"]

_OFFICIAL_NORMALIZE_TEXT: Callable[[str, str], str] | None = None
_OFFICIAL_LEVENSHTEIN: Callable[[str, str], int] | None = None


@dataclass
class Region:
    bbox: tuple[float, float, float, float]
    type: str
    text: str
    index: int
    language: str = "uk"
    legibility: str = "legible"


@dataclass
class PageData:
    image: str
    image_path: Path
    width: float
    height: float
    source: str
    gt_regions: list[Region]
    pred_regions: list[Region]


@dataclass
class MatchRecord:
    image: str
    image_path: Path
    width: float
    height: float
    source: str
    gt: Region | None
    pred: Region | None
    iou: float
    cer: float | None
    status: str
    bucket: str = ""
    note: str = ""

    @property
    def gt_type(self) -> str:
        return self.gt.type if self.gt else "none"

    @property
    def pred_type(self) -> str:
        if self.pred:
            return self.pred.type
        return "missed"

    @property
    def route(self) -> str:
        return route_for_type(self.pred.type if self.pred else self.gt_type)

    @property
    def crop_bbox(self) -> tuple[float, float, float, float]:
        if self.pred:
            return self.pred.bbox
        if self.gt:
            return self.gt.bbox
        raise ValueError("record has neither GT nor prediction")


@dataclass
class PageStats:
    matched: int
    missed: int
    false_positive: int
    det_f1: float
    class_acc: float
    class_correct: int
    type_confusions: int
    mean_cer: float | None
    source: str
    gt_types: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supplementary qualitative result images and LaTeX."
    )
    parser.add_argument("--gt-jsonl", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=Path("06_yolo_trocr_qwen_hybrid_submission.csv"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(r"C:\Users\HP\source\rukopys\train\images"),
    )
    parser.add_argument(
        "--metric-notebook",
        type=Path,
        default=Path("official-evaluation-metric-text-normalization.ipynb"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Supplementary/AnalysResultDetail"),
    )
    parser.add_argument("--num-full-pages", type=int, default=5)
    parser.add_argument("--num-crops", type=int, default=12)
    parser.add_argument("--browser", type=Path, default=None)
    return parser.parse_args()


def image_key(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value).replace("\\", "/")).name


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-cp1252 mojibake without changing normal text."""
    if not any(marker in text for marker in ("Ã", "Ð", "Ñ", "Â")):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text

    def cyrillic_count(value: str) -> int:
        return sum(1 for char in value if "\u0400" <= char <= "\u04ff")

    if cyrillic_count(repaired) > cyrillic_count(text):
        return repaired
    return text


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = repair_mojibake(text)
    return re.sub(r"\s+", " ", text.strip())


def truncate_text(text: str, limit: int = 150) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def route_for_type(region_type: str) -> str:
    if region_type in {"handwritten", "printed", "annotation"}:
        return "TrOCR"
    if region_type in {"formula", "table"}:
        return "Qwen3-VL"
    if region_type in {"image", "graph"}:
        return "empty text"
    if region_type == "missed":
        return "unmatched"
    return "unknown"


def configure_official_metric_helpers(path: Path) -> None:
    global _OFFICIAL_NORMALIZE_TEXT, _OFFICIAL_LEVENSHTEIN

    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    start = source.find("_LATEX_SYMBOLS =")
    end = source.find("def _compute_iou", start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not extract official metric helpers from notebook")

    namespace: dict[str, Any] = {"re": re}
    exec(source[start:end], namespace)
    _OFFICIAL_NORMALIZE_TEXT = namespace["_normalize_text"]
    _OFFICIAL_LEVENSHTEIN = namespace["_levenshtein"]


def official_normalize_text(value: Any, region_type: str) -> str:
    if _OFFICIAL_NORMALIZE_TEXT is None:
        raise RuntimeError("Official metric normalizer was not configured")
    return _OFFICIAL_NORMALIZE_TEXT(clean_text(value), region_type)


def official_levenshtein(a: str, b: str) -> int:
    if _OFFICIAL_LEVENSHTEIN is None:
        raise RuntimeError("Official Levenshtein helper was not configured")
    return _OFFICIAL_LEVENSHTEIN(a, b)


def clamp_bbox(
    bbox: Iterable[Any], width: float, height: float
) -> tuple[float, float, float, float]:
    values = [float(value) for value in bbox]
    if len(values) != 4:
        raise ValueError(f"bbox must have four values, got {bbox!r}")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        x1 = x0 + max(0.0, x1)
        y1 = y0 + max(0.0, y1)
    x0 = min(max(x0, 0.0), width)
    y0 = min(max(y0, 0.0), height)
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    if x1 <= x0:
        x1 = min(width, x0 + 1.0)
    if y1 <= y0:
        y1 = min(height, y0 + 1.0)
    return (x0, y0, x1, y1)


def area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def compute_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = area((ix0, iy0, ix1, iy1))
    union = area(a) + area(b) - intersection
    return intersection / union if union > 0 else 0.0


def is_scorable(region: Region | None) -> bool:
    if region is None:
        return False
    if region.type in {"image", "graph"}:
        return False
    if region.language == "other":
        return False
    if region.legibility == "illegible":
        return False
    return True


def region_cer(pred_text: str, gt_text: str, region_type: str) -> float:
    pred = official_normalize_text(pred_text, region_type)
    gt = official_normalize_text(gt_text, region_type)
    return official_levenshtein(pred, gt) / max(len(gt), 1)


def status_for_match(gt: Region | None, pred: Region | None, cer_value: float | None) -> str:
    if gt is None:
        return "failure"
    if pred is None:
        return "failure"
    if not is_scorable(gt):
        return "correct" if gt.type == pred.type else "type confusion"
    if gt.type != pred.type:
        return "type confusion"
    if cer_value is not None and cer_value <= 0.10:
        return "correct"
    if cer_value is not None and cer_value <= 0.35:
        return "near-correct"
    return "failure"


def parse_regions(value: str) -> list[dict[str, Any]]:
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("regions must be a JSON list")
    return parsed


def read_gt(path: Path, image_dir: Path) -> dict[str, PageData]:
    pages: dict[str, PageData] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = image_key(row.get("file_name") or row.get("image") or row.get("image_id"))
            if not key:
                raise ValueError(f"Cannot infer image key at {path}:{line_no}")
            width = float(row.get("image_width") or row.get("width") or 1)
            height = float(row.get("image_height") or row.get("height") or 1)
            regions: list[Region] = []
            for idx, reg in enumerate(row.get("regions", [])):
                region_type = str(reg.get("type", "")).strip()
                regions.append(
                    Region(
                        bbox=clamp_bbox(reg.get("bbox", [0, 0, 1, 1]), width, height),
                        type=region_type,
                        text=clean_text(reg.get("text", "")),
                        index=idx,
                        language=str(reg.get("language", "uk")),
                        legibility=str(reg.get("legibility", "legible")),
                    )
                )
            pages[key] = PageData(
                image=key,
                image_path=(image_dir / key).resolve(),
                width=width,
                height=height,
                source=str(row.get("source", "")),
                gt_regions=regions,
                pred_regions=[],
            )
    return pages


def attach_predictions(path: Path, pages: dict[str, PageData]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image" not in reader.fieldnames or "regions" not in reader.fieldnames:
            raise ValueError("prediction CSV must contain image and regions columns")
        for row in reader:
            key = image_key(row.get("image"))
            page = pages.get(key)
            if page is None:
                continue
            pred_regions: list[Region] = []
            for idx, reg in enumerate(parse_regions(row.get("regions", ""))):
                region_type = str(reg.get("type", "")).strip()
                pred_regions.append(
                    Region(
                        bbox=clamp_bbox(reg.get("bbox", [0, 0, 1, 1]), page.width, page.height),
                        type=region_type,
                        text=clean_text(reg.get("text", "")),
                        index=idx,
                    )
                )
            page.pred_regions = pred_regions


def match_page(page: PageData, threshold: float = 0.5) -> list[MatchRecord]:
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(page.gt_regions):
        for pi, pred in enumerate(page.pred_regions):
            iou = compute_iou(gt.bbox, pred.bbox)
            if iou >= threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    records: list[MatchRecord] = []
    for iou, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        gt = page.gt_regions[gi]
        pred = page.pred_regions[pi]
        cer_value = region_cer(pred.text, gt.text, gt.type) if is_scorable(gt) else None
        records.append(
            MatchRecord(
                image=page.image,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                source=page.source,
                gt=gt,
                pred=pred,
                iou=iou,
                cer=cer_value,
                status=status_for_match(gt, pred, cer_value),
            )
        )
        used_gt.add(gi)
        used_pred.add(pi)

    for gi, gt in enumerate(page.gt_regions):
        if gi in used_gt:
            continue
        records.append(
            MatchRecord(
                image=page.image,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                source=page.source,
                gt=gt,
                pred=None,
                iou=0.0,
                cer=1.0 if is_scorable(gt) else None,
                status="failure",
            )
        )
    for pi, pred in enumerate(page.pred_regions):
        if pi in used_pred:
            continue
        records.append(
            MatchRecord(
                image=page.image,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                source=page.source,
                gt=None,
                pred=pred,
                iou=0.0,
                cer=None,
                status="failure",
            )
        )
    return records


def compute_page_stats(page: PageData, records: list[MatchRecord]) -> PageStats:
    matched = [record for record in records if record.gt and record.pred]
    missed = [record for record in records if record.gt and not record.pred]
    false_positive = [record for record in records if record.pred and not record.gt]
    class_correct = sum(1 for record in matched if record.gt_type == record.pred_type)
    type_confusions = len(matched) - class_correct
    tp = len(matched)
    fp = len(false_positive)
    fn = len(missed)
    det_f1 = (2 * tp) / max((2 * tp) + fp + fn, 1)
    class_acc = class_correct / max(tp, 1)
    cers = [record.cer for record in matched if record.cer is not None and is_scorable(record.gt)]
    mean_cer = sum(cers) / len(cers) if cers else None
    return PageStats(
        matched=tp,
        missed=fn,
        false_positive=fp,
        det_f1=det_f1,
        class_acc=class_acc,
        class_correct=class_correct,
        type_confusions=type_confusions,
        mean_cer=mean_cer,
        source=page.source,
        gt_types=sorted({region.type for region in page.gt_regions}),
    )


def observation_for_page(stats: PageStats) -> str:
    if stats.type_confusions:
        return (
            f"Shows {stats.type_confusions} matched type confusion case(s); "
            "this is useful for inspecting the router boundary."
        )
    if stats.missed or stats.false_positive:
        return (
            f"Detection is mostly stable, with {stats.missed} missed and "
            f"{stats.false_positive} extra predicted region(s)."
        )
    if stats.det_f1 >= 0.95 and stats.class_acc >= 0.95:
        return "Strong page-level example with high detection and routing agreement."
    return "Representative page-level example for detector and routing inspection."


def full_page_selection_score(page: PageData, stats: PageStats) -> float:
    diversity = len(stats.gt_types)
    structured = 1 if {"formula", "table", "graph", "image"} & set(stats.gt_types) else 0
    manageable = 1 if 5 <= len(page.pred_regions) <= 38 else 0
    image_bonus = 2 if page.image_path.exists() else -100
    return (
        stats.det_f1 * 10
        + stats.class_acc * 6
        + diversity * 1.5
        + structured * 2
        + manageable
        + image_bonus
    )


def error_page_score(stats: PageStats) -> float:
    return (
        stats.type_confusions * 8
        + stats.missed * 1.5
        + stats.false_positive * 1.2
        + (1 - stats.class_acc) * 6
        + (1 - stats.det_f1) * 5
    )


def select_full_pages(
    pages: dict[str, PageData],
    all_records: dict[str, list[MatchRecord]],
    stats_by_image: dict[str, PageStats],
    desired: int,
) -> list[PageData]:
    selected: list[PageData] = []
    used: set[str] = set()

    for source in SOURCE_ORDER:
        candidates = [
            page
            for page in pages.values()
            if page.source == source and page.pred_regions and page.image_path.exists()
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda page: full_page_selection_score(page, stats_by_image[page.image]),
        )
        selected.append(best)
        used.add(best.image)

    error_candidates = [
        page
        for page in pages.values()
        if page.image not in used
        and page.pred_regions
        and page.image_path.exists()
        and stats_by_image[page.image].det_f1 >= 0.50
        and error_page_score(stats_by_image[page.image]) > 0
    ]
    if error_candidates and len(selected) < desired:
        notable = max(
            error_candidates,
            key=lambda page: (
                error_page_score(stats_by_image[page.image]),
                stats_by_image[page.image].det_f1,
                stats_by_image[page.image].class_acc,
            ),
        )
        selected.append(notable)
        used.add(notable.image)

    if len(selected) < desired:
        remaining = [
            page
            for page in pages.values()
            if page.image not in used and page.pred_regions and page.image_path.exists()
        ]
        remaining.sort(
            key=lambda page: full_page_selection_score(page, stats_by_image[page.image]),
            reverse=True,
        )
        selected.extend(remaining[: desired - len(selected)])

    return selected[:desired]


def record_key(record: MatchRecord) -> tuple[str, int, int]:
    return (
        record.image,
        record.gt.index if record.gt else -1,
        record.pred.index if record.pred else -1,
    )


def crop_quality(record: MatchRecord) -> float:
    x0, y0, x1, y1 = record.crop_bbox
    crop_area = max(1.0, (x1 - x0) * (y1 - y0))
    image_area = max(1.0, record.width * record.height)
    area_ratio = crop_area / image_area
    text_len = len(record.gt.text if record.gt else record.pred.text if record.pred else "")
    return (
        record.iou * 3
        + (1.2 if 0.0015 <= area_ratio <= 0.22 else -1.0)
        + (1.0 if 2 <= text_len <= 140 else 0.0)
        + (0.5 if record.image_path.exists() else -5.0)
    )


def pick_best(
    records: list[MatchRecord],
    predicate: Callable[[MatchRecord], bool],
    used: set[tuple[str, int, int]],
    preference: Callable[[MatchRecord], float],
) -> MatchRecord | None:
    candidates = [record for record in records if predicate(record) and record_key(record) not in used]
    if not candidates:
        return None
    return max(candidates, key=preference)


def assign_bucket(record: MatchRecord, bucket: str) -> MatchRecord:
    record.bucket = bucket
    if record.status == "correct":
        record.note = "Matched type with low normalized CER."
    elif record.status == "near-correct":
        record.note = "Recognition is close after official normalization."
    elif record.status == "type confusion":
        record.note = f"Type confusion between {record.gt_type} and {record.pred_type}."
    else:
        record.note = "Representative failure under the official matching rule."
    if "formatting" in bucket:
        record.note = "Prediction misses part of the expected table structure."
    return record


def select_crop_examples(
    all_records: dict[str, list[MatchRecord]], desired: int
) -> tuple[list[MatchRecord], list[str]]:
    records = [record for page_records in all_records.values() for record in page_records]
    selected: list[MatchRecord] = []
    used: set[tuple[str, int, int]] = set()
    warnings: list[str] = []

    def add(bucket: str, record: MatchRecord | None) -> None:
        if record is None:
            warnings.append(f"missing bucket: {bucket}")
            return
        selected.append(assign_bucket(record, bucket))
        used.add(record_key(record))

    add(
        "handwritten success",
        pick_best(
            records,
            lambda r: r.gt_type == "handwritten"
            and r.pred_type == "handwritten"
            and r.cer is not None
            and r.cer <= 0.10,
            used,
            lambda r: crop_quality(r) - abs((r.cer or 0) - 0.04),
        ),
    )
    add(
        "printed success",
        pick_best(
            records,
            lambda r: r.gt_type == "printed"
            and r.pred_type == "printed"
            and r.cer is not None
            and r.cer <= 0.10,
            used,
            crop_quality,
        ),
    )
    add(
        "formula near-correct",
        pick_best(
            records,
            lambda r: r.gt_type == "formula"
            and r.pred_type == "formula"
            and r.cer is not None
            and 0.02 <= r.cer <= 0.35,
            used,
            lambda r: crop_quality(r) - abs((r.cer or 0) - 0.16),
        ),
    )
    add(
        "formula failure",
        pick_best(
            records,
            lambda r: r.gt_type == "formula"
            and is_scorable(r.gt)
            and (r.pred_type != "formula" or (r.cer is not None and r.cer > 0.35)),
            used,
            lambda r: crop_quality(r) + (r.cer or 0),
        ),
    )
    add(
        "table example",
        pick_best(
            records,
            lambda r: r.gt_type == "table" and r.pred_type == "table" and is_scorable(r.gt),
            used,
            lambda r: crop_quality(r) - min(r.cer or 0, 0.5),
        ),
    )
    add(
        "table formatting failure",
        pick_best(
            records,
            lambda r: r.gt_type == "table"
            and is_scorable(r.gt)
            and (
                r.pred_type != "table"
                or (r.cer is not None and r.cer >= 0.20)
                or (r.pred is not None and "|" not in r.pred.text)
            ),
            used,
            lambda r: crop_quality(r) + min(r.cer or 0, 1.0),
        ),
    )
    add(
        "annotation near-correct",
        pick_best(
            records,
            lambda r: r.gt_type == "annotation"
            and r.pred_type == "annotation"
            and r.cer is not None
            and 0.10 < r.cer <= 0.35,
            used,
            crop_quality,
        ),
    )
    add(
        "annotation failure",
        pick_best(
            records,
            lambda r: r.gt_type == "annotation"
            and is_scorable(r.gt)
            and (r.pred_type != "annotation" or (r.cer is not None and r.cer > 0.35)),
            used,
            lambda r: crop_quality(r) + min(r.cer or 0, 1.0),
        ),
    )
    add(
        "formula-handwritten confusion",
        pick_best(
            records,
            lambda r: (r.gt_type, r.pred_type)
            in {("formula", "handwritten"), ("handwritten", "formula")},
            used,
            lambda r: crop_quality(r) + 2,
        ),
    )
    add(
        "type confusion",
        pick_best(
            records,
            lambda r: r.gt is not None
            and r.pred is not None
            and r.gt_type != r.pred_type
            and record_key(r) not in used,
            used,
            lambda r: crop_quality(r) + 1.5,
        ),
    )
    add(
        "empty-text route",
        pick_best(
            records,
            lambda r: r.gt_type in {"image", "graph"} and r.pred_type == r.gt_type,
            used,
            crop_quality,
        ),
    )

    if len(selected) < desired:
        fallback = [
            record
            for record in records
            if record.gt is not None
            and record.pred is not None
            and is_scorable(record.gt)
            and record_key(record) not in used
        ]
        fallback.sort(
            key=lambda r: (
                1 if r.status == "type confusion" else 0,
                1 if r.status == "failure" else 0,
                crop_quality(r),
            ),
            reverse=True,
        )
        for record in fallback:
            if len(selected) >= desired:
                break
            selected.append(assign_bucket(record, f"{record.gt_type} {record.status}"))
            used.add(record_key(record))

    return selected[:desired], warnings


def pct(value: float, total: float) -> str:
    return f"{(value / max(total, 1.0)) * 100:.4f}%"


def box_label(region: Region) -> str:
    return f"{region.type} | {route_for_type(region.type)}"


def render_full_page_html(
    page: PageData,
    stats: PageStats,
    ordinal: int,
    total: int,
) -> str:
    boxes = sorted(page.pred_regions, key=lambda region: (region.bbox[1], region.bbox[0]))
    label_limit = 42
    visible_boxes = boxes[:label_limit]
    box_html: list[str] = []
    for region in visible_boxes:
        x0, y0, x1, y1 = region.bbox
        color = TYPE_COLORS.get(region.type, "#374151")
        box_html.append(
            f"""
            <div class="bbox" style="
              left:{pct(x0, page.width)}; top:{pct(y0, page.height)};
              width:{pct(x1 - x0, page.width)}; height:{pct(y1 - y0, page.height)};
              border-color:{color};">
              <span style="background:{color};">{html_escape(box_label(region))}</span>
            </div>
            """
        )

    cer_text = "--" if stats.mean_cer is None else f"{stats.mean_cer:.3f}"
    type_list = ", ".join(stats.gt_types)
    observation = observation_for_page(stats)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html {{
    width: 1400px;
    height: 1450px;
    overflow: hidden;
  }}
  body {{
    margin: 0;
    width: 1400px;
    height: 1450px;
    padding: 34px;
    overflow: hidden;
    background: #ffffff;
    color: #111827;
    font-family: Arial, "Segoe UI", "DejaVu Sans", sans-serif;
  }}
  .title {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 18px;
    border-bottom: 2px solid #111827;
    padding-bottom: 12px;
  }}
  h1 {{
    margin: 0;
    font-size: 36px;
    letter-spacing: 0;
  }}
  .count {{
    font-size: 22px;
    color: #4b5563;
    font-weight: 700;
  }}
  .layout {{
    display: grid;
    grid-template-columns: 900px 1fr;
    gap: 24px;
    align-items: start;
  }}
  .page-frame {{
    position: relative;
    width: 900px;
    height: 1195px;
    overflow: hidden;
    border: 1px solid #d1d5db;
    background: #f9fafb;
  }}
  .page-frame img {{
    width: 100%;
    height: 100%;
    display: block;
  }}
  .bbox {{
    position: absolute;
    border: 3px solid;
    background: rgba(255, 255, 255, 0.02);
  }}
  .bbox span {{
    position: absolute;
    left: -3px;
    top: -24px;
    max-width: 230px;
    padding: 3px 6px;
    color: #ffffff;
    font-size: 14px;
    line-height: 1.1;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .info {{
    border: 1px solid #d1d5db;
    padding: 20px;
    min-height: 520px;
    font-size: 23px;
    line-height: 1.35;
  }}
  .info h2 {{
    margin: 0 0 16px;
    font-size: 28px;
  }}
  .kv {{
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 10px 14px;
    margin-bottom: 18px;
  }}
  .key {{
    color: #4b5563;
    font-weight: 700;
  }}
  .value {{
    overflow-wrap: anywhere;
  }}
  .metric {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 18px 0;
  }}
  .metric div {{
    border: 1px solid #e5e7eb;
    padding: 12px;
    background: #f9fafb;
  }}
  .metric b {{
    display: block;
    font-size: 32px;
    color: #111827;
  }}
  .obs {{
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #e5e7eb;
    font-size: 22px;
  }}
  .legend {{
    margin-top: 22px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 12px;
    font-size: 18px;
  }}
  .swatch {{
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 7px;
    vertical-align: -3px;
  }}
</style>
</head>
<body>
  <div class="title">
    <h1>Full-Page Example</h1>
    <div class="count">{ordinal} / {total}</div>
  </div>
  <div class="layout">
    <div class="page-frame">
      <img src="{file_uri(page.image_path)}" alt="Full page">
      {''.join(box_html)}
    </div>
    <aside class="info">
      <h2>Page diagnostics</h2>
      <div class="kv">
        <div class="key">Source image</div><div class="value">{html_escape(page.image)}</div>
        <div class="key">Subset</div><div class="value">{html_escape(page.source)}</div>
        <div class="key">GT types</div><div class="value">{html_escape(type_list)}</div>
        <div class="key">Boxes shown</div><div class="value">{len(visible_boxes)} / {len(page.pred_regions)}</div>
        <div class="key">Matched</div><div class="value">{stats.matched}</div>
        <div class="key">Missed</div><div class="value">{stats.missed}</div>
        <div class="key">False positives</div><div class="value">{stats.false_positive}</div>
      </div>
      <div class="metric">
        <div>Det-F1<b>{stats.det_f1:.3f}</b></div>
        <div>ClassAcc<b>{stats.class_acc:.3f}</b></div>
        <div>Matched CER<b>{cer_text}</b></div>
        <div>Type conf.<b>{stats.type_confusions}</b></div>
      </div>
      <div class="obs"><b>Observation:</b> {html_escape(observation)}</div>
      <div class="legend">
        {''.join(f'<div><span class="swatch" style="background:{TYPE_COLORS[k]}"></span>{k} | {route_for_type(k)}</div>' for k in ["handwritten", "printed", "annotation", "formula", "table", "image", "graph"])}
      </div>
    </aside>
  </div>
</body>
</html>"""


def expand_bbox(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
    pad_ratio: float = 0.10,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    pad_x = max(12.0, (x1 - x0) * pad_ratio)
    pad_y = max(12.0, (y1 - y0) * pad_ratio)
    return (
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def overlay_box(
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    class_name: str,
    label: str,
) -> str:
    cx0, cy0, cx1, cy1 = crop
    x0, y0, x1, y1 = bbox
    cw = max(1.0, cx1 - cx0)
    ch = max(1.0, cy1 - cy0)
    return f"""
    <div class="{class_name}" style="
      left:{((x0 - cx0) / cw) * 100:.4f}%; top:{((y0 - cy0) / ch) * 100:.4f}%;
      width:{((x1 - x0) / cw) * 100:.4f}%; height:{((y1 - y0) / ch) * 100:.4f}%;">
      <span>{html_escape(label)}</span>
    </div>
    """


def render_crop_card(record: MatchRecord) -> str:
    crop = expand_bbox(record.crop_bbox, record.width, record.height)
    cx0, cy0, cx1, cy1 = crop
    cw = max(1.0, cx1 - cx0)
    ch = max(1.0, cy1 - cy0)
    image_width_pct = record.width / cw * 100.0
    image_height_pct = record.height / ch * 100.0
    image_left_pct = -cx0 / cw * 100.0
    image_top_pct = -cy0 / ch * 100.0

    overlays: list[str] = []
    if record.gt:
        overlays.append(overlay_box(record.gt.bbox, crop, "gt-box", "GT"))
    if record.pred:
        overlays.append(overlay_box(record.pred.bbox, crop, "pred-box", "Pred"))

    status_class = re.sub(r"[^a-z]+", "-", record.status.lower()).strip("-")
    cer_text = "--" if record.cer is None else f"{record.cer:.3f}"
    gt_text = truncate_text(record.gt.text if record.gt else "", 135)
    pred_text = truncate_text(record.pred.text if record.pred else "", 135)
    return f"""
    <article class="crop-card">
      <div class="crop-header">
        <span>{html_escape(record.bucket)}</span>
        <b class="badge {status_class}">{html_escape(record.status)}</b>
      </div>
      <div class="crop-image" style="aspect-ratio:{cw:.0f}/{ch:.0f};">
        <img src="{file_uri(record.image_path)}" alt="Crop source" style="
          width:{image_width_pct:.4f}%; height:{image_height_pct:.4f}%;
          left:{image_left_pct:.4f}%; top:{image_top_pct:.4f}%;">
        {''.join(overlays)}
      </div>
      <div class="metrics">
        <span>Source: <b>{html_escape(record.image)}</b></span>
        <span>GT: <b>{html_escape(record.gt_type)}</b></span>
        <span>Pred: <b>{html_escape(record.pred_type)}</b></span>
        <span>Route: <b>{html_escape(record.route)}</b></span>
        <span>IoU: <b>{record.iou:.2f}</b></span>
        <span>CER: <b>{cer_text}</b></span>
      </div>
      <div class="text-line"><b>GT text:</b> {html_escape(gt_text)}</div>
      <div class="text-line"><b>Pred text:</b> {html_escape(pred_text)}</div>
      <div class="note">{html_escape(record.note)}</div>
    </article>
    """


def render_crop_grid_html(crops: list[MatchRecord], grid_index: int, total_grids: int) -> str:
    cards = "\n".join(render_crop_card(record) for record in crops)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html {{
    width: 1800px;
    height: 1250px;
    overflow: hidden;
  }}
  body {{
    margin: 0;
    width: 1800px;
    height: 1250px;
    padding: 34px;
    overflow: hidden;
    background: #ffffff;
    color: #111827;
    font-family: Arial, "Segoe UI", "DejaVu Sans", sans-serif;
  }}
  .title {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 18px;
    border-bottom: 2px solid #111827;
    padding-bottom: 12px;
  }}
  h1 {{
    margin: 0;
    font-size: 36px;
  }}
  .count {{
    font-size: 22px;
    color: #4b5563;
    font-weight: 700;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
  }}
  .crop-card {{
    border: 1px solid #d1d5db;
    padding: 14px;
    min-height: 475px;
    min-width: 0;
    overflow: hidden;
    background: #ffffff;
  }}
  .crop-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
  }}
  .crop-header span {{
    min-width: 0;
    overflow-wrap: anywhere;
  }}
  .badge {{
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 17px;
    border: 1px solid #d1d5db;
    background: #f3f4f6;
    color: #374151;
    white-space: nowrap;
  }}
  .badge.correct {{
    color: #065f46;
    background: #ecfdf5;
    border-color: #a7f3d0;
  }}
  .badge.near-correct {{
    color: #92400e;
    background: #fffbeb;
    border-color: #fde68a;
  }}
  .badge.failure,
  .badge.type-confusion {{
    color: #991b1b;
    background: #fef2f2;
    border-color: #fecaca;
  }}
  .crop-image {{
    position: relative;
    width: 100%;
    height: 170px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
  }}
  .crop-image img {{
    position: absolute;
    max-width: none;
    max-height: none;
  }}
  .gt-box,
  .pred-box {{
    position: absolute;
    border: 3px solid;
  }}
  .gt-box {{
    border-color: #111827;
  }}
  .pred-box {{
    border-color: #dc2626;
  }}
  .gt-box span,
  .pred-box span {{
    position: absolute;
    left: -3px;
    top: -22px;
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    padding: 2px 6px;
  }}
  .gt-box span {{
    background: #111827;
  }}
  .pred-box span {{
    background: #dc2626;
  }}
  .metrics {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 12px;
    margin: 11px 0 8px;
    font-size: 16px;
    color: #374151;
  }}
  .metrics span:first-child {{
    grid-column: 1 / -1;
    overflow-wrap: anywhere;
  }}
  .metrics span {{
    min-width: 0;
    overflow-wrap: anywhere;
  }}
  .text-line {{
    margin-top: 6px;
    font-size: 16px;
    line-height: 1.25;
    overflow-wrap: anywhere;
  }}
  .note {{
    margin-top: 8px;
    color: #4b5563;
    font-size: 15px;
    line-height: 1.2;
  }}
</style>
</head>
<body>
  <div class="title">
    <h1>Crop-Level Examples</h1>
    <div class="count">Grid {grid_index} / {total_grids}</div>
  </div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>"""


def find_browser(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(shutil.which("msedge") or "") if shutil.which("msedge") else None,
        Path(shutil.which("chrome") or "") if shutil.which("chrome") else None,
        Path(shutil.which("google-chrome") or "") if shutil.which("google-chrome") else None,
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find Chrome or Edge for HTML-to-PNG rendering")


def render_png(
    browser: Path,
    html_path: Path,
    png_path: Path,
    width: int,
    height: int,
) -> None:
    png_path = png_path.resolve()
    if png_path.exists():
        png_path.unlink()
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--disable-extensions",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=2000",
        f"--window-size={width},{height}",
        f"--screenshot={str(png_path)}",
        html_path.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Browser screenshot failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise RuntimeError(f"Browser did not create a non-empty PNG: {png_path}")


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_tex(
    output_path: Path,
    full_pages: list[PageData],
    stats_by_image: dict[str, PageStats],
    crop_grid_files: list[str],
) -> None:
    full_page_blocks: list[str] = []
    for idx, page in enumerate(full_pages, start=1):
        stats = stats_by_image[page.image]
        filename = f"full_page_example_{idx:02d}.png"
        caption = (
            f"{page.source} page; Det-F1={stats.det_f1:.3f}, "
            f"ClassAcc={stats.class_acc:.3f}, boxes={len(page.pred_regions)}."
        )
        full_page_blocks.append(
            "\\begin{minipage}[t]{0.485\\textwidth}\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{filename}}}\n"
            f"{{\\small {tex_escape(caption)}\\par}}\n"
            "\\end{minipage}"
        )

    paired_blocks: list[str] = []
    for offset in range(0, len(full_page_blocks), 2):
        pair = full_page_blocks[offset : offset + 2]
        if len(pair) == 2:
            paired_blocks.append(pair[0] + "\n\\hfill\n" + pair[1])
        else:
            paired_blocks.append(pair[0])

    crop_figures: list[str] = []
    for idx, filename in enumerate(crop_grid_files, start=1):
        crop_figures.append(
            "\\begin{figure}[p]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{filename}}}\n"
            "\\caption{Crop-level qualitative examples with matched GT/predicted boxes, "
            f"routing, IoU, and normalized CER diagnostics (grid {idx}).}}\n"
            f"\\label{{fig:crop-level-examples-{idx}}}\n"
            "\\end{figure}"
        )

    first_full_page_figure = paired_blocks[0] if paired_blocks else ""
    remaining_full_page_figure = (
        "\n\\par\\vspace{0.7em}\n".join(paired_blocks[1:])
        if len(paired_blocks) > 1
        else ""
    )
    crop_figures_tex = "\n".join(crop_figures)

    content = f"""\\documentclass[10pt]{{article}}
\\usepackage[a4paper,margin=0.55in]{{geometry}}
\\usepackage{{graphicx}}
\\graphicspath{{{{./}}}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{0.45em}}

\\title{{Additional Qualitative Results}}
\\date{{}}

\\begin{{document}}
\\maketitle

This supplementary document provides additional qualitative examples for the final BoustoDoc pipeline. Section~\\ref{{sec:full-page-examples}} shows full-page examples with detection, classification, and routing information. Section~\\ref{{sec:crop-level-examples}} shows crop-level examples from multiple pages, including both successful predictions and representative failure cases.

\\section{{Full-Page Examples}}
\\label{{sec:full-page-examples}}

This section presents page-level qualitative examples from the final pipeline. Each example shows the detected regions, predicted region types, routed recognition branch, and page-level detection/classification diagnostics. Det-F1 and ClassAcc are computed from the same IoU matching protocol used by the official evaluation code.

\\begin{{figure}}[p]
\\centering
{first_full_page_figure}
\\end{{figure}}

\\begin{{figure}}[p]
\\centering
{remaining_full_page_figure}
\\end{{figure}}

\\clearpage
\\section{{Crop-Level Examples}}
\\label{{sec:crop-level-examples}}

This section presents crop-level examples from multiple pages. The examples include both successful predictions and representative failure modes, such as type confusion, annotation over-generation, and malformed table formatting. Each tile reports the source image, GT and predicted type, route, IoU, and normalized CER where applicable.

{crop_figures_tex}

These examples are consistent with the quantitative results reported in the main paper and highlight both the strengths and the remaining failure modes of the type-aware pipeline.

\\end{{document}}
"""
    output_path.write_text(content, encoding="utf-8")


def record_to_json(record: MatchRecord) -> dict[str, Any]:
    gt_norm = (
        official_normalize_text(record.gt.text, record.gt.type)
        if record.gt and is_scorable(record.gt)
        else ""
    )
    pred_norm = (
        official_normalize_text(record.pred.text, record.gt.type)
        if record.gt and record.pred and is_scorable(record.gt)
        else ""
    )
    return {
        "bucket": record.bucket,
        "status": record.status,
        "source_image": record.image,
        "source": record.source,
        "gt_type": record.gt_type,
        "pred_type": record.pred_type,
        "route": record.route,
        "iou": round(record.iou, 4),
        "cer": None if record.cer is None else round(record.cer, 4),
        "gt_text": record.gt.text if record.gt else "",
        "pred_text": record.pred.text if record.pred else "",
        "gt_text_official_normalized": gt_norm,
        "pred_text_official_normalized": pred_norm,
        "gt_bbox": list(record.gt.bbox) if record.gt else None,
        "pred_bbox": list(record.pred.bbox) if record.pred else None,
        "note": record.note,
    }


def write_selection_metadata(
    output_path: Path,
    full_pages: list[PageData],
    crop_examples: list[MatchRecord],
    stats_by_image: dict[str, PageStats],
    warnings: list[str],
) -> None:
    payload = {
        "full_page_examples": [
            {
                "source_image": page.image,
                "source": page.source,
                "gt_region_count": len(page.gt_regions),
                "pred_region_count": len(page.pred_regions),
                "gt_types": stats_by_image[page.image].gt_types,
                "det_f1": round(stats_by_image[page.image].det_f1, 4),
                "class_acc": round(stats_by_image[page.image].class_acc, 4),
                "matched": stats_by_image[page.image].matched,
                "missed": stats_by_image[page.image].missed,
                "false_positive": stats_by_image[page.image].false_positive,
                "observation": observation_for_page(stats_by_image[page.image]),
            }
            for page in full_pages
        ],
        "crop_examples": [record_to_json(record) for record in crop_examples],
        "warnings": warnings,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_inputs(args: argparse.Namespace) -> None:
    for label in ("gt_jsonl", "pred_csv", "image_dir", "metric_notebook"):
        path = getattr(args, label.replace("-", "_"), None)
        if path is None:
            path = getattr(args, label)
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {args.image_dir}")


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_official_metric_helpers(args.metric_notebook)
    pages = read_gt(args.gt_jsonl, args.image_dir)
    attach_predictions(args.pred_csv, pages)

    gt_keys = set(pages)
    pred_keys = {key for key, page in pages.items() if page.pred_regions}
    if gt_keys != pred_keys:
        missing = sorted(gt_keys - pred_keys)[:5]
        raise RuntimeError(f"Predictions do not cover all GT pages; first missing: {missing}")

    all_records = {key: match_page(page) for key, page in pages.items()}
    stats_by_image = {
        key: compute_page_stats(page, all_records[key]) for key, page in pages.items()
    }

    full_pages = select_full_pages(
        pages,
        all_records,
        stats_by_image,
        desired=args.num_full_pages,
    )
    crop_examples, warnings = select_crop_examples(all_records, desired=args.num_crops)
    if len(full_pages) < args.num_full_pages:
        raise RuntimeError(f"Only selected {len(full_pages)} full-page examples")
    if len(crop_examples) < min(args.num_crops, 6):
        raise RuntimeError(f"Only selected {len(crop_examples)} crop examples")

    browser = find_browser(args.browser)
    crop_grid_files: list[str] = []

    with tempfile.TemporaryDirectory(prefix="additional_qualitative_") as tmp:
        tmp_dir = Path(tmp)
        for idx, page in enumerate(full_pages, start=1):
            html_path = tmp_dir / f"full_page_example_{idx:02d}.html"
            png_path = args.output_dir / f"full_page_example_{idx:02d}.png"
            html_path.write_text(
                render_full_page_html(page, stats_by_image[page.image], idx, len(full_pages)),
                encoding="utf-8",
            )
            render_png(browser, html_path, png_path, width=1400, height=1450)

        grid_size = 6
        total_grids = (len(crop_examples) + grid_size - 1) // grid_size
        for grid_idx, offset in enumerate(range(0, len(crop_examples), grid_size), start=1):
            group = crop_examples[offset : offset + grid_size]
            html_path = tmp_dir / f"crop_level_examples_grid_{grid_idx:02d}.html"
            png_path = args.output_dir / f"crop_level_examples_grid_{grid_idx:02d}.png"
            html_path.write_text(
                render_crop_grid_html(group, grid_idx, total_grids),
                encoding="utf-8",
            )
            render_png(browser, html_path, png_path, width=1800, height=1250)
            crop_grid_files.append(png_path.name)

    tex_path = args.output_dir / "AdditionalQualitativeResults.tex"
    build_tex(tex_path, full_pages, stats_by_image, crop_grid_files)
    write_selection_metadata(
        args.output_dir / "qualitative_selected_examples.json",
        full_pages,
        crop_examples,
        stats_by_image,
        warnings,
    )

    print("Generated AdditionalQualitativeResults supplementary assets")
    print(f"Output directory: {args.output_dir}")
    print(f"TeX: {tex_path}")
    print("Full-page examples:")
    for idx, page in enumerate(full_pages, start=1):
        stats = stats_by_image[page.image]
        print(
            f"  {idx}. {page.image} source={page.source} "
            f"Det-F1={stats.det_f1:.3f} ClassAcc={stats.class_acc:.3f}"
        )
    print("Crop grids:")
    for filename in crop_grid_files:
        print(f"  - {filename}")
    if warnings:
        print("Selection warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
