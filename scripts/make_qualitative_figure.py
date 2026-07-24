#!/usr/bin/env python
"""Build a composite qualitative figure for the BoustoDoc paper."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    "handwritten": "#1f77b4",
    "printed": "#4c78a8",
    "formula": "#d62728",
    "table": "#9467bd",
    "annotation": "#ff7f0e",
    "image": "#2ca02c",
    "graph": "#17becf",
    "missed": "#7f7f7f",
}

_OFFICIAL_NORMALIZE_TEXT = None
_OFFICIAL_LEVENSHTEIN = None

LATEX_SNIPPET = r"""
\subsection{Qualitative Results}
\label{sec:qualitative}

Figure~\ref{fig:qualitative-results} shows representative outputs from the
final pipeline. The page-level example illustrates the detector--router--OCR
workflow: detected regions are assigned types, routed to the corresponding
recognition branch, and assembled into the final page output. The crop-level
examples include both successful transcriptions and common failure cases. These
examples provide qualitative evidence for the routing design without making
claims of superiority. The remaining errors are consistent with the quantitative
analysis, including formula--handwriting confusion, annotation over-generation,
malformed table separators, and local reading-order mistakes.

\begin{figure*}[tb]
  \centering
  \includegraphics[width=\linewidth]{figure/qualitative_results.pdf}
  \caption{
  Qualitative examples from BoustoDoc. The page-level example shows detected
  regions, predicted region types, and routed OCR branches. The crop-level
  examples show representative predictions for handwritten text, formulas,
  tables, and annotations. We include both successful cases and common errors,
  such as formula--handwriting confusion, annotation over-generation, malformed
  table separators, and local reading-order mistakes.
  }
  \label{fig:qualitative-results}
\end{figure*}
""".strip()


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
    gt_regions: list[Region]
    pred_regions: list[Region]


@dataclass
class MatchRecord:
    image: str
    image_path: Path
    width: float
    height: float
    gt: Region | None
    pred: Region | None
    iou: float
    cer: float | None
    status: str
    bucket: str = ""

    @property
    def gt_type(self) -> str:
        return self.gt.type if self.gt else "none"

    @property
    def pred_type(self) -> str:
        return self.pred.type if self.pred else "missed"

    @property
    def route(self) -> str:
        return route_for_type(self.pred_type if self.pred else self.gt_type)

    @property
    def bbox_for_crop(self) -> tuple[float, float, float, float]:
        if self.pred:
            return self.pred.bbox
        if self.gt:
            return self.gt.bbox
        raise ValueError("record has neither GT nor prediction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a composite qualitative figure and update paper/main.tex."
    )
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--gt_jsonl", required=True, type=Path)
    parser.add_argument("--pred_csv", required=True, type=Path)
    parser.add_argument("--main_tex", required=True, type=Path)
    parser.add_argument("--output_fig", required=True, type=Path)
    return parser.parse_args()


def find_metric_notebook(main_tex: Path) -> Path:
    candidates = [
        main_tex.resolve().parent.parent / "official-evaluation-metric-text-normalization.ipynb",
        Path.cwd() / "official-evaluation-metric-text-normalization.ipynb",
        Path(__file__).resolve().parent.parent / "official-evaluation-metric-text-normalization.ipynb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find official-evaluation-metric-text-normalization.ipynb. "
        "This script uses the official text normalizer for CER."
    )


def configure_official_metric_helpers(main_tex: Path) -> Path:
    """Load official notebook normalizer without importing pandas-dependent code."""
    global _OFFICIAL_NORMALIZE_TEXT, _OFFICIAL_LEVENSHTEIN
    notebook_path = find_metric_notebook(main_tex)
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    start = source.find("_LATEX_SYMBOLS =")
    end = source.find("# ── IoU", start)
    if start == -1 or end == -1:
        end = source.find("def _compute_iou", start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not extract official text normalizer from metric notebook")
    namespace: dict[str, Any] = {"re": re}
    exec(source[start:end], namespace)
    _OFFICIAL_NORMALIZE_TEXT = namespace["_normalize_text"]
    _OFFICIAL_LEVENSHTEIN = namespace["_levenshtein"]
    return notebook_path


def image_key(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value).replace("\\", "/")).name


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = repair_mojibake(text)
    return re.sub(r"\s+", " ", text.strip())


def official_normalize_text(value: Any, region_type: str) -> str:
    if _OFFICIAL_NORMALIZE_TEXT is None:
        raise RuntimeError("Official metric normalizer was not configured")
    text = repair_mojibake("" if value is None else str(value))
    return _OFFICIAL_NORMALIZE_TEXT(text, region_type)


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


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-cp1252 mojibake without touching normal text."""
    if not any(marker in text for marker in ("Ð", "Ñ", "Â")):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text

    def cyrillic_count(value: str) -> int:
        return sum(1 for ch in value if "\u0400" <= ch <= "\u04ff")

    if cyrillic_count(repaired) > cyrillic_count(text):
        return repaired
    return text


def truncate_text(text: str, limit: int = 78) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def route_for_type(region_type: str) -> str:
    if region_type in {"handwritten", "printed", "annotation"}:
        return "TrOCR"
    if region_type in {"formula", "table"}:
        return "Qwen3-VL"
    if region_type in {"image", "graph"}:
        return "empty text"
    return "unmatched"


def clamp_bbox(
    bbox: Iterable[Any], width: float, height: float
) -> tuple[float, float, float, float]:
    vals = [float(v) for v in bbox]
    if len(vals) != 4:
        raise ValueError(f"bbox must contain 4 values, got {bbox!r}")
    x0, y0, x2, y2 = vals

    # RUKOPYS annotations and the submitted predictions are xyxy. This fallback
    # keeps the script usable if a future metadata file stores xywh boxes.
    if x2 <= x0 or y2 <= y0:
        x2 = x0 + max(0.0, x2)
        y2 = y0 + max(0.0, y2)

    x0 = min(max(x0, 0.0), width)
    y0 = min(max(y0, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 <= x0:
        x2 = min(width, x0 + 1.0)
    if y2 <= y0:
        y2 = min(height, y0 + 1.0)
    return (x0, y0, x2, y2)


def area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = area((ix0, iy0, ix1, iy1))
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def cer(pred_text: str, gt_text: str) -> float:
    pred = official_normalize_text(pred_text, "handwritten")
    gt = official_normalize_text(gt_text, "handwritten")
    distance = _OFFICIAL_LEVENSHTEIN(pred, gt) if _OFFICIAL_LEVENSHTEIN else edit_distance(pred, gt)
    return distance / max(1, len(gt))


def cer_for_region(pred_text: str, gt_text: str, region_type: str) -> float:
    pred = official_normalize_text(pred_text, region_type)
    gt = official_normalize_text(gt_text, region_type)
    distance = _OFFICIAL_LEVENSHTEIN(pred, gt) if _OFFICIAL_LEVENSHTEIN else edit_distance(pred, gt)
    return distance / max(1, len(gt))


def read_gt(path: Path, image_dir: Path) -> dict[str, PageData]:
    pages: dict[str, PageData] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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
                        type=region_type if region_type in VALID_TYPES else region_type,
                        text=normalize_text(reg.get("text", "")),
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
                gt_regions=regions,
                pred_regions=[],
            )
    return pages


def parse_regions(value: str) -> list[dict[str, Any]]:
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("regions column must contain a list")
    return parsed


def attach_predictions(path: Path, pages: dict[str, PageData]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image" not in reader.fieldnames or "regions" not in reader.fieldnames:
            raise ValueError("prediction CSV must contain image and regions columns")
        for row in reader:
            key = image_key(row.get("image"))
            page = pages.get(key)
            if not page:
                continue
            pred_regions = []
            for idx, reg in enumerate(parse_regions(row.get("regions", ""))):
                region_type = str(reg.get("type", "")).strip()
                pred_regions.append(
                    Region(
                        bbox=clamp_bbox(reg.get("bbox", [0, 0, 1, 1]), page.width, page.height),
                        type=region_type if region_type in VALID_TYPES else region_type,
                        text=normalize_text(reg.get("text", "")),
                        index=idx,
                        language=str(reg.get("language", "uk")),
                        legibility=str(reg.get("legibility", "legible")),
                    )
                )
            page.pred_regions = pred_regions


def status_for_match(gt: Region | None, pred: Region | None, score_cer: float | None) -> str:
    if gt is None:
        return "false positive"
    if pred is None:
        return "missed"
    if not is_scorable(gt):
        return "unscored"
    if gt.type != pred.type:
        return "type confusion"
    if score_cer is not None and score_cer <= 0.10:
        return "correct"
    if score_cer is not None and score_cer <= 0.35:
        return "near-correct"
    return "failure"


def match_page(page: PageData, threshold: float = 0.5) -> list[MatchRecord]:
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(page.gt_regions):
        for pi, pred in enumerate(page.pred_regions):
            score = iou(gt.bbox, pred.bbox)
            if score >= threshold:
                pairs.append((score, gi, pi))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    records: list[MatchRecord] = []
    for score, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        gt = page.gt_regions[gi]
        pred = page.pred_regions[pi]
        score_cer = cer_for_region(pred.text, gt.text, gt.type) if is_scorable(gt) else None
        records.append(
            MatchRecord(
                image=page.image,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                gt=gt,
                pred=pred,
                iou=score,
                cer=score_cer,
                status=status_for_match(gt, pred, score_cer),
            )
        )
        used_gt.add(gi)
        used_pred.add(pi)

    for gi, gt in enumerate(page.gt_regions):
        if gi not in used_gt:
            records.append(
                MatchRecord(
                    image=page.image,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    gt=gt,
                    pred=None,
                    iou=0.0,
                    cer=1.0 if is_scorable(gt) else None,
                    status="missed",
                )
            )
    for pi, pred in enumerate(page.pred_regions):
        if pi not in used_pred:
            records.append(
                MatchRecord(
                    image=page.image,
                    image_path=page.image_path,
                    width=page.width,
                    height=page.height,
                    gt=None,
                    pred=pred,
                    iou=0.0,
                    cer=None,
                    status="false positive",
                )
            )
    return records


def match_all(pages: dict[str, PageData]) -> dict[str, list[MatchRecord]]:
    return {key: match_page(page) for key, page in pages.items()}


def page_score(page: PageData, records: list[MatchRecord]) -> tuple[float, float]:
    gt_types = {r.type for r in page.gt_regions}
    pred_count = len(page.pred_regions)
    valid_matches = [r for r in records if r.gt and r.pred]
    match_ratio = len(valid_matches) / max(1, len(page.gt_regions))
    type_errors = [r for r in valid_matches if r.gt_type != r.pred_type]
    text_errors = [
        r
        for r in valid_matches
        if r.gt_type == r.pred_type and r.cer is not None and 0.10 < r.cer <= 0.45
    ]
    missed_annotations = [r for r in records if r.gt_type == "annotation" and not r.pred]
    has_structured = bool(gt_types & {"formula", "table"})
    has_hpa = bool(gt_types & {"handwritten", "printed", "annotation"})
    has_annotation = "annotation" in gt_types
    diversity = len(gt_types)

    score = 0.0
    score += diversity * 2.0
    score += 5.0 if has_structured and has_hpa else 0.0
    score += 2.0 if has_annotation else 0.0
    score += 4.0 * match_ratio
    score += 2.0 if type_errors or text_errors or missed_annotations else 0.0
    if 8 <= pred_count <= 35:
        score += 2.0
    elif pred_count > 45:
        score -= 3.0
    if not page.image_path.exists():
        score -= 100.0
    return (score, match_ratio)


def select_page(pages: dict[str, PageData], all_records: dict[str, list[MatchRecord]]) -> PageData:
    candidates = [
        page
        for page in pages.values()
        if len({r.type for r in page.gt_regions}) >= 3 and page.pred_regions
    ]
    if not candidates:
        candidates = [page for page in pages.values() if page.pred_regions]
    if not candidates:
        raise RuntimeError("No page with predictions was found")
    return max(candidates, key=lambda p: page_score(p, all_records[p.image]))


def crop_quality(record: MatchRecord) -> float:
    x0, y0, x1, y1 = record.bbox_for_crop
    crop_area = max(1.0, (x1 - x0) * (y1 - y0))
    image_area = max(1.0, record.width * record.height)
    ratio = crop_area / image_area
    text_len = len((record.gt.text if record.gt else record.pred.text if record.pred else ""))
    score = 0.0
    score += min(record.iou, 1.0) * 3.0
    score += 1.0 if 0.002 <= ratio <= 0.25 else -1.0
    score += 1.0 if 3 <= text_len <= 100 else 0.0
    score += 0.5 if record.image_path.exists() else -5.0
    return score


def pick_best(
    records: list[MatchRecord],
    predicate,
    used: set[tuple[str, int, int]],
    preference,
) -> MatchRecord | None:
    candidates = [r for r in records if predicate(r) and record_key(r) not in used]
    if not candidates:
        return None
    return max(candidates, key=preference)


def record_key(record: MatchRecord) -> tuple[str, int, int]:
    gt_idx = record.gt.index if record.gt else -1
    pred_idx = record.pred.index if record.pred else -1
    return (record.image, gt_idx, pred_idx)


def select_crops(records_by_page: dict[str, list[MatchRecord]]) -> tuple[list[MatchRecord], list[str]]:
    records = [r for page_records in records_by_page.values() for r in page_records]
    selected: list[MatchRecord] = []
    used: set[tuple[str, int, int]] = set()
    warnings: list[str] = []

    def add(bucket: str, rec: MatchRecord | None) -> None:
        if rec is None:
            warnings.append(f"missing bucket: {bucket}")
            return
        rec.bucket = bucket
        selected.append(rec)
        used.add(record_key(rec))

    add(
        "handwritten success",
        pick_best(
            records,
            lambda r: r.gt_type == "handwritten"
            and r.pred_type == "handwritten"
            and is_scorable(r.gt)
            and r.cer is not None
            and r.cer <= 0.10,
            used,
            lambda r: crop_quality(r) - abs((r.cer or 0.0) - 0.04),
        ),
    )
    add(
        "formula near-correct",
        pick_best(
            records,
            lambda r: r.gt_type == "formula"
            and r.pred_type == "formula"
            and is_scorable(r.gt)
            and r.cer is not None
            and r.cer <= 0.35,
            used,
            lambda r: crop_quality(r) - abs((r.cer or 0.0) - 0.16),
        ),
    )
    add(
        "table example",
        pick_best(
            records,
            lambda r: r.gt_type == "table" and r.pred_type == "table" and is_scorable(r.gt),
            used,
            lambda r: crop_quality(r) + min((r.cer or 0.0), 0.5),
        ),
    )
    add(
        "annotation failure",
        pick_best(
            records,
            lambda r: r.gt_type == "annotation"
            and is_scorable(r.gt)
            and (
                r.pred_type in {"annotation", "handwritten", "missed"}
                and (r.status != "correct" or (r.cer is not None and r.cer >= 0.20))
            ),
            used,
            lambda r: crop_quality(r) + min((r.cer or 0.0), 1.0),
        ),
    )
    add(
        "formula-handwritten confusion",
        pick_best(
            records,
            lambda r: (r.gt_type, r.pred_type)
            in {("formula", "handwritten"), ("handwritten", "formula")}
            and is_scorable(r.gt)
            and r.cer is not None,
            used,
            lambda r: crop_quality(r) + 2.0,
        ),
    )
    add(
        "table formatting failure",
        pick_best(
            records,
            lambda r: r.gt_type == "table"
            and r.pred_type == "table"
            and is_scorable(r.gt)
            and r.cer is not None
            and (r.cer >= 0.20 or "|" not in (r.pred.text if r.pred else "")),
            used,
            lambda r: crop_quality(r) + min((r.cer or 0.0), 1.0),
        ),
    )

    if len(selected) < 4:
        for fallback in sorted(
            (r for r in records if r.gt and r.pred and is_scorable(r.gt) and record_key(r) not in used),
            key=lambda r: (r.status == "correct", crop_quality(r)),
            reverse=True,
        ):
            fallback.bucket = "fallback example"
            selected.append(fallback)
            used.add(record_key(fallback))
            if len(selected) >= 4:
                break

    return selected[:6], warnings


def pct(value: float, total: float) -> str:
    return f"{(value / max(total, 1.0)) * 100:.4f}%"


def expand_bbox(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
    pad_ratio: float = 0.08,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    pad_x = (x1 - x0) * pad_ratio
    pad_y = (y1 - y0) * pad_ratio
    return (
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def label_for_box(region: Region) -> str:
    return f"{region.type} | {route_for_type(region.type)}"


def html_escape(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def render_page_panel(page: PageData) -> str:
    boxes = sorted(page.pred_regions, key=lambda r: (r.bbox[1], r.bbox[0]))
    max_labels = 34
    if len(boxes) > max_labels:
        boxes = boxes[:max_labels]

    box_html = []
    for region in boxes:
        x0, y0, x1, y1 = region.bbox
        color = TYPE_COLORS.get(region.type, "#333333")
        box_html.append(
            f"""
            <div class="bbox" style="
              left:{pct(x0, page.width)}; top:{pct(y0, page.height)};
              width:{pct(x1 - x0, page.width)}; height:{pct(y1 - y0, page.height)};
              border-color:{color};">
              <span style="background:{color};">{html_escape(label_for_box(region))}</span>
            </div>
            """
        )

    return f"""
    <section class="page-section">
      <div class="section-title">Page-level output: detector, region type, and route</div>
      <div class="page-frame" style="aspect-ratio:{page.width:.0f}/{page.height:.0f};">
        <img src="{file_uri(page.image_path)}" alt="Selected page">
        {''.join(box_html)}
      </div>
      <div class="page-meta">{html_escape(page.image)} | boxes shown: {len(boxes)} / {len(page.pred_regions)}</div>
    </section>
    """


def overlay_box(
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    cls: str,
    label: str,
) -> str:
    cx0, cy0, cx1, cy1 = crop
    x0, y0, x1, y1 = bbox
    cw, ch = max(1.0, cx1 - cx0), max(1.0, cy1 - cy0)
    return f"""
      <div class="{cls}" style="
        left:{((x0 - cx0) / cw) * 100:.4f}%; top:{((y0 - cy0) / ch) * 100:.4f}%;
        width:{((x1 - x0) / cw) * 100:.4f}%; height:{((y1 - y0) / ch) * 100:.4f}%;">
        <span>{html_escape(label)}</span>
      </div>
    """


def render_crop_card(record: MatchRecord) -> str:
    display_bbox = expand_bbox(record.bbox_for_crop, record.width, record.height)
    dx0, dy0, dx1, dy1 = display_bbox
    dw, dh = max(1.0, dx1 - dx0), max(1.0, dy1 - dy0)
    img_width_pct = record.width / dw * 100.0
    img_height_pct = record.height / dh * 100.0
    img_left_pct = -dx0 / dw * 100.0
    img_top_pct = -dy0 / dh * 100.0

    overlays = []
    if record.gt:
        overlays.append(overlay_box(record.gt.bbox, display_bbox, "gt-box", "GT"))
    if record.pred:
        overlays.append(overlay_box(record.pred.bbox, display_bbox, "pred-box", "Pred"))

    gt_text = truncate_text(record.gt.text if record.gt else "", 76)
    pred_text = truncate_text(record.pred.text if record.pred else "", 76)
    cer_text = "--" if record.cer is None else f"{record.cer:.3f}"
    status_class = re.sub(r"[^a-z]+", "-", record.status.lower()).strip("-")

    return f"""
    <article class="crop-card">
      <div class="crop-header">
        <span>{html_escape(record.bucket)}</span>
        <b class="status {status_class}">{html_escape(record.status)}</b>
      </div>
      <div class="crop-stage" style="aspect-ratio:{dw:.0f}/{dh:.0f};">
        <img src="{file_uri(record.image_path)}" alt="Crop source" style="
          width:{img_width_pct:.4f}%; height:{img_height_pct:.4f}%;
          left:{img_left_pct:.4f}%; top:{img_top_pct:.4f}%;">
        {''.join(overlays)}
      </div>
      <div class="metrics">
        <span>GT: <b>{html_escape(record.gt_type)}</b></span>
        <span>Pred: <b>{html_escape(record.pred_type)}</b></span>
        <span>Route: <b>{html_escape(record.route)}</b></span>
        <span>IoU: <b>{record.iou:.2f}</b></span>
        <span>CER: <b>{cer_text}</b></span>
      </div>
      <div class="text-line"><b>GT text:</b> {html_escape(gt_text)}</div>
      <div class="text-line"><b>Pred text:</b> {html_escape(pred_text)}</div>
    </article>
    """


def render_html(page: PageData, crops: list[MatchRecord]) -> str:
    crop_cards = "\n".join(render_crop_card(crop) for crop in crops)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BoustoDoc qualitative results</title>
<style>
  @page {{ size: 7.2in 9.6in; margin: 0.12in; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    width: 6.96in;
    color: #111827;
    background: #ffffff;
    font-family: "Segoe UI", Arial, "DejaVu Sans", sans-serif;
    font-size: 8.5px;
    line-height: 1.22;
  }}
  .figure-title {{
    font-size: 13px;
    font-weight: 700;
    margin: 0 0 5px;
  }}
  .section-title {{
    font-size: 9.5px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 3px;
  }}
  .page-section {{
    border: 1px solid #d1d5db;
    padding: 6px;
    margin-bottom: 6px;
  }}
  .page-frame {{
    position: relative;
    width: 3.25in;
    margin: 0 auto;
    overflow: hidden;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
  }}
  .page-frame img {{
    display: block;
    width: 100%;
    height: 100%;
  }}
  .bbox {{
    position: absolute;
    border: 1.15px solid;
    background: rgba(255,255,255,0.02);
  }}
  .bbox span {{
    position: absolute;
    left: -1px;
    top: -9px;
    max-width: 82px;
    padding: 1px 3px;
    color: white;
    font-size: 5.8px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .page-meta {{
    text-align: center;
    color: #4b5563;
    margin-top: 3px;
    font-size: 7.2px;
  }}
  .crop-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 4px;
  }}
  .crop-card {{
    border: 1px solid #d1d5db;
    background: #fff;
    padding: 4px;
    min-height: 1.48in;
    break-inside: avoid;
  }}
  .crop-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 4px;
    margin-bottom: 3px;
    font-weight: 700;
    font-size: 8px;
  }}
  .status {{
    padding: 1px 4px;
    border: 1px solid #d1d5db;
    background: #f3f4f6;
    border-radius: 2px;
    font-size: 6.8px;
    color: #374151;
  }}
  .status.correct {{ color: #065f46; background: #ecfdf5; border-color: #a7f3d0; }}
  .status.near-correct {{ color: #92400e; background: #fffbeb; border-color: #fde68a; }}
  .status.failure, .status.type-confusion, .status.missed {{ color: #991b1b; background: #fef2f2; border-color: #fecaca; }}
  .crop-stage {{
    position: relative;
    width: 100%;
    max-height: 0.68in;
    overflow: hidden;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
  }}
  .crop-stage img {{
    position: absolute;
    max-width: none;
    max-height: none;
  }}
  .gt-box, .pred-box {{
    position: absolute;
    border: 1.4px solid;
    pointer-events: none;
  }}
  .gt-box {{ border-color: #111827; }}
  .pred-box {{ border-color: #d62728; }}
  .gt-box span, .pred-box span {{
    position: absolute;
    left: -1px;
    top: -8px;
    padding: 0 3px;
    color: #ffffff;
    font-size: 5.9px;
    font-weight: 700;
  }}
  .gt-box span {{ background: #111827; }}
  .pred-box span {{ background: #d62728; }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(2, auto);
    gap: 2px 5px;
    margin: 3px 0;
    color: #374151;
    font-size: 6.9px;
  }}
  .text-line {{
    margin-top: 2px;
    color: #111827;
    font-size: 7.15px;
    overflow-wrap: anywhere;
  }}
</style>
</head>
<body>
  <div class="figure-title">Qualitative examples from the final type-aware pipeline</div>
  {render_page_panel(page)}
  <section>
    <div class="section-title">Composite crop examples from multiple pages</div>
    <div class="crop-grid">
      {crop_cards}
    </div>
  </section>
</body>
</html>
"""


def find_browser() -> Path | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    browser = find_browser()
    if not browser:
        raise RuntimeError(
            "Could not find Edge or Chrome for HTML-to-PDF rendering. "
            "The HTML file was still written; open it manually or install a Chromium browser."
        )
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--disable-extensions",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=1500",
        "--no-pdf-header-footer",
        f"--print-to-pdf={str(pdf_path)}",
        html_path.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Browser PDF rendering failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(
            "Browser finished but did not create a non-empty PDF. "
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def record_to_metadata(record: MatchRecord) -> dict[str, Any]:
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
        "image": record.image,
        "gt_type": record.gt_type,
        "pred_type": record.pred_type,
        "route": record.route,
        "iou": round(record.iou, 4),
        "cer": None if record.cer is None else round(record.cer, 4),
        "status": record.status,
        "scorable": is_scorable(record.gt),
        "gt_text": record.gt.text if record.gt else "",
        "pred_text": record.pred.text if record.pred else "",
        "gt_text_official_normalized": gt_norm,
        "pred_text_official_normalized": pred_norm,
        "gt_bbox": list(record.gt.bbox) if record.gt else None,
        "pred_bbox": list(record.pred.bbox) if record.pred else None,
    }


def write_metadata(path: Path, page: PageData, crops: list[MatchRecord], warnings: list[str]) -> None:
    metadata = {
        "selected_page": {
            "image": page.image,
            "path": str(page.image_path),
            "gt_regions": len(page.gt_regions),
            "pred_regions": len(page.pred_regions),
            "gt_types": sorted({r.type for r in page.gt_regions}),
        },
        "selected_crops": [record_to_metadata(crop) for crop in crops],
        "warnings": warnings,
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def find_subsection_end(text: str, start: int) -> int:
    matches = list(re.finditer(r"(?m)^\\(?:subsection|section)\{", text[start + 1 :]))
    if not matches:
        return len(text)
    return start + 1 + matches[0].start()


def update_main_tex(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    snippet = LATEX_SNIPPET + "\n\n"

    for title in ("Qualitative Results", "Competition Comparison"):
        marker = f"\\subsection{{{title}}}"
        start = text.find(marker)
        if start != -1:
            end = find_subsection_end(text, start)
            updated = text[:start].rstrip() + "\n\n" + snippet + text[end:].lstrip()
            path.write_text(updated, encoding="utf-8")
            return f"replaced subsection: {title}"

    marker = "\\subsection{Ablation Study}"
    start = text.find(marker)
    if start == -1:
        raise RuntimeError("Could not find insertion point: \\subsection{Ablation Study}")
    updated = text[:start].rstrip() + "\n\n" + snippet + text[start:].lstrip()
    path.write_text(updated, encoding="utf-8")
    return "inserted before Ablation Study"


def report_unused_leaderboard_refs(main_tex: Path, bib_path: Path) -> list[str]:
    if not bib_path.exists():
        return []
    tex = main_tex.read_text(encoding="utf-8")
    cited: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        cited.update(key.strip() for key in match.group(1).split(","))
    bib = bib_path.read_text(encoding="utf-8")
    leaderboard_keys = {"ebi2026public2", "ernesto2026top8", "konbu2026top10", "anton2026top19"}
    existing = set(re.findall(r"@\w+\{([^,\s]+)", bib))
    return sorted(key for key in leaderboard_keys if key in existing and key not in cited)


def validate_inputs(args: argparse.Namespace) -> None:
    for field in ("image_dir", "gt_jsonl", "pred_csv", "main_tex"):
        path = getattr(args, field)
        if not path.exists():
            raise FileNotFoundError(f"{field} does not exist: {path}")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {args.image_dir}")


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    metric_notebook = configure_official_metric_helpers(args.main_tex)
    args.output_fig.mkdir(parents=True, exist_ok=True)

    pages = read_gt(args.gt_jsonl, args.image_dir)
    attach_predictions(args.pred_csv, pages)
    all_records = match_all(pages)
    selected_page = select_page(pages, all_records)
    crops, warnings = select_crops(all_records)

    if len(crops) < 4:
        raise RuntimeError(f"Only selected {len(crops)} crop examples; expected at least 4")

    html_path = args.output_fig / "qualitative_results.html"
    pdf_path = args.output_fig / "qualitative_results.pdf"
    metadata_path = args.output_fig / "qualitative_selected_examples.json"

    html_path.write_text(render_html(selected_page, crops), encoding="utf-8")
    render_pdf(html_path, pdf_path)
    write_metadata(metadata_path, selected_page, crops, warnings)
    update_status = update_main_tex(args.main_tex)

    bib_path = args.main_tex.with_suffix(".bib")
    removable = report_unused_leaderboard_refs(args.main_tex, bib_path)

    print(f"Selected page: {selected_page.image}")
    print(f"Figure PDF: {pdf_path}")
    print(f"Debug HTML: {html_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Official metric normalizer: {metric_notebook}")
    print(f"LaTeX update: {update_status}")
    print("Selected crop examples:")
    for crop in crops:
        cer_text = "--" if crop.cer is None else f"{crop.cer:.4f}"
        print(
            f"  - {crop.bucket}: {crop.image} | "
            f"GT={crop.gt_type} Pred={crop.pred_type} Route={crop.route} "
            f"IoU={crop.iou:.4f} CER={cer_text} Status={crop.status}"
        )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if removable:
        print("Uncited leaderboard bibliography entries that can be removed:")
        for key in removable:
            print(f"  - {key}")
    else:
        print("No uncited leaderboard bibliography entries found.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
