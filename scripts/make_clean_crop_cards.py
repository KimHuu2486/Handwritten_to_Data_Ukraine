#!/usr/bin/env python
"""Render individual crop-level cards for LaTeX adaptive grids."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


WIDTH_CLASSES = {
    "handwritten success": "full",
    "printed success": "full",
    "formula near-correct": "wide",
    "formula failure": "full",
    "table example": "wide",
    "table formatting failure": "wide",
    "annotation near-correct": "normal",
    "annotation failure": "normal",
    "formula-handwritten confusion": "full",
    "type confusion": "wide",
    "empty-text route": "normal",
    "formula type confusion": "normal",
}

CARD_SIZES = {
    "normal": (900, 900),
    "wide": (1400, 780),
    "full": (1800, 760),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one cleaned PNG card per crop example.")
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
        default=Path("Supplementary/AnalysResultDetail/figure/crops"),
    )
    return parser.parse_args()


def load_generator_module() -> Any:
    module_path = Path("scripts/make_additional_qualitative_results.py")
    spec = importlib.util.spec_from_file_location("make_additional_qualitative_results", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def title_for(record: Any) -> str:
    if record.gt_type in {"handwritten", "printed", "formula", "table", "annotation", "image", "graph"}:
        return record.gt_type
    return record.pred_type


def status_for_display(status: str) -> str:
    return "type confusion" if status == "type confusion" else status


def class_for(record: Any) -> str:
    return WIDTH_CLASSES.get(record.bucket, "normal")


def text_limit(width_class: str) -> int:
    return {"normal": 58, "wide": 105, "full": 145}[width_class]


def display_text(q: Any, text: str, limit: int) -> tuple[str, bool]:
    text = q.clean_text(text)
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 3)].rstrip() + "...", True


def card_filename(index: int, record: Any) -> str:
    semantic = title_for(record)
    status = slugify(status_for_display(record.status))
    return f"crop_{index:02d}_{semantic}_{status}.png"


def overlay_box(q: Any, bbox: tuple[float, float, float, float], crop: tuple[float, float, float, float], class_name: str, label: str) -> str:
    cx0, cy0, cx1, cy1 = crop
    x0, y0, x1, y1 = bbox
    cw = max(1.0, cx1 - cx0)
    ch = max(1.0, cy1 - cy0)
    return f"""
    <div class="{class_name}" style="
      left:{((x0 - cx0) / cw) * 100:.4f}%; top:{((y0 - cy0) / ch) * 100:.4f}%;
      width:{((x1 - x0) / cw) * 100:.4f}%; height:{((y1 - y0) / ch) * 100:.4f}%;">
      <span>{q.html_escape(label)}</span>
    </div>
    """


def render_crop_card_html(q: Any, record: Any, width_class: str) -> str:
    width, height = CARD_SIZES[width_class]
    crop = q.expand_bbox(record.crop_bbox, record.width, record.height, pad_ratio=0.10)
    cx0, cy0, cx1, cy1 = crop
    cw = max(1.0, cx1 - cx0)
    ch = max(1.0, cy1 - cy0)
    image_width_pct = record.width / cw * 100.0
    image_height_pct = record.height / ch * 100.0
    image_left_pct = -cx0 / cw * 100.0
    image_top_pct = -cy0 / ch * 100.0

    overlays: list[str] = []
    if record.gt:
        overlays.append(overlay_box(q, record.gt.bbox, crop, "gt-box", "GT"))
    if record.pred:
        overlays.append(overlay_box(q, record.pred.bbox, crop, "pred-box", "Pred"))

    limit = text_limit(width_class)
    gt_text, gt_truncated = display_text(q, record.gt.text if record.gt else "", limit)
    pred_text, pred_truncated = display_text(q, record.pred.text if record.pred else "", limit)
    truncated = gt_truncated or pred_truncated
    note = record.note
    if truncated:
        note = (
            "Long text is truncated for display; full text is used for CER computation. "
            + note
        )

    cer_text = "--" if record.cer is None else f"{record.cer:.3f}"
    status = status_for_display(record.status)
    status_class = slugify(status)
    image_stage_height = {"normal": 255, "wide": 245, "full": 255}[width_class]
    base_font = {"normal": 31, "wide": 29, "full": 30}[width_class]
    title_font = {"normal": 44, "wide": 42, "full": 42}[width_class]

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html {{
    width: {width}px;
    height: {height}px;
    overflow: hidden;
  }}
  body {{
    margin: 0;
    width: {width}px;
    height: {height}px;
    padding: 24px;
    overflow: hidden;
    background: #ffffff;
    color: #111827;
    font-family: Arial, "Segoe UI", "DejaVu Sans", sans-serif;
  }}
  .card {{
    width: 100%;
    height: 100%;
    border: 1px solid #d1d5db;
    padding: 20px;
    background: #ffffff;
  }}
  .header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 15px;
  }}
  .title {{
    font-size: {title_font}px;
    line-height: 1.0;
    font-weight: 700;
  }}
  .status {{
    font-size: {base_font - 5}px;
    line-height: 1.0;
    color: #374151;
    white-space: nowrap;
  }}
  .status b {{
    color: #111827;
  }}
  .status.type-confusion b,
  .status.failure b {{
    color: #991b1b;
  }}
  .status.near-correct b {{
    color: #92400e;
  }}
  .status.correct b {{
    color: #065f46;
  }}
  .crop-image {{
    position: relative;
    width: 100%;
    height: {image_stage_height}px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
  }}
  .crop-image img {{
    position: absolute;
    max-width: none;
    max-height: none;
    width: {image_width_pct:.4f}%;
    height: {image_height_pct:.4f}%;
    left: {image_left_pct:.4f}%;
    top: {image_top_pct:.4f}%;
  }}
  .gt-box,
  .pred-box {{
    position: absolute;
    border: 4px solid;
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
    left: -4px;
    top: -30px;
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    padding: 2px 8px;
  }}
  .gt-box span {{
    background: #111827;
  }}
  .pred-box span {{
    background: #dc2626;
  }}
  .meta {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px 18px;
    margin-top: 14px;
    font-size: {base_font}px;
    line-height: 1.17;
  }}
  .meta .source {{
    grid-column: 1 / -1;
  }}
  .meta b,
  .text-line b,
  .note b {{
    font-weight: 700;
  }}
  .text-line {{
    margin-top: 9px;
    font-size: {base_font}px;
    line-height: 1.18;
    overflow-wrap: anywhere;
  }}
  .note {{
    margin-top: 11px;
    font-size: {base_font - 2}px;
    line-height: 1.18;
    color: #374151;
    overflow-wrap: anywhere;
  }}
</style>
</head>
<body>
  <article class="card">
    <div class="header">
      <div class="title">{q.html_escape(title_for(record))}</div>
      <div class="status {status_class}">Status: <b>{q.html_escape(status)}</b></div>
    </div>
    <div class="crop-image">
      <img src="{q.file_uri(record.image_path)}" alt="Crop source">
      {''.join(overlays)}
    </div>
    <div class="meta">
      <div class="source"><b>Source image:</b> {q.html_escape(record.image)}</div>
      <div><b>GT type:</b> {q.html_escape(record.gt_type)}</div>
      <div><b>Pred type:</b> {q.html_escape(record.pred_type)}</div>
      <div><b>Route:</b> {q.html_escape(record.route)}</div>
      <div><b>IoU:</b> {record.iou:.2f}</div>
      <div><b>Norm CER:</b> {cer_text}</div>
    </div>
    <div class="text-line"><b>GT text:</b> {q.html_escape(gt_text)}</div>
    <div class="text-line"><b>Pred text:</b> {q.html_escape(pred_text)}</div>
    <div class="note"><b>Diagnostic note:</b> {q.html_escape(note)}</div>
  </article>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    q = load_generator_module()
    q.configure_official_metric_helpers(args.metric_notebook)
    pages = q.read_gt(args.gt_jsonl, args.image_dir)
    q.attach_predictions(args.pred_csv, pages)
    all_records = {key: q.match_page(page) for key, page in pages.items()}
    crops, warnings = q.select_crop_examples(all_records, desired=12)
    if warnings:
        print("Selection warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    browser = q.find_browser(None)

    with tempfile.TemporaryDirectory(prefix="clean_crop_cards_") as tmp:
        tmp_dir = Path(tmp)
        for idx, record in enumerate(crops, start=1):
            width_class = class_for(record)
            width, height = CARD_SIZES[width_class]
            filename = card_filename(idx, record)
            html_path = tmp_dir / f"{filename}.html"
            png_path = args.output_dir / filename
            html_path.write_text(
                render_crop_card_html(q, record, width_class),
                encoding="utf-8",
            )
            q.render_png(browser, html_path, png_path, width=width, height=height)
            print(f"{idx:02d} {width_class:6s} {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
