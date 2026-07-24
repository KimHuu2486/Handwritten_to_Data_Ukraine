#!/usr/bin/env python
"""Render clean full-page qualitative figures for the supplementary paper."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


QUALITATIVE_NOTES = [
    (
        "All regions are detected and matched. The page shows one routing/type "
        "confusion case between visually similar formula and table regions."
    ),
    (
        "The detector covers most handwritten lines correctly. Remaining errors "
        "come from two missed regions and two extra predicted regions, while "
        "matched regions keep correct type labels."
    ),
    (
        "The page contains dense formula and handwritten content. Most regions "
        "are correctly detected and routed, with one matched type confusion case."
    ),
    (
        "A strong page-level example with complete detection and correct routing "
        "across all matched handwritten and annotation regions."
    ),
    (
        "This page illustrates a harder case with sparse writing and several "
        "type confusions between handwritten and formula-like regions."
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clean full-page PNGs without report-style headers."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Supplementary/AnalysResultDetail"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("Supplementary/AnalysResultDetail/qualitative_selected_examples.json"),
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


def render_clean_full_page_html(q: Any, page: Any, stats: Any, note: str) -> str:
    boxes = sorted(page.pred_regions, key=lambda region: (region.bbox[1], region.bbox[0]))
    visible_boxes = boxes[:42]
    box_html: list[str] = []
    for region in visible_boxes:
        x0, y0, x1, y1 = region.bbox
        color = q.TYPE_COLORS.get(region.type, "#374151")
        label = f"{region.type} | {q.route_for_type(region.type)}"
        box_html.append(
            f"""
            <div class="bbox" style="
              left:{q.pct(x0, page.width)}; top:{q.pct(y0, page.height)};
              width:{q.pct(x1 - x0, page.width)}; height:{q.pct(y1 - y0, page.height)};
              border-color:{color};">
              <span style="background:{color};">{q.html_escape(label)}</span>
            </div>
            """
        )

    cer_text = "--" if stats.mean_cer is None else f"{stats.mean_cer:.3f}"
    type_list = ", ".join(stats.gt_types)
    legend_items = "".join(
        (
            f'<div><span class="swatch" style="background:{q.TYPE_COLORS[k]}"></span>'
            f"{k} | {q.route_for_type(k)}</div>"
        )
        for k in ["handwritten", "printed", "annotation", "formula", "table", "image", "graph"]
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html {{
    width: 1400px;
    height: 1250px;
    overflow: hidden;
  }}
  body {{
    margin: 0;
    width: 1400px;
    height: 1250px;
    padding: 28px;
    overflow: hidden;
    background: #ffffff;
    color: #111827;
    font-family: Arial, "Segoe UI", "DejaVu Sans", sans-serif;
  }}
  .layout {{
    display: grid;
    grid-template-columns: 880px 1fr;
    gap: 22px;
    align-items: start;
  }}
  .page-frame {{
    position: relative;
    width: 880px;
    height: 1168px;
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
    max-width: 220px;
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
    padding: 18px;
    font-size: 21px;
    line-height: 1.32;
    background: #ffffff;
  }}
  .info h1 {{
    margin: 0 0 15px;
    font-size: 28px;
  }}
  .kv {{
    display: grid;
    grid-template-columns: 135px 1fr;
    gap: 9px 13px;
    margin-bottom: 17px;
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
    gap: 11px;
    margin: 17px 0;
  }}
  .metric div {{
    border: 1px solid #e5e7eb;
    padding: 11px;
    background: #f9fafb;
  }}
  .metric b {{
    display: block;
    font-size: 30px;
    color: #111827;
  }}
  .note {{
    margin-top: 17px;
    padding-top: 15px;
    border-top: 1px solid #e5e7eb;
    font-size: 20px;
  }}
  .legend {{
    margin-top: 19px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 12px;
    font-size: 17px;
  }}
  .swatch {{
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 7px;
    vertical-align: -3px;
  }}
</style>
</head>
<body>
  <div class="layout">
    <div class="page-frame">
      <img src="{q.file_uri(page.image_path)}" alt="Full page">
      {''.join(box_html)}
    </div>
    <aside class="info">
      <h1>Page diagnostics</h1>
      <div class="kv">
        <div class="key">Image ID</div><div class="value">{q.html_escape(page.image)}</div>
        <div class="key">Source</div><div class="value">{q.html_escape(page.source)}</div>
        <div class="key">GT types</div><div class="value">{q.html_escape(type_list)}</div>
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
      <div class="note"><b>Qualitative note:</b> {q.html_escape(note)}</div>
      <div class="legend">{legend_items}</div>
    </aside>
  </div>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    q = load_generator_module()
    q.configure_official_metric_helpers(args.metric_notebook)

    pages = q.read_gt(args.gt_jsonl, args.image_dir)
    q.attach_predictions(args.pred_csv, pages)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    selected_images = [item["source_image"] for item in metadata["full_page_examples"]]

    all_records = {key: q.match_page(page) for key, page in pages.items()}
    stats_by_image = {
        key: q.compute_page_stats(page, all_records[key]) for key, page in pages.items()
    }

    figure_dir = args.output_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    browser = q.find_browser(None)

    with tempfile.TemporaryDirectory(prefix="clean_full_page_") as tmp:
        tmp_dir = Path(tmp)
        for idx, image_id in enumerate(selected_images, start=1):
            page = pages[image_id]
            html_path = tmp_dir / f"full_page_example_{idx:02d}_clean.html"
            png_path = figure_dir / f"full_page_example_{idx:02d}_clean.png"
            html_path.write_text(
                render_clean_full_page_html(
                    q,
                    page,
                    stats_by_image[image_id],
                    QUALITATIVE_NOTES[idx - 1],
                ),
                encoding="utf-8",
            )
            q.render_png(browser, html_path, png_path, width=1400, height=1250)
            print(f"wrote {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
