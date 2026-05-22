from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from MainPipeline.src.common.io import load_config, read_jsonl, resolve_path
from MainPipeline.src.common.schema import normalize_type


def parse_regions(value: str) -> list[dict[str, Any]]:
    """Parse a regions JSON string from a prediction CSV row."""
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write analysis rows with a stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Create lightweight Phase 1 validation error summaries.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML analysis config.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    manifest_rows = read_jsonl(resolve_path(cfg["manifest_path"]))
    manifest_by_image = {row["submission_image"]: row for row in manifest_rows}

    with resolve_path(cfg["predictions_csv"]).open("r", encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))

    by_source: dict[str, dict[str, Any]] = defaultdict(lambda: {"images": 0, "gt_regions": 0, "pred_regions": 0, "parse_fail": 0})
    gt_type_counts: Counter[str] = Counter()
    pred_type_counts: Counter[str] = Counter()

    for pred in prediction_rows:
        image = pred["image"]
        gt = manifest_by_image.get(image, {})
        source = str(gt.get("source", "unknown"))
        pred_regions = parse_regions(pred.get("regions", "[]"))
        gt_regions = gt.get("regions", []) if isinstance(gt.get("regions"), list) else []

        by_source[source]["images"] += 1
        by_source[source]["gt_regions"] += len(gt_regions)
        by_source[source]["pred_regions"] += len(pred_regions)
        if str(pred.get("parse_ok", "")).lower() != "true":
            by_source[source]["parse_fail"] += 1
        gt_type_counts.update(normalize_type(region.get("type")) for region in gt_regions)
        pred_type_counts.update(normalize_type(region.get("type")) for region in pred_regions)

    source_rows = []
    for source, stats in sorted(by_source.items()):
        source_rows.append(
            {
                "source": source,
                **stats,
                "region_delta": int(stats["pred_regions"]) - int(stats["gt_regions"]),
            }
        )
    type_rows = []
    for region_type in sorted(set(gt_type_counts) | set(pred_type_counts)):
        type_rows.append(
            {
                "type": region_type,
                "gt_count": gt_type_counts.get(region_type, 0),
                "pred_count": pred_type_counts.get(region_type, 0),
                "delta": pred_type_counts.get(region_type, 0) - gt_type_counts.get(region_type, 0),
            }
        )

    write_csv(resolve_path(cfg["source_breakdown_csv"]), source_rows, ["source", "images", "gt_regions", "pred_regions", "region_delta", "parse_fail"])
    write_csv(resolve_path(cfg["type_breakdown_csv"]), type_rows, ["type", "gt_count", "pred_count", "delta"])
    print(json.dumps({"sources": len(source_rows), "types": len(type_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

