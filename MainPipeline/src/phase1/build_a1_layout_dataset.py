from __future__ import annotations

import argparse
import json
from typing import Any

from MainPipeline.src.common.bbox import as_float_bbox, clamp_bbox, is_valid_bbox, pixel_to_grid_bbox
from MainPipeline.src.common.io import load_config, read_jsonl, resolve_path, write_json, write_jsonl
from MainPipeline.src.common.prompts import stage_a_prompt
from MainPipeline.src.common.schema import normalize_type, sorted_regions
from MainPipeline.src.phase1.dataset_common import image_keys, qwen_vl_example, resolve_image_path


def build_layout_target(record: dict[str, Any], grid_size: int) -> list[dict[str, Any]]:
    """Build the Phase 1 Stage A target: bbox/type only, with no text fields."""
    image_width = int(record.get("image_width", 1))
    image_height = int(record.get("image_height", 1))
    regions: list[dict[str, Any]] = []
    for raw_region in record.get("regions", []) or []:
        bbox = as_float_bbox(raw_region.get("bbox"))
        if bbox is None:
            continue
        pixel_bbox = clamp_bbox(bbox, image_width, image_height)
        if not is_valid_bbox(pixel_bbox):
            continue
        regions.append(
            {
                "bbox": pixel_to_grid_bbox(pixel_bbox, image_width, image_height, grid_size),
                "type": normalize_type(raw_region.get("type")),
            }
        )
    return sorted_regions(regions)


def build_examples(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read metadata rows and convert valid pages into A1 layout-only SFT examples."""
    metadata_path = resolve_path(cfg["metadata_path"])
    image_roots = [str(root) for root in cfg.get("image_roots", [])]
    max_records = cfg.get("max_records")
    skip_missing_images = bool(cfg.get("skip_missing_images", True))
    grid_size = int(cfg.get("grid_size", 1000))
    max_pixels = int(cfg.get("max_pixels_page", 850000))

    examples: list[dict[str, Any]] = []
    stats = {
        "metadata_path": str(metadata_path),
        "rows_seen": 0,
        "rows_written": 0,
        "missing_images": 0,
        "empty_targets": 0,
    }

    for record in read_jsonl(metadata_path):
        if max_records is not None and stats["rows_seen"] >= int(max_records):
            break
        stats["rows_seen"] += 1
        image_path = resolve_image_path(record, metadata_path, image_roots)
        if image_path is None:
            stats["missing_images"] += 1
            if skip_missing_images:
                continue
            raise FileNotFoundError(f"Cannot resolve image for {record.get('file_name')}")

        target_regions = build_layout_target(record, grid_size)
        if not target_regions:
            stats["empty_targets"] += 1
            continue

        source = str(record.get("source", "unknown"))
        metadata = {
            "task": "A1_layout_only",
            "source": source,
            "file_name": record.get("file_name"),
            "image_keys": sorted(image_keys(record)),
            "num_regions": len(target_regions),
        }
        examples.append(
            qwen_vl_example(
                image_path=image_path,
                prompt_text=stage_a_prompt(source),
                answer_text=json.dumps(target_regions, ensure_ascii=False),
                max_pixels=max_pixels,
                metadata=metadata,
            )
        )
        stats["rows_written"] += 1

    return examples, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 1 A1 layout-only dataset.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    examples, stats = build_examples(cfg)
    write_jsonl(resolve_path(cfg["output_jsonl"]), examples)
    if cfg.get("stats_json"):
        write_json(resolve_path(cfg["stats_json"]), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

