from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from MainPipeline.src.common.bbox import as_float_bbox, clamp_bbox, crop_region, is_valid_bbox
from MainPipeline.src.common.io import load_config, read_jsonl, resolve_path, write_json, write_jsonl
from MainPipeline.src.common.prompts import stage_b_prompt
from MainPipeline.src.common.schema import normalize_type, text_for_type
from MainPipeline.src.phase1.dataset_common import image_keys, qwen_vl_example, resolve_image_path


def safe_name(value: str) -> str:
    """Sanitize metadata values before using them in crop filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def build_examples(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Crop annotated regions and convert them into B1 type-aware OCR SFT examples."""
    metadata_path = resolve_path(cfg["metadata_path"])
    image_roots = [str(root) for root in cfg.get("image_roots", [])]
    crops_dir = resolve_path(cfg["crops_dir"])
    max_records = cfg.get("max_records")
    max_regions = cfg.get("max_regions")
    max_pixels_crop = int(cfg.get("max_pixels_crop", 262144))
    pad_ratio = float(cfg.get("crop_pad_ratio", 0.02))
    skip_missing_images = bool(cfg.get("skip_missing_images", True))

    examples: list[dict[str, Any]] = []
    stats = {
        "metadata_path": str(metadata_path),
        "rows_seen": 0,
        "regions_seen": 0,
        "regions_written": 0,
        "missing_images": 0,
        "invalid_bboxes": 0,
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

        image_width = int(record.get("image_width", 1))
        image_height = int(record.get("image_height", 1))
        source = str(record.get("source", "unknown"))
        image_stem = safe_name(Path(str(record.get("file_name", "image"))).stem)

        for region_index, raw_region in enumerate(record.get("regions", []) or []):
            if max_regions is not None and stats["regions_written"] >= int(max_regions):
                break
            stats["regions_seen"] += 1
            bbox = as_float_bbox(raw_region.get("bbox"))
            if bbox is None:
                stats["invalid_bboxes"] += 1
                continue
            pixel_bbox = clamp_bbox(bbox, image_width, image_height)
            if not is_valid_bbox(pixel_bbox):
                stats["invalid_bboxes"] += 1
                continue

            region_type = normalize_type(raw_region.get("type"))
            region_id = str(raw_region.get("region_id", f"r{region_index:04d}"))
            crop_path = crops_dir / source / f"{image_stem}__{safe_name(region_id)}__{region_type}.jpg"
            crop_width, crop_height = crop_region(image_path, pixel_bbox, crop_path, pad_ratio)
            target_text = text_for_type(region_type, raw_region.get("text", ""))

            metadata = {
                "task": "B1_crop_ocr",
                "source": source,
                "type": region_type,
                "file_name": record.get("file_name"),
                "region_id": region_id,
                "image_keys": sorted(image_keys(record)),
                "bbox": pixel_bbox,
                "crop_width": crop_width,
                "crop_height": crop_height,
            }
            examples.append(
                qwen_vl_example(
                    image_path=crop_path,
                    prompt_text=stage_b_prompt(source, region_type),
                    answer_text=target_text,
                    max_pixels=max_pixels_crop,
                    metadata=metadata,
                )
            )
            stats["regions_written"] += 1
        if max_regions is not None and stats["regions_written"] >= int(max_regions):
            break

    return examples, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 1 B1 crop OCR dataset.")
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

