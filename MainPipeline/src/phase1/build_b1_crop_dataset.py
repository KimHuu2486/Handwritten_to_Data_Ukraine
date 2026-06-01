from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from MainPipeline.src.common.bbox import as_float_bbox, clamp_bbox, is_valid_bbox, padded_bbox
from MainPipeline.src.common.io import load_config, read_jsonl, resolve_path, write_json, write_jsonl
from MainPipeline.src.common.prompts import stage_b_prompt
from MainPipeline.src.common.schema import normalize_type, text_for_type
from MainPipeline.src.phase1.dataset_common import image_keys, qwen_vl_example, resolve_image_path


def safe_name(value: str) -> str:
    """Sanitize metadata values before using them in crop filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def crop_from_open_image(image, bbox: list[int], output_path: Path, pad_ratio: float) -> tuple[int, int]:
    """Crop one region from an already-open page image and save it to disk."""
    crop_box = padded_bbox(bbox, image.width, image.height, pad_ratio)
    crop = image.crop(tuple(crop_box))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=95)
    return crop.width, crop.height


def crop_record_job(job: dict[str, Any], pad_ratio: float, max_pixels_crop: int) -> list[dict[str, Any]]:
    """Crop every planned region for one page and return B1 SFT examples in region order."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    examples: list[dict[str, Any]] = []
    with Image.open(job["image_path"]) as image:
        image = image.convert("RGB")
        for spec in job["regions"]:
            crop_width, crop_height = crop_from_open_image(image, spec["bbox"], spec["crop_path"], pad_ratio)
            metadata = {
                "task": "B1_crop_ocr",
                "source": job["source"],
                "type": spec["type"],
                "file_name": job["file_name"],
                "region_id": spec["region_id"],
                "image_keys": job["image_keys"],
                "bbox": spec["bbox"],
                "crop_width": crop_width,
                "crop_height": crop_height,
            }
            examples.append(
                qwen_vl_example(
                    image_path=spec["crop_path"],
                    prompt_text=stage_b_prompt(job["source"], spec["type"]),
                    answer_text=spec["target_text"],
                    max_pixels=max_pixels_crop,
                    metadata=metadata,
                )
            )
    return examples


def build_examples(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Crop annotated regions and convert them into B1 type-aware OCR SFT examples."""
    metadata_path = resolve_path(cfg["metadata_path"])
    image_roots = [str(root) for root in cfg.get("image_roots", [])]
    crops_dir = resolve_path(cfg["crops_dir"])
    max_records = cfg.get("max_records")
    max_regions = cfg.get("max_regions")
    max_pixels_crop = int(cfg.get("max_pixels_crop", 262144))
    pad_ratio = float(cfg.get("crop_pad_ratio", 0.0))
    skip_missing_images = bool(cfg.get("skip_missing_images", True))
    num_workers = max(int(cfg.get("num_workers", 1)), 1)

    record_jobs: list[dict[str, Any]] = []
    stats = {
        "metadata_path": str(metadata_path),
        "rows_seen": 0,
        "regions_seen": 0,
        "regions_written": 0,
        "missing_images": 0,
        "invalid_bboxes": 0,
        "num_workers": num_workers,
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
        image_key_values = sorted(image_keys(record))
        crop_specs: list[dict[str, Any]] = []

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
            crop_specs.append(
                {
                    "type": region_type,
                    "region_id": region_id,
                    "bbox": pixel_bbox,
                    "crop_path": crop_path,
                    "target_text": text_for_type(region_type, raw_region.get("text", "")),
                }
            )
            stats["regions_written"] += 1
        if crop_specs:
            record_jobs.append(
                {
                    "image_path": image_path,
                    "source": source,
                    "file_name": record.get("file_name"),
                    "image_keys": image_key_values,
                    "regions": crop_specs,
                }
            )
        if max_regions is not None and stats["regions_written"] >= int(max_regions):
            break

    print(
        f"planned B1 crops rows={len(record_jobs)} regions={stats['regions_written']} num_workers={num_workers}",
        flush=True,
    )
    examples: list[dict[str, Any]] = []
    worker = partial(crop_record_job, pad_ratio=pad_ratio, max_pixels_crop=max_pixels_crop)
    if num_workers == 1:
        for job in record_jobs:
            examples.extend(worker(job))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for record_examples in executor.map(worker, record_jobs):
                examples.extend(record_examples)

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
