#!/usr/bin/env python3
"""Build a small silver subset for rare layout classes and draw bbox previews."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_TARGET_TYPES = ("table", "graph", "image")
STANDARD_TOP_LEVEL_KEYS = (
    "annotation_source",
    "file_name",
    "image_height",
    "image_width",
    "regions",
    "source",
)
STANDARD_REGION_KEYS = ("bbox", "type", "language", "legibility", "text")
TYPE_COLORS = {
    "table": (225, 29, 72),
    "graph": (37, 99, 235),
    "image": (22, 163, 74),
    "formula": (147, 51, 234),
    "annotation": (8, 145, 178),
    "printed": (71, 85, 105),
    "handwritten": (245, 158, 11),
}
FALLBACK_COLOR = (15, 23, 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore a silver metadata JSONL file, filter images that contain rare "
            "classes, copy them into a smaller dataset, and draw bbox previews."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/silver"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/silver_rare_layout"))
    parser.add_argument("--metadata-name", default="metadata.jsonl")
    parser.add_argument("--target-types", nargs="+", default=list(DEFAULT_TARGET_TYPES))
    parser.add_argument(
        "--rare-regions-only",
        action="store_true",
        help="Keep only regions whose type is in --target-types. Default keeps all regions for selected images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Default exports every selected image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place source images into the subset.",
    )
    parser.add_argument(
        "--draw-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw all boxes on previews. Rare target boxes are highlighted.",
    )
    return parser.parse_args()


def load_jsonl(metadata_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {metadata_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {metadata_path}:{line_number}")
            records.append(record)
    return records


def normalize_region(region: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: region.get(key, "") for key in STANDARD_REGION_KEYS}
    bbox = normalized["bbox"]
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox: {bbox}")
    normalized["bbox"] = [int(round(float(value))) for value in bbox]
    return normalized


def normalize_record(
    record: dict[str, Any],
    output_file_name: str,
    target_types: set[str],
    rare_regions_only: bool,
) -> dict[str, Any]:
    normalized = {key: record.get(key) for key in STANDARD_TOP_LEVEL_KEYS}
    normalized["file_name"] = output_file_name

    regions = record.get("regions", [])
    if not isinstance(regions, list):
        regions = []

    normalized_regions: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        if rare_regions_only and region.get("type") not in target_types:
            continue
        normalized_regions.append(normalize_region(region))
    normalized["regions"] = normalized_regions
    return normalized


def record_types(record: dict[str, Any]) -> set[str]:
    regions = record.get("regions", [])
    if not isinstance(regions, list):
        return set()
    return {
        region.get("type")
        for region in regions
        if isinstance(region, dict) and isinstance(region.get("type"), str)
    }


def place_image(source_path: Path, destination_path: Path, copy_mode: str) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()

    if copy_mode == "copy":
        shutil.copy2(source_path, destination_path)
    elif copy_mode == "hardlink":
        destination_path.hardlink_to(source_path)
    elif copy_mode == "symlink":
        destination_path.symlink_to(source_path.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def draw_previews(
    source_path: Path,
    destination_path: Path,
    record: dict[str, Any],
    target_types: set[str],
    draw_all: bool,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        line_width = max(2, round(max(image.size) / 900))

        for region in record.get("regions", []):
            region_type = region.get("type")
            if not draw_all and region_type not in target_types:
                continue

            bbox = region.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            x1, x2 = sorted((max(0, x1), min(image.width - 1, x2)))
            y1, y2 = sorted((max(0, y1), min(image.height - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            color = TYPE_COLORS.get(str(region_type), FALLBACK_COLOR)
            rare = region_type in target_types
            width = line_width * (2 if rare else 1)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

            if rare:
                label = str(region_type)
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                label_width = text_bbox[2] - text_bbox[0] + 6
                label_height = text_bbox[3] - text_bbox[1] + 4
                label_y1 = max(0, y1 - label_height)
                draw.rectangle((x1, label_y1, x1 + label_width, label_y1 + label_height), fill=color)
                draw.text((x1 + 3, label_y1 + 2), label, fill=(255, 255, 255), font=font)

        image.save(destination_path, quality=95)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(records: list[dict[str, Any]], selected_records: list[dict[str, Any]], target_types: set[str]) -> dict[str, Any]:
    region_counts = Counter()
    image_counts = Counter()
    selected_region_counts = Counter()
    selected_image_counts = Counter()
    selected_target_combo_counts = Counter()

    for record in records:
        types_in_image = record_types(record)
        image_counts.update(types_in_image)
        regions = record.get("regions", [])
        if isinstance(regions, list):
            region_counts.update(
                region.get("type")
                for region in regions
                if isinstance(region, dict) and isinstance(region.get("type"), str)
            )

    for record in selected_records:
        types_in_image = record_types(record)
        selected_image_counts.update(types_in_image)
        selected_target_combo_counts.update([",".join(sorted(types_in_image & target_types))])
        regions = record.get("regions", [])
        if isinstance(regions, list):
            selected_region_counts.update(
                region.get("type")
                for region in regions
                if isinstance(region, dict) and isinstance(region.get("type"), str)
            )

    return {
        "target_types": sorted(target_types),
        "total_images": len(records),
        "selected_images": len(selected_records),
        "total_region_counts": dict(region_counts.most_common()),
        "total_image_counts_by_type": dict(image_counts.most_common()),
        "selected_region_counts": dict(selected_region_counts.most_common()),
        "selected_image_counts_by_type": dict(selected_image_counts.most_common()),
        "selected_target_combo_counts": dict(selected_target_combo_counts.most_common()),
    }


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir
    output_dir = args.output_dir
    metadata_path = input_dir / args.metadata_name
    target_types = set(args.target_types)

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_images_dir = output_dir / "images"
    output_annotated_dir = output_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotated_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(metadata_path)
    selected = [record for record in records if record_types(record) & target_types]
    if args.max_images is not None:
        selected = selected[: args.max_images]

    output_records: list[dict[str, Any]] = []
    seen_output_names: set[str] = set()

    for index, record in enumerate(selected, start=1):
        source_file_name = str(record.get("file_name", ""))
        source_path = input_dir / source_file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing image for metadata record: {source_path}")

        output_name = source_path.name
        if output_name in seen_output_names:
            output_name = f"{source_path.stem}-{index}{source_path.suffix}"
        seen_output_names.add(output_name)
        output_file_name = f"images/{output_name}"
        output_image_path = output_images_dir / output_name
        output_annotated_path = output_annotated_dir / output_name

        normalized_record = normalize_record(
            record=record,
            output_file_name=output_file_name,
            target_types=target_types,
            rare_regions_only=args.rare_regions_only,
        )
        place_image(source_path, output_image_path, args.copy_mode)
        draw_previews(
            source_path=source_path,
            destination_path=output_annotated_path,
            record=normalized_record,
            target_types=target_types,
            draw_all=args.draw_all,
        )
        output_records.append(normalized_record)

    write_jsonl(output_dir / args.metadata_name, output_records)
    summary = summarize(records, output_records, target_types)
    summary.update(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "metadata": args.metadata_name,
            "copy_mode": args.copy_mode,
            "rare_regions_only": args.rare_regions_only,
            "draw_all": args.draw_all,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_subset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
