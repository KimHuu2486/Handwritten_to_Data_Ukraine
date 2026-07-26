#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop all region bboxes from a metadata.jsonl file."
    )
    parser.add_argument(
        "--metadata",
        default="metadata.jsonl",
        help="Path to metadata JSONL file. Default: metadata.jsonl",
    )
    parser.add_argument(
        "--image-root",
        default=".",
        help="Directory used to resolve metadata file_name paths. Default: current directory",
    )
    parser.add_argument(
        "--output-dir",
        default="cropped_bboxes",
        help="Directory to write cropped bbox images and manifest.csv. Default: cropped_bboxes",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Optional padding in pixels added around each bbox before cropping.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Skip crops whose width or height is smaller than this value. Default: 1",
    )
    parser.add_argument(
        "--save-by-type",
        action="store_true",
        help="Save crops into subfolders named by region type, e.g. handwritten/ printed/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of metadata lines to process. 0 means no limit.",
    )
    return parser.parse_args()


def safe_name(value):
    value = str(value or "unknown")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "unknown"


def clamp_bbox(bbox, image_width, image_height, padding=0):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    args = parse_args()
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: Pillow. Install it with: python -m pip install Pillow"
        ) from exc

    metadata_path = Path(args.metadata)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "metadata_lines": 0,
        "regions_seen": 0,
        "crops_saved": 0,
        "missing_images": 0,
        "invalid_bboxes": 0,
        "small_crops": 0,
        "json_errors": 0,
    }

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "crop_path",
                "source_file_name",
                "metadata_line",
                "region_index",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "type",
                "language",
                "legibility",
                "text",
            ],
        )
        writer.writeheader()

        with metadata_path.open(encoding="utf-8") as metadata_file:
            for line_no, line in enumerate(metadata_file, 1):
                if args.limit and stats["metadata_lines"] >= args.limit:
                    break
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    continue

                stats["metadata_lines"] += 1
                source_file_name = item.get("file_name")
                image_path = image_root / source_file_name if source_file_name else None
                if not image_path or not image_path.is_file():
                    stats["missing_images"] += 1
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                except OSError:
                    stats["missing_images"] += 1
                    continue

                source_stem = safe_name(Path(source_file_name).stem)
                regions = item.get("regions") or []
                for region_index, region in enumerate(regions):
                    stats["regions_seen"] += 1
                    bbox = clamp_bbox(
                        region.get("bbox"),
                        image.width,
                        image.height,
                        padding=args.padding,
                    )
                    if bbox is None:
                        stats["invalid_bboxes"] += 1
                        continue

                    x1, y1, x2, y2 = bbox
                    if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                        stats["small_crops"] += 1
                        continue

                    region_type = safe_name(region.get("type"))
                    crop_dir = output_dir / region_type if args.save_by_type else output_dir
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{source_stem}_r{region_index:04d}_{x1}_{y1}_{x2}_{y2}.jpg"
                    crop_path = crop_dir / crop_name

                    image.crop((x1, y1, x2, y2)).save(crop_path, quality=95)
                    stats["crops_saved"] += 1

                    writer.writerow(
                        {
                            "crop_path": str(crop_path),
                            "source_file_name": source_file_name,
                            "metadata_line": line_no,
                            "region_index": region_index,
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                            "type": region.get("type", ""),
                            "language": region.get("language", ""),
                            "legibility": region.get("legibility", ""),
                            "text": region.get("text", ""),
                        }
                    )

    print(f"Metadata lines processed: {stats['metadata_lines']}")
    print(f"Regions seen: {stats['regions_seen']}")
    print(f"Crops saved: {stats['crops_saved']}")
    print(f"Missing/unreadable images: {stats['missing_images']}")
    print(f"Invalid bboxes: {stats['invalid_bboxes']}")
    print(f"Small crops skipped: {stats['small_crops']}")
    print(f"JSON errors: {stats['json_errors']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
