from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MainPipeline.src.common.bbox import as_float_bbox, clamp_bbox, is_valid_bbox, padded_bbox
from MainPipeline.src.common.io import iter_jsonl, resolve_path, write_json


OCR_TYPES = ("handwritten", "printed", "formula", "table", "annotation")
SKIPPED_TYPES = ("image", "graph")


def safe_name(value: str) -> str:
    """Make a stable filename segment from metadata values."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def path_variants(file_name: str) -> list[str]:
    """Return common local variants for competition/HF image paths."""
    normalized = file_name.replace("\\", "/").lstrip("/")
    variants = {
        normalized,
        normalized.replace("/images/", "/"),
        re.sub(r"^images/", "", normalized),
    }
    if normalized.startswith("silver/"):
        variants.add("sliver/" + normalized[len("silver/") :])
    if normalized.startswith("sliver/"):
        variants.add("silver/" + normalized[len("sliver/") :])
    return [value for value in variants if value]


def default_image_roots(metadata_path: Path) -> list[Path]:
    dataset_root = metadata_path.parent
    return [
        dataset_root,
        dataset_root / "train",
        dataset_root / "silver",
        dataset_root / "sliver",
        dataset_root / "test",
    ]


def candidate_image_paths(record: dict[str, Any], metadata_path: Path, image_roots: Iterable[Path]) -> list[Path]:
    file_name = str(record.get("file_name", "")).replace("\\", "/")
    basename = Path(file_name).name
    candidates: list[Path] = []
    for root in image_roots:
        for variant in path_variants(file_name):
            candidates.append(root / variant)
        candidates.append(root / basename)
        candidates.append(root / "images" / basename)

    dataset_root = metadata_path.parent
    for variant in path_variants(file_name):
        candidates.append(dataset_root / variant)
    candidates.append(dataset_root / basename)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def resolve_image_path(record: dict[str, Any], metadata_path: Path, image_roots: Iterable[Path]) -> Path | None:
    for path in candidate_image_paths(record, metadata_path, image_roots):
        if path.exists():
            return path
    return None


def ensure_type_dirs(output_dir: Path) -> None:
    for region_type in OCR_TYPES:
        (output_dir / region_type).mkdir(parents=True, exist_ok=True)


def crop_name(file_name: str, region_index: int, region_type: str) -> str:
    image_stem = safe_name(Path(file_name.replace("\\", "/")).stem)
    return f"{image_stem}__r{region_index:04d}__{region_type}.jpg"


def crop_regions(
    metadata_path: Path,
    output_dir: Path,
    image_roots: list[Path],
    max_records: int | None,
    max_regions: int | None,
    pad_ratio: float,
    skip_missing_images: bool,
    progress_every: int,
) -> dict[str, Any]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    ensure_type_dirs(output_dir)
    labels_path = output_dir / "labels.jsonl"

    stats: dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "image_roots": [str(path) for path in image_roots],
        "ocr_types": list(OCR_TYPES),
        "skipped_types": list(SKIPPED_TYPES),
        "rows_seen": 0,
        "rows_with_crops": 0,
        "regions_seen": 0,
        "regions_cropped": 0,
        "missing_images": 0,
        "invalid_bboxes": 0,
        "cropped_by_type": {region_type: 0 for region_type in OCR_TYPES},
        "skipped_by_type": {region_type: 0 for region_type in SKIPPED_TYPES},
        "unknown_types": {},
    }

    with labels_path.open("w", encoding="utf-8", newline="\n") as label_handle:
        for row_index, record in enumerate(iter_jsonl(metadata_path), start=1):
            if max_records is not None and stats["rows_seen"] >= max_records:
                break
            if max_regions is not None and stats["regions_cropped"] >= max_regions:
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
            file_name = str(record.get("file_name", "image"))
            wrote_crop_for_row = False

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                for region_index, raw_region in enumerate(record.get("regions", []) or []):
                    if max_regions is not None and stats["regions_cropped"] >= max_regions:
                        break

                    stats["regions_seen"] += 1
                    region_type = str(raw_region.get("type", "")).strip().lower()
                    if region_type in SKIPPED_TYPES:
                        stats["skipped_by_type"][region_type] += 1
                        continue
                    if region_type not in OCR_TYPES:
                        unknown_types = stats["unknown_types"]
                        unknown_types[region_type or "<missing>"] = unknown_types.get(region_type or "<missing>", 0) + 1
                        continue

                    bbox = as_float_bbox(raw_region.get("bbox"))
                    if bbox is None:
                        stats["invalid_bboxes"] += 1
                        continue
                    pixel_bbox = clamp_bbox(bbox, image_width, image_height)
                    if not is_valid_bbox(pixel_bbox):
                        stats["invalid_bboxes"] += 1
                        continue

                    crop_box = padded_bbox(pixel_bbox, image.width, image.height, pad_ratio)
                    crop = image.crop(tuple(crop_box))
                    crop_path = output_dir / region_type / crop_name(file_name, region_index, region_type)
                    crop.save(crop_path, quality=95)
                    label_crop_path = crop_path.relative_to(output_dir.parent).as_posix()

                    label = {
                        "crop_path": str(label_crop_path),
                        "type": region_type,
                        "text": "" if raw_region.get("text") is None else str(raw_region.get("text", "")),
                        "source_file_name": file_name,
                        "source": record.get("source"),
                        "row_index": row_index,
                        "region_index": region_index,
                        "bbox": pixel_bbox,
                        "crop_bbox": crop_box,
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                    }
                    label_handle.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")

                    stats["regions_cropped"] += 1
                    stats["cropped_by_type"][region_type] += 1
                    wrote_crop_for_row = True
                    if progress_every > 0 and stats["regions_cropped"] % progress_every == 0:
                        print(
                            "cropped "
                            f"{stats['regions_cropped']} regions "
                            f"(rows_seen={stats['rows_seen']}, "
                            f"missing_images={stats['missing_images']}, "
                            f"by_type={stats['cropped_by_type']})",
                            flush=True,
                        )

            if wrote_crop_for_row:
                stats["rows_with_crops"] += 1

    write_json(output_dir / "stats.json", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop OCR regions into one folder per type.")
    parser.add_argument("--metadata", default="dataset/metadata_hybrid.jsonl", help="Input JSONL metadata path.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/ocr_region_crops_hybrid",
        help="Directory that will contain the 5 type folders plus labels.jsonl/stats.json.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=None,
        help="Image root to search. May be passed multiple times. Defaults to dataset roots near metadata.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--max-regions", type=int, default=None, help="Optional crop limit for smoke tests.")
    parser.add_argument("--crop-pad-ratio", type=float, default=0.0, help="Padding ratio around each bbox.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress after this many cropped regions.")
    parser.add_argument("--fail-on-missing-image", action="store_true", help="Raise instead of skipping missing images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    output_dir = resolve_path(args.output_dir)
    image_roots = (
        [resolve_path(path) for path in args.image_root]
        if args.image_root
        else default_image_roots(metadata_path)
    )
    stats = crop_regions(
        metadata_path=metadata_path,
        output_dir=output_dir,
        image_roots=image_roots,
        max_records=args.max_records,
        max_regions=args.max_regions,
        pad_ratio=args.crop_pad_ratio,
        skip_missing_images=not args.fail_on_missing_image,
        progress_every=max(int(args.progress_every), 0),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
