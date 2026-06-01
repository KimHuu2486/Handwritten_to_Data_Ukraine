from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_crop_path(input_root: Path, row: dict[str, Any]) -> Path:
    raw = Path(str(row.get("image_path") or row.get("relative_image_path") or ""))
    if raw.is_absolute():
        return raw
    return input_root / raw


def output_crop_name(row: dict[str, Any], source_path: Path, index: int) -> str:
    suffix = source_path.suffix.lower() or ".jpg"
    return f"{index:06d}_{row.get('region_type', 'crop')}_aug{row.get('aug_id', 0)}{suffix}"


def convert_manifest(input_root: Path, output_dir: Path, manifest_name: str) -> dict[str, Any]:
    manifest_path = input_root / manifest_name
    rows = read_jsonl(manifest_path)
    labels_path = output_dir / "labels.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    copied_by_type = Counter()
    augment_by_type = Counter()
    missing_images = 0
    empty_text = 0

    with labels_path.open("w", encoding="utf-8", newline="\n") as label_handle:
        for index, row in enumerate(rows):
            region_type = str(row.get("region_type") or row.get("type") or "unknown").strip().lower()
            text = "" if row.get("answer") is None else str(row.get("answer"))
            if not text.strip():
                empty_text += 1
                continue

            src = source_crop_path(input_root, row)
            if not src.exists():
                missing_images += 1
                continue

            type_dir = output_dir / region_type
            type_dir.mkdir(parents=True, exist_ok=True)
            dst = type_dir / output_crop_name(row, src, index)
            if not dst.exists():
                shutil.copy2(src, dst)

            with Image.open(dst) as image:
                crop_width, crop_height = image.size

            rel_crop_path = dst.relative_to(output_dir.parent).as_posix()
            bbox = row.get("bbox") or []
            label = {
                "crop_path": rel_crop_path,
                "type": region_type,
                "text": text,
                "source_file_name": row.get("source_file_name", ""),
                "source": row.get("source"),
                "row_index": row.get("row_index"),
                "region_index": row.get("region_index"),
                "bbox": bbox,
                "crop_bbox": bbox,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "augment": bool(row.get("augment")),
                "aug_id": int(row.get("aug_id") or 0),
                "crop_pad_ratio": float(row.get("crop_pad_ratio") or 0.0),
                "error_cer": row.get("error_cer"),
                "error_wer": row.get("error_wer"),
                "error_prediction": row.get("error_prediction", ""),
                "error_crop_path": row.get("error_crop_path", ""),
            }
            label_handle.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")

            copied_by_type[region_type] += 1
            if row.get("augment"):
                augment_by_type[region_type] += 1

    stats = {
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "format": "ocr_region_crops_hybrid",
        "ocr_types": sorted(copied_by_type),
        "regions_seen": len(rows),
        "regions_cropped": sum(copied_by_type.values()),
        "missing_images": missing_images,
        "empty_text_skipped": empty_text,
        "cropped_by_type": dict(sorted(copied_by_type.items())),
        "augmented_by_type": dict(sorted(augment_by_type.items())),
        "labels_path": str(labels_path),
    }
    write_json(output_dir / "stats.json", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Stage2B augment manifest into ocr_region_crops_hybrid-style folders.")
    parser.add_argument("--input-root", default="artifacts/stage2b_wrong_sample_aug_data")
    parser.add_argument("--output-dir", default="artifacts/ocr_region_crops_wrong_aug")
    parser.add_argument("--manifest-name", default="stage2b_hardtype_aug_samples.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = convert_manifest(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        manifest_name=args.manifest_name,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
