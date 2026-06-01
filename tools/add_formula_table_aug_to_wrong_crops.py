from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageEnhance, ImageFilter


Image.MAX_IMAGE_PIXELS = None

TARGET_TYPES = {"formula", "table"}
CROP_PAD_RATIO = 0.0
BASE_JPEG_QUALITY = 95
WRONG_AUG_COPIES_BY_TYPE = {
    "formula": 2,
    "table": 4,
}

AUG_RANGES = {
    "formula": {
        "rotate_degrees": 1.0,
        "rotate_prob": 0.85,
        "brightness": (0.92, 1.08),
        "contrast": (0.90, 1.15),
        "sharpness": (0.90, 1.20),
        "sharpness_prob": 0.60,
        "blur_radius": (0.10, 0.35),
        "blur_prob": 0.18,
        "jpeg_quality": (85, 98),
    },
    "table": {
        "rotate_degrees": 0.4,
        "rotate_prob": 0.55,
        "brightness": (0.95, 1.05),
        "contrast": (0.92, 1.10),
        "sharpness": (0.95, 1.15),
        "sharpness_prob": 0.45,
        "blur_radius": (0.08, 0.20),
        "blur_prob": 0.05,
        "jpeg_quality": (88, 98),
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def path_variants(file_name: str) -> list[str]:
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


def default_image_roots(metadata_like_path: Path) -> list[Path]:
    roots = [
        metadata_like_path.parent,
        metadata_like_path.parent / "train",
        metadata_like_path.parent / "silver",
        metadata_like_path.parent / "sliver",
        Path(r"C:\Users\HP\source\rukopys_data"),
    ]
    return [path for path in roots if path.exists()]


def candidate_image_paths(file_name: str, image_roots: Iterable[Path]) -> list[Path]:
    normalized = file_name.replace("\\", "/")
    basename = Path(normalized).name
    candidates = []
    for root in image_roots:
        for variant in path_variants(normalized):
            candidates.append(root / variant)
        candidates.append(root / basename)
        candidates.append(root / "images" / basename)

    deduped = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def resolve_image_path(file_name: str, image_roots: Iterable[Path]) -> Path | None:
    for path in candidate_image_paths(file_name, image_roots):
        if path.exists():
            return path
    return None


def clamp_bbox(value: Any, image_width: int, image_height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(image_width, round(x1)))
    x2 = max(0, min(image_width, round(x2)))
    y1 = max(0, min(image_height, round(y1)))
    y2 = max(0, min(image_height, round(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def stable_rng(*parts: Any) -> random.Random:
    return random.Random("|".join(str(part) for part in parts))


def apply_formula_table_augment(image: Image.Image, region_type: str, rng: random.Random) -> tuple[Image.Image, int]:
    cfg = AUG_RANGES[region_type]
    if rng.random() < cfg["rotate_prob"]:
        max_degrees = float(cfg["rotate_degrees"])
        angle = rng.uniform(-max_degrees, max_degrees)
        if abs(angle) > 0.03:
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))

    low, high = cfg["brightness"]
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(low, high))
    low, high = cfg["contrast"]
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(low, high))

    if rng.random() < cfg["sharpness_prob"]:
        low, high = cfg["sharpness"]
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(low, high))

    if rng.random() < cfg["blur_prob"]:
        low, high = cfg["blur_radius"]
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(low, high)))

    q_low, q_high = cfg["jpeg_quality"]
    jpeg_quality = int(round(rng.uniform(q_low, q_high)))
    return image, jpeg_quality


def safe_reset_type_dirs(output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for region_type in TARGET_TYPES:
        target = (output_dir / region_type).resolve()
        if target.exists():
            if target.parent != output_root:
                raise RuntimeError(f"Refusing to delete outside output_dir: {target}")
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def existing_non_target_labels(output_dir: Path) -> list[dict[str, Any]]:
    labels_path = output_dir / "labels.jsonl"
    labels = []
    for row in read_jsonl(labels_path):
        if str(row.get("type") or "").strip().lower() not in TARGET_TYPES:
            labels.append(row)
    return labels


def crop_from_source(source_image_path: Path, bbox: list[int]) -> tuple[Image.Image | None, list[int] | None]:
    with Image.open(source_image_path) as image:
        image = image.convert("RGB")
        crop_box = clamp_bbox(bbox, image.width, image.height)
        if crop_box is None:
            return None, None
        return image.crop(tuple(crop_box)), crop_box


def output_name(prefix: str, index: int, row: dict[str, Any], aug_id: int) -> str:
    region_type = str(row.get("type") or row.get("region_type") or "crop").strip().lower()
    source_stem = Path(str(row.get("crop_path") or row.get("source_file_name") or "sample")).stem
    return f"{prefix}_{index:06d}_{safe_name(source_stem)}_aug{aug_id}.jpg"


def make_label(
    output_dir: Path,
    output_path: Path,
    row: dict[str, Any],
    text: str,
    crop_box: list[int],
    image_size: tuple[int, int],
    augment: bool,
    aug_id: int,
    source_kind: str,
) -> dict[str, Any]:
    region_type = str(row.get("type") or row.get("region_type") or "unknown").strip().lower()
    label = {
        "crop_path": output_path.relative_to(output_dir.parent).as_posix(),
        "type": region_type,
        "text": text,
        "source_file_name": row.get("source_file_name", ""),
        "source": row.get("source"),
        "row_index": row.get("row_index"),
        "region_index": row.get("region_index"),
        "bbox": crop_box,
        "crop_bbox": crop_box,
        "crop_width": image_size[0],
        "crop_height": image_size[1],
        "augment": augment,
        "aug_id": aug_id,
        "crop_pad_ratio": CROP_PAD_RATIO,
        "source_kind": source_kind,
    }
    if "cer" in row:
        label["error_cer"] = row.get("cer")
    elif "error_cer" in row:
        label["error_cer"] = row.get("error_cer")
    if "wer" in row:
        label["error_wer"] = row.get("wer")
    elif "error_wer" in row:
        label["error_wer"] = row.get("error_wer")
    if "prediction" in row:
        label["error_prediction"] = row.get("prediction", "")
    if "crop_path" in row:
        label["error_crop_path"] = row.get("crop_path", "")
    return label


def materialize_base_formula_table(
    hybrid_labels_path: Path,
    output_dir: Path,
    image_roots: list[Path],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    labels = []
    counts = Counter()
    skipped = Counter()
    source_cache: dict[str, Path | None] = {}

    for index, row in enumerate(read_jsonl(hybrid_labels_path)):
        region_type = str(row.get("type") or "").strip().lower()
        if region_type not in TARGET_TYPES:
            continue
        text = "" if row.get("text") is None else str(row.get("text"))
        if not text.strip():
            skipped[f"{region_type}_empty_text"] += 1
            continue
        source_file_name = str(row.get("source_file_name") or "")
        if source_file_name not in source_cache:
            source_cache[source_file_name] = resolve_image_path(source_file_name, image_roots)
        source_image_path = source_cache[source_file_name]
        if source_image_path is None:
            skipped[f"{region_type}_missing_image"] += 1
            continue
        bbox = row.get("bbox")
        if dry_run:
            crop_box = bbox if isinstance(bbox, list) and len(bbox) == 4 else None
            if crop_box is None:
                skipped[f"{region_type}_invalid_bbox"] += 1
                continue
            image_size = (max(0, int(crop_box[2]) - int(crop_box[0])), max(0, int(crop_box[3]) - int(crop_box[1])))
        else:
            crop, crop_box = crop_from_source(source_image_path, bbox)
            if crop is None or crop_box is None:
                skipped[f"{region_type}_invalid_bbox"] += 1
                continue
            image_size = crop.size
            output_path = output_dir / region_type / output_name("base", counts[region_type], row, 0)
            crop.save(output_path, quality=BASE_JPEG_QUALITY, optimize=True)

        if dry_run:
            output_path = output_dir / region_type / output_name("base", counts[region_type], row, 0)
        labels.append(make_label(output_dir, output_path, row, text, crop_box, image_size, False, 0, "hybrid_base"))
        counts[region_type] += 1
    return labels, counts, skipped


def load_wrong_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("wrong_samples") or []
    return samples if isinstance(samples, list) else []


def materialize_wrong_augments(
    wrong_samples_path: Path,
    output_dir: Path,
    image_roots: list[Path],
    seed: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    labels = []
    counts = Counter()
    skipped = Counter()
    source_cache: dict[str, Path | None] = {}

    for sample_index, row in enumerate(load_wrong_samples(wrong_samples_path)):
        region_type = str(row.get("type") or "").strip().lower()
        if region_type not in TARGET_TYPES:
            continue
        text = "" if row.get("label_text") is None else str(row.get("label_text"))
        if not text.strip():
            skipped[f"{region_type}_empty_label_text"] += 1
            continue
        source_file_name = str(row.get("source_file_name") or "")
        if source_file_name not in source_cache:
            source_cache[source_file_name] = resolve_image_path(source_file_name, image_roots)
        source_image_path = source_cache[source_file_name]
        if source_image_path is None:
            skipped[f"{region_type}_missing_image"] += 1
            continue

        bbox = row.get("bbox")
        if dry_run:
            crop_box = bbox if isinstance(bbox, list) and len(bbox) == 4 else None
            if crop_box is None:
                skipped[f"{region_type}_invalid_bbox"] += 1
                continue
            base_size = (max(0, int(crop_box[2]) - int(crop_box[0])), max(0, int(crop_box[3]) - int(crop_box[1])))
        else:
            base_crop, crop_box = crop_from_source(source_image_path, bbox)
            if base_crop is None or crop_box is None:
                skipped[f"{region_type}_invalid_bbox"] += 1
                continue
            base_size = base_crop.size

        for aug_id in range(1, WRONG_AUG_COPIES_BY_TYPE[region_type] + 1):
            output_path = output_dir / region_type / output_name("wrong", counts[region_type], row, aug_id)
            if dry_run:
                image_size = base_size
            else:
                rng = stable_rng(seed, row.get("source_file_name"), row.get("bbox"), region_type, sample_index, aug_id)
                aug_crop, jpeg_quality = apply_formula_table_augment(base_crop.copy(), region_type, rng)
                image_size = aug_crop.size
                aug_crop.save(output_path, quality=jpeg_quality, optimize=True)

            labels.append(make_label(output_dir, output_path, row, text, crop_box, image_size, True, aug_id, "formula_table_wrong_aug"))
            counts[region_type] += 1
    return labels, counts, skipped


def build_stats(
    output_dir: Path,
    existing_labels: list[dict[str, Any]],
    base_labels: list[dict[str, Any]],
    aug_labels: list[dict[str, Any]],
    base_counts: Counter,
    aug_counts: Counter,
    skipped: Counter,
    dry_run: bool,
) -> dict[str, Any]:
    all_labels = existing_labels + base_labels + aug_labels
    cropped_by_type = Counter(str(row.get("type") or "").strip().lower() for row in all_labels)
    augmented_by_type = Counter(str(row.get("type") or "").strip().lower() for row in all_labels if row.get("augment"))
    return {
        "output_dir": str(output_dir),
        "format": "ocr_region_crops_hybrid",
        "stage": "wrong_aug_plus_formula_table",
        "dry_run": dry_run,
        "ocr_types": sorted(cropped_by_type),
        "regions_seen": len(all_labels),
        "regions_cropped": len(all_labels),
        "missing_images": sum(value for key, value in skipped.items() if key.endswith("_missing_image")),
        "empty_text_skipped": sum(value for key, value in skipped.items() if "empty" in key),
        "crop_pad_ratio": CROP_PAD_RATIO,
        "cropped_by_type": dict(sorted(cropped_by_type.items())),
        "augmented_by_type": dict(sorted(augmented_by_type.items())),
        "formula_table_base_by_type": dict(sorted(base_counts.items())),
        "formula_table_wrong_aug_by_type": dict(sorted(aug_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "labels_path": str(output_dir / "labels.jsonl"),
    }


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = Path(args.output_dir)
    hybrid_labels_path = Path(args.hybrid_labels)
    wrong_samples_path = Path(args.wrong_samples)
    image_roots = [Path(path) for path in args.image_root] if args.image_root else default_image_roots(Path("dataset/metadata_hybrid.jsonl"))

    existing_labels = existing_non_target_labels(output_dir)
    base_labels, base_counts, base_skipped = materialize_base_formula_table(
        hybrid_labels_path=hybrid_labels_path,
        output_dir=output_dir,
        image_roots=image_roots,
        dry_run=args.dry_run,
    )
    aug_labels, aug_counts, aug_skipped = materialize_wrong_augments(
        wrong_samples_path=wrong_samples_path,
        output_dir=output_dir,
        image_roots=image_roots,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    skipped = Counter()
    skipped.update(base_skipped)
    skipped.update(aug_skipped)
    labels = existing_labels + base_labels + aug_labels
    stats = build_stats(output_dir, existing_labels, base_labels, aug_labels, base_counts, aug_counts, skipped, args.dry_run)
    stats["image_roots"] = [str(path) for path in image_roots]
    stats["hybrid_labels"] = str(hybrid_labels_path)
    stats["wrong_samples"] = str(wrong_samples_path)
    stats["seed"] = args.seed
    return labels, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add no-padding formula/table base crops and wrong-sample augments.")
    parser.add_argument("--output-dir", default="artifacts/ocr_region_crops_wrong_aug")
    parser.add_argument("--hybrid-labels", default="artifacts/ocr_region_crops_hybrid/labels.jsonl")
    parser.add_argument("--wrong-samples", default="dataset/formula_table_wrong_samples.json")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        labels, stats = run(args)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_reset_type_dirs(output_dir)
    labels, stats = run(args)
    write_jsonl(output_dir / "labels.jsonl", labels)
    write_json(output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
