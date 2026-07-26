#!/usr/bin/env python3
"""Export capped silver crops for rare layout classes with text previews."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None

DEFAULT_TARGET_TYPES = ("annotation", "table", "formula")
STANDARD_REGION_KEYS = ("bbox", "type", "language", "legibility", "text")
TYPE_COLORS = {
    "annotation": (8, 145, 178),
    "table": (225, 29, 72),
    "formula": (147, 51, 234),
}
FALLBACK_COLOR = (15, 23, 42)


@dataclass(frozen=True)
class CropSpec:
    record_index: int
    region_index: int
    source_file_name: str
    source_path: Path
    source_width: int
    source_height: int
    source: str
    annotation_source: str
    region: dict[str, Any]
    source_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    region_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop annotation/table/formula regions from dataset/silver, cap each "
            "class, normalize metadata, and write preview images with detected text."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/silver"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/silver_rare_crops"))
    parser.add_argument("--metadata-name", default="metadata.jsonl")
    parser.add_argument("--target-types", nargs="+", default=list(DEFAULT_TARGET_TYPES))
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.0,
        help="Extra padding around each source bbox, as a fraction of bbox width/height.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle candidates before applying --max-per-class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview-max-side",
        type=int,
        default=1600,
        help="Resize previews only so the longest crop side is at most this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_type(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def resolve_image_path(input_dir: Path, file_name: str) -> Path:
    normalized = file_name.replace("\\", "/")
    basename = Path(normalized).name
    candidates = (
        input_dir / normalized,
        input_dir / basename,
        input_dir / "images" / basename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image {file_name!r} under {input_dir}")


def clamp_bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None

    left = max(0, min(width, math.floor(min(x1, x2))))
    top = max(0, min(height, math.floor(min(y1, y2))))
    right = max(0, min(width, math.ceil(max(x1, x2))))
    bottom = max(0, min(height, math.ceil(max(y1, y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def padded_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    if pad_ratio <= 0:
        return bbox
    pad_x = int(round((right - left) * pad_ratio))
    pad_y = int(round((bottom - top) * pad_ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def normalize_region(
    region: dict[str, Any],
    region_type: str,
    bbox_in_crop: tuple[int, int, int, int],
) -> dict[str, Any]:
    normalized = {key: region.get(key, "") for key in STANDARD_REGION_KEYS}
    normalized["bbox"] = [int(value) for value in bbox_in_crop]
    normalized["type"] = region_type
    normalized["language"] = str(normalized.get("language") or "")
    normalized["legibility"] = str(normalized.get("legibility") or "")
    normalized["text"] = normalize_text(normalized.get("text"))
    return normalized


def collect_crop_specs(
    records: list[dict[str, Any]],
    input_dir: Path,
    target_types: set[str],
    pad_ratio: float,
) -> tuple[dict[str, list[CropSpec]], Counter[str], Counter[str]]:
    specs_by_type: dict[str, list[CropSpec]] = defaultdict(list)
    total_region_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for record_index, record in enumerate(records):
        source_file_name = str(record.get("file_name") or "")
        regions = record.get("regions", [])
        if not source_file_name or not isinstance(regions, list):
            continue

        try:
            source_path = resolve_image_path(input_dir, source_file_name)
        except FileNotFoundError:
            skipped_counts["missing_image"] += 1
            continue

        source_width = int(record.get("image_width") or 0)
        source_height = int(record.get("image_height") or 0)
        if source_width <= 0 or source_height <= 0:
            with Image.open(source_path) as image:
                source_width, source_height = image.size

        for region_index, region in enumerate(regions):
            if not isinstance(region, dict):
                skipped_counts["non_object_region"] += 1
                continue
            region_type = normalize_type(region.get("type"))
            if not region_type:
                skipped_counts["missing_type"] += 1
                continue
            total_region_counts[region_type] += 1
            if region_type not in target_types:
                continue

            source_bbox = clamp_bbox(region.get("bbox"), source_width, source_height)
            if source_bbox is None:
                skipped_counts[f"invalid_bbox:{region_type}"] += 1
                continue
            crop_bbox = padded_bbox(source_bbox, source_width, source_height, pad_ratio)
            specs_by_type[region_type].append(
                CropSpec(
                    record_index=record_index,
                    region_index=region_index,
                    source_file_name=source_file_name,
                    source_path=source_path,
                    source_width=source_width,
                    source_height=source_height,
                    source=str(record.get("source") or ""),
                    annotation_source=str(record.get("annotation_source") or ""),
                    region=region,
                    source_bbox=source_bbox,
                    crop_bbox=crop_bbox,
                    region_type=region_type,
                )
            )

    return specs_by_type, total_region_counts, skipped_counts


def select_specs(
    specs_by_type: dict[str, list[CropSpec]],
    target_types: list[str],
    max_per_class: int,
    shuffle: bool,
    seed: int,
) -> list[CropSpec]:
    rng = random.Random(seed)
    selected: list[CropSpec] = []
    for target_type in target_types:
        candidates = list(specs_by_type.get(target_type, []))
        if shuffle:
            rng.shuffle(candidates)
        selected.extend(candidates[:max_per_class])
    return selected


def load_unicode_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for font_path in candidates:
        if font_path and Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_one_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if text_width(draw, text, font) <= max_width:
        return [text]

    words = text.split(" ")
    if len(words) > 1:
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    for char in text:
        candidate = current + char
        if not current or text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        lines.extend(wrap_one_line(draw, raw_line, font, max_width))

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip(". ") + "..."
    return lines


def resize_for_preview(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image.copy()
    longest = max(image.size)
    if longest <= max_side:
        return image.copy()
    scale = max_side / longest
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize((width, height), resampling)


def draw_preview(
    crop: Image.Image,
    output_path: Path,
    spec: CropSpec,
    preview_max_side: int,
) -> None:
    preview_crop = resize_for_preview(crop.convert("RGB"), preview_max_side)
    color = TYPE_COLORS.get(spec.region_type, FALLBACK_COLOR)

    base_width = max(360, preview_crop.width)
    font_size = max(16, min(34, base_width // 34))
    title_font = load_unicode_font(font_size + 2, bold=True)
    body_font = load_unicode_font(font_size)

    probe = Image.new("RGB", (base_width, 64), "white")
    probe_draw = ImageDraw.Draw(probe)
    padding = max(12, font_size)
    max_text_width = base_width - padding * 2
    text = normalize_text(spec.region.get("text"))
    title = f"{spec.region_type} | r{spec.region_index:04d}"
    source_lines = wrap_text(
        probe_draw,
        f"source: {Path(spec.source_file_name).name}",
        body_font,
        max_text_width,
        2,
    )
    bbox_lines = wrap_text(probe_draw, f"bbox: {list(spec.source_bbox)}", body_font, max_text_width, 2)
    text_lines = wrap_text(probe_draw, f"text: {text}" if text else "text:", body_font, max_text_width, 8)
    lines = [title, *source_lines, *bbox_lines, *text_lines]

    line_gap = max(4, font_size // 4)
    line_heights = []
    for index, line in enumerate(lines):
        font = title_font if index == 0 else body_font
        bbox = probe_draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    panel_height = padding * 2 + sum(line_heights) + max(0, len(lines) - 1) * line_gap
    canvas = Image.new("RGB", (base_width, preview_crop.height + panel_height), (255, 255, 255))
    canvas.paste(preview_crop, ((base_width - preview_crop.width) // 2, 0))

    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(max(preview_crop.size) / 260))
    draw.rectangle((0, 0, base_width - 1, preview_crop.height - 1), outline=color, width=line_width)
    draw.rectangle((0, preview_crop.height, base_width, preview_crop.height + panel_height), fill=(248, 250, 252))
    draw.rectangle((0, preview_crop.height, max(8, font_size // 2), preview_crop.height + panel_height), fill=color)

    y = preview_crop.height + padding
    for index, line in enumerate(lines):
        font = title_font if index == 0 else body_font
        fill = color if index == 0 else (15, 23, 42)
        draw.text((padding, y), line, fill=fill, font=font)
        y += line_heights[index] + line_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def make_output_name(spec: CropSpec, used_names: set[str]) -> str:
    stem = Path(spec.source_file_name).stem
    suffix = Path(spec.source_file_name).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    base = f"{stem}__r{spec.region_index:04d}__{spec.region_type}{suffix}"
    output_name = base
    duplicate_index = 1
    while output_name in used_names:
        output_name = f"{stem}__r{spec.region_index:04d}__{spec.region_type}__{duplicate_index}{suffix}"
        duplicate_index += 1
    used_names.add(output_name)
    return output_name


def build_crop_record(
    spec: CropSpec,
    output_file_name: str,
    preview_file_name: str,
    crop_size: tuple[int, int],
) -> dict[str, Any]:
    crop_left, crop_top, _, _ = spec.crop_bbox
    src_left, src_top, src_right, src_bottom = spec.source_bbox
    bbox_in_crop = (
        src_left - crop_left,
        src_top - crop_top,
        src_right - crop_left,
        src_bottom - crop_top,
    )
    normalized_region = normalize_region(spec.region, spec.region_type, bbox_in_crop)
    crop_width, crop_height = crop_size
    text = normalized_region["text"]

    return {
        "file_name": output_file_name,
        "preview_file_name": preview_file_name,
        "image_width": int(crop_width),
        "image_height": int(crop_height),
        "source_split": "silver",
        "source": spec.source,
        "annotation_source": spec.annotation_source,
        "type": spec.region_type,
        "label": spec.region_type,
        "language": normalized_region["language"],
        "legibility": normalized_region["legibility"],
        "text": text,
        "regions": [normalized_region],
        "source_file_name": spec.source_file_name,
        "source_image_width": int(spec.source_width),
        "source_image_height": int(spec.source_height),
        "source_region_index": int(spec.region_index),
        "source_bbox": [int(value) for value in spec.source_bbox],
        "crop_bbox": [int(value) for value in spec.crop_bbox],
        "bbox_in_crop": [int(value) for value in bbox_in_crop],
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)


def export_crops(args: argparse.Namespace) -> dict[str, Any]:
    target_types = [normalize_type(item) for item in args.target_types]
    target_types = [item for item in target_types if item]
    if not target_types:
        raise ValueError("No valid --target-types were provided.")
    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive.")

    input_dir = args.input_dir
    metadata_path = input_dir / args.metadata_name
    records = load_jsonl(metadata_path)
    specs_by_type, total_region_counts, skipped_counts = collect_crop_specs(
        records=records,
        input_dir=input_dir,
        target_types=set(target_types),
        pad_ratio=float(args.pad_ratio),
    )
    selected_specs = select_specs(
        specs_by_type=specs_by_type,
        target_types=target_types,
        max_per_class=int(args.max_per_class),
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
    )

    prepare_output_dir(args.output_dir, args.overwrite)

    output_records: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    used_names: set[str] = set()

    for spec in selected_specs:
        output_name = make_output_name(spec, used_names)
        rel_image = f"images/{spec.region_type}/{output_name}"
        rel_preview = f"previews/{spec.region_type}/{output_name}"
        output_image_path = args.output_dir / rel_image
        output_preview_path = args.output_dir / rel_preview

        with Image.open(spec.source_path) as source_image:
            crop = source_image.convert("RGB").crop(spec.crop_bbox)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_image_path, quality=95)
        draw_preview(crop, output_preview_path, spec, int(args.preview_max_side))

        output_records.append(
            build_crop_record(
                spec=spec,
                output_file_name=rel_image,
                preview_file_name=rel_preview,
                crop_size=crop.size,
            )
        )
        selected_counts[spec.region_type] += 1

    write_jsonl(args.output_dir / args.metadata_name, output_records)

    available_counts = {target_type: len(specs_by_type.get(target_type, [])) for target_type in target_types}
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(args.output_dir),
        "metadata": args.metadata_name,
        "target_types": target_types,
        "max_per_class": int(args.max_per_class),
        "pad_ratio": float(args.pad_ratio),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "total_images": len(records),
        "total_region_counts": dict(total_region_counts.most_common()),
        "available_target_counts": available_counts,
        "selected_counts": {target_type: selected_counts.get(target_type, 0) for target_type in target_types},
        "selected_total": len(output_records),
        "skipped_counts": dict(skipped_counts.most_common()),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = export_crops(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
