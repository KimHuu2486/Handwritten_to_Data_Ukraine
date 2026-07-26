from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
TARGET_TYPES = ("annotation", "printed")


@dataclass(frozen=True)
class CropInfo:
    crop_id: str
    source_file_name: str
    source_image_path: Path
    source_width: int
    source_height: int
    source_region_index: int
    source_bbox: tuple[int, int, int, int]
    background_rgb: tuple[int, int, int]
    region_type: str
    region: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_image_path(record: dict[str, Any], input_dir: Path) -> Path:
    file_name = str(record.get("file_name", "")).replace("\\", "/")
    basename = Path(file_name).name
    candidates = [
        input_dir / file_name,
        input_dir / basename,
        input_dir / "images" / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image for {file_name!r} under {input_dir}")


def normalized_bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width, math.floor(min(x1, x2))))
    top = max(0, min(height, math.floor(min(y1, y2))))
    right = max(0, min(width, math.ceil(max(x1, x2))))
    bottom = max(0, min(height, math.ceil(max(y1, y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def clipped_rgb(values: np.ndarray) -> tuple[int, int, int]:
    clipped = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def estimate_background_rgb(image: Image.Image, valid_mask: np.ndarray | None = None) -> tuple[int, int, int]:
    """Estimate paper/background color from crop borders."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    border = max(2, min(16, width // 8, height // 8))

    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    if valid_mask is not None:
        edge_mask &= valid_mask

    edge_pixels = rgb[edge_mask]
    if edge_pixels.size == 0:
        edge_pixels = rgb.reshape(-1, 3)

    luma = edge_pixels @ np.array([0.299, 0.587, 0.114])
    cutoff = np.percentile(luma, 45)
    paper_pixels = edge_pixels[luma >= cutoff]
    if len(paper_pixels) < 16:
        paper_pixels = edge_pixels
    return clipped_rgb(np.median(paper_pixels, axis=0))


def foreground_mask_array(
    image: Image.Image,
    background_rgb: tuple[int, int, int],
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    background = np.array(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    diff = rgb - background
    color_distance = np.sqrt(np.sum(diff * diff, axis=2))

    luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    background_luma = background_rgb[0] * 0.299 + background_rgb[1] * 0.587 + background_rgb[2] * 0.114
    mask = (color_distance >= threshold) | (np.abs(luma - background_luma) >= threshold * 0.72)
    if valid_mask is not None:
        mask &= valid_mask

    coverage = float(mask.mean()) if mask.size else 0.0
    if coverage < 0.0005 or coverage > 0.85:
        return np.zeros(mask.shape, dtype=bool)
    return mask


def bbox_from_mask(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape[:2]
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + 1 + padding)
    bottom = min(height, int(ys.max()) + 1 + padding)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def trim_to_foreground(
    crop: Image.Image,
    background_rgb: tuple[int, int, int],
    threshold: float,
    padding: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    mask = foreground_mask_array(crop, background_rgb, threshold)
    bbox = bbox_from_mask(mask, padding=padding)
    if bbox is None:
        bbox = (0, 0, crop.width, crop.height)
    return crop.crop(bbox), bbox


def region_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for region in record.get("regions", []) or []:
            counts[str(region.get("type", "handwritten"))] += 1
    return dict(counts.most_common())


def page_counts_by_augmentation(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter("original" if row.get("augmentation_source") is None else "augmented" for row in records)
    return dict(counts)


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> str:
    if dst.exists():
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    return images_dir


def copy_original_records(
    records: list[dict[str, Any]],
    input_dir: Path,
    output_images_dir: Path,
    copy_mode: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    copied: list[dict[str, Any]] = []
    copy_counts: Counter[str] = Counter()
    used_names: set[str] = set()

    for record in records:
        src = resolve_image_path(record, input_dir)
        basename = Path(str(record.get("file_name", src.name))).name
        if basename in used_names:
            raise ValueError(f"Duplicate image basename in train metadata: {basename}")
        used_names.add(basename)

        dst = output_images_dir / basename
        copy_counts[link_or_copy(src, dst, copy_mode)] += 1
        with Image.open(dst) as image:
            width, height = image.size

        copied_record = dict(record)
        copied_record["file_name"] = f"images/{basename}"
        copied_record["image_width"] = int(width)
        copied_record["image_height"] = int(height)
        copied_record["augmentation_source"] = None
        copied.append(copied_record)

    return copied, copy_counts


def collect_crop_infos(
    records: list[dict[str, Any]],
    input_dir: Path,
    target_types: tuple[str, ...],
    min_width: int,
    min_height: int,
    min_area: int,
) -> dict[str, list[CropInfo]]:
    pools = {region_type: [] for region_type in target_types}

    for record_index, record in enumerate(records):
        image_path = resolve_image_path(record, input_dir)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

        for region_index, region in enumerate(record.get("regions", []) or []):
            region_type = str(region.get("type", "")).strip().lower()
            if region_type not in pools:
                continue
            bbox = normalized_bbox(region.get("bbox"), width, height)
            if bbox is None:
                continue
            left, top, right, bottom = bbox
            crop_width = right - left
            crop_height = bottom - top
            if crop_width < min_width or crop_height < min_height or crop_width * crop_height < min_area:
                continue
            crop = image.crop(bbox)
            background_rgb = estimate_background_rgb(crop)
            pools[region_type].append(
                CropInfo(
                    crop_id=f"r{record_index:04d}_{region_index:04d}",
                    source_file_name=str(record.get("file_name")),
                    source_image_path=image_path,
                    source_width=width,
                    source_height=height,
                    source_region_index=region_index,
                    source_bbox=bbox,
                    background_rgb=background_rgb,
                    region_type=region_type,
                    region=dict(region),
                )
            )

    return pools


@lru_cache(maxsize=16)
def load_source_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def fair_schedule(pool: list[CropInfo], needed: int, rng: random.Random) -> list[CropInfo]:
    schedule: list[CropInfo] = []
    while len(schedule) < needed:
        chunk = list(pool)
        rng.shuffle(chunk)
        schedule.extend(chunk)
    return schedule[:needed]


def choose_page_size(rng: random.Random) -> tuple[int, int]:
    sizes = [
        (1600, 2200),
        (1700, 2338),
        (1800, 2400),
        (1600, 2000),
        (1920, 2560),
    ]
    return rng.choice(sizes)


def page_background_rgb(infos: list[CropInfo], rng: random.Random, jitter: int) -> tuple[int, int, int]:
    if not infos:
        base = rng.randint(240, 254)
        return base, base, base
    colors = np.array([info.background_rgb for info in infos], dtype=np.int16)
    color = np.median(colors, axis=0)
    color += np.array([rng.randint(-jitter, jitter) for _ in range(3)], dtype=np.int16)
    return clipped_rgb(color)


def paper_background(
    width: int,
    height: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    color = np.array(background_rgb, dtype=np.int16)
    pixels = np.zeros((height, width, 3), dtype=np.int16)
    pixels[:, :] = color
    pixels += np_rng.normal(0, rng.uniform(1.2, 3.5), pixels.shape).astype(np.int16)

    gradient = np.linspace(rng.uniform(-3, 1), rng.uniform(0, 4), height, dtype=np.float32).reshape(height, 1, 1)
    pixels = pixels + gradient
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")

    if rng.random() < 0.35:
        draw = ImageDraw.Draw(image)
        line_gap = rng.randint(52, 82)
        line_color = tuple(int(np.clip(channel - rng.randint(8, 16), 0, 255)) for channel in background_rgb)
        for y in range(rng.randint(80, 130), height - 60, line_gap):
            draw.line((40, y, width - 40, y), fill=line_color, width=1)

    return image


def resize_rgba(image: Image.Image, scale: float) -> Image.Image:
    new_width = max(2, int(round(image.width * scale)))
    new_height = max(2, int(round(image.height * scale)))
    return image.resize((new_width, new_height), BICUBIC)


def recolor_crop_background(
    crop: Image.Image,
    target_background_rgb: tuple[int, int, int],
    threshold: float,
    feather: float,
) -> tuple[Image.Image, tuple[int, int, int]]:
    """Blend crop paper pixels toward the synthetic page color while preserving ink."""
    rgba = np.asarray(crop.convert("RGBA")).astype(np.float32)
    alpha = rgba[:, :, 3] > 0
    source_background_rgb = estimate_background_rgb(Image.fromarray(rgba[:, :, :3].astype(np.uint8), mode="RGB"), alpha)

    source_background = np.array(source_background_rgb, dtype=np.float32).reshape(1, 1, 3)
    target_background = np.array(target_background_rgb, dtype=np.float32).reshape(1, 1, 3)
    diff = rgba[:, :, :3] - source_background
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    background_weight = np.clip((threshold - distance) / max(1.0, threshold), 0.0, 1.0)
    background_weight *= alpha.astype(np.float32)
    background_weight *= float(feather)

    rgba[:, :, :3] = rgba[:, :, :3] * (1.0 - background_weight[:, :, None]) + target_background * background_weight[:, :, None]
    rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA"), source_background_rgb


def foreground_bbox_for_crop(
    crop: Image.Image,
    background_rgb: tuple[int, int, int],
    threshold: float,
    padding: int,
) -> tuple[int, int, int, int]:
    alpha = np.asarray(crop.getchannel("A")) > 0 if crop.mode == "RGBA" else None
    mask = foreground_mask_array(crop.convert("RGB"), background_rgb, threshold, alpha)
    bbox = bbox_from_mask(mask, padding=padding)
    if bbox is not None:
        return bbox
    alpha_bbox = crop.getchannel("A").getbbox() if crop.mode == "RGBA" else None
    if alpha_bbox is not None:
        return alpha_bbox
    return 0, 0, crop.width, crop.height


def transform_crop(
    info: CropInfo,
    rng: random.Random,
    np_rng: np.random.Generator,
    page_background: tuple[int, int, int],
    max_width: int,
    max_height: int,
    max_rotation_degrees: float,
    foreground_threshold: float,
    background_recolor_threshold: float,
    background_feather: float,
    bbox_padding: int,
) -> tuple[Image.Image, tuple[int, int, int, int], dict[str, Any]]:
    source = load_source_image(str(info.source_image_path))
    crop = source.crop(info.source_bbox).convert("RGBA")

    requested_scale = rng.uniform(0.85, 1.2)
    fit_scale = min(max_width / max(1, crop.width), max_height / max(1, crop.height), requested_scale)
    scale = max(0.12, fit_scale)
    crop = resize_rgba(crop, scale)

    brightness = rng.uniform(0.94, 1.07)
    contrast = rng.uniform(0.92, 1.08)
    rgb = ImageEnhance.Brightness(crop.convert("RGB")).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    alpha = crop.getchannel("A")
    crop = Image.merge("RGBA", (*rgb.split(), alpha))

    crop, transformed_background_rgb = recolor_crop_background(
        crop,
        target_background_rgb=page_background,
        threshold=background_recolor_threshold,
        feather=background_feather,
    )

    rotation_degrees = rng.uniform(-max_rotation_degrees, max_rotation_degrees)
    crop = crop.rotate(
        rotation_degrees,
        resample=BICUBIC,
        expand=True,
        fillcolor=(*page_background, 255),
    )

    if crop.width > max_width or crop.height > max_height:
        final_scale = min(max_width / crop.width, max_height / crop.height)
        crop = resize_rgba(crop, max(0.12, final_scale))
    else:
        final_scale = 1.0

    blur_radius = 0.0
    if rng.random() < 0.25:
        blur_radius = rng.uniform(0.15, 0.45)
        crop = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    noise_sigma = rng.uniform(0.0, 2.8)
    if noise_sigma > 0.2:
        arr = np.asarray(crop).astype(np.int16).copy()
        arr[:, :, :3] += np_rng.normal(0, noise_sigma, arr[:, :, :3].shape).astype(np.int16)
        arr[:, :, :3] = np.clip(arr[:, :, :3], 0, 255)
        crop = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    output_bbox = (0, 0, crop.width, crop.height)

    transform_meta = {
        "scale": round(scale * final_scale, 4),
        "rotation_degrees": round(rotation_degrees, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "blur_radius": round(blur_radius, 3),
        "noise_sigma": round(noise_sigma, 3),
        "output_width": crop.width,
        "output_height": crop.height,
        "source_background_rgb": list(info.background_rgb),
        "transformed_background_rgb": list(transformed_background_rgb),
        "page_background_rgb": list(page_background),
        "source_crop_bbox": [0, 0, info.source_bbox[2] - info.source_bbox[0], info.source_bbox[3] - info.source_bbox[1]],
        "bbox_in_crop": list(output_bbox),
        "rotation_fill_rgb": list(page_background),
        "bbox_policy": "full_transformed_crop",
    }
    return crop, output_bbox, transform_meta


def make_augmented_page(
    target_type: str,
    page_index: int,
    schedule: list[CropInfo],
    schedule_index: int,
    remaining: int,
    target_ratio: float,
    max_rotation_degrees: float,
    foreground_threshold: float,
    background_recolor_threshold: float,
    background_feather: float,
    background_jitter: int,
    bbox_padding: int,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[Image.Image, dict[str, Any], int, int]:
    width, height = choose_page_size(rng)
    margin = rng.randint(58, 96)
    y = rng.randint(60, 120)
    desired_regions = min(remaining, rng.randint(6, 14))
    page_infos = schedule[schedule_index : schedule_index + desired_regions]
    page_background = page_background_rgb(page_infos, rng, background_jitter)
    page = paper_background(width, height, rng, np_rng, page_background)
    regions: list[dict[str, Any]] = []
    provenance_regions: list[dict[str, Any]] = []

    while len(regions) < desired_regions and schedule_index < len(schedule):
        info = schedule[schedule_index]
        available_height = max(80, height - margin - y)
        crop, foreground_bbox, transform_meta = transform_crop(
            info,
            rng,
            np_rng,
            page_background=page_background,
            max_width=width - 2 * margin,
            max_height=min(max(160, height // 5), available_height),
            max_rotation_degrees=max_rotation_degrees,
            foreground_threshold=foreground_threshold,
            background_recolor_threshold=background_recolor_threshold,
            background_feather=background_feather,
            bbox_padding=bbox_padding,
        )
        if y + crop.height > height - margin:
            if regions:
                break
            fit_scale = (height - margin - y) / max(1, crop.height)
            crop = resize_rgba(crop, max(0.12, fit_scale))
            transform_meta["scale"] = round(float(transform_meta["scale"]) * max(0.12, fit_scale), 4)
            transform_meta["output_width"] = crop.width
            transform_meta["output_height"] = crop.height
            foreground_bbox = (0, 0, crop.width, crop.height)
            transform_meta["bbox_in_crop"] = list(foreground_bbox)

        x_max = max(margin, width - margin - crop.width)
        x = rng.randint(margin, x_max)
        page.paste(crop, (x, y), crop)
        bbox_left, bbox_top, bbox_right, bbox_bottom = foreground_bbox

        synthetic_region = dict(info.region)
        synthetic_region["bbox"] = [
            int(x + bbox_left),
            int(y + bbox_top),
            int(x + bbox_right),
            int(y + bbox_bottom),
        ]
        synthetic_region["type"] = info.region_type
        regions.append(synthetic_region)

        provenance_regions.append(
            {
                "synthetic_region_index": len(regions) - 1,
                "source_file_name": info.source_file_name,
                "source_region_index": info.source_region_index,
                "source_bbox": list(info.source_bbox),
                "source_image_width": info.source_width,
                "source_image_height": info.source_height,
                "type": info.region_type,
                "transform": transform_meta,
            }
        )
        schedule_index += 1
        y += crop.height + rng.randint(18, 58)

    if not regions:
        raise RuntimeError(f"Could not place any {target_type} crop on augmented page {page_index}")

    target_percent = int(round(float(target_ratio) * 100))
    file_name = f"images/aug_train{target_percent}_v3_{target_type}_{page_index:05d}.jpg"
    record = {
        "file_name": file_name,
        "image_width": width,
        "image_height": height,
        "source": "synthetic_train",
        "annotation_source": "synthetic_train_augment",
        "augmentation_source": {
            "method": "rare_region_crop_paste_v1",
            "version": "v3_background_matched_full_crop_bbox",
            "target_type": target_type,
            "target_ratio": float(target_ratio),
            "source_split": "train",
            "page_background_rgb": list(page_background),
            "regions": provenance_regions,
        },
        "regions": regions,
    }
    return page, record, len(regions), schedule_index


def validate_dataset(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    def add_issue(kind: str, index: int, detail: str) -> None:
        issues[kind] += 1
        if len(examples) < 20:
            examples.append({"kind": kind, "record_index": index, "detail": detail})

    for record_index, record in enumerate(records):
        image_path = output_dir / str(record.get("file_name", ""))
        if not image_path.exists():
            add_issue("missing_image", record_index, str(image_path))
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            add_issue("unreadable_image", record_index, f"{image_path}: {exc}")
            continue

        if int(record.get("image_width", -1)) != width or int(record.get("image_height", -1)) != height:
            add_issue(
                "dimension_mismatch",
                record_index,
                f"{record.get('file_name')} metadata=({record.get('image_width')},{record.get('image_height')}) actual=({width},{height})",
            )

        for region_index, region in enumerate(record.get("regions", []) or []):
            bbox = normalized_bbox(region.get("bbox"), width, height)
            if bbox is None:
                add_issue("invalid_bbox", record_index, f"{record.get('file_name')} region={region_index}")
                continue
            if list(bbox) != [int(round(float(v))) for v in region.get("bbox", [])]:
                add_issue("bbox_outside_or_non_integer", record_index, f"{record.get('file_name')} region={region_index}")
            if not str(region.get("type", "")).strip():
                add_issue("missing_region_type", record_index, f"{record.get('file_name')} region={region_index}")

    return {
        "ok": not issues,
        "issue_counts": dict(issues),
        "examples": examples,
    }


def build_augmented_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    metadata_path = input_dir / args.metadata
    output_dir = Path(args.output_dir)
    output_images_dir = prepare_output_dir(output_dir, args.overwrite)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    records = read_jsonl(metadata_path)
    original_counts = region_type_counts(records)
    handwritten_count = int(original_counts.get("handwritten", 0))
    target_count = math.ceil(handwritten_count * float(args.target_ratio))
    needed_by_type = {
        region_type: max(0, target_count - int(original_counts.get(region_type, 0)))
        for region_type in TARGET_TYPES
    }

    original_output_records, copy_counts = copy_original_records(records, input_dir, output_images_dir, args.copy_mode)
    crop_pools = collect_crop_infos(
        records=records,
        input_dir=input_dir,
        target_types=TARGET_TYPES,
        min_width=int(args.min_crop_width),
        min_height=int(args.min_crop_height),
        min_area=int(args.min_crop_area),
    )
    for region_type, needed in needed_by_type.items():
        if needed > 0 and not crop_pools[region_type]:
            raise ValueError(f"No usable crop pool for {region_type}; cannot add {needed} regions.")

    augmented_records: list[dict[str, Any]] = []
    added_by_type: dict[str, int] = {}
    pages_by_type: dict[str, int] = {}

    for region_type in TARGET_TYPES:
        needed = needed_by_type[region_type]
        schedule = fair_schedule(crop_pools[region_type], needed, rng) if needed else []
        schedule_index = 0
        added = 0
        page_number = 0
        while added < needed:
            page, record, placed, schedule_index = make_augmented_page(
                target_type=region_type,
                page_index=page_number,
                schedule=schedule,
                schedule_index=schedule_index,
                remaining=needed - added,
                target_ratio=float(args.target_ratio),
                max_rotation_degrees=float(args.max_rotation_degrees),
                foreground_threshold=float(args.foreground_threshold),
                background_recolor_threshold=float(args.background_recolor_threshold),
                background_feather=float(args.background_feather),
                background_jitter=int(args.background_jitter),
                bbox_padding=int(args.bbox_padding),
                rng=rng,
                np_rng=np_rng,
            )
            output_path = output_dir / record["file_name"]
            page.save(output_path, format="JPEG", quality=int(args.jpeg_quality), optimize=True)
            augmented_records.append(record)
            added += placed
            page_number += 1
        added_by_type[region_type] = added
        pages_by_type[region_type] = page_number

    output_records = original_output_records + augmented_records
    write_jsonl(output_dir / "metadata.jsonl", output_records)

    output_counts = region_type_counts(output_records)
    validation = validate_dataset(output_records, output_dir)
    summary = {
        "input_dir": str(input_dir),
        "input_metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "output_metadata": str(output_dir / "metadata.jsonl"),
        "target_ratio": float(args.target_ratio),
        "target_types": list(TARGET_TYPES),
        "seed": int(args.seed),
        "augmentation_version": "v3_background_matched_full_crop_bbox",
        "augmentation_parameters": {
            "max_rotation_degrees": float(args.max_rotation_degrees),
            "background_recolor_threshold": float(args.background_recolor_threshold),
            "background_feather": float(args.background_feather),
            "background_jitter": int(args.background_jitter),
            "rotation_fill": "page_background_rgb",
            "bbox_policy": "full_transformed_crop",
        },
        "copy_mode_requested": args.copy_mode,
        "copy_mode_actual_counts": dict(copy_counts),
        "page_counts": {
            "original": len(original_output_records),
            "augmented": len(augmented_records),
            "total": len(output_records),
            "augmented_by_type": pages_by_type,
        },
        "region_counts_before": original_counts,
        "handwritten_reference_count": handwritten_count,
        "target_count_per_rare_type": target_count,
        "augmentation_plan": {
            region_type: {
                "existing": int(original_counts.get(region_type, 0)),
                "target": target_count,
                "added": int(added_by_type.get(region_type, 0)),
                "usable_crop_pool": len(crop_pools[region_type]),
            }
            for region_type in TARGET_TYPES
        },
        "region_counts_after": output_counts,
        "page_counts_by_augmentation_source": page_counts_by_augmentation(output_records),
        "validation": validation,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-only rare-class augmented dataset.")
    parser.add_argument("--input-dir", default="dataset/train", help="Input train split directory.")
    parser.add_argument("--metadata", default="metadata.jsonl", help="Metadata JSONL name under input-dir.")
    parser.add_argument("--output-dir", default="dataset/train_augmented_20pct_v3", help="Output dataset directory.")
    parser.add_argument("--target-ratio", type=float, default=0.20, help="Rare class target as a ratio of handwritten regions.")
    parser.add_argument("--seed", type=int, default=20260526, help="Deterministic augmentation seed.")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink", help="How to place original images in output.")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for augmented images.")
    parser.add_argument("--max-rotation-degrees", type=float, default=1.5, help="Maximum absolute crop rotation in degrees.")
    parser.add_argument("--foreground-threshold", type=float, default=30.0, help="Reserved for earlier bbox-mask experiments; v3 does not use it.")
    parser.add_argument("--background-recolor-threshold", type=float, default=44.0, help="Color-distance threshold for crop paper recoloring.")
    parser.add_argument("--background-feather", type=float, default=0.9, help="Blend strength for crop paper recoloring.")
    parser.add_argument("--background-jitter", type=int, default=5, help="RGB jitter around per-page crop background median.")
    parser.add_argument("--bbox-padding", type=int, default=0, help="Reserved for earlier bbox-mask experiments; v3 does not use it.")
    parser.add_argument("--min-crop-width", type=int, default=8, help="Minimum source crop width.")
    parser.add_argument("--min-crop-height", type=int, default=8, help="Minimum source crop height.")
    parser.add_argument("--min-crop-area", type=int, default=64, help="Minimum source crop area.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output-dir if it already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_augmented_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
