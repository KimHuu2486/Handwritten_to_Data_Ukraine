from __future__ import annotations

from pathlib import Path
from typing import Any


def as_float_bbox(value: Any) -> list[float] | None:
    """Convert a user/model bbox value to four floats, or return None if invalid."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def clamp_bbox(bbox: list[float], image_width: int, image_height: int) -> list[int]:
    """Clamp a bbox to image bounds and fix inverted coordinates."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_width, round(x1)))
    x2 = max(0, min(image_width, round(x2)))
    y1 = max(0, min(image_height, round(y1)))
    y2 = max(0, min(image_height, round(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(x1), int(y1), int(x2), int(y2)]


def is_valid_bbox(bbox: list[int], min_size: int = 2) -> bool:
    """Return True when a bbox has meaningful positive area."""
    return (bbox[2] - bbox[0]) >= min_size and (bbox[3] - bbox[1]) >= min_size


def pixel_to_grid_bbox(bbox: list[int], image_width: int, image_height: int, grid_size: int = 1000) -> list[int]:
    """Convert pixel coordinates into the 0-grid_size coordinate system used in prompts."""
    w = max(int(image_width), 1)
    h = max(int(image_height), 1)
    return [
        int(max(0, min(w, bbox[0])) / w * grid_size),
        int(max(0, min(h, bbox[1])) / h * grid_size),
        int(max(0, min(w, bbox[2])) / w * grid_size),
        int(max(0, min(h, bbox[3])) / h * grid_size),
    ]


def grid_to_pixel_bbox(bbox: list[float], image_width: int, image_height: int, grid_size: int = 1000) -> list[int]:
    """Convert normalized/grid model coordinates back to pixels, with pixel fallback."""
    max_value = max(bbox) if bbox else 0
    if max_value <= 1.0:
        return clamp_bbox(
            [bbox[0] * image_width, bbox[1] * image_height, bbox[2] * image_width, bbox[3] * image_height],
            image_width,
            image_height,
        )
    if max_value > grid_size * 1.05:
        return clamp_bbox(bbox, image_width, image_height)
    return clamp_bbox(
        [
            bbox[0] / grid_size * image_width,
            bbox[1] / grid_size * image_height,
            bbox[2] / grid_size * image_width,
            bbox[3] / grid_size * image_height,
        ],
        image_width,
        image_height,
    )


def padded_bbox(bbox: list[int], image_width: int, image_height: int, pad_ratio: float = 0.02) -> list[int]:
    """Expand a bbox by a small ratio to avoid cutting off handwritten strokes."""
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return clamp_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], image_width, image_height)


def crop_region(image_path: Path, bbox: list[int], output_path: Path, pad_ratio: float = 0.02) -> tuple[int, int]:
    """Crop one region to disk and return the crop width/height."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop_box = padded_bbox(bbox, image.width, image.height, pad_ratio)
        crop = image.crop(tuple(crop_box))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, quality=95)
        return crop.width, crop.height
