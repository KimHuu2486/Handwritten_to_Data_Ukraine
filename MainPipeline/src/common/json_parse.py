from __future__ import annotations

import json
import re
import warnings
from typing import Any

from .bbox import as_float_bbox, grid_to_pixel_bbox, is_valid_bbox
from .schema import enforce_submission_policy, normalize_type


def extract_json_array(text: str) -> tuple[list[Any], str | None]:
    """Extract a JSON array from a model response with code-fence and wrapper fallbacks."""
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed, None
        if isinstance(parsed, dict):
            for key in ("regions", "results", "data"):
                if isinstance(parsed.get(key), list):
                    return parsed[key], None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed, None
        except json.JSONDecodeError:
            pass

    objects: list[Any] = []
    pattern = r'\{[^{}]*"bbox"[^{}]*"type"[^{}]*(?:"text"[^{}]*)?\}'
    for object_match in re.finditer(pattern, text):
        try:
            objects.append(json.loads(object_match.group(0)))
        except json.JSONDecodeError:
            continue
    if objects:
        return objects, None
    return [], "json_parse_failed"


def normalize_layout_regions(raw_regions: list[Any], image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Normalize Stage A model output into pixel-space bbox/type records."""
    regions: list[dict[str, Any]] = []
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        bbox = as_float_bbox(item.get("bbox"))
        if bbox is None:
            continue
        if max(bbox) > 1050:
            warnings.warn(
                "Stage A bbox exceeded the 0-1000 grid contract; treating it as pixel coordinates.",
                RuntimeWarning,
                stacklevel=2,
            )
        pixel_bbox = grid_to_pixel_bbox(bbox, image_width, image_height)
        if not is_valid_bbox(pixel_bbox):
            continue
        regions.append({"bbox": pixel_bbox, "type": normalize_type(item.get("type"))})
    return enforce_submission_policy(regions)
