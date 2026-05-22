from __future__ import annotations

from typing import Any


VALID_REGION_TYPES = {
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
}

STRUCTURAL_TYPES = {"image", "graph"}
SCORABLE_TYPES = VALID_REGION_TYPES - STRUCTURAL_TYPES


def normalize_type(value: Any) -> str:
    """Normalize model/data type labels into the official 7-class schema."""
    region_type = str(value or "handwritten").strip().lower()
    return region_type if region_type in VALID_REGION_TYPES else "handwritten"


def text_for_type(region_type: str, text: Any) -> str:
    """Apply Phase 1 structural-type policy to target or predicted text."""
    normalized_type = normalize_type(region_type)
    if normalized_type in STRUCTURAL_TYPES:
        return ""
    return "" if text is None else str(text)


def sorted_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort regions in the simple Phase 1 reading order: top-to-bottom, then left-to-right."""
    return sorted(regions, key=lambda r: (r.get("bbox", [0, 0, 0, 0])[1], r.get("bbox", [0, 0, 0, 0])[0]))


def enforce_submission_policy(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only official fields, normalize types, and force image/graph text to empty."""
    cleaned: list[dict[str, Any]] = []
    for region in regions:
        region_type = normalize_type(region.get("type"))
        cleaned.append(
            {
                "bbox": region.get("bbox", [0, 0, 0, 0]),
                "type": region_type,
                "text": text_for_type(region_type, region.get("text", "")),
            }
        )
    return sorted_regions(cleaned)

