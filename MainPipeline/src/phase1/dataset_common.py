from __future__ import annotations

from pathlib import Path
from typing import Any

from MainPipeline.src.common.io import resolve_path


def image_keys(record: dict[str, Any]) -> set[str]:
    """Return stable identifiers used to filter train rows against validation manifests."""
    file_name = str(record.get("file_name", "")).replace("\\", "/")
    basename = Path(file_name).name
    keys = {file_name, basename}
    if record.get("image_id"):
        keys.add(str(record["image_id"]).replace("\\", "/"))
    if record.get("submission_image"):
        keys.add(str(record["submission_image"]).replace("\\", "/"))
    return {key for key in keys if key}


def candidate_image_paths(
    record: dict[str, Any],
    metadata_path: Path,
    image_roots: list[str],
) -> list[Path]:
    """Build path candidates for local samples, VM mounts, and HF-style image folders."""
    file_name = str(record.get("file_name", "")).replace("\\", "/")
    basename = Path(file_name).name
    candidates: list[Path] = []
    for root_value in image_roots:
        root = resolve_path(root_value)
        candidates.append(root / file_name)
        candidates.append(root / basename)
    metadata_parent = metadata_path.parent
    candidates.extend(
        [
            metadata_parent / file_name,
            metadata_parent / basename,
            metadata_parent / "images" / basename,
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def resolve_image_path(record: dict[str, Any], metadata_path: Path, image_roots: list[str]) -> Path | None:
    """Resolve an image path for a metadata row, returning None when the image is unavailable."""
    for path in candidate_image_paths(record, metadata_path, image_roots):
        if path.exists():
            return path
    return None


def qwen_vl_example(
    image_path: Path,
    prompt_text: str,
    answer_text: str,
    max_pixels: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create one Qwen-VL chat example with image + prompt as user and target answer as assistant."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path), "max_pixels": int(max_pixels)},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ],
        "meta": metadata,
    }

