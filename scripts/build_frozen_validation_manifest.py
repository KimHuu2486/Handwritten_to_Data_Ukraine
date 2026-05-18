#!/usr/bin/env python3
"""Stage A: build the frozen validation manifest for the RUKOPYS pipeline.

This script intentionally mirrors the validation split used by
`finetune/rukopys_qwen3vl_stage2_gold_finetune.ipynb`:

- prefer an existing `gold_validation_records.jsonl` from Stage 2;
- otherwise split `train/metadata.jsonl` once with seed=42, val_ratio=0.12,
  stratified by `source`, using the same shuffle order as the notebook.

It writes:

- artifacts/manifests/frozen_validation_manifest.v1.jsonl
- artifacts/manifests/frozen_validation_manifest.v1.config.json
- artifacts/manifests/frozen_validation_manifest.v1.gate_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the metadata usually has dimensions.
    Image = None  # type: ignore[assignment]


ARTIFACT_NAME = "frozen_validation_manifest"
ARTIFACT_VERSION = "frozen_validation_manifest.v1"
SCRIPT_VERSION = "stage-a-manifest-v1"
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.12
DEFAULT_SPLIT = "train"
DEFAULT_STRATIFY_FIELD = "source"
NOTEBOOK_REFERENCE = "finetune/rukopys_qwen3vl_stage2_gold_finetune.ipynb"

ALLOWED_SOURCES = {"archive", "school", "dictation", "university", "unknown"}
ALLOWED_REGION_TYPES = {
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
    "unknown",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DETAIL_LIMIT = 200


@dataclass(frozen=True)
class ResolvedImage:
    path: Path
    exists: bool
    tried: list[str]


@dataclass(frozen=True)
class BBoxResult:
    bbox: list[int] | None
    original_format: str | None
    warnings: list[str]
    error: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_display_path(path: Path) -> str:
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve()
        base = repo_root().parent.resolve()
        try:
            return f"./{resolved.relative_to(base).as_posix()}"
        except ValueError:
            return str(resolved)
    except OSError:
        normalized = expanded.as_posix()
        return normalized if normalized.startswith(".") else f"./{normalized}"


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def read_jsonl_head(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(obj).__name__}")
    return rows


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_posix(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    path = path.strip("/")
    return path


def normalize_file_name(value: Any, dataset_root: Path, split: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""

    path = Path(raw)
    if path.is_absolute():
        for base in (dataset_root / split, dataset_root):
            try:
                return Path(path).relative_to(base).as_posix()
            except ValueError:
                pass
        parts = PurePosixPath(raw).parts
        if split in parts:
            split_idx = parts.index(split)
            return PurePosixPath(*parts[split_idx + 1 :]).as_posix()
        return PurePosixPath(raw).name

    normalized = normalize_posix(raw)
    split_prefix = f"{split}/"
    if normalized.startswith(split_prefix):
        normalized = normalized[len(split_prefix) :]
    return normalized


def candidate_image_names(file_name: str) -> list[str]:
    raw = PurePosixPath(file_name)
    name = raw.name
    stem = PurePosixPath(name).stem
    names = [name]
    for ext in IMAGE_EXTENSIONS:
        candidate = stem + ext
        if candidate not in names:
            names.append(candidate)
    return names


def resolve_image_path(dataset_root: Path, split: str, file_name: str) -> ResolvedImage:
    raw = Path(file_name)
    tried: list[Path] = []

    if raw.is_absolute():
        tried.append(raw)

    normalized = normalize_posix(file_name)
    if normalized:
        tried.extend(
            [
                dataset_root / split / normalized,
                dataset_root / normalized,
            ]
        )

    for name in candidate_image_names(normalized):
        tried.extend(
            [
                dataset_root / split / "images" / name,
                dataset_root / split / name,
                dataset_root / "images" / name,
                dataset_root / name,
            ]
        )

    seen: set[str] = set()
    unique_tried: list[Path] = []
    for path in tried:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_tried.append(path)

    for path in unique_tried:
        if path.exists():
            return ResolvedImage(path=path, exists=True, tried=[str(p) for p in unique_tried])

    fallback = unique_tried[0] if unique_tried else dataset_root / split / normalized
    return ResolvedImage(path=fallback, exists=False, tried=[str(p) for p in unique_tried])


def metadata_path_for_root(root: Path, split: str) -> Path:
    return root / split / "metadata.jsonl"


def unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser().resolve()) if path.exists() else str(path.expanduser())
        if key not in seen:
            seen.add(key)
            out.append(path.expanduser())
    return out


def dataset_root_candidates(root: Path, cwd: Path) -> list[Path]:
    env_root = os.getenv("RUKOPYS_DATASET_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            cwd / "data",
            cwd / "dataset",
            root / "data",
            root / "dataset",
            root.parent / "data",
            root.parent / "dataset",
            Path("/kaggle/input/datasets/quii29/rukopys-dataset"),
        ]
    )
    return unique_paths(candidates)


def score_dataset_root(candidate: Path, split: str, sample_size: int = 100) -> tuple[int, int, int]:
    metadata = metadata_path_for_root(candidate, split)
    if not metadata.exists():
        return (-1, -1, -1)
    try:
        sample = read_jsonl_head(metadata, sample_size)
    except Exception:
        return (-1, -1, -1)
    resolved = 0
    for row in sample:
        file_name = normalize_file_name(row.get("file_name"), candidate, split)
        if file_name and resolve_image_path(candidate, split, file_name).exists:
            resolved += 1
    image_count = sum(1 for _ in (candidate / split).glob("**/*") if _.suffix.lower() in IMAGE_EXTENSIONS)
    return (resolved, image_count, len(sample))


def discover_dataset_root(root: Path, cwd: Path, split: str) -> Path:
    scored: list[tuple[tuple[int, int, int], Path]] = []
    for candidate in dataset_root_candidates(root, cwd):
        score = score_dataset_root(candidate, split)
        if score[0] >= 0:
            scored.append((score, candidate))
    if not scored:
        raise FileNotFoundError(
            "Could not find a dataset root with train/metadata.jsonl. "
            "Pass --dataset-root or set RUKOPYS_DATASET_ROOT."
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def find_stage2_validation_records(root: Path) -> Path | None:
    search_paths = [
        root / "qwen3vl_rukopys_stage2_gold" / "gold_validation_records.jsonl",
        root / "finetune" / "qwen3vl_rukopys_stage2_gold" / "gold_validation_records.jsonl",
        root.parent / "qwen3vl_rukopys_stage2_gold" / "gold_validation_records.jsonl",
        Path("/kaggle/working/qwen3vl_rukopys_stage2_gold/gold_validation_records.jsonl"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    for path in sorted(root.glob("**/gold_validation_records.jsonl")):
        if ".git" not in path.parts:
            return path
    return None


def stratified_split(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    stratify_field: str = DEFAULT_STRATIFY_FIELD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact split logic from the Stage 2 notebook."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_source[row.get(stratify_field, "unknown")].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for _source, items in by_source.items():
        items = list(items)
        rng.shuffle(items)
        if len(items) <= 1:
            n_val = 0
        else:
            n_val = min(max(1, int(round(len(items) * val_ratio))), len(items) - 1)
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def to_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def image_size(path: Path) -> tuple[int, int] | None:
    if not path.exists() or Image is None:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def normalize_source(value: Any) -> tuple[str, str | None]:
    source = str(value or "unknown").strip().lower()
    if source in ALLOWED_SOURCES:
        return source, None
    return "unknown", source


def normalize_region_type(value: Any) -> tuple[str, str | None]:
    region_type = str(value or "unknown").strip().lower()
    if region_type in ALLOWED_REGION_TYPES:
        return region_type, None
    return "unknown", region_type


def numeric_bbox_values(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def clip_bbox(values: list[float], width: int, height: int) -> tuple[list[float], bool]:
    x1, y1, x2, y2 = values
    clipped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    changed = any(abs(a - b) > 1e-6 for a, b in zip(values, clipped))
    return clipped, changed


def valid_xyxy(values: list[float]) -> bool:
    return values[2] > values[0] and values[3] > values[1]


def round_bbox(values: list[float]) -> list[int]:
    return [int(round(v)) for v in values]


def normalize_bbox(
    value: Any,
    width: int | None,
    height: int | None,
    input_format: str,
) -> BBoxResult:
    if width is None or height is None:
        return BBoxResult(None, None, [], "missing_image_dimensions")

    values = numeric_bbox_values(value)
    if values is None:
        return BBoxResult(None, None, [], "bbox_must_be_numeric_list_of_4")

    warnings: list[str] = []
    fmt = input_format

    if input_format == "pixel_xyxy":
        xyxy = values
    elif input_format == "normalized_xyxy":
        xyxy = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    elif input_format == "grid1000_xyxy":
        xyxy = [values[0] / 1000 * width, values[1] / 1000 * height, values[2] / 1000 * width, values[3] / 1000 * height]
    elif input_format == "pixel_xywh":
        xyxy = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    elif input_format == "auto":
        if all(0.0 <= v <= 1.0 for v in values):
            fmt = "normalized_xyxy"
            xyxy = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
        else:
            fmt = "pixel_xyxy"
            xyxy = values
    else:
        return BBoxResult(None, None, [], f"unsupported_input_bbox_format:{input_format}")

    if not valid_xyxy(xyxy):
        reordered = [min(xyxy[0], xyxy[2]), min(xyxy[1], xyxy[3]), max(xyxy[0], xyxy[2]), max(xyxy[1], xyxy[3])]
        if valid_xyxy(reordered):
            xyxy = reordered
            fmt = f"{fmt}_reordered"
            warnings.append("bbox_coordinate_order_repaired")
        elif input_format == "auto" and values[2] > 0 and values[3] > 0:
            xywh = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
            if valid_xyxy(xywh):
                xyxy = xywh
                fmt = "pixel_xywh_inferred"
                warnings.append("bbox_format_inferred_as_xywh")

    clipped, was_clipped = clip_bbox(xyxy, width, height)
    if was_clipped:
        warnings.append("bbox_clipped_to_image_bounds")

    if not valid_xyxy(clipped):
        return BBoxResult(None, fmt, warnings, "bbox_invalid_after_clipping")

    return BBoxResult(round_bbox(clipped), fmt, warnings, None)


def cap_details(items: list[Any], limit: int = DETAIL_LIMIT) -> dict[str, Any]:
    return {
        "count": len(items),
        "truncated": len(items) > limit,
        "items": items[:limit],
    }


def build_manifest_records(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    split: str,
    input_bbox_format: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    missing_images: list[dict[str, Any]] = []
    width_height_fallbacks: list[dict[str, Any]] = []
    invalid_dimensions: list[dict[str, Any]] = []
    unknown_sources: list[dict[str, Any]] = []
    unknown_region_types: list[dict[str, Any]] = []
    bbox_errors: list[dict[str, Any]] = []
    bbox_warnings: list[dict[str, Any]] = []
    generated_region_ids = 0
    duplicate_region_ids: list[dict[str, Any]] = []
    non_dict_regions: list[dict[str, Any]] = []

    image_id_counts: Counter[str] = Counter()
    region_key_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    region_type_counts: Counter[str] = Counter()
    region_count = 0

    for record_index, row in enumerate(rows):
        file_name = normalize_file_name(row.get("file_name"), dataset_root, split)
        image_id = file_name
        submission_image = PurePosixPath(file_name).name
        image_id_counts[image_id] += 1

        resolved = resolve_image_path(dataset_root, split, file_name)
        if not resolved.exists:
            missing_images.append(
                {
                    "record_index": record_index,
                    "image_id": image_id,
                    "file_name": file_name,
                    "expected_path": str(resolved.path),
                    "tried": resolved.tried[:8],
                }
            )

        width = to_positive_int(row.get("image_width"))
        height = to_positive_int(row.get("image_height"))
        if width is None or height is None:
            detected_size = image_size(resolved.path)
            if detected_size is not None:
                width, height = detected_size
                width_height_fallbacks.append(
                    {
                        "record_index": record_index,
                        "image_id": image_id,
                        "source": "image_file",
                        "image_width": width,
                        "image_height": height,
                    }
                )
            else:
                invalid_dimensions.append(
                    {
                        "record_index": record_index,
                        "image_id": image_id,
                        "image_width": row.get("image_width"),
                        "image_height": row.get("image_height"),
                    }
                )

        source, source_original = normalize_source(row.get("source", "unknown"))
        if source_original is not None:
            unknown_sources.append({"record_index": record_index, "image_id": image_id, "source_original": source_original})
        source_counts[source] += 1

        out_record = dict(row)
        out_record["image_id"] = image_id
        out_record["file_name"] = file_name
        out_record["submission_image"] = submission_image
        out_record["source"] = source
        out_record["image_width"] = width
        out_record["image_height"] = height
        if source_original is not None:
            out_record["source_original"] = source_original

        out_regions: list[dict[str, Any]] = []
        seen_region_ids_for_image: set[str] = set()
        raw_regions = row.get("regions") or []
        if not isinstance(raw_regions, list):
            raw_regions = []
            non_dict_regions.append({"record_index": record_index, "image_id": image_id, "error": "regions_is_not_a_list"})

        for region_index, raw_region in enumerate(raw_regions):
            region_count += 1
            if not isinstance(raw_region, dict):
                region_id = f"r{region_index:04d}"
                out_region = {"region_id": region_id, "bbox": None, "type": "unknown", "text": "", "region_error": "region_is_not_an_object"}
                non_dict_regions.append(
                    {
                        "record_index": record_index,
                        "region_index": region_index,
                        "image_id": image_id,
                        "region_id": region_id,
                    }
                )
                out_regions.append(out_region)
                region_key_counts[(image_id, region_id)] += 1
                continue

            out_region = dict(raw_region)
            raw_region_id = out_region.get("region_id")
            if raw_region_id is None or str(raw_region_id).strip() == "":
                region_id = f"r{region_index:04d}"
                generated_region_ids += 1
            else:
                region_id = str(raw_region_id)
            out_region["region_id"] = region_id

            if region_id in seen_region_ids_for_image:
                duplicate_region_ids.append(
                    {
                        "record_index": record_index,
                        "region_index": region_index,
                        "image_id": image_id,
                        "region_id": region_id,
                    }
                )
            seen_region_ids_for_image.add(region_id)
            region_key_counts[(image_id, region_id)] += 1

            region_type, type_original = normalize_region_type(out_region.get("type", "unknown"))
            if type_original is not None:
                unknown_region_types.append(
                    {
                        "record_index": record_index,
                        "region_index": region_index,
                        "image_id": image_id,
                        "region_id": region_id,
                        "type_original": type_original,
                    }
                )
                out_region["type_original"] = type_original
            out_region["type"] = region_type
            region_type_counts[region_type] += 1

            bbox_result = normalize_bbox(out_region.get("bbox"), width, height, input_bbox_format)
            if bbox_result.error is not None:
                out_region["bbox_original"] = out_region.get("bbox")
                out_region["bbox"] = None
                out_region["bbox_error"] = bbox_result.error
                if bbox_result.original_format is not None:
                    out_region["bbox_format_original"] = bbox_result.original_format
                bbox_errors.append(
                    {
                        "record_index": record_index,
                        "region_index": region_index,
                        "image_id": image_id,
                        "region_id": region_id,
                        "bbox_original": out_region.get("bbox_original"),
                        "error": bbox_result.error,
                        "warnings": bbox_result.warnings,
                    }
                )
            else:
                original_bbox = out_region.get("bbox")
                out_region["bbox"] = bbox_result.bbox
                if bbox_result.original_format != "pixel_xyxy" or bbox_result.warnings:
                    out_region["bbox_original"] = original_bbox
                    out_region["bbox_format_original"] = bbox_result.original_format
                if bbox_result.warnings:
                    out_region["bbox_warnings"] = bbox_result.warnings
                    bbox_warnings.append(
                        {
                            "record_index": record_index,
                            "region_index": region_index,
                            "image_id": image_id,
                            "region_id": region_id,
                            "bbox_original": original_bbox,
                            "bbox": bbox_result.bbox,
                            "warnings": bbox_result.warnings,
                        }
                    )

            out_regions.append(out_region)

        out_record["regions"] = out_regions
        manifest.append(out_record)

    duplicate_image_ids = [
        {"image_id": image_id, "count": count}
        for image_id, count in sorted(image_id_counts.items())
        if count > 1
    ]
    duplicate_region_keys = [
        {"image_id": image_id, "region_id": region_id, "count": count}
        for (image_id, region_id), count in sorted(region_key_counts.items())
        if count > 1
    ]
    duplicate_region_ids.extend(duplicate_region_keys)

    diagnostics = {
        "row_count": len(manifest),
        "region_count": region_count,
        "source_distribution": dict(sorted(source_counts.items())),
        "region_type_distribution": dict(sorted(region_type_counts.items())),
        "generated_region_id_count": generated_region_ids,
        "duplicate_image_ids": cap_details(duplicate_image_ids),
        "duplicate_region_ids": cap_details(duplicate_region_ids),
        "missing_images": cap_details(missing_images),
        "invalid_dimensions": cap_details(invalid_dimensions),
        "width_height_fallbacks": cap_details(width_height_fallbacks),
        "unknown_sources": cap_details(unknown_sources),
        "unknown_region_types": cap_details(unknown_region_types),
        "non_dict_regions": cap_details(non_dict_regions),
        "bbox_errors": cap_details(bbox_errors),
        "bbox_warnings": cap_details(bbox_warnings),
        "image_order_sha256": sha256_text("\n".join(row.get("image_id", "") for row in manifest) + "\n"),
    }
    return manifest, diagnostics


def output_paths(output_dir: Path, artifact_version: str) -> dict[str, Path]:
    return {
        "manifest": output_dir / f"{artifact_version}.jsonl",
        "config": output_dir / f"{artifact_version}.config.json",
        "gate_report": output_dir / f"{artifact_version}.gate_report.json",
    }


def ensure_outputs_can_be_written(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Stage A outputs already exist. Use --overwrite only when intentionally regenerating "
            f"the same artifact version.\n{formatted}"
        )


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), **details}


def build_gate_report(
    manifest: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    expected_row_count: int,
    source_config: dict[str, Any],
    paths: dict[str, Path],
    artifact_version: str,
) -> dict[str, Any]:
    checks = [
        check(
            "validation_source_known",
            bool(source_config.get("source_split_kind")),
            source_split_kind=source_config.get("source_split_kind"),
        ),
        check(
            "row_count_matches_validation_split",
            len(manifest) == expected_row_count,
            expected=expected_row_count,
            actual=len(manifest),
        ),
        check(
            "all_image_files_exist",
            diagnostics["missing_images"]["count"] == 0,
            missing_count=diagnostics["missing_images"]["count"],
        ),
        check(
            "image_id_unique",
            diagnostics["duplicate_image_ids"]["count"] == 0,
            duplicate_count=diagnostics["duplicate_image_ids"]["count"],
        ),
        check(
            "region_keys_unique",
            diagnostics["duplicate_region_ids"]["count"] == 0,
            duplicate_count=diagnostics["duplicate_region_ids"]["count"],
        ),
        check(
            "image_dimensions_available",
            diagnostics["invalid_dimensions"]["count"] == 0,
            invalid_dimension_count=diagnostics["invalid_dimensions"]["count"],
        ),
        check(
            "bbox_errors_reported",
            True,
            bbox_error_count=diagnostics["bbox_errors"]["count"],
        ),
        check(
            "bbox_all_normalized",
            diagnostics["bbox_errors"]["count"] == 0,
            bbox_error_count=diagnostics["bbox_errors"]["count"],
        ),
        check(
            "output_order_stable",
            True,
            image_order_sha256=diagnostics["image_order_sha256"],
        ),
    ]
    passed = all(item["pass"] for item in checks)

    report = {
        "artifact_name": ARTIFACT_NAME,
        "artifact_version": artifact_version,
        "status": "pass" if passed else "fail",
        "pass": passed,
        "created_at": utc_now(),
        "source_split": source_config,
        "row_count": len(manifest),
        "expected_validation_row_count": expected_row_count,
        "region_count": diagnostics["region_count"],
        "source_distribution": diagnostics["source_distribution"],
        "region_type_distribution": diagnostics["region_type_distribution"],
        "generated_region_id_count": diagnostics["generated_region_id_count"],
        "checks": checks,
        "details": {
            "missing_images": diagnostics["missing_images"],
            "duplicate_image_ids": diagnostics["duplicate_image_ids"],
            "duplicate_region_ids": diagnostics["duplicate_region_ids"],
            "invalid_dimensions": diagnostics["invalid_dimensions"],
            "width_height_fallbacks": diagnostics["width_height_fallbacks"],
            "unknown_sources": diagnostics["unknown_sources"],
            "unknown_region_types": diagnostics["unknown_region_types"],
            "non_dict_regions": diagnostics["non_dict_regions"],
            "bbox_errors": diagnostics["bbox_errors"],
            "bbox_warnings": diagnostics["bbox_warnings"],
        },
        "outputs": {name: as_display_path(path) for name, path in paths.items()},
    }
    return report


def prepare_source_records(args: argparse.Namespace, root: Path, dataset_root: Path, metadata_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.force_rebuild_split and args.validation_records:
        raise ValueError("--force-rebuild-split cannot be combined with --validation-records")

    validation_records_path: Path | None = None
    if args.validation_records:
        validation_records_path = args.validation_records.expanduser()
        if not validation_records_path.exists():
            raise FileNotFoundError(f"--validation-records does not exist: {validation_records_path}")
    elif not args.force_rebuild_split:
        validation_records_path = find_stage2_validation_records(root)

    if validation_records_path is not None:
        validation_records = read_jsonl(validation_records_path)
        source_config = {
            "source_split_kind": "promoted_stage2_gold_validation_records",
            "validation_records_path": as_display_path(validation_records_path),
            "metadata_path": as_display_path(metadata_path),
            "dataset_root": as_display_path(dataset_root),
            "split": args.split,
            "stage2_output_name": "gold_validation_records.jsonl",
            "preserve_input_order": True,
            "fallback_split_seed": args.seed,
            "fallback_val_ratio": args.val_ratio,
            "fallback_stratify_field": DEFAULT_STRATIFY_FIELD,
        }
        return validation_records, source_config

    all_records = read_jsonl(metadata_path)
    _train_records, validation_records = stratified_split(
        all_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_field=DEFAULT_STRATIFY_FIELD,
    )
    source_config = {
        "source_split_kind": "deterministic_stage2_split_rebuild",
        "metadata_path": as_display_path(metadata_path),
        "dataset_root": as_display_path(dataset_root),
        "split": args.split,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "stratify_field": DEFAULT_STRATIFY_FIELD,
        "source_shuffle": "random.Random(seed).shuffle(items) per source",
        "validation_rule": "round(len(source_items) * val_ratio), min 1 when source has >1 record, max len-1",
        "final_shuffle": "random.Random(seed).shuffle(validation_rows)",
        "preserve_input_order": False,
        "notebook_reference": NOTEBOOK_REFERENCE,
        "split_function": "stratified_split(gold_records, VAL_RATIO)",
    }
    return validation_records, source_config


def build_config(
    args: argparse.Namespace,
    dataset_root: Path,
    metadata_path: Path,
    source_config: dict[str, Any],
    paths: dict[str, Path],
    manifest_sha256: str | None,
    gate_report_sha256: str | None,
) -> dict[str, Any]:
    root = repo_root()
    return {
        "artifact_name": ARTIFACT_NAME,
        "artifact_version": args.artifact_version,
        "stage": "Stage A - Frozen Validation Manifest",
        "created_at": utc_now(),
        "script_path": str(Path(__file__).relative_to(root)),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "code_version": run_git_commit(root),
        "notebook_reference": NOTEBOOK_REFERENCE,
        "dataset_root": as_display_path(dataset_root),
        "metadata_path": as_display_path(metadata_path),
        "source_split": source_config,
        "schema": {
            "required_image_fields": [
                "image_id",
                "file_name",
                "submission_image",
                "source",
                "image_width",
                "image_height",
                "regions",
            ],
            "image_id_rule": "normalized file_name with forward slashes, no dataset root",
            "submission_image_rule": "basename(file_name)",
            "source_allowed_values": sorted(ALLOWED_SOURCES),
            "bbox_internal_format": "xyxy_pixel_original_image",
            "region_id_rule": "preserve existing region_id, otherwise r0000/r0001 by original region order",
        },
        "normalization": {
            "input_bbox_format": args.input_bbox_format,
            "clip_bbox_to_image_bounds": True,
            "unknown_source_policy": "map_to_unknown_and_report",
            "unknown_region_type_policy": "map_to_unknown_and_report",
            "drop_images": False,
            "drop_regions": False,
        },
        "outputs": {name: as_display_path(path) for name, path in paths.items()},
        "checksums": {
            "manifest_sha256": manifest_sha256,
            "gate_report_sha256": gate_report_sha256,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build artifacts/manifests/frozen_validation_manifest.v1 for Stage A.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root containing train/metadata.jsonl and images.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Metadata JSONL. Defaults to <dataset-root>/<split>/metadata.jsonl.")
    parser.add_argument("--validation-records", type=Path, default=None, help="Existing Stage 2 gold_validation_records.jsonl to promote.")
    parser.add_argument("--force-rebuild-split", action="store_true", help="Ignore any existing gold_validation_records.jsonl and rebuild the Stage 2 split.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for manifest artifacts.")
    parser.add_argument("--artifact-version", default=ARTIFACT_VERSION, help="Artifact version basename for output files.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split that owns the validation images.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Stage 2 split seed.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Stage 2 validation ratio.")
    parser.add_argument(
        "--input-bbox-format",
        choices=("pixel_xyxy", "normalized_xyxy", "grid1000_xyxy", "pixel_xywh", "auto"),
        default="pixel_xyxy",
        help="Expected bbox format in the source validation records.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files for this artifact version.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate in memory without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    cwd = Path.cwd()

    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser()
    elif args.metadata_path is not None:
        dataset_root = args.metadata_path.expanduser().parent.parent
    else:
        dataset_root = discover_dataset_root(root, cwd, args.split)

    if args.metadata_path is not None:
        metadata_path = args.metadata_path.expanduser()
    else:
        metadata_path = metadata_path_for_root(dataset_root, args.split)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")

    output_dir = args.output_dir.expanduser() if args.output_dir else root / "artifacts" / "manifests"
    paths = output_paths(output_dir, args.artifact_version)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_outputs_can_be_written(paths, args.overwrite)

    validation_records, source_config = prepare_source_records(args, root, dataset_root, metadata_path)
    manifest, diagnostics = build_manifest_records(
        rows=validation_records,
        dataset_root=dataset_root,
        split=args.split,
        input_bbox_format=args.input_bbox_format,
    )

    gate_report = build_gate_report(
        manifest=manifest,
        diagnostics=diagnostics,
        expected_row_count=len(validation_records),
        source_config=source_config,
        paths=paths,
        artifact_version=args.artifact_version,
    )

    manifest_sha256: str | None = None
    gate_report_sha256: str | None = None
    if not args.dry_run:
        write_jsonl(paths["manifest"], manifest)
        manifest_sha256 = sha256_file(paths["manifest"])
        gate_report["outputs_sha256"] = {"manifest": manifest_sha256}
        write_json(paths["gate_report"], gate_report)
        gate_report_sha256 = sha256_file(paths["gate_report"])
        config = build_config(
            args=args,
            dataset_root=dataset_root,
            metadata_path=metadata_path,
            source_config=source_config,
            paths=paths,
            manifest_sha256=manifest_sha256,
            gate_report_sha256=gate_report_sha256,
        )
        write_json(paths["config"], config)

    status = "PASS" if gate_report["pass"] else "FAIL"
    print(f"Stage A gate: {status}")
    print(f"Dataset root: {as_display_path(dataset_root)}")
    print(f"Source split: {source_config['source_split_kind']}")
    print(f"Rows: {len(manifest)} images, {diagnostics['region_count']} regions")
    print(f"Source distribution: {diagnostics['source_distribution']}")
    if not args.dry_run:
        print(f"Manifest: {as_display_path(paths['manifest'])}")
        print(f"Config: {as_display_path(paths['config'])}")
        print(f"Gate report: {as_display_path(paths['gate_report'])}")

    if not gate_report["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
