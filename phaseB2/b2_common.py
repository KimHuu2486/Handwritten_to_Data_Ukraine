from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = {
    "image_id",
    "file_name",
    "submission_image",
    "source",
    "image_width",
    "image_height",
    "regions",
}

PREDICTION_FIELDS = [
    "image_id",
    "file_name",
    "submission_image",
    "regions",
    "parse_ok",
    "error_type",
    "raw_output_id",
    "checkpoint_id",
    "prompt_version",
    "runtime_sec",
]

VALID_REGION_TYPES = {
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path, base: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base or repo_root()) / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_runtime_config(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    defaults = {
        "run_id": "b2_stage2_gold__crop-none__prompt-v1__val-v1",
        "checkpoint_id": "b2_stage2_gold",
        "artifact_version": "frozen_validation_manifest.v1",
        "prompt_version": "prompt-v1",
        "metric_version": "official-compatible-v1",
        "crop_mode": "none",
        "manifest_path": "artifacts/manifests/frozen_validation_manifest.v1.jsonl",
        "gate_report_path": "artifacts/manifests/frozen_validation_manifest.v1.gate_report.json",
        "output_dir": "artifacts/baseline_predictions_cache/b2_stage2_gold__crop-none__prompt-v1__val-v1",
        "official_metric_notebook_path": "official-evaluation-metric-text-normalization.ipynb",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    cfg.setdefault("image_roots", [])
    cfg.setdefault("model_load", {})
    cfg.setdefault("generation_params", {})
    cfg.setdefault("runtime", {})
    cfg["generation_params"].setdefault("do_sample", False)
    cfg["generation_params"].setdefault("num_beams", 1)
    cfg["generation_params"].setdefault("max_new_tokens_page", 4096)
    cfg["generation_params"].setdefault("max_new_tokens_crop", 0)
    cfg["generation_params"].setdefault("max_pixels_page", 850000)
    cfg["generation_params"].setdefault("max_pixels_crop", 0)
    repetition_stop = cfg["generation_params"].setdefault("repetition_eos_stop", {})
    repetition_stop.setdefault("enabled", True)
    repetition_stop.setdefault("min_new_tokens", 512)
    repetition_stop.setdefault("ngram_size", 32)
    repetition_stop.setdefault("repeat_count", 4)
    repetition_stop.setdefault("check_interval", 32)
    return cfg


def validate_manifest_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    image_ids: set[str] = set()
    region_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_MANIFEST_FIELDS - set(row))
        if missing:
            errors.append(f"row {index} missing fields: {', '.join(missing)}")
        image_id = str(row.get("image_id", ""))
        if image_id in image_ids:
            errors.append(f"duplicate image_id: {image_id}")
        image_ids.add(image_id)
        regions = row.get("regions")
        if not isinstance(regions, list):
            errors.append(f"row {index} regions is not a list")
            continue
        for region_index, region in enumerate(regions):
            if not isinstance(region, dict):
                errors.append(f"row {index} region {region_index} is not an object")
                continue
            region_id = str(region.get("region_id", f"__missing_{region_index}"))
            key = (image_id, region_id)
            if key in region_keys:
                errors.append(f"duplicate region key: {image_id}/{region_id}")
            region_keys.add(key)
            bbox = region.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                errors.append(f"row {index} region {region_id} has invalid bbox")
    return errors


def load_prompt_config(adapter_path: Path) -> dict[str, Any]:
    prompt_path = adapter_path / "rukopys_prompt_config.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt config: {prompt_path}")
    data = load_json(prompt_path)
    if not data.get("page_prompt"):
        raise ValueError(f"Missing page_prompt in {prompt_path}")
    return data


def candidate_image_paths(row: dict[str, Any], image_roots: list[str]) -> list[Path]:
    file_name = str(row.get("file_name", "")).replace("\\", "/")
    basename = Path(file_name).name
    candidates: list[Path] = []
    for root_value in image_roots:
        root = resolve_project_path(root_value)
        candidates.append(root / file_name)
        candidates.append(root / basename)
    root = repo_root()
    candidates.extend(
        [
            root / file_name,
            root / "data" / "train" / file_name,
            root / "dataset" / "train" / file_name,
            root / "dataset" / "train" / basename,
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def resolve_image_path(row: dict[str, Any], image_roots: list[str]) -> Path:
    for path in candidate_image_paths(row, image_roots):
        if path.exists():
            return path
    tried = ", ".join(str(p) for p in candidate_image_paths(row, image_roots)[:6])
    raise FileNotFoundError(f"Cannot resolve image for {row.get('file_name')}; tried {tried}")


def extract_json_from_response(text: str) -> tuple[list[Any], str | None]:
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    for candidate in (clean,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed, None
            if isinstance(parsed, dict):
                for key in ("regions", "results", "data"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return value, None
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
    pattern = r'\{[^{}]*"bbox"[^{}]*"type"[^{}]*"text"[^{}]*\}'
    for match in re.finditer(pattern, text):
        try:
            objects.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    if objects:
        return objects, None
    return [], "json_parse_failed"


def _to_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def normalize_regions(
    raw_regions: list[Any],
    image_width: int,
    image_height: int,
    max_pixels_page: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(raw_regions, list):
        return [], "regions_not_list"

    total_pixels = max(image_width * image_height, 1)
    if total_pixels > max_pixels_page > 0:
        scale_down = math.sqrt(max_pixels_page / total_pixels)
        resized_w = max(int(image_width * scale_down), 1)
        resized_h = max(int(image_height * scale_down), 1)
    else:
        resized_w, resized_h = image_width, image_height

    scale_x = image_width / max(resized_w, 1)
    scale_y = image_height / max(resized_h, 1)
    normalized: list[dict[str, Any]] = []

    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        bbox = _to_float_list(item.get("bbox"))
        if bbox is None:
            continue

        max_val = max(bbox)
        if max_val <= 1.0 and all(isinstance(v, float) for v in bbox):
            x1 = bbox[0] * image_width
            y1 = bbox[1] * image_height
            x2 = bbox[2] * image_width
            y2 = bbox[3] * image_height
        elif max_val > 1005:
            x1 = bbox[0] * scale_x
            y1 = bbox[1] * scale_y
            x2 = bbox[2] * scale_x
            y2 = bbox[3] * scale_y
        else:
            x1 = bbox[0] / 1000.0 * image_width
            y1 = bbox[1] / 1000.0 * image_height
            x2 = bbox[2] / 1000.0 * image_width
            y2 = bbox[3] / 1000.0 * image_height

        x1 = int(max(0, min(image_width, round(x1))))
        y1 = int(max(0, min(image_height, round(y1))))
        x2 = int(max(0, min(image_width, round(x2))))
        y2 = int(max(0, min(image_height, round(y2))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        region_type = str(item.get("type", "handwritten")).strip().lower()
        if region_type not in VALID_REGION_TYPES:
            region_type = "handwritten"

        normalized.append(
            {
                "bbox": [x1, y1, x2, y2],
                "type": region_type,
                "text": "" if item.get("text") is None else str(item.get("text")),
            }
        )

    normalized.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    return normalized, None


def manifest_solution_dataframe(rows: list[dict[str, Any]]):
    import pandas as pd

    records = []
    for row in rows:
        gt_regions = []
        for region in row.get("regions", []):
            gt_regions.append(
                {
                    "bbox": region.get("bbox", [0, 0, 0, 0]),
                    "type": region.get("type", "handwritten"),
                    "text": "" if region.get("text") is None else str(region.get("text")),
                }
            )
        records.append(
            {
                "image": row["submission_image"],
                "regions": json.dumps(gt_regions, ensure_ascii=False),
            }
        )
    return pd.DataFrame(records)


def predictions_submission_dataframe(prediction_rows: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "image": row["submission_image"],
                "regions": row["regions"],
            }
            for row in prediction_rows
        ]
    )


def ensure_metric_module(notebook_path: Path, metric_path: Path):
    if not metric_path.exists():
        if not notebook_path.exists():
            raise FileNotFoundError(
                f"Missing {metric_path} and cannot extract it because {notebook_path} does not exist"
            )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metric_code: str | None = None
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            if "%%writefile kaggle_metric.py" in source and "def score_detailed" in source:
                lines = source.splitlines()
                if lines and lines[0].startswith("%%writefile"):
                    lines = lines[1:]
                metric_code = "\n".join(lines).rstrip() + "\n"
                break
        if metric_code is None:
            raise ValueError(f"Could not find kaggle_metric.py cell in {notebook_path}")
        metric_path.write_text(metric_code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("phaseB2_kaggle_metric", metric_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import metric module from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_predictions(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    runtime_total_sec: float,
    runtime_per_page_sec: float,
    parse_fail_count: int,
    metric_version: str,
    metric_notebook_path: Path,
    metric_path: Path,
) -> dict[str, Any]:
    metric = ensure_metric_module(metric_notebook_path, metric_path)
    solution_df = manifest_solution_dataframe(manifest_rows)
    submission_df = predictions_submission_dataframe(prediction_rows)
    detailed = metric.score_detailed(solution_df, submission_df, "image")
    total_score = detailed.get("total_score", detailed.get("composite_score"))
    if total_score is None:
        raise KeyError(f"Metric result has no total_score/composite_score key: {sorted(detailed)}")
    return {
        "total_score": float(total_score),
        "detection_f1": float(detailed["detection_f1"]),
        "class_acc": float(detailed["classification_accuracy"]),
        "region_cer": float(detailed["region_cer"]),
        "page_cer": float(detailed["page_cer"]),
        "row_count": len(prediction_rows),
        "parse_fail_count": parse_fail_count,
        "runtime_total_sec": runtime_total_sec,
        "runtime_per_page_sec": runtime_per_page_sec,
        "metric_version": metric_version,
        "metric_details": {
            "detection_precision": float(detailed.get("detection_precision", 0.0)),
            "detection_recall": float(detailed.get("detection_recall", 0.0)),
            "matched_regions": int(detailed.get("matched_regions", detailed.get("n_matched_regions", 0))),
            "false_positives": int(detailed.get("n_false_positives", 0)),
            "false_negatives": int(detailed.get("n_false_negatives", 0)),
            "n_images": int(detailed.get("n_images", len(prediction_rows))),
        },
    }


def read_existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PREDICTION_FIELDS})
    tmp_path.replace(path)
