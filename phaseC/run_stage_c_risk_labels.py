from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEXT_TYPES = {"handwritten", "printed", "formula", "table", "annotation"}
NON_TEXT_TYPES = {"image", "graph"}
DEFAULT_TEXT_CANDIDATE_TYPES = {"handwritten", "printed", "annotation"}
DEFAULT_P2_CANDIDATE_TYPES = {"formula", "table"}

RUN_ID = "b2_stage2_gold__crop-none__prompt-v1__val-v1"
CHECKPOINT_ID = "b2_stage2_gold"
PROMPT_VERSION = "prompt-v1"
ARTIFACT_VERSION = "frozen_validation_manifest.v1"
RISK_VERSION = "risk-v1"
SELECTION_VERSION = "candidate-v1"

RISK_FIELDS = [
    "image_id",
    "region_id",
    "join_kind",
    "source",
    "region_type",
    "bbox",
    "ground_truth",
    "first_pass_prediction",
    "cer",
    "cer_not_computed_reason",
    "exact_match",
    "risk_label_binary",
    "risk_label_ordinal",
    "failure_tags",
    "checkpoint_id",
    "prompt_version",
    "matched_prediction_index",
    "match_iou",
    "prediction_region_type",
    "bbox_prediction",
    "prediction_runtime_sec",
    "raw_output_id",
    "parse_ok",
    "error_type",
    "recoverable_for_crop",
]


LATEX_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\pi": "π",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\sqrt": "√",
    r"\cdot": "·",
    r"\times": "×",
    r"\div": "÷",
    r"\pm": "±",
    r"\leq": "≤",
    r"\le": "≤",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\neq": "≠",
    r"\ne": "≠",
    r"\approx": "≈",
    r"\rightarrow": "→",
    r"\to": "→",
    r"\leftarrow": "←",
    r"\angle": "∠",
    r"\perp": "⊥",
    r"\parallel": "∥",
}

LOOKALIKE_LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "I": "І",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "a": "а",
        "c": "с",
        "e": "е",
        "i": "і",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
    }
)

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


@dataclass(frozen=True)
class RiskRules:
    safe_cer: float = 0.05
    risky_cer: float = 0.15
    catastrophic_cer: float = 0.60
    iou_threshold: float = 0.50
    too_short_ratio: float = 0.50
    too_long_ratio: float = 2.00
    hallucination_ratio: float = 4.00
    min_text_len_for_ratio: int = 8
    repetition_min_repeats: int = 8
    small_area_ratio: float = 0.015
    min_candidate_gt_len: int = 3
    source_specific_cer: float = 0.30


@dataclass(frozen=True)
class CandidateRules:
    large_validation_region_threshold: int = 1000
    p0_cap: int = 500
    p1_cap: int = 500
    p2_cap: int = 100
    risky_fraction_cap: float = 0.30
    p2_fraction_cap: float = 0.10
    max_source_share_warn: float = 0.75
    max_type_share_warn: float = 0.85


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(*parts: Any, length: int = 20) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
    tmp.replace(path)


def read_prediction_csv(path: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return stable_json(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    tmp.replace(path)


def write_parquet_if_available(path: Path, rows: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if not rows:
        return False, "no rows to write"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        return False, f"pyarrow unavailable: {exc}"

    serializable_rows = []
    for row in rows:
        serializable = dict(row)
        for key in ("bbox", "bbox_prediction", "failure_tags"):
            serializable[key] = stable_json(serializable.get(key))
        serializable_rows.append(serializable)

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(serializable_rows)
    pq.write_table(table, path)
    return True, None


def normalize_fraction_once(text: str) -> str:
    return re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)


def normalize_text(text: Any, region_type: str = "handwritten") -> str:
    text = str(text or "")
    text = re.sub(r"~~([^~{}]+)~~\{([^{}]+)\}", r"\2", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(left|right)\s*([()|\[\]{}.])", r"\2", text)
    text = re.sub(r"\\[,;:!]|\\quad|\\qquad|\\hspace\{[^}]*\}", " ", text)

    for _ in range(4):
        new = normalize_fraction_once(text)
        if new == text:
            break
        text = new

    for key in sorted(LATEX_SYMBOLS, key=len, reverse=True):
        text = text.replace(key, LATEX_SYMBOLS[key])

    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√\1", text)
    text = re.sub(r"\\(bar|hat|vec|overline|widetilde)\{([^{}]+)\}", r"\2", text)
    text = re.sub(r"\^\{([^{}])\}", r"^\1", text)
    text = re.sub(r"_\{([^{}])\}", r"_\1", text)
    text = text.translate(SUPERSCRIPTS).translate(SUBSCRIPTS)
    text = re.sub(r"[\u0300-\u036f]", "", text)

    text = text.translate(LOOKALIKE_LATIN_TO_CYRILLIC)
    text = re.sub(r"[\u2010-\u2015−]", "-", text)
    text = text.translate(str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "ʼ": "'", "’": "'", "`": "'"}))
    text = text.replace("\u00a0", " ")

    if region_type == "formula":
        text = text.replace("*", "·").replace("⋅", "·").replace("∗", "·")
    if region_type == "table":
        text = re.sub(r"\s*\|\s*", "|", text)

    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(prediction: str, ground_truth: str) -> float:
    return levenshtein(prediction, ground_truth) / max(1, len(ground_truth))


def bbox_valid(bbox: Any) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def bbox_area(bbox: Any) -> float:
    if not bbox_valid(bbox):
        return 0.0
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def compute_iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def greedy_match(gt_regions: list[dict[str, Any]], pred_regions: list[dict[str, Any]], threshold: float) -> dict[int, tuple[int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(gt_regions):
        gt_bbox = gt.get("bbox")
        if not bbox_valid(gt_bbox):
            continue
        for pred_index, pred in enumerate(pred_regions):
            pred_bbox = pred.get("bbox")
            if bbox_valid(pred_bbox):
                pairs.append((compute_iou(gt_bbox, pred_bbox), gt_index, pred_index))
    pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}
    for score, gt_index, pred_index in pairs:
        if score < threshold:
            break
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches[gt_index] = (pred_index, score)
    return matches


def has_repetition(text: str, min_repeats: int) -> bool:
    clean = normalize_text(text).lower()
    if not clean:
        return False
    tokens = re.findall(r"\w+|[^\w\s]", clean, flags=re.UNICODE)
    if not tokens:
        return False
    token_counts = Counter(tokens)
    if token_counts.most_common(1)[0][1] >= min_repeats:
        return True
    for size in (2, 3, 4):
        if len(tokens) < size * min_repeats:
            continue
        ngrams = Counter(tuple(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1))
        if ngrams and ngrams.most_common(1)[0][1] >= min_repeats:
            return True
    return False


def parse_regions(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        regions = json.loads(str(row.get("regions") or "[]"))
    except json.JSONDecodeError as exc:
        return [], f"regions_json_error:{exc}"
    if not isinstance(regions, list):
        return [], "regions_not_list"
    cleaned = []
    for region in regions:
        if isinstance(region, dict):
            cleaned.append(region)
    return cleaned, None


def load_inputs(
    manifest_path: Path,
    baseline_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    predictions_path = baseline_dir / "validation_predictions.csv"
    raw_outputs_path = baseline_dir / "validation_raw_outputs.jsonl"
    score_path = baseline_dir / "validation_score.json"
    config_path = baseline_dir / "config.json"

    required = [manifest_path, predictions_path, raw_outputs_path, score_path, config_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage C inputs: " + ", ".join(missing))

    manifest_rows = read_jsonl(manifest_path)
    prediction_rows = read_prediction_csv(predictions_path)
    raw_rows = read_jsonl(raw_outputs_path)
    raw_by_id = {str(row.get("raw_output_id") or ""): row for row in raw_rows if row.get("raw_output_id")}
    score = read_json(score_path)
    baseline_config = read_json(config_path)
    return manifest_rows, prediction_rows, raw_by_id, score, baseline_config


def validate_baseline(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    score: dict[str, Any],
    allow_failed_baseline: bool,
) -> list[str]:
    errors: list[str] = []
    manifest_ids = [str(row.get("image_id") or "") for row in manifest_rows]
    prediction_ids = [str(row.get("image_id") or "") for row in prediction_rows]
    if len(set(manifest_ids)) != len(manifest_ids):
        errors.append("manifest image_id is not unique")
    if len(set(prediction_ids)) != len(prediction_ids):
        errors.append("prediction image_id is not unique")
    if set(manifest_ids) != set(prediction_ids):
        errors.append(
            f"manifest/prediction image_id mismatch: missing={len(set(manifest_ids) - set(prediction_ids))}, "
            f"extra={len(set(prediction_ids) - set(manifest_ids))}"
        )
    missing_raw = sum(1 for row in prediction_rows if str(row.get("raw_output_id") or "") not in raw_by_id)
    if missing_raw:
        errors.append(f"{missing_raw} prediction rows cannot join to raw outputs")

    gate = score.get("pass_gate") or {}
    if not allow_failed_baseline and gate and not all(bool(value) for value in gate.values()):
        errors.append(f"baseline pass_gate is not fully true: {gate}")
    return errors


def risk_label_from_tags(tags: list[str], exact_match: bool, cer_value: float | None, rules: RiskRules) -> str:
    catastrophic_tags = {
        "malformed_json",
        "empty_prediction",
        "bbox_invalid",
        "repetition",
        "hallucination_long",
    }
    if any(tag in tags for tag in catastrophic_tags):
        return "catastrophic"
    if cer_value is not None and cer_value >= rules.catastrophic_cer:
        return "catastrophic"
    if not tags and (exact_match or (cer_value is not None and cer_value <= rules.safe_cer)):
        return "safe"
    return "risky"


def append_ratio_tags(
    tags: list[str],
    gt_norm: str,
    pred_norm: str,
    rules: RiskRules,
) -> None:
    gt_len = len(gt_norm)
    pred_len = len(pred_norm)
    if gt_len < rules.min_text_len_for_ratio:
        return
    if pred_len == 0:
        return
    ratio = pred_len / max(1, gt_len)
    if ratio <= rules.too_short_ratio:
        tags.append("too_short")
    if ratio >= rules.too_long_ratio:
        tags.append("too_long")
    if ratio >= rules.hallucination_ratio:
        tags.append("hallucination_long")


def source_specific_tag(source: str, cer_value: float | None, tags: list[str], threshold: float) -> None:
    if source in {"dictation", "school", "university"} and cer_value is not None and cer_value >= threshold:
        tags.append("source_specific_error")


def dedupe_tags(tags: list[str]) -> list[str]:
    return sorted(set(tag for tag in tags if tag))


def build_risk_rows(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    rules: RiskRules,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_by_image = {str(row["image_id"]): row for row in prediction_rows}
    risk_rows: list[dict[str, Any]] = []
    match_stats = Counter()

    for manifest in manifest_rows:
        image_id = str(manifest["image_id"])
        source = str(manifest.get("source") or "unknown")
        image_area = max(1.0, float(manifest.get("image_width") or 1) * float(manifest.get("image_height") or 1))
        gt_regions = [region for region in manifest.get("regions", []) if isinstance(region, dict)]
        prediction = predictions_by_image[image_id]
        pred_regions, parse_error = parse_regions(prediction)
        parse_ok = str(prediction.get("parse_ok", "")).lower() == "true" and parse_error is None
        raw = raw_by_id.get(str(prediction.get("raw_output_id") or ""), {})
        raw_text = str(raw.get("raw_text") or "")
        image_error_type = str(prediction.get("error_type") or parse_error or "")
        raw_repetition = (not parse_ok) and has_repetition(raw_text, rules.repetition_min_repeats)

        matches = greedy_match(gt_regions, pred_regions, rules.iou_threshold) if parse_ok else {}
        matched_pred_indexes = {pred_index for pred_index, _score in matches.values()}
        match_stats["matched_gt_regions"] += len(matches)
        match_stats["missed_gt_regions"] += max(0, len(gt_regions) - len(matches))
        match_stats["false_positive_regions"] += max(0, len(pred_regions) - len(matched_pred_indexes))

        for gt_index, gt in enumerate(gt_regions):
            region_id = str(gt.get("region_id") or f"r{gt_index:04d}")
            region_type = str(gt.get("type") or "unknown").lower()
            gt_bbox = gt.get("bbox") or []
            gt_text = "" if gt.get("text") is None else str(gt.get("text"))
            matched = matches.get(gt_index)
            pred_index: int | None = None
            pred: dict[str, Any] | None = None
            match_iou: float | None = None
            if matched is not None:
                pred_index, match_iou = matched
                pred = pred_regions[pred_index]

            pred_text = "" if pred is None or pred.get("text") is None else str(pred.get("text"))
            pred_type = "" if pred is None else str(pred.get("type") or "unknown").lower()
            pred_bbox = [] if pred is None else pred.get("bbox") or []
            gt_norm = normalize_text(gt_text, region_type)
            pred_norm = normalize_text(pred_text, pred_type or region_type)
            tags: list[str] = []
            cer_value: float | None = None
            cer_reason = ""

            if not bbox_valid(gt_bbox):
                tags.append("bbox_invalid")
            if not parse_ok:
                tags.append("malformed_json")
            if raw_repetition:
                tags.append("repetition")
            if pred is None:
                tags.append("missed_detection")
            elif pred_type and pred_type != region_type:
                tags.append("invalid_region_type")

            if region_type in TEXT_TYPES:
                cer_value = cer(pred_norm, gt_norm)
                if gt_norm and not pred_norm:
                    tags.append("empty_prediction")
                if cer_value >= rules.risky_cer:
                    tags.append("high_cer")
                elif cer_value > rules.safe_cer:
                    tags.append("minor_cer")
                if cer_value >= rules.catastrophic_cer:
                    tags.append("very_high_cer")
                append_ratio_tags(tags, gt_norm, pred_norm, rules)
                if pred_norm and has_repetition(pred_norm, rules.repetition_min_repeats):
                    tags.append("repetition")
                source_specific_tag(source, cer_value, tags, rules.source_specific_cer)
            else:
                cer_reason = "non_text_region"
                if pred is None:
                    tags.append("missed_non_text_region")
                elif pred_type != region_type:
                    tags.append("invalid_region_type")

            exact_match = bool(region_type in TEXT_TYPES and pred is not None and gt_norm == pred_norm)
            tags = dedupe_tags(tags)
            ordinal = risk_label_from_tags(tags, exact_match, cer_value, rules)
            recoverable = is_recoverable_for_crop(region_type, gt_bbox, gt_norm, tags, cer_value, image_area, rules)

            risk_rows.append(
                {
                    "image_id": image_id,
                    "region_id": region_id,
                    "join_kind": "ground_truth",
                    "source": source,
                    "region_type": region_type,
                    "bbox": gt_bbox,
                    "ground_truth": gt_text,
                    "first_pass_prediction": pred_text,
                    "cer": None if cer_value is None else round(float(cer_value), 6),
                    "cer_not_computed_reason": cer_reason,
                    "exact_match": exact_match,
                    "risk_label_binary": 0 if ordinal == "safe" else 1,
                    "risk_label_ordinal": ordinal,
                    "failure_tags": tags,
                    "checkpoint_id": str(prediction.get("checkpoint_id") or CHECKPOINT_ID),
                    "prompt_version": str(prediction.get("prompt_version") or PROMPT_VERSION),
                    "matched_prediction_index": pred_index,
                    "match_iou": None if match_iou is None else round(float(match_iou), 6),
                    "prediction_region_type": pred_type,
                    "bbox_prediction": pred_bbox,
                    "prediction_runtime_sec": float(prediction.get("runtime_sec") or 0.0),
                    "raw_output_id": str(prediction.get("raw_output_id") or ""),
                    "parse_ok": parse_ok,
                    "error_type": image_error_type,
                    "recoverable_for_crop": recoverable,
                }
            )

        for pred_index, pred in enumerate(pred_regions):
            if pred_index in matched_pred_indexes:
                continue
            pred_type = str(pred.get("type") or "unknown").lower()
            pred_text = "" if pred.get("text") is None else str(pred.get("text"))
            pred_norm = normalize_text(pred_text, pred_type)
            tags = ["false_positive"]
            if pred_type not in TEXT_TYPES | NON_TEXT_TYPES:
                tags.append("invalid_region_type")
            if pred_norm and has_repetition(pred_norm, rules.repetition_min_repeats):
                tags.append("repetition")
            if len(pred_norm) >= rules.min_text_len_for_ratio * rules.hallucination_ratio:
                tags.append("hallucination_long")
            tags = dedupe_tags(tags)
            ordinal = risk_label_from_tags(tags, False, None, rules)
            risk_rows.append(
                {
                    "image_id": image_id,
                    "region_id": f"__fp_{pred_index:04d}",
                    "join_kind": "false_positive",
                    "source": source,
                    "region_type": pred_type,
                    "bbox": [],
                    "ground_truth": "",
                    "first_pass_prediction": pred_text,
                    "cer": None,
                    "cer_not_computed_reason": "false_positive_no_ground_truth",
                    "exact_match": False,
                    "risk_label_binary": 0 if ordinal == "safe" else 1,
                    "risk_label_ordinal": ordinal,
                    "failure_tags": tags,
                    "checkpoint_id": str(prediction.get("checkpoint_id") or CHECKPOINT_ID),
                    "prompt_version": str(prediction.get("prompt_version") or PROMPT_VERSION),
                    "matched_prediction_index": pred_index,
                    "match_iou": None,
                    "prediction_region_type": pred_type,
                    "bbox_prediction": pred.get("bbox") or [],
                    "prediction_runtime_sec": float(prediction.get("runtime_sec") or 0.0),
                    "raw_output_id": str(prediction.get("raw_output_id") or ""),
                    "parse_ok": parse_ok,
                    "error_type": image_error_type,
                    "recoverable_for_crop": False,
                }
            )

    return risk_rows, dict(match_stats)


def is_recoverable_for_crop(
    region_type: str,
    bbox: Any,
    gt_norm: str,
    tags: list[str],
    cer_value: float | None,
    image_area: float,
    rules: RiskRules,
) -> bool:
    if region_type not in DEFAULT_TEXT_CANDIDATE_TYPES | DEFAULT_P2_CANDIDATE_TYPES:
        return False
    if not bbox_valid(bbox):
        return False
    if len(gt_norm) < rules.min_candidate_gt_len:
        return False
    if "malformed_json" in tags or "repetition" in tags or "hallucination_long" in tags:
        return False
    if "empty_prediction" in tags or "missed_detection" in tags:
        return True
    if cer_value is not None and cer_value >= rules.risky_cer:
        return True
    area_ratio = bbox_area(bbox) / max(1.0, image_area)
    return area_ratio <= rules.small_area_ratio and cer_value is not None and cer_value > rules.safe_cer


def candidate_priority(row: dict[str, Any], rules: RiskRules) -> tuple[str | None, list[str]]:
    if row.get("join_kind") != "ground_truth":
        return None, []
    if not row.get("recoverable_for_crop"):
        return None, []
    if bool(row.get("exact_match")):
        return None, []

    region_type = str(row.get("region_type") or "")
    tags = list(row.get("failure_tags") or [])
    cer_value = row.get("cer")
    cer_float = float(cer_value) if cer_value is not None else None
    reasons = list(tags)

    bbox = row.get("bbox") or []
    if bbox_valid(bbox):
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        if width <= 220 or height <= 80:
            reasons.append("small_text")

    if region_type in DEFAULT_TEXT_CANDIDATE_TYPES:
        if "empty_prediction" in tags or "missed_detection" in tags:
            return "P0", dedupe_tags(reasons + ["recoverable_missed_text"])
        if cer_float is not None and cer_float >= rules.risky_cer:
            return "P0", dedupe_tags(reasons + ["recoverable_high_cer"])
        return "P1", dedupe_tags(reasons + ["recoverable_text_error"])

    if region_type in DEFAULT_P2_CANDIDATE_TYPES:
        if cer_float is not None and cer_float >= rules.risky_cer:
            return "P2", dedupe_tags(reasons + ["formula_or_table_probe"])

    return None, []


def rank_candidate(row: dict[str, Any]) -> tuple[int, float, int, str]:
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    cer_value = row.get("first_pass_cer")
    score = float(cer_value) if cer_value is not None else 0.0
    text_len = len(normalize_text(row.get("ground_truth") or "", str(row.get("region_type") or "handwritten")))
    return (priority_order.get(str(row.get("candidate_priority")), 9), -score, -text_len, str(row.get("candidate_id")))


def cap_candidates(candidates: list[dict[str, Any]], total_risky_regions: int, rules: CandidateRules) -> list[dict[str, Any]]:
    if total_risky_regions <= rules.large_validation_region_threshold:
        return sorted(candidates, key=rank_candidate)

    caps = {
        "P0": min(rules.p0_cap, max(1, math.ceil(total_risky_regions * rules.risky_fraction_cap))),
        "P1": min(rules.p1_cap, max(1, math.ceil(total_risky_regions * rules.risky_fraction_cap))),
        "P2": min(rules.p2_cap, max(1, math.ceil(total_risky_regions * rules.p2_fraction_cap))),
    }
    selected: list[dict[str, Any]] = []
    by_priority: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_priority[str(candidate.get("candidate_priority"))].append(candidate)
    for priority in ("P0", "P1", "P2"):
        selected.extend(sorted(by_priority.get(priority, []), key=rank_candidate)[: caps[priority]])
    return sorted(selected, key=rank_candidate)


def build_candidates(
    risk_rows: list[dict[str, Any]],
    risk_version: str,
    selection_version: str,
    rules: RiskRules,
    candidate_rules: CandidateRules,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    risky_regions = sum(1 for row in risk_rows if row.get("join_kind") == "ground_truth" and row.get("risk_label_binary") == 1)

    for row in risk_rows:
        priority, reasons = candidate_priority(row, rules)
        if priority is None:
            continue
        candidate_id = stable_id(selection_version, row["image_id"], row["region_id"], row["checkpoint_id"])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "image_id": row["image_id"],
                "region_id": row["region_id"],
                "source": row["source"],
                "region_type": row["region_type"],
                "bbox_region": row["bbox"],
                "ground_truth": row["ground_truth"],
                "first_pass_prediction": row["first_pass_prediction"],
                "first_pass_cer": row["cer"],
                "risk_reason": reasons,
                "candidate_priority": priority,
                "risk_version": risk_version,
                "selection_version": selection_version,
                "checkpoint_id": row["checkpoint_id"],
                "prompt_version": row["prompt_version"],
            }
        )

    return cap_candidates(candidates, risky_regions, candidate_rules)


def count_by(rows: list[dict[str, Any]], *keys: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = "/".join(str(row.get(item) or "unknown") for item in keys)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def count_tags(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("failure_tags") or [])
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def validate_outputs(
    manifest_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rules: RiskRules,
) -> dict[str, bool]:
    gt_region_count = sum(len(row.get("regions") or []) for row in manifest_rows)
    gt_risk_rows = [row for row in risk_rows if row.get("join_kind") == "ground_truth"]
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    candidate_keys = {(row["image_id"], row["region_id"]) for row in candidates}
    risk_keys = {(row["image_id"], row["region_id"]) for row in gt_risk_rows}
    rows_have_cer_or_reason = all(
        row.get("cer") is not None or bool(row.get("cer_not_computed_reason")) for row in risk_rows
    )

    return {
        "all_gt_regions_have_risk_rows": len(gt_risk_rows) == gt_region_count,
        "risk_rows_have_cer_or_reason": rows_have_cer_or_reason,
        "thresholds_recorded_in_config": bool(asdict(rules)),
        "candidate_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "candidates_join_risk_labels": candidate_keys.issubset(risk_keys),
        "candidate_bboxes_valid": all(bbox_valid(row.get("bbox_region")) for row in candidates),
        "candidate_exact_matches_excluded": all(bool(row.get("first_pass_cer") is None or float(row["first_pass_cer"]) > rules.safe_cer) for row in candidates),
    }


def make_paths(output_root: Path, baseline_checkpoint_id: str, risk_version: str, selection_version: str) -> dict[str, Path]:
    risk_stem = f"risk_labels__{baseline_checkpoint_id}__{risk_version}"
    candidate_stem = f"refine_candidates__{baseline_checkpoint_id}__{risk_version}"
    return {
        "risk_parquet": output_root / "risk_labels" / f"{risk_stem}.parquet",
        "risk_jsonl": output_root / "risk_labels" / f"{risk_stem}.jsonl",
        "risk_csv": output_root / "risk_labels" / f"{risk_stem}.csv",
        "risk_config": output_root / "risk_labels" / f"{risk_stem}.config.json",
        "candidates_jsonl": output_root / "candidates" / f"{candidate_stem}.jsonl",
        "candidates_config": output_root / "candidates" / f"{candidate_stem}.config.json",
        "summary": output_root / "candidates" / f"{candidate_stem}.summary.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C: build risk labels and refine candidates from locked B2 baseline cache.")
    parser.add_argument("--manifest", default="artifacts/manifests/frozen_validation_manifest.v1.jsonl")
    parser.add_argument("--baseline-dir", default=f"artifacts/baseline_predictions_cache/{RUN_ID}")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--risk-version", default=RISK_VERSION)
    parser.add_argument("--selection-version", default=SELECTION_VERSION)
    parser.add_argument("--allow-failed-baseline", action="store_true")
    parser.add_argument("--require-parquet", action="store_true", help="Fail if pyarrow is unavailable.")

    parser.add_argument("--safe-cer", type=float, default=RiskRules.safe_cer)
    parser.add_argument("--risky-cer", type=float, default=RiskRules.risky_cer)
    parser.add_argument("--catastrophic-cer", type=float, default=RiskRules.catastrophic_cer)
    parser.add_argument("--iou-threshold", type=float, default=RiskRules.iou_threshold)
    parser.add_argument("--too-short-ratio", type=float, default=RiskRules.too_short_ratio)
    parser.add_argument("--too-long-ratio", type=float, default=RiskRules.too_long_ratio)
    parser.add_argument("--hallucination-ratio", type=float, default=RiskRules.hallucination_ratio)
    parser.add_argument("--small-area-ratio", type=float, default=RiskRules.small_area_ratio)
    parser.add_argument("--min-text-len-for-ratio", type=int, default=RiskRules.min_text_len_for_ratio)
    parser.add_argument("--repetition-min-repeats", type=int, default=RiskRules.repetition_min_repeats)
    parser.add_argument("--min-candidate-gt-len", type=int, default=RiskRules.min_candidate_gt_len)
    parser.add_argument("--source-specific-cer", type=float, default=RiskRules.source_specific_cer)
    parser.add_argument("--large-validation-region-threshold", type=int, default=CandidateRules.large_validation_region_threshold)
    parser.add_argument("--p0-cap", type=int, default=CandidateRules.p0_cap)
    parser.add_argument("--p1-cap", type=int, default=CandidateRules.p1_cap)
    parser.add_argument("--p2-cap", type=int, default=CandidateRules.p2_cap)
    parser.add_argument("--risky-fraction-cap", type=float, default=CandidateRules.risky_fraction_cap)
    parser.add_argument("--p2-fraction-cap", type=float, default=CandidateRules.p2_fraction_cap)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    manifest_path = resolve_path(args.manifest, root)
    baseline_dir = resolve_path(args.baseline_dir, root)
    output_root = resolve_path(args.output_root, root)

    rules = RiskRules(
        safe_cer=args.safe_cer,
        risky_cer=args.risky_cer,
        catastrophic_cer=args.catastrophic_cer,
        iou_threshold=args.iou_threshold,
        too_short_ratio=args.too_short_ratio,
        too_long_ratio=args.too_long_ratio,
        hallucination_ratio=args.hallucination_ratio,
        small_area_ratio=args.small_area_ratio,
        min_text_len_for_ratio=args.min_text_len_for_ratio,
        repetition_min_repeats=args.repetition_min_repeats,
        min_candidate_gt_len=args.min_candidate_gt_len,
        source_specific_cer=args.source_specific_cer,
    )
    candidate_rules = CandidateRules(
        large_validation_region_threshold=args.large_validation_region_threshold,
        p0_cap=args.p0_cap,
        p1_cap=args.p1_cap,
        p2_cap=args.p2_cap,
        risky_fraction_cap=args.risky_fraction_cap,
        p2_fraction_cap=args.p2_fraction_cap,
    )

    manifest_rows, prediction_rows, raw_by_id, score, baseline_config = load_inputs(manifest_path, baseline_dir)
    baseline_errors = validate_baseline(
        manifest_rows=manifest_rows,
        prediction_rows=prediction_rows,
        raw_by_id=raw_by_id,
        score=score,
        allow_failed_baseline=args.allow_failed_baseline,
    )
    if baseline_errors:
        raise RuntimeError("Stage B baseline is not ready for Stage C: " + "; ".join(baseline_errors))

    paths = make_paths(output_root, str(baseline_config.get("checkpoint_id") or CHECKPOINT_ID), args.risk_version, args.selection_version)
    risk_rows, match_stats = build_risk_rows(manifest_rows, prediction_rows, raw_by_id, rules)
    candidates = build_candidates(risk_rows, args.risk_version, args.selection_version, rules, candidate_rules)

    parquet_written, parquet_error = write_parquet_if_available(paths["risk_parquet"], risk_rows)
    if args.require_parquet and not parquet_written:
        raise RuntimeError(f"Could not write parquet: {parquet_error}")

    write_jsonl(paths["risk_jsonl"], risk_rows)
    write_csv(paths["risk_csv"], risk_rows, RISK_FIELDS)
    write_jsonl(paths["candidates_jsonl"], candidates)

    pass_gate = validate_outputs(manifest_rows, risk_rows, candidates, rules)
    summary = {
        "created_at": utc_now_iso(),
        "stage": "Stage C - Risk Labels and Refine Candidates",
        "risk_version": args.risk_version,
        "selection_version": args.selection_version,
        "source_manifest": ARTIFACT_VERSION,
        "baseline_run_id": baseline_config.get("run_id", RUN_ID),
        "checkpoint_id": baseline_config.get("checkpoint_id", CHECKPOINT_ID),
        "prompt_version": baseline_config.get("prompt_version", PROMPT_VERSION),
        "manifest_row_count": len(manifest_rows),
        "prediction_row_count": len(prediction_rows),
        "gt_region_count": sum(len(row.get("regions") or []) for row in manifest_rows),
        "risk_row_count": len(risk_rows),
        "ground_truth_risk_row_count": sum(1 for row in risk_rows if row.get("join_kind") == "ground_truth"),
        "false_positive_risk_row_count": sum(1 for row in risk_rows if row.get("join_kind") == "false_positive"),
        "candidate_count": len(candidates),
        "baseline_score": {
            "total_score": score.get("total_score"),
            "detection_f1": score.get("detection_f1"),
            "class_acc": score.get("class_acc"),
            "region_cer": score.get("region_cer"),
            "page_cer": score.get("page_cer"),
            "parse_fail_count": score.get("parse_fail_count"),
        },
        "match_stats": match_stats,
        "risk_distribution": count_by(risk_rows, "risk_label_ordinal"),
        "risk_by_source": count_by(risk_rows, "source", "risk_label_ordinal"),
        "risk_by_type": count_by(risk_rows, "region_type", "risk_label_ordinal"),
        "top_failure_tags": count_tags(risk_rows),
        "candidate_by_priority": count_by(candidates, "candidate_priority"),
        "candidate_by_source": count_by(candidates, "source"),
        "candidate_by_type": count_by(candidates, "region_type"),
        "pass_gate": pass_gate,
        "parquet_written": parquet_written,
        "parquet_error": parquet_error,
    }

    config = {
        **summary,
        "inputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "baseline_dir": str(baseline_dir),
            "validation_predictions_sha256": sha256_file(baseline_dir / "validation_predictions.csv"),
            "validation_raw_outputs_sha256": sha256_file(baseline_dir / "validation_raw_outputs.jsonl"),
            "validation_score_sha256": sha256_file(baseline_dir / "validation_score.json"),
        },
        "risk_rules": asdict(rules),
        "candidate_rules": asdict(candidate_rules),
        "outputs": {name: str(path) for name, path in paths.items()},
    }

    write_json(paths["risk_config"], config)
    write_json(paths["candidates_config"], config)
    write_json(paths["summary"], summary)

    print("Stage C complete")
    print(f"risk rows: {summary['risk_row_count']} ({summary['ground_truth_risk_row_count']} GT, {summary['false_positive_risk_row_count']} FP)")
    print(f"candidates: {summary['candidate_count']} {summary['candidate_by_priority']}")
    print(f"risk distribution: {summary['risk_distribution']}")
    print(f"top failure tags: {dict(list(summary['top_failure_tags'].items())[:8])}")
    print(f"parquet_written: {parquet_written} {parquet_error or ''}")
    print(f"risk config: {paths['risk_config']}")
    print(f"candidates: {paths['candidates_jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
