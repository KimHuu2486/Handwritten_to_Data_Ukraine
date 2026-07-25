#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-phase fine-tuning for Kansallisarkisto/cyrillic-htr-model on the
RUKOPYS HPA branch: handwritten, printed, annotation.

Expected dataset layout:

DATA_ROOT/
  train/
    metadata.jsonl
    images/
      ...

This script:
  1. Reads DATA_ROOT/train/metadata.jsonl.
  2. Keeps only HPA labels: handwritten, printed, annotation.
  3. Builds/reuses a fixed stratified train/val split.
  4. Fine-tunes once from Kansallisarkisto/cyrillic-htr-model or RESUME_MODEL_DIR.
  5. Selects the best model by validation CER through HuggingFace Trainer.
  6. Runs typed generation evaluation on the validation split.
  7. Saves the final model and reports.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from jiwer import cer, wer
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
    set_seed,
)

ImageFile.LOAD_TRUNCATED_IMAGES = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

TARGET_LABELS = ("handwritten", "printed", "annotation")

MODEL_NAME = "Kansallisarkisto/cyrillic-htr-model"
BASE_PROCESSOR_MODEL = "microsoft/trocr-base-handwritten"
RUN_NAME = "single_train_finetune"

# =============================================================================
# CONFIG: edit these values before running on Linux/Thunder Compute.
# =============================================================================

# Optional .env file for HF_TOKEN and paths. You can also set HPA_ENV_FILE.
ENV_FILE: Optional[Path] = None
HF_TOKEN_ENV = "HF_TOKEN"

# Expected:
#   DATA_ROOT/train/metadata.jsonl
#   DATA_ROOT/train/images/...
DATA_ROOT = Path("/home/ubuntu/dataset")
TRAIN_DIR: Optional[Path] = None        # None -> DATA_ROOT / "train"
OUTPUT_DIR = Path("/home/ubuntu/outputs_hpa_single")
FINAL_DIR: Optional[Path] = None        # None -> OUTPUT_DIR / "final_cyrillic_htr_model"
HF_CACHE_DIR: Optional[Path] = Path("/home/ubuntu/hf_cache")
RESUME_MODEL_DIR: Optional[Path] = None # Local model/checkpoint dir, or None to start from MODEL_NAME.

SEED = 42
VAL_RATIO = 0.10
REBUILD_SPLITS = False

MAX_TARGET_LENGTH = 192
TRAIN_EPOCHS = 10.0
TRAIN_LR = 1e-5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
DATALOADER_NUM_WORKERS = 2
LOGGING_STEPS = 50

# Early stopping needs checkpoint saving + load_best_model_at_end.
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT: Optional[int] = 2
EARLY_STOPPING_PATIENCE = 2

# AdamW optimizer for HuggingFace Trainer.
OPTIM = "adamw_torch"
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"
REPORT_TO = "none"
NO_FP16 = False
BF16 = False
GRADIENT_CHECKPOINTING = True

# 0 means use all train split. Set >0 for smoke test/debug, per label.
TRAIN_MAX_PER_CLASS = 0

# Typed validation generation after training.
SKIP_TYPED_EVAL = False
TYPED_EVAL_BATCH_SIZE = 8

# Fail fast on bad mounts/data.
FAIL_ON_MISSING_IMAGE = True
FAIL_ON_IMAGE_OPEN_ERROR = True
VERIFY_IMAGE_OPEN_IN_REPORT = False

# Keep training labels close to raw labels. The official competition normalizer
# should be used for final leaderboard scoring; this is a lightweight text cleanup
# used for train/val labels and local CER/WER reports.
COLLAPSE_WHITESPACE = False

# Ukrainian is not Russian: keep these characters in the tokenizer explicitly.
UKRAINIAN_EXTRA_TOKENS = ["і", "ї", "є", "ґ", "І", "Ї", "Є", "Ґ"]

# Typed generation budgets: annotation should not use the same long generation
# budget as handwritten text lines.
GEN_CONFIG_BY_TYPE: Dict[str, Dict[str, Any]] = {
    "handwritten": {
        "num_beams": 3,
        "max_new_tokens": 192,
        "length_penalty": 1.0,
        "early_stopping": True,
    },
    "printed": {
        "num_beams": 3,
        "max_new_tokens": 128,
        "length_penalty": 1.0,
        "early_stopping": True,
    },
    "annotation": {
        "num_beams": 1,
        "max_new_tokens": 16,
        "length_penalty": 0.8,
        "early_stopping": True,
    },
}


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def env_int(names: Sequence[str], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def env_float(names: Sequence[str], default: float) -> float:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def env_bool(names: Sequence[str], default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return str_to_bool(value)
    return default


def build_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_name=MODEL_NAME,
        base_processor_model=BASE_PROCESSOR_MODEL,
        run_name=RUN_NAME,
        env_file=ENV_FILE,
        hf_token_env=HF_TOKEN_ENV,
        hf_cache_dir=HF_CACHE_DIR,
        data_root=DATA_ROOT,
        train_dir=TRAIN_DIR,
        output_dir=OUTPUT_DIR,
        final_dir=FINAL_DIR,
        resume_model_dir=RESUME_MODEL_DIR,
        seed=SEED,
        val_ratio=VAL_RATIO,
        rebuild_splits=REBUILD_SPLITS,
        max_target_length=MAX_TARGET_LENGTH,
        train_epochs=TRAIN_EPOCHS,
        train_lr=TRAIN_LR,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to=REPORT_TO,
        no_fp16=NO_FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        train_max_per_class=TRAIN_MAX_PER_CLASS,
        skip_typed_eval=SKIP_TYPED_EVAL,
        typed_eval_batch_size=TYPED_EVAL_BATCH_SIZE,
        fail_on_missing_image=FAIL_ON_MISSING_IMAGE,
        fail_on_image_open_error=FAIL_ON_IMAGE_OPEN_ERROR,
        verify_image_open_in_report=VERIFY_IMAGE_OPEN_IN_REPORT,
        collapse_whitespace=COLLAPSE_WHITESPACE,
    )


def load_optional_env_file(env_file: Optional[Path]) -> None:
    if env_file is None:
        value = os.environ.get("HPA_ENV_FILE")
        env_file = Path(value) if value else None
    if env_file is None:
        return
    if not env_file.exists():
        raise FileNotFoundError(f"ENV_FILE does not exist: {env_file}")

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        with env_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    print(f"Loaded env file: {env_file}")


def path_from_env(
    config_value: Optional[Path],
    env_names: Sequence[str],
    default: Optional[Path],
) -> Optional[Path]:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return Path(value)
    if config_value is not None:
        return config_value
    return default


def finalize_paths(args: SimpleNamespace) -> None:
    args.data_root = path_from_env(
        args.data_root,
        ("HPA_DATA_ROOT", "DATA_ROOT"),
        REPO_ROOT / "dataset",
    )
    args.train_dir = path_from_env(
        args.train_dir,
        ("HPA_TRAIN_DIR",),
        args.data_root / "train",
    )
    args.output_dir = path_from_env(
        args.output_dir,
        ("HPA_OUTPUT_DIR", "OUTPUT_DIR"),
        SCRIPT_DIR / "outputs_kansallisarkisto_hpa_single",
    )
    args.final_dir = path_from_env(
        args.final_dir,
        ("HPA_FINAL_DIR",),
        args.output_dir / "final_cyrillic_htr_model",
    )
    args.hf_cache_dir = path_from_env(
        args.hf_cache_dir,
        ("HF_CACHE_DIR", "HF_HOME"),
        None,
    )
    args.resume_model_dir = path_from_env(
        args.resume_model_dir,
        ("HPA_RESUME_MODEL_DIR", "RESUME_MODEL_DIR"),
        None,
    )

    # Optional environment overrides for common debug runs.
    args.val_ratio = env_float(("HPA_VAL_RATIO", "VAL_RATIO"), args.val_ratio)
    args.rebuild_splits = env_bool(("HPA_REBUILD_SPLITS", "REBUILD_SPLITS"), args.rebuild_splits)
    args.train_max_per_class = env_int(("HPA_TRAIN_MAX_PER_CLASS", "TRAIN_MAX_PER_CLASS"), args.train_max_per_class)
    args.train_epochs = env_float(("HPA_TRAIN_EPOCHS", "TRAIN_EPOCHS"), args.train_epochs)
    args.train_lr = env_float(("HPA_TRAIN_LR", "TRAIN_LR"), args.train_lr)
    args.train_batch_size = env_int(("HPA_TRAIN_BATCH_SIZE", "TRAIN_BATCH_SIZE"), args.train_batch_size)
    args.eval_batch_size = env_int(("HPA_EVAL_BATCH_SIZE", "EVAL_BATCH_SIZE"), args.eval_batch_size)
    args.bf16 = env_bool(("HPA_BF16", "BF16"), args.bf16)

    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError(f"VAL_RATIO must be between 0 and 1. Got: {args.val_ratio}")
    if args.save_strategy == "no":
        raise ValueError(
            "SAVE_STRATEGY='no' disables load_best_model_at_end and breaks early stopping. "
            "Use SAVE_STRATEGY='epoch'."
        )


def resolve_hf_token(args: SimpleNamespace) -> Optional[str]:
    env_names = [
        args.hf_token_env,
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    for env_name in env_names:
        token = os.environ.get(env_name)
        if token:
            print(f"HF token: found in ${env_name}")
            return token
    print("HF token: not set. Public model downloads may still work.")
    return None


def from_pretrained_hf(loader, name: str, hf_token: Optional[str], **kwargs):
    if hf_token:
        try:
            return loader.from_pretrained(name, token=hf_token, **kwargs)
        except TypeError:
            return loader.from_pretrained(name, use_auth_token=hf_token, **kwargs)
    return loader.from_pretrained(name, **kwargs)


def normalize_text(text: Any, collapse_whitespace: bool = COLLAPSE_WHITESPACE) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ").strip()
    if collapse_whitespace:
        text = " ".join(text.split())
    return text


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip bad JSON {path}:{line_no}: {exc}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_record_label(item: Dict[str, Any]) -> str:
    return str(item.get("label", item.get("type", "")))


def get_record_image_value(item: Dict[str, Any]) -> Optional[str]:
    for key in ("image", "file_name", "filename", "path", "image_path"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def resolve_image_path(split_dir: Path, image_value: str) -> Tuple[Path, List[str]]:
    """Resolve metadata image value against train dir and train/images dir."""
    raw = Path(image_value)
    if raw.is_absolute():
        return raw, [str(raw)]

    candidates: List[Path] = []
    candidates.append(split_dir / raw)
    if not raw.parts or raw.parts[0] != "images":
        candidates.append(split_dir / "images" / raw)
    if raw.name != str(raw):
        candidates.append(split_dir / "images" / raw.name)

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return candidate, [str(x) for x in deduped]
    # Prefer the common DATA_ROOT/train/images/... layout in error messages.
    preferred = deduped[1] if len(deduped) > 1 else deduped[0]
    return preferred, [str(x) for x in deduped]


def compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": record["image"],
        "label": record["label"],
        "text": record["text"],
        "source_split": "train",
    }


def hydrate_record(row: Dict[str, Any], train_dir: Path) -> Dict[str, Any]:
    image = get_record_image_value(row)
    if not image:
        raise ValueError(f"Bad split row without image/file_name/path: {row}")
    image_path, attempts = resolve_image_path(train_dir, image)
    return {
        "image": image,
        "image_path": str(image_path),
        "image_path_attempts": attempts,
        "label": get_record_label(row),
        "text": normalize_text(row.get("text", "")),
        "source_split": "train",
    }


def load_metadata_records(split_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing train metadata: {metadata_path}. Expected DATA_ROOT/train/metadata.jsonl."
        )

    records: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "source_split": "train",
        "total_rows": 0,
        "bad_json_rows": 0,
        "kept_rows": 0,
        "skipped_non_hpa_label": 0,
        "skipped_missing_image_field": 0,
        "skipped_empty_text": 0,
        "raw_label_counts": {},
        "kept_label_counts": {},
        "missing_image_files": 0,
        "missing_image_examples": [],
    }
    raw_label_counts: Counter[str] = Counter()
    kept_label_counts: Counter[str] = Counter()

    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            report["total_rows"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                report["bad_json_rows"] += 1
                if report["bad_json_rows"] <= 5:
                    print(f"[WARN] Bad JSON {metadata_path}:{line_no}: {exc}")
                continue

            label = get_record_label(item)
            raw_label_counts[label] += 1
            if label not in TARGET_LABELS:
                report["skipped_non_hpa_label"] += 1
                continue

            image = get_record_image_value(item)
            if not image:
                report["skipped_missing_image_field"] += 1
                continue

            text = normalize_text(item.get("text", ""))
            if not text:
                report["skipped_empty_text"] += 1
                continue

            image_path, attempts = resolve_image_path(split_dir, image)
            if not image_path.exists():
                report["missing_image_files"] += 1
                if len(report["missing_image_examples"]) < 20:
                    report["missing_image_examples"].append(
                        {
                            "line_no": line_no,
                            "image": image,
                            "resolved_image_path": str(image_path),
                            "attempts": attempts,
                        }
                    )

            kept_label_counts[label] += 1
            records.append(
                {
                    "image": image,
                    "image_path": str(image_path),
                    "image_path_attempts": attempts,
                    "label": label,
                    "text": text,
                    "source_split": "train",
                }
            )

    report["kept_rows"] = len(records)
    report["raw_label_counts"] = dict(sorted(raw_label_counts.items()))
    report["kept_label_counts"] = {label: kept_label_counts.get(label, 0) for label in TARGET_LABELS}
    return records, report


def count_by_label(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(r["label"] for r in records)
    return {label: counts.get(label, 0) for label in TARGET_LABELS}


def print_counts(name: str, records: Sequence[Dict[str, Any]]) -> None:
    print(f"{name}: {len(records):,} samples | {count_by_label(records)}")


def summarize_numeric(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(p: float) -> int:
        index = int((len(ordered) - 1) * p)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def add_truncation_summary(section: Dict[str, Any], lengths: Sequence[int], max_target_length: int) -> None:
    truncated = sum(length > max_target_length for length in lengths)
    section["token_lengths"] = summarize_numeric(lengths)
    section["num_truncated"] = truncated
    section["truncated_rate"] = truncated / max(1, len(lengths))


def build_records_report(
    name: str,
    records: Sequence[Dict[str, Any]],
    processor: Optional[TrOCRProcessor],
    max_target_length: int,
    verify_image_open: bool,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "name": name,
        "num_samples": len(records),
        "label_counts": count_by_label(records),
        "missing_image_files": 0,
        "missing_image_examples": [],
        "image_open_errors": 0,
        "image_open_error_examples": [],
        "text_char_lengths": {},
        "tokenizer_max_target_length": max_target_length,
        "normalization": {
            "nbsp_to_space": True,
            "strip_edges": True,
            "collapse_whitespace": COLLAPSE_WHITESPACE,
        },
        "overall": {},
        "by_label": {},
    }

    char_lengths: List[int] = []
    char_lengths_by_label: Dict[str, List[int]] = {label: [] for label in TARGET_LABELS}

    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.exists():
            report["missing_image_files"] += 1
            if len(report["missing_image_examples"]) < 20:
                report["missing_image_examples"].append(
                    {
                        "image": record.get("image"),
                        "image_path": str(image_path),
                        "attempts": record.get("image_path_attempts", []),
                    }
                )
        elif verify_image_open:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                report["image_open_errors"] += 1
                if len(report["image_open_error_examples"]) < 20:
                    report["image_open_error_examples"].append(
                        {"image_path": str(image_path), "error": str(exc)}
                    )

        text_len = len(record["text"])
        char_lengths.append(text_len)
        char_lengths_by_label[record["label"]].append(text_len)

    report["text_char_lengths"]["overall"] = summarize_numeric(char_lengths)
    report["text_char_lengths"]["by_label"] = {
        label: summarize_numeric(char_lengths_by_label[label]) for label in TARGET_LABELS
    }

    if processor is not None:
        tokenizer = processor.tokenizer
        token_lengths: List[int] = []
        token_lengths_by_label: Dict[str, List[int]] = {label: [] for label in TARGET_LABELS}
        batch_size = 512
        for start in tqdm(range(0, len(records), batch_size), desc=f"token length report {name}"):
            batch = records[start : start + batch_size]
            encoded = tokenizer(
                [r["text"] for r in batch],
                add_special_tokens=True,
                truncation=False,
            )
            for record, input_ids in zip(batch, encoded["input_ids"]):
                length = len(input_ids)
                token_lengths.append(length)
                token_lengths_by_label[record["label"]].append(length)

        add_truncation_summary(report["overall"], token_lengths, max_target_length)
        for label in TARGET_LABELS:
            report["by_label"][label] = {}
            add_truncation_summary(report["by_label"][label], token_lengths_by_label[label], max_target_length)

    return report


def save_report(args: SimpleNamespace, name: str, report: Dict[str, Any]) -> None:
    save_json(args.reports_dir / f"{name}.json", report)
    if report.get("missing_image_files", 0) and args.fail_on_missing_image:
        raise FileNotFoundError(
            f"{name}: {report['missing_image_files']} image files are missing. "
            f"Examples: {report.get('missing_image_examples', [])[:3]}"
        )
    if report.get("image_open_errors", 0) and args.fail_on_image_open_error:
        raise RuntimeError(
            f"{name}: {report['image_open_errors']} images could not be opened. "
            f"Examples: {report.get('image_open_error_examples', [])[:3]}"
        )


def stratified_split(
    records: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    for label in TARGET_LABELS:
        rows = list(by_label.get(label, []))
        rng.shuffle(rows)
        if len(rows) <= 1:
            train_rows.extend(rows)
            continue
        n_val = max(1, round(len(rows) * val_ratio))
        n_val = min(n_val, len(rows) - 1)
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def cap_per_class(records: Sequence[Dict[str, Any]], max_per_class: int, seed: int) -> List[Dict[str, Any]]:
    if max_per_class <= 0:
        return list(records)
    rng = random.Random(seed)
    capped: List[Dict[str, Any]] = []
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    for label in TARGET_LABELS:
        rows = list(by_label.get(label, []))
        rng.shuffle(rows)
        capped.extend(rows[:max_per_class])
    rng.shuffle(capped)
    return capped


def validate_non_empty_splits(train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]]) -> None:
    if not train_rows:
        raise RuntimeError("After filtering HPA samples, train split is empty.")
    if not val_rows:
        raise RuntimeError(
            "Validation split is empty. Increase data size or adjust VAL_RATIO. "
            "Each label with only one sample is kept in train only."
        )


def prepare_fixed_splits(args: SimpleNamespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_split_path = args.splits_dir / "train_fixed_hpa.jsonl"
    val_split_path = args.splits_dir / "val_fixed_hpa.jsonl"

    if train_split_path.exists() and val_split_path.exists() and not args.rebuild_splits:
        print(f"Reusing fixed split: {args.splits_dir}")
        train_rows = [hydrate_record(r, args.train_dir) for r in read_jsonl(train_split_path)]
        val_rows = [hydrate_record(r, args.train_dir) for r in read_jsonl(val_split_path)]
    else:
        print("Building fixed stratified train/val split from train/metadata.jsonl...")
        all_records, load_report = load_metadata_records(args.train_dir)
        save_report(args, "train_metadata_load_report", load_report)
        if not all_records:
            raise RuntimeError(
                "After filtering HPA labels, no samples remain. Check labels/type and text fields in metadata.jsonl."
            )
        train_rows, val_rows = stratified_split(all_records, args.val_ratio, args.seed)
        validate_non_empty_splits(train_rows, val_rows)
        write_jsonl(train_split_path, (compact_record(r) for r in train_rows))
        write_jsonl(val_split_path, (compact_record(r) for r in val_rows))
        print(f"Saved train_fixed: {train_split_path}")
        print(f"Saved val_fixed  : {val_split_path}")

    validate_non_empty_splits(train_rows, val_rows)
    print_counts("train_fixed", train_rows)
    print_counts("val_fixed", val_rows)
    return train_rows, val_rows


def open_rgb_or_raise(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        if FAIL_ON_IMAGE_OPEN_ERROR:
            raise RuntimeError(f"Cannot open image: {image_path}") from exc
        raise


class HpaDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        processor: TrOCRProcessor,
        max_target_length: int,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = open_rgb_or_raise(record["image_path"])

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        encoding = self.processor.tokenizer(
            record["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = [
            token_id if mask == 1 else -100
            for token_id, mask in zip(encoding["input_ids"], encoding["attention_mask"])
        ]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_processor(
    model_name: str,
    base_processor_model: str,
    hf_token: Optional[str],
    cache_dir: Optional[Path],
) -> TrOCRProcessor:
    errors: List[str] = []
    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}

    for kwargs in ({}, {"subfolder": "processor"}):
        try:
            processor = from_pretrained_hf(
                TrOCRProcessor,
                model_name,
                hf_token,
                **cache_kwargs,
                **kwargs,
            )
            print(f"Loaded processor from {model_name} with kwargs={kwargs}")
            return processor
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")

    print("[WARN] Could not load full TrOCRProcessor from model repo.")
    for error in errors:
        print(f"       {error}")
    print(f"Falling back to image processor from {base_processor_model}")
    image_processor = from_pretrained_hf(
        ViTImageProcessor,
        base_processor_model,
        hf_token,
        **cache_kwargs,
    )

    tokenizer = None
    tokenizer_errors: List[str] = []
    for kwargs in ({}, {"subfolder": "processor"}):
        try:
            tokenizer = from_pretrained_hf(
                AutoTokenizer,
                model_name,
                hf_token,
                **cache_kwargs,
                **kwargs,
            )
            print(f"Loaded tokenizer from {model_name} with kwargs={kwargs}")
            break
        except Exception as exc:
            tokenizer_errors.append(f"{kwargs}: {exc}")

    if tokenizer is None:
        print("[WARN] Could not load tokenizer from model repo.")
        for error in tokenizer_errors:
            print(f"       {error}")
        print(f"Falling back to tokenizer from {base_processor_model}")
        tokenizer = from_pretrained_hf(
            AutoTokenizer,
            base_processor_model,
            hf_token,
            **cache_kwargs,
        )

    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def find_processor_source_for_resume(resume_dir: Path) -> Optional[Path]:
    for candidate in [resume_dir, *resume_dir.parents]:
        if (candidate / "preprocessor_config.json").exists():
            return candidate
    return None


def load_processor_for_resume(
    resume_dir: Path,
    base_processor_model: str,
    hf_token: Optional[str],
    cache_dir: Optional[Path],
) -> TrOCRProcessor:
    try:
        processor = TrOCRProcessor.from_pretrained(resume_dir)
        print(f"Loaded full processor from resume dir: {resume_dir}")
        return processor
    except Exception as exc:
        print(f"[WARN] Could not load full processor from resume dir: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(str(resume_dir))
    processor_source = find_processor_source_for_resume(resume_dir)
    if processor_source is not None:
        print(f"Loaded tokenizer from resume dir: {resume_dir}")
        print(f"Loading image processor from nearest artifact: {processor_source}")
        image_processor = ViTImageProcessor.from_pretrained(processor_source)
    else:
        print(f"Loaded tokenizer from resume dir: {resume_dir}")
        print(f"Loading image processor fallback from: {base_processor_model}")
        cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        image_processor = from_pretrained_hf(
            ViTImageProcessor,
            base_processor_model,
            hf_token,
            **cache_kwargs,
        )
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def resize_decoder_embeddings_if_needed(model: VisionEncoderDecoderModel, tokenizer) -> None:
    new_size = len(tokenizer)
    old_size = model.decoder.get_input_embeddings().weight.shape[0]
    if new_size != old_size:
        print(f"Resizing decoder embeddings: {old_size} -> {new_size}")
        model.decoder.resize_token_embeddings(new_size)
    model.config.vocab_size = new_size
    if hasattr(model.config, "decoder"):
        model.config.decoder.vocab_size = new_size


def reset_generation_config(
    model: VisionEncoderDecoderModel,
    tokenizer,
    decoder_start_token_id: int,
) -> None:
    model.generation_config = GenerationConfig(
        decoder_start_token_id=decoder_start_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model_config_generation_defaults = {
        "max_length": 20,
        "early_stopping": False,
        "num_beams": 1,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }
    for key, value in model_config_generation_defaults.items():
        if hasattr(model.config, key):
            setattr(model.config, key, value)


def configure_tokenizer_and_model(processor: TrOCRProcessor, model: VisionEncoderDecoderModel) -> None:
    tokenizer = processor.tokenizer
    changed_vocab = False

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        changed_vocab = True
        print("Added missing [PAD] token.")
    elif tokenizer.pad_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != tokenizer.bos_token_id:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Using EOS token as PAD: id={tokenizer.pad_token_id}")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            changed_vocab = True
            print("Added [PAD] token because pad_token_id == bos_token_id.")

    vocab = tokenizer.get_vocab()
    missing_ukrainian = [token for token in UKRAINIAN_EXTRA_TOKENS if token not in vocab]
    if missing_ukrainian:
        added = tokenizer.add_tokens(missing_ukrainian)
        changed_vocab = changed_vocab or added > 0
        print(f"Added Ukrainian tokens: {missing_ukrainian} (added={added})")
    else:
        print("Ukrainian-specific tokens already present in tokenizer vocab.")

    resize_decoder_embeddings_if_needed(model, tokenizer)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.cls_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no bos_token_id or cls_token_id for decoder start.")

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer still has no pad_token_id after pad_token fix.")
    if tokenizer.eos_token_id is None:
        print("[WARN] Tokenizer has no eos_token_id. Generation may rely on max length only.")

    model.config.decoder_start_token_id = bos_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model.config, "decoder"):
        model.config.decoder.decoder_start_token_id = bos_id
        model.config.decoder.pad_token_id = tokenizer.pad_token_id
        model.config.decoder.eos_token_id = tokenizer.eos_token_id

    reset_generation_config(model, tokenizer, bos_id)

    print(
        "Token IDs | "
        f"bos={bos_id} pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id} "
        f"vocab={len(tokenizer)} changed_vocab={changed_vocab}"
    )


def print_ukrainian_tokenization(processor: TrOCRProcessor) -> None:
    tokenizer = processor.tokenizer
    print("Ukrainian tokenization check:")
    for token in UKRAINIAN_EXTRA_TOKENS:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  {token!r}: ids={ids} decoded={decoded!r}")


def build_compute_metrics(processor: TrOCRProcessor):
    tokenizer = processor.tokenizer

    def compute_metrics(eval_pred) -> Dict[str, float]:
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_ids = pred_ids.copy()
        label_ids = eval_pred.label_ids.copy()

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

        vocab_size = len(tokenizer)
        pred_ids[(pred_ids < 0) | (pred_ids >= vocab_size)] = pad_id
        label_ids[(label_ids < 0) | (label_ids >= vocab_size)] = pad_id

        pred_str = [normalize_text(x) for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        label_str = [normalize_text(x) for x in processor.batch_decode(label_ids, skip_special_tokens=True)]
        exact = sum(p == r for p, r in zip(pred_str, label_str)) / max(1, len(label_str))
        return {
            "cer": round(cer(label_str, pred_str), 6),
            "wer": round(wer(label_str, pred_str), 6),
            "exact_match": round(exact, 6),
        }

    return compute_metrics


def filter_training_args_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    valid_keys = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_keys}
    dropped = sorted(set(kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"[WARN] Dropped unsupported TrainingArguments keys: {dropped}")
    return filtered_kwargs


def build_training_args(args: SimpleNamespace) -> Seq2SeqTrainingArguments:
    valid_keys = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in valid_keys else "evaluation_strategy"

    fp16 = torch.cuda.is_available() and not args.no_fp16 and not args.bf16
    bf16 = args.bf16

    kwargs: Dict[str, Any] = {
        "output_dir": str(args.trainer_dir),
        "run_name": args.run_name,
        "learning_rate": args.train_lr,
        "num_train_epochs": args.train_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "save_strategy": args.save_strategy,
        strategy_key: "epoch",
        "logging_steps": args.logging_steps,
        "report_to": args.report_to,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "generation_num_beams": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": True,
    }

    if args.dataloader_num_workers >= 0:
        kwargs["dataloader_num_workers"] = args.dataloader_num_workers
        if args.dataloader_num_workers > 0:
            kwargs["dataloader_prefetch_factor"] = 2

    filtered = filter_training_args_kwargs(kwargs)

    critical_keys = ["load_best_model_at_end", "metric_for_best_model", "greater_is_better", "save_strategy", strategy_key]
    missing_critical = [key for key in critical_keys if key not in filtered]
    if missing_critical:
        raise RuntimeError(
            "This Transformers version does not support critical TrainingArguments keys needed "
            f"for reliable early stopping/best-model selection: {missing_critical}"
        )

    return Seq2SeqTrainingArguments(**filtered)


def build_trainer(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: Seq2SeqTrainingArguments,
    patience: int,
) -> Seq2SeqTrainer:
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": default_data_collator,
        "compute_metrics": build_compute_metrics(processor),
        "callbacks": callbacks,
    }
    try:
        return Seq2SeqTrainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        return Seq2SeqTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)


def train_with_resume_compat(trainer: Seq2SeqTrainer, resume_from_checkpoint: Optional[str]):
    if resume_from_checkpoint is None:
        return trainer.train()

    # PyTorch 2.6 changed torch.load default to weights_only=True. Older
    # Transformers Trainer resumes rng_state.pth with torch.load(path), which
    # can fail because RNG state contains NumPy objects. This checkpoint is local
    # and produced by this run, so allow full pickle loading only during resume.
    original_torch_load = torch.load

    def torch_load_resume_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_resume_compat
    try:
        return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        torch.load = original_torch_load


def compute_row_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def one(sub_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        refs = [r["reference"] for r in sub_rows]
        preds = [r["prediction"] for r in sub_rows]
        exact = sum(r == p for r, p in zip(refs, preds)) / max(1, len(sub_rows))
        return {
            "num_samples": len(sub_rows),
            "cer": cer(refs, preds) if sub_rows else None,
            "wer": wer(refs, preds) if sub_rows else None,
            "exact_match": exact,
        }

    metrics = {"overall": one(rows), "by_label": {}}
    for label in TARGET_LABELS:
        sub = [r for r in rows if r["label"] == label]
        metrics["by_label"][label] = one(sub)
    return metrics


@torch.inference_mode()
def evaluate_typed_generation(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    records: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    output_jsonl: Path,
) -> Dict[str, Any]:
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    old_decoder_use_cache = getattr(model.decoder.config, "use_cache", None)

    model.eval()
    model.config.use_cache = True
    if hasattr(model.decoder.config, "use_cache"):
        model.decoder.config.use_cache = True

    rows: List[Dict[str, Any]] = []
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    for label in TARGET_LABELS:
        label_records = by_label.get(label, [])
        if not label_records:
            continue
        gen_kwargs = GEN_CONFIG_BY_TYPE[label]
        for start in tqdm(range(0, len(label_records), batch_size), desc=f"typed eval {label}"):
            batch = label_records[start : start + batch_size]
            images = [open_rgb_or_raise(r["image_path"]) for r in batch]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, **gen_kwargs)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for record, pred in zip(batch, preds):
                rows.append(
                    {
                        "image": record["image"],
                        "image_path": record["image_path"],
                        "label": record["label"],
                        "reference": normalize_text(record["text"]),
                        "prediction": normalize_text(pred),
                    }
                )

    write_jsonl(output_jsonl, rows)
    metrics = compute_row_metrics(rows)

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if old_decoder_use_cache is not None and hasattr(model.decoder.config, "use_cache"):
        model.decoder.config.use_cache = old_decoder_use_cache
    if was_training:
        model.train()
    return metrics


def save_model_artifact(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    old_use_cache = getattr(model.config, "use_cache", None)
    old_decoder_use_cache = getattr(model.decoder.config, "use_cache", None)

    model.config.use_cache = True
    if hasattr(model.decoder.config, "use_cache"):
        model.decoder.config.use_cache = True

    model.save_pretrained(path)
    processor.save_pretrained(path)
    save_json(path / "generation_config_by_type.json", GEN_CONFIG_BY_TYPE)

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if old_decoder_use_cache is not None and hasattr(model.decoder.config, "use_cache"):
        model.decoder.config.use_cache = old_decoder_use_cache


def load_model_and_processor(args: SimpleNamespace, hf_token: Optional[str]) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    model_cache_kwargs = {"cache_dir": str(args.hf_cache_dir)} if args.hf_cache_dir is not None else {}

    if args.resume_model_dir is not None:
        if not args.resume_model_dir.exists():
            raise FileNotFoundError(f"RESUME_MODEL_DIR does not exist: {args.resume_model_dir}")
        print(f"\nLoading model and processor from resume artifact: {args.resume_model_dir}")
        processor = load_processor_for_resume(
            resume_dir=args.resume_model_dir,
            base_processor_model=args.base_processor_model,
            hf_token=hf_token,
            cache_dir=args.hf_cache_dir,
        )
        model = VisionEncoderDecoderModel.from_pretrained(args.resume_model_dir)
    else:
        print("\nLoading model and processor from HuggingFace...")
        processor = load_processor(
            args.model_name,
            args.base_processor_model,
            hf_token=hf_token,
            cache_dir=args.hf_cache_dir,
        )
        model = from_pretrained_hf(
            VisionEncoderDecoderModel,
            args.model_name,
            hf_token,
            **model_cache_kwargs,
        )

    return model, processor


def create_output_dirs(args: SimpleNamespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir = args.output_dir / "reports"
    args.splits_dir = args.output_dir / "splits"
    args.single_run_dir = args.output_dir / args.run_name
    args.trainer_dir = args.single_run_dir / "trainer"
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.single_run_dir.mkdir(parents=True, exist_ok=True)


def save_run_config(args: SimpleNamespace) -> None:
    serializable_config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    serializable_config["gen_config_by_type"] = GEN_CONFIG_BY_TYPE
    serializable_config["target_labels"] = TARGET_LABELS
    serializable_config["ukrainian_extra_tokens"] = UKRAINIAN_EXTRA_TOKENS
    save_json(args.output_dir / "run_config.json", serializable_config)
    save_json(args.output_dir / "generation_config_by_type.json", GEN_CONFIG_BY_TYPE)


def main() -> None:
    # 1. Build config.
    args = build_config()

    # 2. Load optional .env.
    load_optional_env_file(args.env_file)

    # 3. Finalize paths.
    finalize_paths(args)

    # 4. Resolve HF token.
    hf_token = resolve_hf_token(args)

    # 5. Set seed.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # 6. Create OUTPUT_DIR, reports, splits.
    create_output_dirs(args)
    save_run_config(args)

    print(f"Output dir : {args.output_dir}")
    print(f"Train dir  : {args.train_dir}")
    print(f"Final dir  : {args.final_dir}")
    print(f"Start model: {args.model_name}")
    print(f"Resume dir : {args.resume_model_dir or '(none)'}")
    print(f"HF cache   : {args.hf_cache_dir or '(default Hugging Face cache)'}")
    print(f"Run name   : {args.run_name}")
    print(f"Optimizer  : {args.optim}")
    print(f"Save strat : {args.save_strategy}")

    # 7. Prepare fixed splits.
    train_fixed_records, val_fixed_records = prepare_fixed_splits(args)

    # 8. Save basic report.
    save_report(
        args,
        "train_fixed_basic_report",
        build_records_report(
            "train_fixed",
            train_fixed_records,
            processor=None,
            max_target_length=args.max_target_length,
            verify_image_open=args.verify_image_open_in_report,
        ),
    )
    save_report(
        args,
        "val_fixed_basic_report",
        build_records_report(
            "val_fixed",
            val_fixed_records,
            processor=None,
            max_target_length=args.max_target_length,
            verify_image_open=args.verify_image_open_in_report,
        ),
    )

    # 9. Load processor and model.
    model, processor = load_model_and_processor(args, hf_token)

    # 10. Configure tokenizer/model.
    configure_tokenizer_and_model(processor, model)

    # 11. Ukrainian tokenization check.
    print_ukrainian_tokenization(processor)

    # 12. Enable gradient checkpointing if needed.
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model.decoder.config, "use_cache"):
            model.decoder.config.use_cache = False

    # 13. Move model to CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # 14. Save token reports.
    save_report(
        args,
        "train_fixed_token_report",
        build_records_report(
            "train_fixed",
            train_fixed_records,
            processor=processor,
            max_target_length=args.max_target_length,
            verify_image_open=False,
        ),
    )
    save_report(
        args,
        "val_fixed_token_report",
        build_records_report(
            "val_fixed",
            val_fixed_records,
            processor=processor,
            max_target_length=args.max_target_length,
            verify_image_open=False,
        ),
    )

    # Optional cap for smoke tests/debug. Validation remains unchanged.
    train_records = cap_per_class(train_fixed_records, args.train_max_per_class, args.seed)
    if args.train_max_per_class > 0:
        print(f"TRAIN_MAX_PER_CLASS={args.train_max_per_class}: using capped train subset.")
        print_counts("capped_train", train_records)
        save_report(
            args,
            "capped_train_token_report",
            build_records_report(
                "capped_train",
                train_records,
                processor=processor,
                max_target_length=args.max_target_length,
                verify_image_open=False,
            ),
        )

    # 15. Create datasets.
    train_dataset = HpaDataset(train_records, processor, args.max_target_length)
    val_dataset = HpaDataset(val_fixed_records, processor, args.max_target_length)

    # 16. Seq2SeqTrainingArguments single phase.
    training_args = build_training_args(args)

    # 17. Seq2SeqTrainer.
    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        patience=args.early_stopping_patience,
    )

    # 18. Train once. If RESUME_MODEL_DIR is a Trainer checkpoint, resume state.
    resume_from_checkpoint = None
    if args.resume_model_dir is not None and (args.resume_model_dir / "trainer_state.json").exists():
        resume_from_checkpoint = str(args.resume_model_dir)
        print(f"Resuming Trainer state from checkpoint: {resume_from_checkpoint}")

    train_result = train_with_resume_compat(trainer, resume_from_checkpoint)
    save_json(args.single_run_dir / "train_result_metrics.json", train_result.metrics)

    trainer_eval_metrics = trainer.evaluate()
    save_json(args.single_run_dir / "trainer_eval_metrics.json", trainer_eval_metrics)
    save_json(
        args.single_run_dir / "best_model_selection.json",
        {
            "selected_by": "trainer_eval_cer",
            "metric_for_best_model": "cer",
            "greater_is_better": False,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "final_trainer_eval_metrics": trainer_eval_metrics,
        },
    )
    print(f"Trainer eval metrics: {trainer_eval_metrics}")
    print(f"Best checkpoint by eval CER: {trainer.state.best_model_checkpoint}")
    print(f"Best eval CER: {trainer.state.best_metric}")

    # After load_best_model_at_end=True, trainer.model is already the best model
    # according to eval CER.
    model = trainer.model
    model.to(device)

    # 19. Typed generation evaluation on val_fixed_records.
    if not args.skip_typed_eval:
        final_predictions = args.output_dir / "final_typed_val_predictions.jsonl"
        final_metrics = evaluate_typed_generation(
            model=model,
            processor=processor,
            records=val_fixed_records,
            device=device,
            batch_size=args.typed_eval_batch_size,
            output_jsonl=final_predictions,
        )
        save_json(args.output_dir / "final_typed_val_metrics.json", final_metrics)
        print(f"Final typed validation metrics: {final_metrics['overall']}")

    # 20. Save generation config, run config, final model.
    save_json(args.output_dir / "generation_config_by_type.json", GEN_CONFIG_BY_TYPE)
    save_run_config(args)
    save_model_artifact(model, processor, args.final_dir)

    # 21. Print final path.
    print(f"Saved final model: {args.final_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
