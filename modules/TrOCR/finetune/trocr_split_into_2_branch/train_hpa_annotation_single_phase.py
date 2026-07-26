#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-phase fine-tuning for the RUKOPYS HPA annotation branch.

Branch:
  - labels: annotation
  - base model: microsoft/trocr-base-handwritten
  - optimizer: AdamW via HuggingFace Trainer optim='adamw_torch'
  - LR control: warmup + cosine scheduler

Expected dataset layout:

DATA_ROOT/
  train/
    metadata.jsonl
    images/
      ...

Supported text markers are preserved as literal target text:
  ~~word~~          -> strikethrough text
  ~~old~~{new}     -> strikethrough with correction
  [illegible]      -> unreadable word within a legible line
"""

from __future__ import annotations

import inspect
import csv
import json
import os
import random
import re
import unicodedata
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

# -----------------------------------------------------------------------------
# Branch constants
# -----------------------------------------------------------------------------
BRANCH_NAME = "annotation"
TARGET_LABELS = ("annotation",)
MODEL_NAME = "microsoft/trocr-base-handwritten"
BASE_PROCESSOR_MODEL = "microsoft/trocr-base-handwritten"
RUN_NAME = "hpa_annotation_single_phase"

# Keep marker tokens explicit, because annotation is usually short and exact
# marker generation matters.
SPECIAL_TEXT_MARKER_TOKENS = ["~~", "{", "}", "[illegible]"]
UKRAINIAN_EXTRA_TOKENS = ["і", "ї", "є", "ґ", "І", "Ї", "Є", "Ґ"]

GEN_CONFIG_BY_TYPE: Dict[str, Dict[str, Any]] = {
    "annotation": {
        "num_beams": 1,
        "max_new_tokens": 32,
        "no_repeat_ngram_size": 2,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "early_stopping": True,
    },
}

# -----------------------------------------------------------------------------
# Default config, overridable through environment variables.
# -----------------------------------------------------------------------------
ENV_FILE: Optional[Path] = None
HF_TOKEN_ENV = "HF_TOKEN"

DATA_ROOT = Path("/home/ubuntu/dataset")
TRAIN_DIR: Optional[Path] = None
CROP_MANIFEST: Optional[Path] = None
USE_CROPPED_MANIFEST = True
OUTPUT_DIR = Path("/home/ubuntu/outputs_hpa_annotation_single")
FINAL_DIR: Optional[Path] = None
HF_CACHE_DIR: Optional[Path] = Path("/home/ubuntu/hf_cache")
RESUME_MODEL_DIR: Optional[Path] = None

SEED = 42
VAL_RATIO = 0.10
REBUILD_SPLITS = False

MAX_TARGET_LENGTH = 64
TRAIN_EPOCHS = 10.0
TRAIN_LR = 1e-5
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
DATALOADER_NUM_WORKERS = 2
LOGGING_STEPS = 25

SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT: Optional[int] = 2
EARLY_STOPPING_PATIENCE = 4

OPTIM = "adamw_torch"
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"
REPORT_TO = "none"
NO_FP16 = False
BF16 = False
GRADIENT_CHECKPOINTING = True

TRAIN_MAX_PER_CLASS = 0
SKIP_GENERATION_EVAL = False
GEN_EVAL_BATCH_SIZE = 16

FAIL_ON_MISSING_IMAGE = True
FAIL_ON_IMAGE_OPEN_ERROR = True
VERIFY_IMAGE_OPEN_IN_REPORT = False
COLLAPSE_WHITESPACE = False
ADD_SPECIAL_MARKER_TOKENS = False
ADD_UKRAINIAN_TOKENS = False

# Do not strip official markers by default. They should remain in the target if
# the competition label contains them.
STRIP_TEXT_MARKERS_FOR_TRAIN = False

STRIKETHROUGH_RE = re.compile(r"~~.+?~~")
CORRECTION_RE = re.compile(r"~~.+?~~\{.*?\}")
ILLEGIBLE_TOKEN = "[illegible]"


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def env_path(names: Sequence[str], default: Optional[Path]) -> Optional[Path]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return default


def build_config() -> SimpleNamespace:
    return SimpleNamespace(
        branch_name=BRANCH_NAME,
        target_labels=TARGET_LABELS,
        model_name=MODEL_NAME,
        base_processor_model=BASE_PROCESSOR_MODEL,
        run_name=RUN_NAME,
        env_file=ENV_FILE,
        hf_token_env=HF_TOKEN_ENV,
        data_root=DATA_ROOT,
        train_dir=TRAIN_DIR,
        crop_manifest=CROP_MANIFEST,
        use_cropped_manifest=USE_CROPPED_MANIFEST,
        output_dir=OUTPUT_DIR,
        final_dir=FINAL_DIR,
        hf_cache_dir=HF_CACHE_DIR,
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
        skip_generation_eval=SKIP_GENERATION_EVAL,
        gen_eval_batch_size=GEN_EVAL_BATCH_SIZE,
        fail_on_missing_image=FAIL_ON_MISSING_IMAGE,
        fail_on_image_open_error=FAIL_ON_IMAGE_OPEN_ERROR,
        verify_image_open_in_report=VERIFY_IMAGE_OPEN_IN_REPORT,
        collapse_whitespace=COLLAPSE_WHITESPACE,
        add_special_marker_tokens=ADD_SPECIAL_MARKER_TOKENS,
        add_ukrainian_tokens=ADD_UKRAINIAN_TOKENS,
        strip_text_markers_for_train=STRIP_TEXT_MARKERS_FOR_TRAIN,
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


def finalize_config(args: SimpleNamespace) -> None:
    args.data_root = env_path(("HPA_DATA_ROOT", "DATA_ROOT"), args.data_root)
    args.train_dir = env_path(("HPA_TRAIN_DIR",), args.train_dir) or args.data_root / "train"
    args.crop_manifest = env_path(
        ("HPA_CROP_MANIFEST", "CROP_MANIFEST"),
        args.crop_manifest or args.train_dir / "manifest.csv",
    )
    if not args.crop_manifest.exists() and not env_path(("HPA_CROP_MANIFEST", "CROP_MANIFEST"), None):
        legacy_manifest = args.train_dir / "cropped_bboxes" / "manifest.csv"
        if legacy_manifest.exists():
            args.crop_manifest = legacy_manifest
    args.use_cropped_manifest = env_bool(("HPA_USE_CROPPED_MANIFEST", "USE_CROPPED_MANIFEST"), args.use_cropped_manifest)
    args.output_dir = env_path(("HPA_OUTPUT_DIR", "OUTPUT_DIR"), args.output_dir)
    args.final_dir = env_path(("HPA_FINAL_DIR",), args.final_dir) or args.output_dir / "final_trocr_base_annotation_model"
    args.hf_cache_dir = env_path(("HF_CACHE_DIR", "HF_HOME"), args.hf_cache_dir)
    args.resume_model_dir = env_path(("HPA_RESUME_MODEL_DIR", "RESUME_MODEL_DIR"), args.resume_model_dir)

    args.val_ratio = env_float(("HPA_VAL_RATIO", "VAL_RATIO"), args.val_ratio)
    args.rebuild_splits = env_bool(("HPA_REBUILD_SPLITS", "REBUILD_SPLITS"), args.rebuild_splits)
    args.max_target_length = env_int(("HPA_MAX_TARGET_LENGTH", "MAX_TARGET_LENGTH"), args.max_target_length)
    args.train_epochs = env_float(("HPA_TRAIN_EPOCHS", "TRAIN_EPOCHS"), args.train_epochs)
    args.train_lr = env_float(("HPA_TRAIN_LR", "TRAIN_LR"), args.train_lr)
    args.train_batch_size = env_int(("HPA_TRAIN_BATCH_SIZE", "TRAIN_BATCH_SIZE"), args.train_batch_size)
    args.eval_batch_size = env_int(("HPA_EVAL_BATCH_SIZE", "EVAL_BATCH_SIZE"), args.eval_batch_size)
    args.gen_eval_batch_size = env_int(
        ("HPA_GEN_EVAL_BATCH_SIZE", "GEN_EVAL_BATCH_SIZE", "HPA_GENERATION_EVAL_BATCH_SIZE", "GENERATION_EVAL_BATCH_SIZE"),
        args.gen_eval_batch_size,
    )
    args.train_max_per_class = env_int(("HPA_TRAIN_MAX_PER_CLASS", "TRAIN_MAX_PER_CLASS"), args.train_max_per_class)
    args.bf16 = env_bool(("HPA_BF16", "BF16"), args.bf16)
    args.no_fp16 = env_bool(("HPA_NO_FP16", "NO_FP16"), args.no_fp16)
    args.skip_generation_eval = env_bool(("HPA_SKIP_GENERATION_EVAL", "SKIP_GENERATION_EVAL"), args.skip_generation_eval)
    args.add_special_marker_tokens = env_bool(
        ("HPA_ADD_SPECIAL_MARKER_TOKENS", "ADD_SPECIAL_MARKER_TOKENS", "HPA_ADD_STRIKETHROUGH_TOKENS", "ADD_STRIKETHROUGH_TOKENS"),
        args.add_special_marker_tokens,
    )
    args.add_ukrainian_tokens = env_bool(("HPA_ADD_UKRAINIAN_TOKENS", "ADD_UKRAINIAN_TOKENS"), args.add_ukrainian_tokens)
    args.strip_text_markers_for_train = env_bool(
        ("HPA_STRIP_TEXT_MARKERS_FOR_TRAIN", "STRIP_TEXT_MARKERS_FOR_TRAIN", "HPA_STRIP_STRIKETHROUGH_MARKERS", "STRIP_STRIKETHROUGH_MARKERS"),
        args.strip_text_markers_for_train,
    )

    if not 0 < args.val_ratio < 1:
        raise ValueError(f"VAL_RATIO must be between 0 and 1. Got: {args.val_ratio}")


def resolve_hf_token(args: SimpleNamespace) -> Optional[str]:
    for env_name in [args.hf_token_env, "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
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


def load_vision_encoder_decoder_model(model_source: str, hf_token: Optional[str], cache_kwargs: Dict[str, Any]) -> VisionEncoderDecoderModel:
    try:
        print(f"[INFO] Loading model with safetensors: {model_source}")
        return from_pretrained_hf(
            VisionEncoderDecoderModel,
            model_source,
            hf_token,
            use_safetensors=True,
            **cache_kwargs,
        )
    except Exception as exc:
        print(f"[WARN] Could not load safetensors for {model_source}: {exc}")
        print("[WARN] Falling back to default model loading. If torch < 2.6 and only pytorch_model.bin is available, this may fail.")
        return from_pretrained_hf(VisionEncoderDecoderModel, model_source, hf_token, **cache_kwargs)


def load_trocr_processor(model_name: str, base_processor_model: str, hf_token: Optional[str], cache_kwargs: Dict[str, Any]) -> TrOCRProcessor:
    errors: List[str] = []
    for kwargs in ({}, {"subfolder": "processor"}):
        try:
            processor = from_pretrained_hf(
                TrOCRProcessor,
                model_name,
                hf_token,
                use_fast=False,
                **cache_kwargs,
                **kwargs,
            )
            print(f"[INFO] Loaded TrOCRProcessor from {model_name} with kwargs={kwargs}")
            return processor
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")

    print("[WARN] Could not load full TrOCRProcessor from model repo:")
    for error in errors:
        print(f"       {error}")
    print(f"[INFO] Falling back to image processor from: {base_processor_model}")
    image_processor = from_pretrained_hf(ViTImageProcessor, base_processor_model, hf_token, use_fast=False, **cache_kwargs)

    tokenizer_errors: List[str] = []
    for kwargs in ({}, {"subfolder": "processor"}):
        try:
            tokenizer = from_pretrained_hf(
                AutoTokenizer,
                model_name,
                hf_token,
                use_fast=True,
                **cache_kwargs,
                **kwargs,
            )
            print(f"[INFO] Loaded tokenizer from {model_name} with kwargs={kwargs}")
            return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        except Exception as exc:
            tokenizer_errors.append(f"{kwargs}: {exc}")

    print("[WARN] Could not load tokenizer from model repo:")
    for error in tokenizer_errors:
        print(f"       {error}")
    if model_name != base_processor_model:
        raise RuntimeError(
            f"Could not load tokenizer from {model_name}. Refusing to fall back to "
            f"{base_processor_model}, because that tokenizer does not match the decoder."
        )
    print(f"[INFO] Falling back to tokenizer from: {base_processor_model}")
    tokenizer = from_pretrained_hf(AutoTokenizer, base_processor_model, hf_token, use_fast=True, **cache_kwargs)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def strip_text_markers(text: str) -> str:
    """Optional fallback if you intentionally do not want to train on markup."""
    # ~~old~~{new} -> new
    text = re.sub(r"~~(.*?)~~\{(.*?)\}", r"\2", text)
    # ~~word~~ -> word
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    # [illegible] -> removed
    text = text.replace(ILLEGIBLE_TOKEN, "")
    return " ".join(text.split())


def sanitize_ocr_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    cleaned: List[str] = []
    for char in text:
        if char == "\ufffd":
            continue
        category = unicodedata.category(char)
        if category[0] == "C":
            if char in "\t\n\r\f\v":
                cleaned.append(" ")
            continue
        cleaned.append(char)
    return "".join(cleaned)


def normalize_text(text: Any, collapse_whitespace: bool = False, strip_markers: bool = False) -> str:
    text = sanitize_ocr_text(text).strip()
    if strip_markers:
        text = strip_text_markers(text)
    if collapse_whitespace:
        text = " ".join(text.split())
    return text


def marker_stats(texts: Sequence[str]) -> Dict[str, int]:
    return {
        "num_texts": len(texts),
        "texts_with_strikethrough": sum(bool(STRIKETHROUGH_RE.search(t)) for t in texts),
        "texts_with_correction": sum(bool(CORRECTION_RE.search(t)) for t in texts),
        "texts_with_illegible": sum(ILLEGIBLE_TOKEN in t for t in texts),
        "strikethrough_marker_count": sum(len(STRIKETHROUGH_RE.findall(t)) for t in texts),
        "correction_marker_count": sum(len(CORRECTION_RE.findall(t)) for t in texts),
        "illegible_marker_count": sum(t.count(ILLEGIBLE_TOKEN) for t in texts),
    }


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
    for key in ("image", "file_name", "filename", "path", "image_path", "crop_path"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def resolve_image_path(split_dir: Path, image_value: str, extra_base_dirs: Sequence[Path] = ()) -> Tuple[Path, List[str]]:
    raw = Path(image_value)
    if raw.is_absolute():
        return raw, [str(raw)]

    candidates: List[Path] = [split_dir / raw]
    for base_dir in extra_base_dirs:
        candidates.append(base_dir / raw)
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
    preferred = deduped[1] if len(deduped) > 1 else deduped[0]
    return preferred, [str(x) for x in deduped]


def compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": record["image"],
        "label": record["label"],
        "text": record["text"],
        "source_split": record.get("source_split", "train"),
        "source_file_name": record.get("source_file_name", ""),
        "source_region_index": record.get("source_region_index", ""),
        "source_manifest": record.get("source_manifest", ""),
    }


def hydrate_record(row: Dict[str, Any], train_dir: Path, args: SimpleNamespace) -> Dict[str, Any]:
    image = get_record_image_value(row)
    if not image:
        raise ValueError(f"Bad split row without image/file_name/path: {row}")
    extra_base_dirs: List[Path] = []
    if row.get("source_split") == "train_crop" and args.crop_manifest:
        extra_base_dirs = [args.crop_manifest.parent, args.crop_manifest.parent.parent]
    image_path, attempts = resolve_image_path(train_dir, image, extra_base_dirs)
    return {
        "image": image,
        "image_path": str(image_path),
        "image_path_attempts": attempts,
        "label": get_record_label(row),
        "text": normalize_text(row.get("text", ""), args.collapse_whitespace, args.strip_text_markers_for_train),
        "raw_text": normalize_text(row.get("text", ""), False, False),
        "source_split": row.get("source_split", "train"),
        "source_file_name": row.get("source_file_name", ""),
        "source_region_index": row.get("source_region_index", ""),
        "source_manifest": row.get("source_manifest", ""),
    }


def load_cropped_manifest_records(split_dir: Path, manifest_path: Path, args: SimpleNamespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "metadata_path": str(manifest_path),
        "dataset_format": "cropped_manifest_csv",
        "branch_name": args.branch_name,
        "target_labels": list(args.target_labels),
        "total_rows": 0,
        "bad_json_rows": 0,
        "kept_rows": 0,
        "skipped_non_target_label": 0,
        "skipped_missing_image_field": 0,
        "skipped_empty_text": 0,
        "raw_label_counts": {},
        "kept_label_counts": {},
        "missing_image_files": 0,
        "missing_image_examples": [],
        "marker_stats_kept_raw_text": {},
        "strip_text_markers_for_train": args.strip_text_markers_for_train,
    }
    raw_label_counts: Counter[str] = Counter()
    kept_label_counts: Counter[str] = Counter()
    raw_texts_kept: List[str] = []
    crop_base_dirs = [split_dir, manifest_path.parent, manifest_path.parent.parent]

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for line_no, item in enumerate(csv.DictReader(f), start=2):
            report["total_rows"] += 1
            label = get_record_label(item)
            raw_label_counts[label] += 1
            if label not in args.target_labels:
                report["skipped_non_target_label"] += 1
                continue

            image = get_record_image_value(item)
            if not image:
                report["skipped_missing_image_field"] += 1
                continue

            raw_text = normalize_text(item.get("text", ""), False, False)
            text = normalize_text(item.get("text", ""), args.collapse_whitespace, args.strip_text_markers_for_train)
            if not text:
                report["skipped_empty_text"] += 1
                continue

            image_path, attempts = resolve_image_path(split_dir, image, crop_base_dirs)
            if not image_path.exists():
                report["missing_image_files"] += 1
                if len(report["missing_image_examples"]) < 20:
                    report["missing_image_examples"].append(
                        {"line_no": line_no, "image": image, "resolved_image_path": str(image_path), "attempts": attempts}
                    )

            kept_label_counts[label] += 1
            raw_texts_kept.append(raw_text)
            records.append(
                {
                    "image": image,
                    "image_path": str(image_path),
                    "image_path_attempts": attempts,
                    "label": label,
                    "text": text,
                    "raw_text": raw_text,
                    "source_split": "train_crop",
                    "source_file_name": item.get("source_file_name", ""),
                    "source_region_index": item.get("region_index", ""),
                    "source_manifest": str(manifest_path),
                }
            )

    report["kept_rows"] = len(records)
    report["raw_label_counts"] = dict(sorted(raw_label_counts.items()))
    report["kept_label_counts"] = {label: kept_label_counts.get(label, 0) for label in args.target_labels}
    report["marker_stats_kept_raw_text"] = marker_stats(raw_texts_kept)
    return records, report


def load_metadata_records(split_dir: Path, args: SimpleNamespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if args.use_cropped_manifest and args.crop_manifest and args.crop_manifest.exists():
        print(f"[INFO] Loading cropped bbox manifest: {args.crop_manifest}")
        return load_cropped_manifest_records(split_dir, args.crop_manifest, args)

    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing train metadata: {metadata_path}")

    records: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "branch_name": args.branch_name,
        "target_labels": list(args.target_labels),
        "total_rows": 0,
        "bad_json_rows": 0,
        "kept_rows": 0,
        "skipped_non_target_label": 0,
        "skipped_missing_image_field": 0,
        "skipped_empty_text": 0,
        "raw_label_counts": {},
        "kept_label_counts": {},
        "missing_image_files": 0,
        "missing_image_examples": [],
        "marker_stats_kept_raw_text": {},
        "strip_text_markers_for_train": args.strip_text_markers_for_train,
    }
    raw_label_counts: Counter[str] = Counter()
    kept_label_counts: Counter[str] = Counter()
    raw_texts_kept: List[str] = []

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
            if label not in args.target_labels:
                report["skipped_non_target_label"] += 1
                continue

            image = get_record_image_value(item)
            if not image:
                report["skipped_missing_image_field"] += 1
                continue

            raw_text = normalize_text(item.get("text", ""), False, False)
            text = normalize_text(item.get("text", ""), args.collapse_whitespace, args.strip_text_markers_for_train)
            if not text:
                report["skipped_empty_text"] += 1
                continue

            image_path, attempts = resolve_image_path(split_dir, image)
            if not image_path.exists():
                report["missing_image_files"] += 1
                if len(report["missing_image_examples"]) < 20:
                    report["missing_image_examples"].append(
                        {"line_no": line_no, "image": image, "resolved_image_path": str(image_path), "attempts": attempts}
                    )

            kept_label_counts[label] += 1
            raw_texts_kept.append(raw_text)
            records.append(
                {
                    "image": image,
                    "image_path": str(image_path),
                    "image_path_attempts": attempts,
                    "label": label,
                    "text": text,
                    "raw_text": raw_text,
                    "source_split": "train",
                }
            )

    report["kept_rows"] = len(records)
    report["raw_label_counts"] = dict(sorted(raw_label_counts.items()))
    report["kept_label_counts"] = {label: kept_label_counts.get(label, 0) for label in args.target_labels}
    report["marker_stats_kept_raw_text"] = marker_stats(raw_texts_kept)
    return records, report


def stratified_split(records: Sequence[Dict[str, Any]], val_ratio: float, seed: int, labels: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for label in labels:
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


def count_by_label(records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Dict[str, int]:
    counts = Counter(r["label"] for r in records)
    return {label: counts.get(label, 0) for label in labels}


def limit_per_class(records: Sequence[Dict[str, Any]], max_per_class: int, seed: int, labels: Sequence[str]) -> List[Dict[str, Any]]:
    if max_per_class <= 0:
        return list(records)
    rng = random.Random(seed)
    result: List[Dict[str, Any]] = []
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    for label in labels:
        rows = list(by_label.get(label, []))
        rng.shuffle(rows)
        result.extend(rows[:max_per_class])
    rng.shuffle(result)
    return result


def validate_split(train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]], args: SimpleNamespace) -> Tuple[bool, str]:
    if not train_rows:
        return False, "train split is empty"
    if not val_rows:
        return False, "val split is empty"
    expected_manifest = str(args.crop_manifest) if args.use_cropped_manifest and args.crop_manifest and args.crop_manifest.exists() else ""
    if expected_manifest:
        stale_rows = [
            r.get("image")
            for r in [*train_rows, *val_rows]
            if r.get("source_split") == "train_crop" and r.get("source_manifest") != expected_manifest
        ]
        if stale_rows:
            return False, f"split was built from a different crop manifest, first stale image={stale_rows[0]}"
    missing_images = [r.get("image_path") for r in [*train_rows, *val_rows] if not Path(str(r.get("image_path", ""))).exists()]
    if missing_images:
        return False, f"split has missing crop images, first missing={missing_images[0]}"

    all_counts = count_by_label([*train_rows, *val_rows], args.target_labels)
    train_counts = count_by_label(train_rows, args.target_labels)
    val_counts = count_by_label(val_rows, args.target_labels)
    for label in args.target_labels:
        if all_counts[label] == 0:
            return False, f"no samples found for label={label!r}"
        if all_counts[label] < 2:
            return False, f"need at least 2 samples for label={label!r} to keep it in both train and val"
        if train_counts[label] == 0:
            return False, f"train split has no samples for label={label!r}"
        if val_counts[label] == 0:
            return False, f"val split has no samples for label={label!r}"
    return True, "ok"


def load_or_create_splits(records: Sequence[Dict[str, Any]], args: SimpleNamespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        raise RuntimeError(
            "No records available for training. Check HPA_CROP_MANIFEST path and that manifest.csv has type/text/crop_path rows for "
            f"{args.target_labels}."
        )

    split_dir = args.output_dir / "splits"
    split_source = "crop" if args.use_cropped_manifest and args.crop_manifest and args.crop_manifest.exists() else "metadata"
    train_split_path = split_dir / f"{args.branch_name}_{split_source}_train.jsonl"
    val_split_path = split_dir / f"{args.branch_name}_{split_source}_val.jsonl"

    if not args.rebuild_splits and train_split_path.exists() and val_split_path.exists():
        print(f"[INFO] Reusing fixed split: {split_dir}")
        train_rows = [hydrate_record(r, args.train_dir, args) for r in read_jsonl(train_split_path)]
        val_rows = [hydrate_record(r, args.train_dir, args) for r in read_jsonl(val_split_path)]
        valid, reason = validate_split(train_rows, val_rows, args)
        if valid:
            return train_rows, val_rows
        print(f"[WARN] Existing split is invalid ({reason}). Rebuilding split from current train data.")

    print("[INFO] Creating new fixed stratified split")
    train_rows, val_rows = stratified_split(records, args.val_ratio, args.seed, args.target_labels)
    valid, reason = validate_split(train_rows, val_rows, args)
    if not valid:
        raise RuntimeError(
            f"Could not create a valid train/val split: {reason}. "
            f"Available counts: {count_by_label(records, args.target_labels)}"
        )
    write_jsonl(train_split_path, (compact_record(r) for r in train_rows))
    write_jsonl(val_split_path, (compact_record(r) for r in val_rows))
    return train_rows, val_rows


def open_rgb_or_raise(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


class HpaDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], processor: TrOCRProcessor, max_target_length: int):
        self.records = list(records)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = open_rgb_or_raise(Path(record["image_path"]))
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        encoding = self.processor.tokenizer(
            record["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_attention_mask=True,
        )
        labels = [
            token_id if mask == 1 else -100
            for token_id, mask in zip(encoding["input_ids"], encoding["attention_mask"])
        ]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels, dtype=torch.long)}


class OcrDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = [f["labels"] for f in features]
        batch_labels = self.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        labels_tensor = batch_labels["input_ids"]
        labels_tensor = labels_tensor.masked_fill(labels_tensor == self.tokenizer.pad_token_id, -100)
        return {"pixel_values": pixel_values, "labels": labels_tensor}


def configure_tokenizer_and_model(processor: TrOCRProcessor, model: VisionEncoderDecoderModel, args: SimpleNamespace) -> List[str]:
    tokenizer = processor.tokenizer
    added_tokens: List[str] = []
    changed_vocab = False

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_tokens.append("[PAD]")
        changed_vocab = True

    tokens_to_add: List[str] = []
    if args.add_ukrainian_tokens:
        tokens_to_add.extend(UKRAINIAN_EXTRA_TOKENS)
    if args.add_special_marker_tokens:
        tokens_to_add.extend(SPECIAL_TEXT_MARKER_TOKENS)

    vocab = tokenizer.get_vocab()
    missing_tokens = [token for token in tokens_to_add if token not in vocab]
    if missing_tokens:
        tokenizer.add_tokens(missing_tokens)
        added_tokens.extend(missing_tokens)
        changed_vocab = True

    if changed_vocab:
        new_size = len(tokenizer)
        model.decoder.resize_token_embeddings(new_size)
        model.config.vocab_size = new_size
        if hasattr(model.config, "decoder"):
            model.config.decoder.vocab_size = new_size

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no bos_token_id or cls_token_id for decoder_start_token_id")

    model.config.decoder_start_token_id = bos_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model.config, "decoder"):
        model.config.decoder.decoder_start_token_id = bos_id
        model.config.decoder.pad_token_id = tokenizer.pad_token_id
        model.config.decoder.eos_token_id = tokenizer.eos_token_id

    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = bos_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    print(
        "Token IDs | "
        f"bos={bos_id} pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id} "
        f"vocab={len(tokenizer)}"
    )
    return added_tokens


def summarize_text_lengths(records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    def summary(values: Sequence[int]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "p95": None, "p99": None}
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "max": ordered[-1],
            "mean": sum(ordered) / len(ordered),
            "p95": ordered[int((len(ordered) - 1) * 0.95)],
            "p99": ordered[int((len(ordered) - 1) * 0.99)],
        }

    by_label = {label: [] for label in labels}
    overall: List[int] = []
    for record in records:
        length = len(record["text"])
        overall.append(length)
        by_label[record["label"]].append(length)
    return {"overall": summary(overall), "by_label": {label: summary(by_label[label]) for label in labels}}


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
            "cer": round(float(cer(label_str, pred_str)), 6),
            "wer": round(float(wer(label_str, pred_str)), 6),
            "exact_match": round(exact, 6),
        }

    return compute_metrics


def build_training_args(args: SimpleNamespace) -> Seq2SeqTrainingArguments:
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir / "trainer_checkpoints"),
        "run_name": args.run_name,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.train_lr,
        "num_train_epochs": args.train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "generation_num_beams": 1,
        "remove_unused_columns": True,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": args.report_to,
        "fp16": bool(torch.cuda.is_available() and (not args.no_fp16) and (not args.bf16)),
        "bf16": bool(args.bf16),
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_safetensors" in signature.parameters:
        kwargs["save_safetensors"] = True
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    dropped = sorted(set(kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"[WARN] Dropped unsupported TrainingArguments keys: {dropped}")
    return Seq2SeqTrainingArguments(**filtered_kwargs)


@torch.inference_mode()
def evaluate_generation(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    records: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    output_jsonl: Path,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()

    old_use_cache = getattr(model.config, "use_cache", None)
    old_decoder_use_cache = getattr(model.decoder.config, "use_cache", None) if hasattr(model, "decoder") else None

    try:
        model.config.use_cache = True
        if hasattr(model, "decoder"):
            model.decoder.config.use_cache = True

        rows: List[Dict[str, Any]] = []
        by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_label[record["label"]].append(record)

        for label in TARGET_LABELS:
            label_records = by_label.get(label, [])
            if not label_records:
                continue
            gen_kwargs = dict(GEN_CONFIG_BY_TYPE[label])
            for start in tqdm(range(0, len(label_records), batch_size), desc=f"generate {label}"):
                batch_records = label_records[start : start + batch_size]
                images = [open_rgb_or_raise(Path(r["image_path"])) for r in batch_records]
                pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values, **gen_kwargs)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                for record, pred in zip(batch_records, preds):
                    pred = normalize_text(pred, False, False)
                    rows.append(
                        {
                            "image": record.get("image"),
                            "label": record["label"],
                            "target": record["text"],
                            "prediction": pred,
                            "raw_text": record.get("raw_text", record["text"]),
                        }
                    )
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if old_decoder_use_cache is not None and hasattr(model, "decoder"):
            model.decoder.config.use_cache = old_decoder_use_cache
        if was_training:
            model.train()

    write_jsonl(output_jsonl, rows)

    def compute_subset(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        refs = [r["target"] for r in subset]
        hyps = [r["prediction"] for r in subset]
        exact = sum(r == h for r, h in zip(refs, hyps)) / max(1, len(refs))
        return {
            "num_samples": len(refs),
            "cer": float(cer(refs, hyps)) if refs else None,
            "wer": float(wer(refs, hyps)) if refs else None,
            "exact_match": exact,
            "marker_stats_targets": marker_stats(refs),
            "marker_stats_predictions": marker_stats(hyps),
        }

    metrics = {"overall": compute_subset(rows), "by_label": {}}
    for label in TARGET_LABELS:
        subset = [r for r in rows if r["label"] == label]
        metrics["by_label"][label] = compute_subset(subset)
    return metrics


def print_config(args: SimpleNamespace) -> None:
    print("=" * 80)
    print(f"Branch: {args.branch_name}")
    print(f"Target labels: {args.target_labels}")
    print(f"Model: {args.model_name}")
    print(f"Train dir: {args.train_dir}")
    print(f"Crop manifest: {args.crop_manifest if args.use_cropped_manifest else 'disabled'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Final dir: {args.final_dir}")
    print(f"Optimizer: {args.optim}")
    print(f"LR/scheduler: {args.train_lr} / {args.lr_scheduler_type}, warmup={args.warmup_ratio}")
    print(f"Special marker tokens: {SPECIAL_TEXT_MARKER_TOKENS}")
    print(f"Add special marker tokens: {args.add_special_marker_tokens}")
    print(f"Add Ukrainian tokens: {args.add_ukrainian_tokens}")
    print(f"Strip text markers for train: {args.strip_text_markers_for_train}")
    print("=" * 80)


def main() -> None:
    args = build_config()
    load_optional_env_file(args.env_file)
    finalize_config(args)
    print_config(args)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)
    if args.hf_cache_dir:
        args.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    hf_token = resolve_hf_token(args)
    cache_kwargs = {"cache_dir": str(args.hf_cache_dir)} if args.hf_cache_dir else {}

    metadata_records, metadata_report = load_metadata_records(args.train_dir, args)
    if metadata_report["missing_image_files"] and args.fail_on_missing_image:
        save_json(args.output_dir / "reports" / "metadata_report.json", metadata_report)
        raise FileNotFoundError(f"Missing images found. See {args.output_dir / 'reports' / 'metadata_report.json'}")

    train_records, val_records = load_or_create_splits(metadata_records, args)
    train_records = limit_per_class(train_records, args.train_max_per_class, args.seed, args.target_labels)

    data_report = {
        "metadata_report": metadata_report,
        "train_counts": count_by_label(train_records, args.target_labels),
        "val_counts": count_by_label(val_records, args.target_labels),
        "train_text_lengths": summarize_text_lengths(train_records, args.target_labels),
        "val_text_lengths": summarize_text_lengths(val_records, args.target_labels),
        "train_marker_stats": marker_stats([r["raw_text"] for r in train_records]),
        "val_marker_stats": marker_stats([r["raw_text"] for r in val_records]),
    }
    save_json(args.output_dir / "reports" / "data_report.json", data_report)
    print(json.dumps(data_report["train_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(data_report["val_counts"], ensure_ascii=False, indent=2))

    processor = load_trocr_processor(args.model_name, args.base_processor_model, hf_token, cache_kwargs)
    model_source = str(args.resume_model_dir) if args.resume_model_dir else args.model_name
    model = load_vision_encoder_decoder_model(model_source, hf_token, cache_kwargs)
    added_tokens = configure_tokenizer_and_model(processor, model, args)
    save_json(args.output_dir / "reports" / "tokenizer_added_tokens.json", {"added_tokens": added_tokens})

    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "decoder"):
            model.decoder.config.use_cache = False
        model.gradient_checkpointing_enable()

    train_dataset = HpaDataset(train_records, processor, args.max_target_length)
    val_dataset = HpaDataset(val_records, processor, args.max_target_length)
    data_collator = default_data_collator

    training_args = build_training_args(args)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics(processor),
        "callbacks": callbacks,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("[INFO] Starting single-phase training...")
    trainer.train()
    print("[INFO] Saving best/final model...")
    trainer.save_model(str(args.final_dir))
    processor.save_pretrained(str(args.final_dir))

    if not args.skip_generation_eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        metrics = evaluate_generation(
            model=model,
            processor=processor,
            records=val_records,
            device=device,
            batch_size=args.gen_eval_batch_size,
            output_jsonl=args.output_dir / "reports" / "val_predictions.jsonl",
        )
        save_json(args.output_dir / "reports" / "final_val_metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"[DONE] Final model saved to: {args.final_dir}")


if __name__ == "__main__":
    main()
