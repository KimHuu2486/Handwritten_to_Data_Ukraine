#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Kansallisarkisto/cyrillic-htr-model for the RUKOPYS HPA branch.

Phases:
  0. Build/reuse a fixed validation split from dataset/train.
  1. Silver warm-up.
  2. Gold/train fine-tune.
  3. Gold/train recovery.

The model is trained as one shared HPA model for handwritten, printed, and
annotation crops. Validation inference is decoded by region type so annotation
does not get the same long generation budget as full handwritten lines.
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
    TrainerCallback,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

ImageFile.LOAD_TRUNCATED_IMAGES = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

TARGET_LABELS = ("handwritten", "printed", "annotation")

MODEL_NAME = "Kansallisarkisto/cyrillic-htr-model"
BASE_PROCESSOR_MODEL = "microsoft/trocr-base-handwritten"

# ============================================================================
# CONFIG: edit these variables before running on Thunder Compute.
# ============================================================================

# Optional .env file for HF_TOKEN and private paths. Keep tokens outside git.
# Example: Path("/workspace/.env.hpa") or None.
ENV_FILE: Optional[Path] = None
HF_TOKEN_ENV = "HF_TOKEN"

# Cloud paths. Expected:
#   DATA_ROOT/train/metadata.jsonl
#   DATA_ROOT/train/images/...
#   DATA_ROOT/silver/metadata.jsonl
#   DATA_ROOT/silver/images/...
DATA_ROOT = Path("/home/ubuntu/dataset")
TRAIN_DIR: Optional[Path] = None  # None -> DATA_ROOT / "train"
SILVER_DIR: Optional[Path] = None  # None -> DATA_ROOT / "silver"
OUTPUT_DIR = Path("/home/ubuntu/second-try/outputs")
FINAL_DIR: Optional[Path] = None  # None -> OUTPUT_DIR / "final_cyrillic_htr_model"
HF_CACHE_DIR: Optional[Path] = Path("/home/ubuntu/second-try/hf_cache")

# Resume controls. Leave RESUME_MODEL_DIR=None and START_PHASE=1 for a fresh
# full 3-phase run. To continue from a saved artifact, point RESUME_MODEL_DIR
# to a directory containing config.json/model.safetensors and set START_PHASE
# to the next phase that still needs training.
RESUME_MODEL_DIR: Optional[Path] = None
START_PHASE = 1

# For local smoke tests only, you can switch these to repo-relative paths:
# DATA_ROOT = REPO_ROOT / "dataset"
# OUTPUT_DIR = SCRIPT_DIR / "outputs_kansallisarkisto_hpa"
# HF_CACHE_DIR = None

SEED = 42
VAL_RATIO = 0.10
REBUILD_SPLITS = False

MAX_TARGET_LENGTH = 192
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
DATALOADER_NUM_WORKERS = 2
LOGGING_STEPS = 50
# Avoid disk-heavy Trainer checkpoints. TypedEvalCheckpointCallback saves the
# selected checkpoint separately.
TRAINER_SAVE_STRATEGY = "no"
SAVE_TOTAL_LIMIT: Optional[int] = 1
EARLY_STOPPING_PATIENCE = 2
OPTIM = "adafactor"
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"
REPORT_TO = "none"
NO_FP16 = False
BF16 = False
GRADIENT_CHECKPOINTING = True

PHASE1_EPOCHS = 1.0
PHASE1_LR = 3e-5
PHASE2_EPOCHS = 6.0
PHASE2_LR = 1e-5
PHASE3_EPOCHS = 2.0
PHASE3_LR = 5e-6

SILVER_MAX_PER_CLASS = 0  # 0 means use all silver HPA samples.
TRAIN_MAX_PER_CLASS = 0  # 0 means use all train_fixed HPA samples.
SKIP_TYPED_EVAL = False
TYPED_EVAL_BATCH_SIZE = 8
TYPED_EVAL_EACH_EPOCH = True
SELECT_BEST_BY_TYPED_EVAL = True

# Fail fast on dataset problems. A wrong mount should crash before training.
FAIL_ON_MISSING_IMAGE = True
FAIL_ON_IMAGE_OPEN_ERROR = True
VERIFY_IMAGE_OPEN_IN_REPORT = False

# Keep target text close to raw labels. Official metric normalization should
# stay in validation/evaluation code, not mutate training labels aggressively.
COLLAPSE_WHITESPACE = False

PHASE_NAMES = {
    1: "silver_warmup",
    2: "gold_finetune",
    3: "gold_recovery",
}

# Ukrainian is not Russian: keep these characters in the tokenizer explicitly.
UKRAINIAN_EXTRA_TOKENS = ["і", "ї", "є", "ґ", "І", "Ї", "Є", "Ґ"]

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


def build_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_name=MODEL_NAME,
        base_processor_model=BASE_PROCESSOR_MODEL,
        env_file=ENV_FILE,
        hf_token_env=HF_TOKEN_ENV,
        hf_cache_dir=HF_CACHE_DIR,
        data_root=DATA_ROOT,
        train_dir=TRAIN_DIR,
        silver_dir=SILVER_DIR,
        output_dir=OUTPUT_DIR,
        final_dir=FINAL_DIR,
        seed=SEED,
        resume_model_dir=RESUME_MODEL_DIR,
        start_phase=START_PHASE,
        val_ratio=VAL_RATIO,
        rebuild_splits=REBUILD_SPLITS,
        max_target_length=MAX_TARGET_LENGTH,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        logging_steps=LOGGING_STEPS,
        trainer_save_strategy=TRAINER_SAVE_STRATEGY,
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
        phase1_epochs=PHASE1_EPOCHS,
        phase1_lr=PHASE1_LR,
        phase2_epochs=PHASE2_EPOCHS,
        phase2_lr=PHASE2_LR,
        phase3_epochs=PHASE3_EPOCHS,
        phase3_lr=PHASE3_LR,
        silver_max_per_class=SILVER_MAX_PER_CLASS,
        train_max_per_class=TRAIN_MAX_PER_CLASS,
        skip_typed_eval=SKIP_TYPED_EVAL,
        typed_eval_batch_size=TYPED_EVAL_BATCH_SIZE,
        typed_eval_each_epoch=TYPED_EVAL_EACH_EPOCH,
        select_best_by_typed_eval=SELECT_BEST_BY_TYPED_EVAL,
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
    args.silver_dir = path_from_env(
        args.silver_dir,
        ("HPA_SILVER_DIR",),
        args.data_root / "silver",
    )
    args.output_dir = path_from_env(
        args.output_dir,
        ("HPA_OUTPUT_DIR", "OUTPUT_DIR"),
        SCRIPT_DIR / "outputs_kansallisarkisto_hpa",
    )
    args.final_dir = path_from_env(
        args.final_dir,
        ("HPA_FINAL_DIR",),
        args.output_dir / "final_cyrillic_htr_model",
    )
    args.resume_model_dir = path_from_env(
        args.resume_model_dir,
        ("HPA_RESUME_MODEL_DIR", "RESUME_MODEL_DIR"),
        None,
    )
    args.hf_cache_dir = path_from_env(
        args.hf_cache_dir,
        ("HF_CACHE_DIR", "HF_HOME"),
        None,
    )
    env_start_phase = os.environ.get("HPA_START_PHASE") or os.environ.get("START_PHASE")
    if env_start_phase:
        args.start_phase = int(env_start_phase)
    if args.start_phase not in (1, 2, 3):
        raise ValueError(f"START_PHASE must be 1, 2, or 3. Got: {args.start_phase}")


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


def compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": record["image"],
        "label": record["label"],
        "text": record["text"],
        "source_split": record["source_split"],
    }


def hydrate_record(row: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    source_split = row.get("source_split", "train")
    image = row.get("image") or row.get("file_name") or row.get("path")
    return {
        "image": image,
        "image_path": str(data_root / source_split / image),
        "label": row.get("label", row.get("type", "")),
        "text": normalize_text(row.get("text", "")),
        "source_split": source_split,
    }


def load_metadata_records(split_dir: Path, source_split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    records: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "source_split": source_split,
        "total_rows": 0,
        "bad_json_rows": 0,
        "kept_rows": 0,
        "skipped_non_hpa_label": 0,
        "skipped_missing_image": 0,
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
            except json.JSONDecodeError:
                report["bad_json_rows"] += 1
                continue

            label = item.get("label", item.get("type", ""))
            raw_label_counts[label] += 1
            if label not in TARGET_LABELS:
                report["skipped_non_hpa_label"] += 1
                continue

            image = item.get("image") or item.get("file_name") or item.get("path")
            if not image:
                report["skipped_missing_image"] += 1
                continue

            text = normalize_text(item.get("text", ""))
            if not text:
                report["skipped_empty_text"] += 1
                continue

            image_path = split_dir / image
            if not image_path.exists():
                report["missing_image_files"] += 1
                if len(report["missing_image_examples"]) < 20:
                    report["missing_image_examples"].append(
                        {"line_no": line_no, "image_path": str(image_path)}
                    )

            kept_label_counts[label] += 1
            records.append(
                {
                    "image": image,
                    "image_path": str(image_path),
                    "label": label,
                    "text": text,
                    "source_split": source_split,
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


def save_report(args: SimpleNamespace, name: str, report: Dict[str, Any]) -> None:
    save_json(args.output_dir / "reports" / f"{name}.json", report)
    if report.get("missing_image_files", 0) and FAIL_ON_MISSING_IMAGE:
        raise FileNotFoundError(
            f"{name}: {report['missing_image_files']} image files are missing. "
            f"Examples: {report.get('missing_image_examples', [])[:3]}"
        )
    if report.get("image_open_errors", 0) and FAIL_ON_IMAGE_OPEN_ERROR:
        raise RuntimeError(
            f"{name}: {report['image_open_errors']} images could not be opened. "
            f"Examples: {report.get('image_open_error_examples', [])[:3]}"
        )


def summarize_numeric(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p95": None,
            "p99": None,
        }
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


def add_truncation_summary(
    section: Dict[str, Any],
    lengths: Sequence[int],
    max_target_length: int,
) -> None:
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
                report["missing_image_examples"].append(str(image_path))
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
        label: summarize_numeric(char_lengths_by_label[label])
        for label in TARGET_LABELS
    }

    token_lengths_by_label: Dict[str, List[int]] = {label: [] for label in TARGET_LABELS}
    token_lengths: List[int] = []
    if processor is not None:
        tokenizer = processor.tokenizer
        batch_size = 512
        for start in tqdm(
            range(0, len(records), batch_size),
            desc=f"token length report {name}",
        ):
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
            add_truncation_summary(
                report["by_label"][label],
                token_lengths_by_label[label],
                max_target_length,
            )

    return report


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


def cap_per_class(
    records: Sequence[Dict[str, Any]],
    max_per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
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


def prepare_fixed_splits(args: SimpleNamespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_dir = args.output_dir / "splits"
    train_split_path = split_dir / "train_fixed_hpa.jsonl"
    val_split_path = split_dir / "val_fixed_hpa.jsonl"

    if (
        train_split_path.exists()
        and val_split_path.exists()
        and not args.rebuild_splits
    ):
        print(f"Reusing fixed split: {split_dir}")
        train_rows = [hydrate_record(r, args.data_root) for r in read_jsonl(train_split_path)]
        val_rows = [hydrate_record(r, args.data_root) for r in read_jsonl(val_split_path)]
    else:
        print("Building fixed validation split from train metadata...")
        all_train, load_report = load_metadata_records(args.train_dir, "train")
        save_report(args, "train_metadata_load_report", load_report)
        train_rows, val_rows = stratified_split(all_train, args.val_ratio, args.seed)
        write_jsonl(train_split_path, (compact_record(r) for r in train_rows))
        write_jsonl(val_split_path, (compact_record(r) for r in val_rows))
        print(f"Saved train_fixed: {train_split_path}")
        print(f"Saved val_fixed  : {val_split_path}")

    print_counts("train_fixed", train_rows)
    print_counts("val_fixed", val_rows)
    return train_rows, val_rows


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
            for token_id, mask in zip(
                encoding["input_ids"],
                encoding["attention_mask"],
            )
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

    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    tokenizer = from_pretrained_hf(
        AutoTokenizer,
        str(resume_dir),
        hf_token=None,
    )

    processor_source = find_processor_source_for_resume(resume_dir)
    if processor_source is not None:
        print(f"Loaded tokenizer from resume dir: {resume_dir}")
        print(f"Loading image processor from nearest artifact: {processor_source}")
        image_processor = ViTImageProcessor.from_pretrained(processor_source)
    else:
        print(f"Loaded tokenizer from resume dir: {resume_dir}")
        print(f"Loading image processor fallback from: {base_processor_model}")
        image_processor = from_pretrained_hf(
            ViTImageProcessor,
            base_processor_model,
            hf_token,
            **cache_kwargs,
        )

    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def resize_decoder_embeddings_if_needed(
    model: VisionEncoderDecoderModel,
    tokenizer,
) -> None:
    new_size = len(tokenizer)
    old_size = model.decoder.get_input_embeddings().weight.shape[0]
    if new_size != old_size:
        print(f"Resizing decoder embeddings: {old_size} -> {new_size}")
        model.decoder.resize_token_embeddings(new_size)
    model.config.vocab_size = new_size
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
    # Older checkpoints can store decoding parameters in model.config. Newer
    # Transformers versions warn on save, and future versions may error.
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


def configure_tokenizer_and_model(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
) -> None:
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

    if changed_vocab:
        resize_decoder_embeddings_if_needed(model, tokenizer)
    else:
        resize_decoder_embeddings_if_needed(model, tokenizer)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.cls_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no bos_token_id or cls_token_id for decoder start.")

    model.config.decoder_start_token_id = bos_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.decoder_start_token_id = bos_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id
    reset_generation_config(model, tokenizer, bos_id)

    print(
        "Token IDs | "
        f"bos={bos_id} pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id} "
        f"vocab={len(tokenizer)}"
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

        pred_str = [
            normalize_text(x)
            for x in processor.batch_decode(pred_ids, skip_special_tokens=True)
        ]
        label_str = [
            normalize_text(x)
            for x in processor.batch_decode(label_ids, skip_special_tokens=True)
        ]
        return {
            "cer": round(cer(label_str, pred_str), 6),
            "wer": round(wer(label_str, pred_str), 6),
        }

    return compute_metrics


def training_args_for_phase(
    args: SimpleNamespace,
    phase_dir: Path,
    run_name: str,
    learning_rate: float,
    num_epochs: float,
) -> Seq2SeqTrainingArguments:
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )

    fp16 = torch.cuda.is_available() and not args.no_fp16 and not args.bf16
    load_best_model_at_end = args.trainer_save_strategy != "no"
    kwargs: Dict[str, Any] = {
        "output_dir": str(phase_dir / "trainer"),
        "run_name": run_name,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "fp16": fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "save_strategy": args.trainer_save_strategy,
        strategy_key: "epoch",
        "logging_steps": args.logging_steps,
        "report_to": args.report_to,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "generation_num_beams": 1,
        "load_best_model_at_end": load_best_model_at_end,
        "remove_unused_columns": True,
    }
    if load_best_model_at_end:
        kwargs["metric_for_best_model"] = "cer"
        kwargs["greater_is_better"] = False
    if args.save_total_limit is not None:
        kwargs["save_total_limit"] = args.save_total_limit
    if args.dataloader_num_workers >= 0:
        kwargs["dataloader_num_workers"] = args.dataloader_num_workers
        if args.dataloader_num_workers > 0:
            kwargs["dataloader_prefetch_factor"] = 2
    valid_keys = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_keys}
    dropped = sorted(set(kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"[WARN] Dropped unsupported TrainingArguments keys: {dropped}")
    return Seq2SeqTrainingArguments(**filtered_kwargs)


def build_trainer(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: Seq2SeqTrainingArguments,
    patience: int,
    extra_callbacks: Optional[Sequence[TrainerCallback]] = None,
) -> Seq2SeqTrainer:
    callbacks: List[TrainerCallback] = []
    if getattr(training_args, "load_best_model_at_end", False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": default_data_collator,
        "compute_metrics": build_compute_metrics(processor),
        "callbacks": callbacks,
    }
    try:
        return Seq2SeqTrainer(processing_class=processor, **kwargs)
    except TypeError:
        return Seq2SeqTrainer(tokenizer=processor.tokenizer, **kwargs)


def train_with_resume_compat(
    trainer: Seq2SeqTrainer,
    resume_from_checkpoint: Optional[str],
):
    if resume_from_checkpoint is None:
        return trainer.train()

    # PyTorch 2.6 changed torch.load default to weights_only=True. Older
    # Transformers Trainer resumes rng_state.pth with torch.load(path), which
    # fails because RNG state contains NumPy objects. This checkpoint is local
    # and produced by this training run, so allow full pickle loading only
    # during Trainer resume.
    original_torch_load = torch.load

    def torch_load_resume_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_resume_compat
    try:
        return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        torch.load = original_torch_load


def open_rgb_or_raise(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        if FAIL_ON_IMAGE_OPEN_ERROR:
            raise RuntimeError(f"Cannot open image: {image_path}") from exc
        raise


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
        for start in tqdm(
            range(0, len(label_records), batch_size),
            desc=f"typed eval {label}",
        ):
            batch = label_records[start : start + batch_size]
            images = [open_rgb_or_raise(r["image_path"]) for r in batch]
            pixel_values = processor(
                images=images,
                return_tensors="pt",
            ).pixel_values.to(device)
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


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_model_artifact(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    path: Path,
) -> None:
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


class TypedEvalCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        processor: TrOCRProcessor,
        val_records: Sequence[Dict[str, Any]],
        device: torch.device,
        batch_size: int,
        phase_dir: Path,
    ) -> None:
        self.processor = processor
        self.val_records = list(val_records)
        self.device = device
        self.batch_size = batch_size
        self.phase_dir = phase_dir
        self.eval_dir = phase_dir / "typed_epoch_eval"
        self.best_model_dir = phase_dir / "best_typed_model"
        self.best_cer = float("inf")
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.seen_steps: set[int] = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = int(state.global_step)
        if step in self.seen_steps:
            return control
        self.seen_steps.add(step)

        model = kwargs.get("model")
        if model is None:
            return control

        epoch_label = "none" if state.epoch is None else f"{state.epoch:.4f}"
        stem = f"step_{step:08d}_epoch_{epoch_label}"
        prediction_path = self.eval_dir / f"{stem}.jsonl"
        metrics_path = self.eval_dir / f"{stem}.metrics.json"

        print(f"\nRunning typed validation for checkpoint step={step}, epoch={state.epoch}")
        model.to(self.device)
        typed_metrics = evaluate_typed_generation(
            model=model,
            processor=self.processor,
            records=self.val_records,
            device=self.device,
            batch_size=self.batch_size,
            output_jsonl=prediction_path,
        )
        typed_metrics["trainer"] = {
            "global_step": step,
            "epoch": state.epoch,
            "eval_metrics": metrics or {},
        }
        save_json(metrics_path, typed_metrics)

        typed_cer = typed_metrics["overall"]["cer"]
        print(f"Typed validation CER: {typed_cer}")
        if typed_cer is not None and typed_cer < self.best_cer:
            self.best_cer = typed_cer
            self.best_metrics = typed_metrics
            save_model_artifact(model, self.processor, self.best_model_dir)
            save_json(self.best_model_dir / "typed_val_metrics.json", typed_metrics)
            print(f"New best typed checkpoint saved: {self.best_model_dir}")
        return control


def run_phase(
    args: SimpleNamespace,
    phase_id: int,
    phase_name: str,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    learning_rate: float,
    num_epochs: float,
    device: torch.device,
) -> VisionEncoderDecoderModel:
    phase_dir = args.output_dir / f"phase{phase_id}_{phase_name}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Phase {phase_id}: {phase_name}")
    print(f"LR={learning_rate} epochs={num_epochs}")
    print_counts("phase train", train_records)
    print_counts("phase val", val_records)
    print("=" * 80)

    model.config.use_cache = False
    if hasattr(model.decoder.config, "use_cache"):
        model.decoder.config.use_cache = False

    train_dataset = HpaDataset(train_records, processor, args.max_target_length)
    val_dataset = HpaDataset(val_records, processor, args.max_target_length)
    train_args = training_args_for_phase(
        args=args,
        phase_dir=phase_dir,
        run_name=f"phase{phase_id}_{phase_name}",
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    typed_callback = None
    extra_callbacks: List[TrainerCallback] = []
    if (
        not args.skip_typed_eval
        and args.typed_eval_each_epoch
    ):
        typed_callback = TypedEvalCheckpointCallback(
            processor=processor,
            val_records=val_records,
            device=device,
            batch_size=args.typed_eval_batch_size,
            phase_dir=phase_dir,
        )
        extra_callbacks.append(typed_callback)

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=train_args,
        patience=args.early_stopping_patience,
        extra_callbacks=extra_callbacks,
    )
    resume_from_checkpoint = None
    if (
        args.resume_model_dir is not None
        and phase_id == args.start_phase
        and (args.resume_model_dir / "trainer_state.json").exists()
    ):
        resume_from_checkpoint = str(args.resume_model_dir)
        print(f"Resuming Trainer state from checkpoint: {resume_from_checkpoint}")
    train_with_resume_compat(trainer, resume_from_checkpoint)
    trainer_metrics = trainer.evaluate()
    save_json(phase_dir / "trainer_eval_metrics.json", trainer_metrics)
    print(f"Trainer eval metrics: {trainer_metrics}")

    if (
        args.select_best_by_typed_eval
        and typed_callback is not None
        and typed_callback.best_metrics is not None
        and typed_callback.best_model_dir.exists()
    ):
        selected_model_source = typed_callback.best_model_dir
        selection_report = {
            "selected_by": "typed_val_cer_overall",
            "selected_model_source": str(selected_model_source),
            "typed_val_metrics": typed_callback.best_metrics,
            "trainer_eval_metrics": trainer_metrics,
        }
        print(f"Selecting typed-best checkpoint: {selected_model_source}")
    else:
        trainer_best_dir = phase_dir / "best_trainer_model"
        save_model_artifact(model, processor, trainer_best_dir)
        print(f"Saved trainer-selected phase artifact: {trainer_best_dir}")
        selected_model_source = trainer_best_dir
        selection_report = {
            "selected_by": "trainer_eval_cer",
            "selected_model_source": str(trainer_best_dir),
            "trainer_eval_metrics": trainer_metrics,
        }

    model = VisionEncoderDecoderModel.from_pretrained(selected_model_source)
    model.to(device)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    best_model_dir = phase_dir / "best_model"
    save_model_artifact(model, processor, best_model_dir)
    selection_report["exported_best_model_dir"] = str(best_model_dir)
    save_json(phase_dir / "checkpoint_selection.json", selection_report)
    print(f"Saved selected phase artifact: {best_model_dir}")

    if not args.skip_typed_eval:
        model.to(device)
        typed_predictions = phase_dir / "typed_val_predictions.jsonl"
        typed_metrics = evaluate_typed_generation(
            model=model,
            processor=processor,
            records=val_records,
            device=device,
            batch_size=args.typed_eval_batch_size,
            output_jsonl=typed_predictions,
        )
        save_json(phase_dir / "typed_val_metrics.json", typed_metrics)
        print(f"Typed validation metrics: {typed_metrics['overall']}")

    return model


def main() -> None:
    args = build_config()
    load_optional_env_file(args.env_file)
    finalize_paths(args)
    hf_token = resolve_hf_token(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        args.output_dir / "run_config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "gen_config_by_type": GEN_CONFIG_BY_TYPE,
            "target_labels": TARGET_LABELS,
            "ukrainian_extra_tokens": UKRAINIAN_EXTRA_TOKENS,
        },
    )
    save_json(args.output_dir / "generation_config_by_type.json", GEN_CONFIG_BY_TYPE)

    print(f"Output dir : {args.output_dir}")
    print(f"Train dir  : {args.train_dir}")
    print(f"Silver dir : {args.silver_dir}")
    print(f"Final dir  : {args.final_dir}")
    print(f"Start model: {args.model_name}")
    print(f"Resume dir : {args.resume_model_dir or '(none)'}")
    print(f"Start phase: {args.start_phase}")
    print(f"HF cache   : {args.hf_cache_dir or '(default Hugging Face cache)'}")

    train_fixed_records, val_fixed_records = prepare_fixed_splits(args)
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
    else:
        print("\nLoading model and processor...")
        processor = load_processor(
            args.model_name,
            args.base_processor_model,
            hf_token=hf_token,
            cache_dir=args.hf_cache_dir,
        )
    model_cache_kwargs = (
        {"cache_dir": str(args.hf_cache_dir)} if args.hf_cache_dir is not None else {}
    )
    if args.resume_model_dir is not None:
        model = VisionEncoderDecoderModel.from_pretrained(args.resume_model_dir)
    else:
        model = from_pretrained_hf(
            VisionEncoderDecoderModel,
            args.model_name,
            hf_token,
            **model_cache_kwargs,
        )
    configure_tokenizer_and_model(processor, model)
    print_ukrainian_tokenization(processor)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

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

    silver_records, silver_load_report = load_metadata_records(args.silver_dir, "silver")
    save_report(args, "silver_metadata_load_report", silver_load_report)
    silver_records = cap_per_class(silver_records, args.silver_max_per_class, args.seed)
    save_report(
        args,
        "silver_token_report",
        build_records_report(
            "silver",
            silver_records,
            processor=processor,
            max_target_length=args.max_target_length,
            verify_image_open=args.verify_image_open_in_report,
        ),
    )

    train_records = cap_per_class(train_fixed_records, args.train_max_per_class, args.seed)
    save_report(
        args,
        "phase_train_token_report",
        build_records_report(
            "phase_train",
            train_records,
            processor=processor,
            max_target_length=args.max_target_length,
            verify_image_open=False,
        ),
    )
    if args.start_phase <= 1:
        model = run_phase(
            args=args,
            phase_id=1,
            phase_name=PHASE_NAMES[1],
            model=model,
            processor=processor,
            train_records=silver_records,
            val_records=val_fixed_records,
            learning_rate=args.phase1_lr,
            num_epochs=args.phase1_epochs,
            device=device,
        )
    else:
        print(f"Skipping phase 1 because START_PHASE={args.start_phase}")

    if args.start_phase <= 2:
        model = run_phase(
            args=args,
            phase_id=2,
            phase_name=PHASE_NAMES[2],
            model=model,
            processor=processor,
            train_records=train_records,
            val_records=val_fixed_records,
            learning_rate=args.phase2_lr,
            num_epochs=args.phase2_epochs,
            device=device,
        )
    else:
        print(f"Skipping phase 2 because START_PHASE={args.start_phase}")

    if args.start_phase <= 3:
        model = run_phase(
            args=args,
            phase_id=3,
            phase_name=PHASE_NAMES[3],
            model=model,
            processor=processor,
            train_records=train_records,
            val_records=val_fixed_records,
            learning_rate=args.phase3_lr,
            num_epochs=args.phase3_epochs,
            device=device,
        )
    else:
        print(f"Skipping phase 3 because START_PHASE={args.start_phase}")

    save_model_artifact(model, processor, args.final_dir)
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
    print(f"Saved final model: {args.final_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
