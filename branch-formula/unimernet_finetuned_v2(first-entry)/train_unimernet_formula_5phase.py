#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curriculum fine-tune UniMERNet cho nhánh formula RUKOPYS đã crop sẵn.

Input metadata format:
{"image": "images/xxx.jpg", "label": "formula", "text": "...", "formula_group": "inline_simple"}

Pipeline 3 phase:
1. Phase 1: gold clean inline_simple.
2. Phase 2: resume phase 1, train gold inline_simple + medium_long.
3. Phase 3: resume phase 2, train silver_filtered ngắn, rồi gold recovery.

Chuẩn bị metadata trước:
python prepare_formula_curriculum_metadata.py \
  --input /home/ubuntu/dataset/train/metadata.jsonl \
  --output-dir /home/ubuntu/dataset/train_formula_curriculum \
  --split gold

python prepare_formula_curriculum_metadata.py \
  --input /home/ubuntu/dataset/silver/metadata.jsonl \
  --output-dir /home/ubuntu/dataset/silver_formula_curriculum \
  --split silver --silver-filter

Chạy train:
python train_unimernet_formula_3phase.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
from jiwer import cer
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# CONFIG - chỉnh ở đây cho Thunder Compute
# =============================================================================

# Defaults mirror train_unimernet_formula.py; override with env vars on Thunder if needed.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models/unimernet_base"))

_PRETRAINED_ENV = os.environ.get("PRETRAINED")
PRETRAINED = Path(_PRETRAINED_ENV) if _PRETRAINED_ENV else MODEL_DIR / "unimernet_base.pth"

# Dataset root chứa thư mục images/
GOLD_ROOT = Path(os.environ.get("GOLD_ROOT", "train"))
SILVER_ROOT = Path(os.environ.get("SILVER_ROOT", "silver"))

# Metadata đã tạo bằng prepare_formula_curriculum_metadata.py
GOLD_CURRICULUM_DIR = Path(os.environ.get("GOLD_CURRICULUM_DIR", str(GOLD_ROOT.parent / f"{GOLD_ROOT.name}_formula_curriculum")))
SILVER_CURRICULUM_DIR = Path(os.environ.get("SILVER_CURRICULUM_DIR", str(SILVER_ROOT.parent / f"{SILVER_ROOT.name}_formula_curriculum")))

PHASE1_GOLD_METADATA = GOLD_CURRICULUM_DIR / "phase1_gold_inline_simple.jsonl"
PHASE2_GOLD_METADATA = GOLD_CURRICULUM_DIR / "phase2_gold_inline_medium.jsonl"
PHASE3_GOLD_METADATA = GOLD_CURRICULUM_DIR / "phase3_gold_recovery.jsonl"
PHASE3_SILVER_METADATA = SILVER_CURRICULUM_DIR / "phase3_silver_filtered.jsonl"
PHASE5_STRUCTURED_METADATA = GOLD_CURRICULUM_DIR / "optional_gold_structured.jsonl"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/home/ubuntu/unimernet_formula_curriculum_3phase"))

# Processor/model
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 672
MAX_SEQ_LEN = 384
MAX_TEXT_CHARS = 1536

# Same train processor choice as train_unimernet_formula.py.
USE_TRAIN_AUGMENT = True

# Batch size tối ưu cho RTX A6000 48GB. Nếu OOM: giảm BATCH_SIZE xuống 16 và đặt GRAD_ACCUM_STEPS = 2.
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = None
NUM_WORKERS_CAP = 16

AMP = True
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
SEED = 42
DEVICE = None

# Split validation trên gold theo image_path để giảm leak style/page.
GOLD_VAL_RATIO = 0.10

# Checkpoint/save/eval
SAVE_EVERY_STEPS = 0
VAL_MAX_BATCHES = 0
TOKENIZER_PAD_BOS_POLICY = "warn"

# Phase config
RUN_PHASE1 = True
RUN_PHASE2 = True
RUN_PHASE3 = True
RUN_PHASE3_SILVER = True
RUN_PHASE3_GOLD_RECOVERY = True
RUN_PHASE5_STRUCTURED = True

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 3
PHASE3_SILVER_EPOCHS = 1
PHASE3_GOLD_EPOCHS = 3
PHASE5_STRUCTURED_EPOCHS = 2

PHASE1_LR = 3e-6
PHASE2_LR = 1e-6
PHASE3_SILVER_LR = 5e-6
PHASE3_GOLD_LR = 1e-6
PHASE5_STRUCTURED_LR = 5e-7
MIN_LR = 1e-7

PHASE1_WARMUP_STEPS = 50
PHASE2_WARMUP_STEPS = 30
PHASE3_SILVER_WARMUP_STEPS = 50
PHASE3_GOLD_WARMUP_STEPS = 30
PHASE5_STRUCTURED_WARMUP_STEPS = 20

# 0 = dùng toàn bộ. Dùng để test nhanh.
MAX_PHASE1_SAMPLES = 0
MAX_PHASE2_SAMPLES = 0
MAX_PHASE3_SILVER_SAMPLES = 0
MAX_PHASE3_GOLD_SAMPLES = 0
MAX_PHASE5_STRUCTURED_SAMPLES = 0

# Early stopping theo CER sau mỗi epoch.
EARLY_STOP_PATIENCE = 2

# Nếu muốn train tiếp từ checkpoint bất kỳ, đặt path tại đây.
RESUME_CHECKPOINT = None

# =============================================================================


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


# Metric-aware normalizer rút gọn nhưng giữ các rule quan trọng của RUKOPYS formula.
_LATEX_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "ϑ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\upsilon": "υ", r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ",
    r"\psi": "ψ", r"\omega": "ω", r"\Gamma": "Γ", r"\Delta": "Δ",
    r"\Theta": "Θ", r"\Lambda": "Λ", r"\Xi": "Ξ", r"\Pi": "Π",
    r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←",
    r"\leftrightarrow": "↔", r"\Rightarrow": "⇒", r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔", r"\implies": "⇒", r"\iff": "⇔",
    r"\cap": "∩", r"\cup": "∪", r"\subset": "⊂", r"\supset": "⊃",
    r"\subseteq": "⊆", r"\supseteq": "⊇", r"\in": "∈", r"\notin": "∉",
    r"\emptyset": "∅", r"\varnothing": "∅", r"\oplus": "⊕", r"\otimes": "⊗",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\forall": "∀", r"\exists": "∃", r"\neg": "¬",
    r"\sqrt": "√", r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\ldots": "…", r"\dots": "…", r"\cdots": "⋯",
    r"\therefore": "∴", r"\because": "∵", r"\perp": "⊥",
    r"\angle": "∠", r"\parallel": "∥", r"\square": "□",
    r"\Box": "□", r"\triangle": "△",
}
_LATEX_COMMANDS_RE = re.compile("|".join(re.escape(k) for k in sorted(_LATEX_SYMBOLS, key=len, reverse=True)))
_LATEX_TEXT_WRAPPER = re.compile(r"\\text\{([^}]*)\}")
_LATEX_LEFT_RIGHT = re.compile(r"\\(left|right)\s*([()|\[\]{}.])")
_LATEX_SPACING = re.compile(r"\\[,;:!]|\\quad|\\qquad|\\hspace\{[^}]*\}")
_LATEX_TABLE_ENV = re.compile(r"\\begin\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix)\}(?:\{[^}]*\})?\s*")
_LATEX_TABLE_ENV_END = re.compile(r"\s*\\end\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix)\}")
_LATEX_TABLE_ROW_SEP = re.compile(r"\s*\\\\\s*")
_LATEX_TABLE_COL_SEP = re.compile(r"\s*&\s*")
_FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_BRACES = re.compile(r"√\s*\{([^{}]*)\}")
_ARROW_RIGHT_DECOR = re.compile(r"\\(?:overrightarrow|vec)\s*\{([^{}]*)\}")
_ARROW_LEFT_DECOR = re.compile(r"\\overleftarrow\s*\{([^{}]*)\}")
_DECORATOR = re.compile(r"\\(?:bar|hat|overline|widetilde|widehat|dot|ddot)\s*\{([^{}]*)\}")
_LATEX_BRACE = re.compile(r"([_^])\{([^}]+)\}")
_MULT_SIGNS = re.compile(r"[*∗⋅×‧]")
_COMBINING_MARKS = re.compile(r"[\u0300-\u036F\u20D0-\u20FF]")
_DASHES = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]")
_MULTI_SPACE = re.compile(r"[ \t\u00A0\u2000-\u200B\u3000]+")
_QUOTES_DOUBLE = re.compile(r'["\u201C\u201D\u201E\u00AB\u00BB\u2033]')
_QUOTES_SINGLE = re.compile(r"['\u2018\u2019\u02BC\u0027\u2032]")
_LATIN_TO_CYRILLIC = str.maketrans({
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х",
})
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ", "0123456789+-=()n")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎", "0123456789+-=()")


def normalize_formula_for_metric(text: Any) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    text = _LATEX_TABLE_ENV.sub("", text)
    text = _LATEX_TABLE_ENV_END.sub("", text)
    text = _LATEX_TABLE_ROW_SEP.sub("\n", text)
    text = _LATEX_TABLE_COL_SEP.sub("|", text)
    text = _LATEX_TEXT_WRAPPER.sub(r"\1", text)
    text = _LATEX_LEFT_RIGHT.sub(r"\2", text)
    text = _LATEX_SPACING.sub(" ", text)
    prev = None
    while text != prev:
        prev = text
        text = _FRAC.sub(r"\1/\2", text)
    text = _ARROW_RIGHT_DECOR.sub(r"→\1", text)
    text = _ARROW_LEFT_DECOR.sub(r"←\1", text)
    text = _DECORATOR.sub(r"\1", text)
    text = _LATEX_COMMANDS_RE.sub(lambda m: _LATEX_SYMBOLS[m.group()], text)
    text = _SQRT_BRACES.sub(r"√\1", text)
    text = _COMBINING_MARKS.sub("", text)
    text = _MULT_SIGNS.sub("·", text)

    converted = []
    for ch in text:
        if ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ":
            converted.append("^" + ch.translate(_SUPERSCRIPTS))
        elif ch in "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎":
            converted.append("_" + ch.translate(_SUBSCRIPTS))
        else:
            converted.append(ch)
    text = "".join(converted)

    text = _LATEX_BRACE.sub(r"\1\2", text)
    text = text.replace("->", "→")
    text = text.translate(_LATIN_TO_CYRILLIC)
    text = _DASHES.sub("-", text)
    text = _QUOTES_DOUBLE.sub('"', text)
    text = _QUOTES_SINGLE.sub("'", text)
    text = _MULTI_SPACE.sub(" ", text)
    return "\n".join(part.strip() for part in text.strip().splitlines())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sample_cer(ref: str, pred: str) -> float:
    return levenshtein(ref, pred) / max(1, len(ref))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_num_workers(num_workers: Optional[int], cap: int) -> int:
    """None = dùng CPU count nhưng giới hạn để tránh tạo quá nhiều process đọc ảnh."""
    if num_workers is not None:
        return int(num_workers)
    cpu_count = os.cpu_count() or 4
    return max(2, min(int(cap), cpu_count))


def resolve_image_path(image_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else image_root / p


@dataclass
class FormulaSample:
    image_path: Path
    text: str
    line_no: int
    group: str
    source: str


class FormulaJsonlDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
        image_root: Path,
        processor,
        max_text_chars: int = 1536,
        drop_missing: bool = True,
    ) -> None:
        self.metadata_path = metadata_path
        self.image_root = image_root
        self.processor = processor
        self.samples = self._read_samples(max_text_chars, drop_missing)

    def _read_samples(self, max_text_chars: int, drop_missing: bool) -> List[FormulaSample]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        samples: List[FormulaSample] = []
        skipped = {"bad_json": 0, "not_formula": 0, "empty_text": 0, "too_long": 0, "missing_image": 0}

        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped["bad_json"] += 1
                    continue

                if item.get("label", item.get("type", "")) != "formula":
                    skipped["not_formula"] += 1
                    continue

                text = normalize_text(item.get("text", ""))
                if not text:
                    skipped["empty_text"] += 1
                    continue
                if len(text) > max_text_chars:
                    skipped["too_long"] += 1
                    continue

                rel_image = item.get("image", item.get("file_name", item.get("path", "")))
                if not rel_image:
                    skipped["missing_image"] += 1
                    continue
                image_path = resolve_image_path(self.image_root, rel_image)
                if drop_missing and not image_path.exists():
                    skipped["missing_image"] += 1
                    continue

                samples.append(
                    FormulaSample(
                        image_path=image_path,
                        text=text,
                        line_no=int(item.get("line_no", line_no)),
                        group=item.get("formula_group", "unknown"),
                        source=item.get("split_source", ""),
                    )
                )

        print(f"\nLoaded {len(samples)} samples from {self.metadata_path}")
        print("Skipped:", ", ".join(f"{k}={v}" for k, v in skipped.items()))
        if not samples:
            raise ValueError(f"Empty dataset: {self.metadata_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.samples[idx]
        try:
            image = Image.open(sample.image_path).convert("RGB")
            tensor = self.processor(image)
            if tensor is None:
                return None
            return {
                "image": tensor,
                "text_input": sample.text,
                "image_path": str(sample.image_path),
                "line_no": sample.line_no,
                "group": sample.group,
            }
        except Exception as exc:
            print(f"[WARN] skip {sample.image_path}: {exc}")
            return None


def collate_formula(samples: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    return {
        "image": torch.stack([s["image"] for s in samples], dim=0),
        "text_input": [s["text_input"] for s in samples],
        "image_path": [s["image_path"] for s in samples],
        "line_no": [s["line_no"] for s in samples],
        "group": [s["group"] for s in samples],
    }


def split_by_image_path(dataset: FormulaJsonlDataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    if val_ratio <= 0 or len(dataset) < 2:
        return Subset(dataset, list(range(len(dataset)))), Subset(dataset, [])

    image_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        image_to_indices.setdefault(str(sample.image_path), []).append(idx)

    image_keys = list(image_to_indices)
    rng = random.Random(seed)
    rng.shuffle(image_keys)
    n_val_images = max(1, int(round(len(image_keys) * val_ratio)))
    val_images = set(image_keys[:n_val_images])

    train_indices, val_indices = [], []
    for image_key, indices in image_to_indices.items():
        if image_key in val_images:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices and val_indices:
        train_indices.append(val_indices.pop())
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def maybe_subsample(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle and len(dataset) >= batch_size,
        collate_fn=collate_formula,
    )


def find_weights(model_dir: Path, explicit_path: Optional[Path]) -> Optional[Path]:
    if explicit_path:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {explicit_path}")
        return explicit_path

    for name in ("unimernet_base.pth", "pytorch_model.pth", "model.pth", "checkpoint_best.pth"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def load_unimernet_model(model_dir: Path, weights_path: Optional[Path], max_seq_len: int):
    import unimernet.models  # noqa
    import unimernet.processors  # noqa
    from unimernet.models.unimernet.unimernet import UniMERModel

    cfg = OmegaConf.create(
        {
            "model_name": str(model_dir),
            "model_config": {
                "model_name": str(model_dir),
                "max_seq_len": max_seq_len,
            },
            "tokenizer_name": "nougat",
            "tokenizer_config": {"path": str(model_dir)},
            "load_pretrained": weights_path is not None,
            "pretrained": str(weights_path) if weights_path else "",
            "load_finetuned": False,
            "finetuned": "",
        }
    )
    return UniMERModel.from_config(cfg)


def load_processor(name: str, image_size: Sequence[int]):
    import unimernet.processors  # noqa
    from unimernet.processors import load_processor

    cfg = OmegaConf.create({"name": name, "image_size": list(image_size)})
    return load_processor(name, cfg)


def check_unimernet_tokenizer(model, policy: str) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        print("[WARN] no tokenizer attr")
        return
    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    print(f"Tokenizer ids: pad={pad_id}, bos={bos_id}, eos={eos_id}")
    if pad_id is not None and bos_id is not None and pad_id == bos_id:
        msg = "pad_token_id == bos_token_id; generation masks may be ambiguous."
        if policy == "raise":
            raise ValueError(msg)
        print(f"[WARN] {msg}")


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing.missing_keys:
        print(f"Missing keys: {len(missing.missing_keys)}")
    if missing.unexpected_keys:
        print(f"Unexpected keys: {len(missing.unexpected_keys)}")


def save_checkpoint(
    output_dir: Path,
    name: str,
    model,
    optimizer,
    scaler,
    step: int,
    epoch: int,
    metric: Optional[float],
    args: SimpleNamespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scaler": scaler.state_dict() if scaler else None,
            "step": step,
            "epoch": epoch,
            "metric": metric,
            "args": vars(args),
        },
        path,
    )
    return path


def make_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: torch.device, max_batches: int, output_path: Optional[Path]) -> Dict[str, float]:
    model.eval()
    refs: List[str] = []
    preds: List[str] = []
    groups: List[str] = []
    rows: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Validate", leave=False)):
        if batch is None:
            continue
        if max_batches > 0 and batch_idx >= max_batches:
            break

        image = batch["image"].to(device, non_blocking=True)
        out = model.generate({"image": image}, temperature=0.0, do_sample=False)
        batch_preds = out["pred_str"] if isinstance(out, dict) and "pred_str" in out else out

        batch_preds = [normalize_formula_for_metric(x) for x in batch_preds]
        batch_refs = [normalize_formula_for_metric(x) for x in batch["text_input"]]

        refs.extend(batch_refs)
        preds.extend(batch_preds)
        groups.extend(batch["group"])

        if output_path is not None:
            for image_path, line_no, group, ref, pred in zip(batch["image_path"], batch["line_no"], batch["group"], batch_refs, batch_preds):
                rows.append(
                    {
                        "image_path": image_path,
                        "line_no": line_no,
                        "group": group,
                        "reference": ref,
                        "prediction": pred,
                        "sample_cer": sample_cer(ref, pred),
                        "ref_len": len(ref),
                        "pred_len": len(pred),
                    }
                )

    corpus_cer = cer(refs, preds) if refs else 1.0
    sample_cers = [sample_cer(r, p) for r, p in zip(refs, preds)]
    exact = sum(r == p for r, p in zip(refs, preds)) / max(1, len(refs))

    metrics: Dict[str, float] = {
        "cer": float(corpus_cer),
        "mean_sample_cer": float(np.mean(sample_cers)) if sample_cers else 1.0,
        "exact_match": float(exact),
        "num_samples": float(len(refs)),
    }

    # Metric theo group
    for group in sorted(set(groups)):
        idxs = [i for i, g in enumerate(groups) if g == group]
        if not idxs:
            continue
        g_refs = [refs[i] for i in idxs]
        g_preds = [preds[i] for i in idxs]
        g_cers = [sample_cers[i] for i in idxs]
        metrics[f"{group}_cer"] = float(cer(g_refs, g_preds))
        metrics[f"{group}_mean_sample_cer"] = float(np.mean(g_cers))
        metrics[f"{group}_num"] = float(len(idxs))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def run_train_stage(
    stage_name: str,
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    args: SimpleNamespace,
    epochs: int,
    lr: float,
    warmup_steps: int,
    min_lr: float,
) -> Tuple[Dict[str, float], Path]:
    if len(train_loader) == 0:
        raise ValueError(f"{stage_name} loader is empty")

    total_update_steps = math.ceil((epochs * len(train_loader)) / args.grad_accum_steps)
    min_lr_ratio = min_lr / lr if lr > 0 else 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(total_update_steps, warmup_steps, min_lr_ratio),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    global_step = 0
    best_cer = float("inf")
    best_metrics = {"cer": float("inf"), "exact_match": 0.0, "num_samples": 0.0}
    best_path = args.output_dir / f"checkpoint_{stage_name}_best.pth"
    log_path = args.output_dir / f"{stage_name}_log.jsonl"
    no_improve_epochs = 0

    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        accum_count = 0

        pbar = tqdm(train_loader, desc=f"{stage_name} epoch {epoch}/{epochs}")
        for batch in pbar:
            if batch is None:
                continue
            image = batch["image"].to(device, non_blocking=True)
            samples = {"image": image, "text_input": batch["text_input"]}

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                loss = model(samples)["loss"]
                loss_to_backward = loss / args.grad_accum_steps

            scaler.scale(loss_to_backward).backward()
            running_loss += float(loss.detach().cpu())
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                pbar.set_postfix(loss=running_loss / max(1, accum_count), lr=f"{scheduler.get_last_lr()[0]:.2e}", step=global_step)

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    save_checkpoint(args.output_dir, f"checkpoint_{stage_name}_step{global_step}.pth", model, optimizer, scaler, global_step, epoch, None, args)

        # Flush gradient còn dư
        if accum_count % args.grad_accum_steps != 0:
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        avg_loss = running_loss / max(1, accum_count)
        metrics: Dict[str, float] = {}
        if val_loader is not None and len(val_loader) > 0:
            pred_path = args.output_dir / f"{stage_name}_epoch{epoch}_predictions.jsonl"
            metrics = evaluate_model(model, val_loader, device, args.val_max_batches, pred_path)
            current_cer = metrics["cer"]
            if current_cer < best_cer:
                best_cer = current_cer
                best_metrics = metrics
                best_path = save_checkpoint(args.output_dir, f"checkpoint_{stage_name}_best.pth", model, optimizer, scaler, global_step, epoch, best_cer, args)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "step": global_step,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"[EarlyStop] {stage_name}: no improvement for {no_improve_epochs} epochs.")
            break

    latest_path = save_checkpoint(args.output_dir, f"checkpoint_{stage_name}_latest.pth", model, optimizer, scaler, global_step, epoch, best_cer if math.isfinite(best_cer) else None, args)
    elapsed = (time.time() - start) / 60
    print(f"{stage_name} done: best_cer={best_metrics['cer']:.6f}, exact={best_metrics.get('exact_match', 0):.6f}, time={elapsed:.1f} min")
    return best_metrics, best_path if best_path.exists() else latest_path


def build_args() -> SimpleNamespace:
    resolved_num_workers = resolve_num_workers(NUM_WORKERS, NUM_WORKERS_CAP)
    return SimpleNamespace(
        model_dir=MODEL_DIR.resolve(),
        pretrained=PRETRAINED.resolve() if PRETRAINED is not None else None,
        resume_checkpoint=Path(RESUME_CHECKPOINT).resolve() if RESUME_CHECKPOINT else None,
        gold_root=GOLD_ROOT,
        silver_root=SILVER_ROOT,
        phase1_gold_metadata=PHASE1_GOLD_METADATA,
        phase2_gold_metadata=PHASE2_GOLD_METADATA,
        phase3_gold_metadata=PHASE3_GOLD_METADATA,
        phase3_silver_metadata=PHASE3_SILVER_METADATA,
        phase5_structured_metadata=PHASE5_STRUCTURED_METADATA,
        output_dir=OUTPUT_DIR.resolve(),
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        max_seq_len=MAX_SEQ_LEN,
        max_text_chars=MAX_TEXT_CHARS,
        use_train_augment=USE_TRAIN_AUGMENT,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        num_workers=resolved_num_workers,
        amp=AMP,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        device=DEVICE,
        gold_val_ratio=GOLD_VAL_RATIO,
        save_every_steps=SAVE_EVERY_STEPS,
        val_max_batches=VAL_MAX_BATCHES,
        tokenizer_pad_bos_policy=TOKENIZER_PAD_BOS_POLICY,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )


def build_gold_loaders(metadata_path: Path, root: Path, train_processor, eval_processor, args: SimpleNamespace, max_samples: int = 0):
    full_train = FormulaJsonlDataset(metadata_path, root, train_processor, max_text_chars=args.max_text_chars)
    full_eval = FormulaJsonlDataset(metadata_path, root, eval_processor, max_text_chars=args.max_text_chars)

    train_subset, _ = split_by_image_path(full_train, args.gold_val_ratio, args.seed)
    _, val_subset = split_by_image_path(full_eval, args.gold_val_ratio, args.seed)

    train_subset = maybe_subsample(train_subset, max_samples, args.seed)

    train_loader = make_loader(train_subset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_subset, args.eval_batch_size, args.num_workers, shuffle=False)
    print(f"Gold metadata: {metadata_path}")
    print(f"  train samples: {len(train_subset)}")
    print(f"  val samples  : {len(val_subset)}")
    return train_loader, val_loader


def build_single_loader(metadata_path: Path, root: Path, processor, args: SimpleNamespace, max_samples: int = 0, shuffle: bool = True):
    dataset = FormulaJsonlDataset(metadata_path, root, processor, max_text_chars=args.max_text_chars)
    dataset = maybe_subsample(dataset, max_samples, args.seed)
    print(f"Dataset: {metadata_path} | samples={len(dataset)}")
    return make_loader(dataset, args.batch_size, args.num_workers, shuffle=shuffle)


def main() -> None:
    args = build_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Batch: train={args.batch_size}, eval={args.eval_batch_size}, accum={args.grad_accum_steps}")
    print(f"Num workers: {args.num_workers}")
    print(f"Train augment: {args.use_train_augment}")

    weights_path = find_weights(args.model_dir, args.pretrained)
    if weights_path:
        print(f"Initial UniMERNet weights: {weights_path}")
    else:
        print("[WARN] No .pth checkpoint found. The model will start from its local transformer weights only.")

    train_processor_name = "formula_image_train" if args.use_train_augment else "formula_image_eval"
    train_processor = load_processor(train_processor_name, [args.image_height, args.image_width])
    eval_processor = load_processor("formula_image_eval", [args.image_height, args.image_width])

    model = load_unimernet_model(args.model_dir, weights_path, args.max_seq_len)
    check_unimernet_tokenizer(model, args.tokenizer_pad_bos_policy)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint, device)
    model.to(device)

    phase_best_paths: Dict[str, str] = {}

    # Phase 1
    if RUN_PHASE1:
        train_loader, val_loader = build_gold_loaders(args.phase1_gold_metadata, args.gold_root, train_processor, eval_processor, args, MAX_PHASE1_SAMPLES)
        metrics, best_path = run_train_stage("phase1_inline_gold", model, train_loader, val_loader, device, args, PHASE1_EPOCHS, PHASE1_LR, PHASE1_WARMUP_STEPS, MIN_LR)
        phase_best_paths["phase1"] = str(best_path)
        load_checkpoint(model, best_path, device)

    # Phase 2
    if RUN_PHASE2:
        train_loader, val_loader = build_gold_loaders(args.phase2_gold_metadata, args.gold_root, train_processor, eval_processor, args, MAX_PHASE2_SAMPLES)
        metrics, best_path = run_train_stage("phase2_medium_gold", model, train_loader, val_loader, device, args, PHASE2_EPOCHS, PHASE2_LR, PHASE2_WARMUP_STEPS, MIN_LR)
        phase_best_paths["phase2"] = str(best_path)
        load_checkpoint(model, best_path, device)

    # Phase 3: silver filtered then gold recovery
    if RUN_PHASE3:
        if RUN_PHASE3_SILVER:
            silver_loader = build_single_loader(args.phase3_silver_metadata, args.silver_root, train_processor, args, MAX_PHASE3_SILVER_SAMPLES, shuffle=True)
            # Validate vẫn dùng phase2 gold validation để silver không làm lệch checkpoint.
            _, val_loader = build_gold_loaders(args.phase2_gold_metadata, args.gold_root, train_processor, eval_processor, args, 0)
            metrics, best_path = run_train_stage("phase3_silver_filtered", model, silver_loader, val_loader, device, args, PHASE3_SILVER_EPOCHS, PHASE3_SILVER_LR, PHASE3_SILVER_WARMUP_STEPS, MIN_LR)
            phase_best_paths["phase3_silver"] = str(best_path)
            load_checkpoint(model, best_path, device)

        if RUN_PHASE3_GOLD_RECOVERY:
            train_loader, val_loader = build_gold_loaders(args.phase3_gold_metadata, args.gold_root, train_processor, eval_processor, args, MAX_PHASE3_GOLD_SAMPLES)
            metrics, best_path = run_train_stage("phase3_gold_recovery", model, train_loader, val_loader, device, args, PHASE3_GOLD_EPOCHS, PHASE3_GOLD_LR, PHASE3_GOLD_WARMUP_STEPS, MIN_LR)
            phase_best_paths["phase3_gold"] = str(best_path)
            load_checkpoint(model, best_path, device)

    # Phase 5: structured gold, low LR to learn matrix/table-like formulas without overwriting recovery.
    if RUN_PHASE5_STRUCTURED:
        train_loader, val_loader = build_gold_loaders(args.phase5_structured_metadata, args.gold_root, train_processor, eval_processor, args, MAX_PHASE5_STRUCTURED_SAMPLES)
        metrics, best_path = run_train_stage("phase5_structured_gold", model, train_loader, val_loader, device, args, PHASE5_STRUCTURED_EPOCHS, PHASE5_STRUCTURED_LR, PHASE5_STRUCTURED_WARMUP_STEPS, MIN_LR)
        phase_best_paths["phase5_structured"] = str(best_path)
        load_checkpoint(model, best_path, device)

    final_path = save_checkpoint(args.output_dir, "checkpoint_curriculum_final_best_loaded.pth", model, None, None, -1, -1, None, args)
    manifest = {
        "final_path": str(final_path),
        "phase_best_paths": phase_best_paths,
        "config": vars(args),
    }
    with (args.output_dir / "curriculum_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nTraining complete.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
