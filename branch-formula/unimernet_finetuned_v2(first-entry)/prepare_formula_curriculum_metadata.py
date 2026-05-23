#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tạo metadata formula đã phân nhóm cho UniMERNet curriculum training.

Input metadata format:
{"image": "images/xxx.jpg", "label": "formula", "text": "..."}

Output:
- all:        formula_all.jsonl
- simple:     formula_inline_simple.jsonl
- medium:     formula_medium_long.jsonl
- structured: formula_structured.jsonl
- train phase jsonl:
    phase1_gold.jsonl
    phase2_gold.jsonl
    phase3_gold.jsonl / phase3_silver_filtered.jsonl

Cách dùng:
1. Sửa các biến trong khối CONFIG.
2. Chạy trực tiếp:
   python prepare_formula_curriculum_metadata.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict


# =============================================================================
# CONFIG: sửa các biến ở đây rồi chạy trực tiếp file này.
# =============================================================================

# Nếu chạy trong project hiện tại:
#   dataset/train/metadata.jsonl
#   dataset/silver/metadata.jsonl
DATASET_ROOT = Path("C:/Users/HABAO/hoc_code/Lap_trinh_Python/TDTT_Finalterm/dataset")

RUN_GOLD = True
RUN_SILVER = True

GOLD_INPUT = DATASET_ROOT / "train" / "metadata.jsonl"
GOLD_OUTPUT_DIR = DATASET_ROOT / "train_formula_curriculum"

SILVER_INPUT = DATASET_ROOT / "silver" / "metadata.jsonl"
SILVER_OUTPUT_DIR = DATASET_ROOT / "silver_formula_curriculum"

SIMPLE_MAX_CHARS = 80
MEDIUM_MAX_CHARS = 180
MAX_SILVER_CHARS = 120
SILVER_FILTER = True

# =============================================================================


STRUCTURE_PATTERNS = [
    r"\n",
    r"\|",
    r"\\begin\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix)\}",
    r"\\end\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix)\}",
    r"\\cline",
    r"\\hline",
    r"\\hlіnе",   # nhãn trong dataset có thể dùng chữ і Cyrillic
    r"\\сlіnе",   # c Cyrillic + i Cyrillic
]

CHEM_PATTERN = re.compile(
    r"(?:[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ]?\d*){2,}|(?:ОН|Н2О|СО2|СН|Сl|Cl|Br|Вr|Na|Nа|Ca|Са|Al|Аl)",
    re.UNICODE,
)

BAD_CHARS_RE = re.compile(r"[�\uFFFD]")


def normalize_spaces(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def has_structured_layout(text: str) -> bool:
    if any(re.search(p, text) for p in STRUCTURE_PATTERNS):
        return True

    # Nhiều segment dạng "a / b / c / d" hoặc nhiều cụm số liên tiếp thường là phép chia dọc/bảng nhỏ.
    slash_count = text.count("/")
    pipe_like_count = text.count("|")
    if pipe_like_count >= 2:
        return True
    if slash_count >= 6 and len(text) > 40:
        return True

    # Nhiều dòng bị mã hóa bằng khoảng trắng: ví dụ truth table / ma trận toàn số.
    tokens = text.split()
    digitish = sum(1 for t in tokens if re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?", t))
    if len(tokens) >= 12 and digitish / max(1, len(tokens)) > 0.65:
        return True

    return False


def classify_formula(text: str, simple_max_chars: int, medium_max_chars: int) -> str:
    text = normalize_spaces(text)
    n = len(text)

    if has_structured_layout(text):
        return "structured"

    # Chuỗi quá dài thường là multi-step derivation; UniMERNet dễ hallucinate nếu trộn ngay từ đầu.
    if n > medium_max_chars:
        return "structured"

    if n <= simple_max_chars:
        return "inline_simple"

    return "medium_long"


def is_good_silver_sample(item: Dict[str, Any], group: str, max_silver_chars: int) -> bool:
    """Lọc silver bảo thủ: chỉ giữ sample ít nhiễu để curriculum không phá model."""
    text = normalize_spaces(item.get("text", ""))
    if not text:
        return False
    if BAD_CHARS_RE.search(text):
        return False
    if len(text) > max_silver_chars:
        return False

    # Với silver, phase 3 ưu tiên simple/medium; structured silver thường nhiễu, dễ làm model hallucinate.
    if group == "structured":
        return False

    # Loại bỏ nhãn có quá nhiều chữ tự nhiên so với ký hiệu/số.
    letters = sum(ch.isalpha() for ch in text)
    mathish = sum(ch.isdigit() or ch in "+-=*/·×÷^_()[]{}.,:;<>%√→←↔∈∉⊂⊃⊆⊇∪∩∞∂∇∀∃¬≈≠≤≥| " for ch in text)
    if len(text) >= 20 and letters > mathish * 1.5:
        return False

    return True


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_metadata(
    input_path: Path,
    output_dir: Path,
    split: str,
    simple_max_chars: int,
    medium_max_chars: int,
    max_silver_chars: int,
    silver_filter: bool,
) -> None:
    if split not in {"gold", "silver"}:
        raise ValueError(f"Invalid split: {split}. Use 'gold' or 'silver'.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_path}")

    groups = {
        "inline_simple": [],
        "medium_long": [],
        "structured": [],
    }
    all_formula = []
    skipped = Counter()
    label_counts = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped["bad_json"] += 1
                continue

            label = item.get("label", item.get("type", ""))
            label_counts[label] += 1
            if label != "formula":
                skipped["not_formula"] += 1
                continue

            text = normalize_spaces(item.get("text", ""))
            image = item.get("image", item.get("file_name", item.get("path", "")))
            if not image:
                skipped["missing_image"] += 1
                continue
            if not text:
                skipped["empty_text"] += 1
                continue

            group = classify_formula(text, simple_max_chars, medium_max_chars)
            out = dict(item)
            out["label"] = "formula"
            out["text"] = text
            out["formula_group"] = group
            out["split_source"] = split
            out["line_no"] = item.get("line_no", line_no)
            out["text_len"] = len(text)
            out["has_newline"] = "\n" in text
            out["has_pipe"] = "|" in text
            out["is_chemistry_like"] = bool(CHEM_PATTERN.search(text))

            all_formula.append(out)
            groups[group].append(out)

    # Metadata theo group
    write_jsonl(output_dir / "formula_all.jsonl", all_formula)
    for group, rows in groups.items():
        write_jsonl(output_dir / f"formula_{group}.jsonl", rows)

    # Phase files:
    # Phase 1: chỉ inline_simple gold.
    # Phase 2: inline_simple + medium_long gold.
    # Phase 3 gold recovery: inline_simple + medium_long, có thể thêm structured sau này nếu muốn model gánh structured.
    if split == "gold":
        phase1 = groups["inline_simple"]
        phase2 = groups["inline_simple"] + groups["medium_long"]
        phase3 = groups["inline_simple"] + groups["medium_long"]
        write_jsonl(output_dir / "phase1_gold_inline_simple.jsonl", phase1)
        write_jsonl(output_dir / "phase2_gold_inline_medium.jsonl", phase2)
        write_jsonl(output_dir / "phase3_gold_recovery.jsonl", phase3)

        # File structured riêng để train checkpoint riêng nếu sau này cần.
        write_jsonl(output_dir / "optional_gold_structured.jsonl", groups["structured"])

    if split == "silver":
        if silver_filter:
            filtered = [x for x in all_formula if is_good_silver_sample(x, x["formula_group"], max_silver_chars)]
        else:
            filtered = groups["inline_simple"] + groups["medium_long"]
        write_jsonl(output_dir / "phase3_silver_filtered.jsonl", filtered)

    print("\nDone.")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print("Label counts:", dict(label_counts))
    print("Skipped:", dict(skipped))
    print("Formula groups:")
    for group, rows in groups.items():
        print(f"  {group:15s}: {len(rows)}")
    print(f"  all_formula     : {len(all_formula)}")

    if split == "gold":
        print("\nCreated:")
        print("  phase1_gold_inline_simple.jsonl")
        print("  phase2_gold_inline_medium.jsonl")
        print("  phase3_gold_recovery.jsonl")
        print("  optional_gold_structured.jsonl")
    else:
        print("\nCreated:")
        print("  phase3_silver_filtered.jsonl")


def main() -> None:
    if RUN_GOLD:
        process_metadata(
            input_path=GOLD_INPUT,
            output_dir=GOLD_OUTPUT_DIR,
            split="gold",
            simple_max_chars=SIMPLE_MAX_CHARS,
            medium_max_chars=MEDIUM_MAX_CHARS,
            max_silver_chars=MAX_SILVER_CHARS,
            silver_filter=False,
        )

    if RUN_SILVER:
        process_metadata(
            input_path=SILVER_INPUT,
            output_dir=SILVER_OUTPUT_DIR,
            split="silver",
            simple_max_chars=SIMPLE_MAX_CHARS,
            medium_max_chars=MEDIUM_MAX_CHARS,
            max_silver_chars=MAX_SILVER_CHARS,
            silver_filter=SILVER_FILTER,
        )


if __name__ == "__main__":
    main()
