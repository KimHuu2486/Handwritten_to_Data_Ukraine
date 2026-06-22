import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def nb_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


DEPS = r'''# Kaggle dependency cell.
INSTALL_DEPS = True

if INSTALL_DEPS:
    import subprocess
    import sys

    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade-strategy",
            "only-if-needed",
            "accelerate",
            "datasets",
            "peft",
            "bitsandbytes",
            "qwen-vl-utils",
            "trl",
            "pandas==2.2.2",
            "pillow<12",
        ],
        [sys.executable, "-m", "pip", "install", "-q", "-U", "git+https://github.com/huggingface/transformers.git"],
    ]
    for cmd in commands:
        print("Running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
'''


COMMON_CONFIG = r'''import gc
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_MODEL_CANDIDATES = [
    "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
    "/kaggle/input/qwen3-vl-8b-instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

DATASET_ROOT_CANDIDATES = [
    "/kaggle/input/datasets/quii29/rukopys-dataset",
    "/kaggle/input/rukopys-dataset",
    "/kaggle/input/handwritten-to-data",
    "/kaggle/input",
]

BAD_MANIFEST_FILENAME = "manifest_labelerror_prederror_or_labelerror_predtrue.csv"
BAD_MANIFEST_CANDIDATES = [
    f"/kaggle/input/{BAD_MANIFEST_FILENAME}",
    f"/kaggle/input/rukopys-clean-manifest/{BAD_MANIFEST_FILENAME}",
    f"/kaggle/input/rukopys-oof-audit/{BAD_MANIFEST_FILENAME}",
    f"/kaggle/input/manifest-labelerror-prederror/{BAD_MANIFEST_FILENAME}",
    f"/kaggle/working/{BAD_MANIFEST_FILENAME}",
    BAD_MANIFEST_FILENAME,
]
REQUIRE_BAD_MANIFEST = True

RANDOM_SEED = 42
VAL_RATIO = 0.03
MAX_VAL_SAMPLES = 128
MAX_TEXT_CHARS = 800
MAX_PIXELS_CROP = 131_072
CROP_PAD_RATIO = 0.04
MAX_SEQ_LENGTH = 2048
USE_GRADIENT_CHECKPOINTING = True
COLLATOR_LOG_FIRST_N = 5
COLLATOR_LOG_EVERY = 25
HEARTBEAT_SUBSTEPS = 4
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

VALID_TYPES = {"handwritten", "printed", "formula", "table", "annotation", "image", "graph"}
TEXT_TYPES = {"handwritten", "printed", "formula", "table", "annotation"}
TYPE_ALIASES = {
    "handwriting": "handwritten",
    "handwritten_text": "handwritten",
    "text": "handwritten",
    "plain_text": "printed",
    "printed_text": "printed",
    "print": "printed",
    "math": "formula",
    "equation": "formula",
    "chemical": "formula",
    "tabular": "table",
    "note": "annotation",
    "mark": "annotation",
}

SPECIAL_TEXT_MARKER_RULES = (
    "Use ~~word~~ only when strikethrough is visible, ~~old~~{new} when a visible correction replaces crossed-out text, "
    "and [illegible] only for unreadable words inside otherwise legible text. "
)

OUTPUT_NOTATION_RULES = (
    "Use stable plain-text notation for visible symbols: straight quotes, hyphen for dash variants, "
    "^ and _ for visible superscripts/subscripts, and compact whitespace. "
    "These notation choices must not change spelling, grammar, meaning, or content. "
)

COMMON_OCR_RULES = (
    "Return only the transcription. No JSON, no Markdown, no explanation. "
    "Read only what is visually present. Preserve original script, spelling, capitalization, punctuation, line breaks, "
    "digits, abbreviations, corrections, teacher marks, and old orthography. "
    "Do not translate, correct grammar, normalize spelling, expand abbreviations, autocomplete, or infer hidden text. "
    + SPECIAL_TEXT_MARKER_RULES
    + OUTPUT_NOTATION_RULES
)

CROP_PROMPTS = {
    "handwritten": "Transcribe this cropped handwritten Ukrainian/Cyrillic text exactly. " + COMMON_OCR_RULES,
    "printed": "Transcribe this cropped printed or typed text exactly. " + COMMON_OCR_RULES,
    "annotation": "Read this short annotation, teacher mark, grade, correction, or numbering exactly. " + COMMON_OCR_RULES,
    "formula": (
        "Read this standalone math, logic, physics, chemistry, or technical expression exactly as written. "
        "Use LaTeX only when it is the clearest compact representation; do not wrap the answer in dollar signs. "
        "Do not solve, simplify, normalize notation beyond plain-text transcription, or explain. "
        + SPECIAL_TEXT_MARKER_RULES
        + OUTPUT_NOTATION_RULES
    ),
    "table": (
        "Read this table region exactly. Return only pipe-separated table text. Use one output line per visual row "
        "and | between cells. Preserve empty cells with empty fields, row order, column order, wrapped cell text, "
        "numbers, units, punctuation, visible spelling mistakes, corrections, and strikethrough markers. "
        "Do not infer missing cells, rebalance columns, summarize, or explain. "
        + OUTPUT_NOTATION_RULES
    ),
    "default": "Transcribe the visible content exactly. " + COMMON_OCR_RULES,
}
'''


COMMON_HELPERS = r'''def first_existing(paths):
    for item in paths:
        p = Path(item)
        if p.exists():
            return p
    return None


def find_model_id():
    for item in BASE_MODEL_CANDIDATES:
        if item.startswith("/") and Path(item).exists():
            return item
        if not item.startswith("/"):
            return item
    raise FileNotFoundError("No Qwen3-VL base model found. Add a Kaggle model input or enable internet.")


def normalize_type(value):
    value = str(value or "handwritten").strip().lower().replace("-", "_").replace(" ", "_")
    value = TYPE_ALIASES.get(value, value)
    return value if value in VALID_TYPES else "handwritten"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_dataset_root():
    for item in DATASET_ROOT_CANDIDATES:
        root = Path(item)
        if (root / "train" / "metadata.jsonl").exists():
            return root
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for meta in sorted(input_root.rglob("train/metadata.jsonl")):
            return meta.parent.parent
    local = Path("dataset")
    if (local / "train" / "metadata.jsonl").exists():
        return local
    raise FileNotFoundError("Could not find train/metadata.jsonl. Set DATASET_ROOT_CANDIDATES.")


def find_bad_manifest_path():
    direct = first_existing(BAD_MANIFEST_CANDIDATES)
    if direct is not None:
        return direct
    for root in [Path("/kaggle/input"), Path("/kaggle/working"), Path(".")]:
        if not root.exists():
            continue
        hits = sorted(root.rglob(BAD_MANIFEST_FILENAME))
        if hits:
            return hits[0]
    if REQUIRE_BAD_MANIFEST:
        raise FileNotFoundError(
            f"Could not find {BAD_MANIFEST_FILENAME}. Upload it as a Kaggle dataset/input or add its path to BAD_MANIFEST_CANDIDATES."
        )
    return None


def as_bool_series(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_bad_image_basenames():
    path = find_bad_manifest_path()
    if path is None:
        return set(), {"manifest_path": None, "rows": 0}
    df = pd.read_csv(path)
    image_col = None
    for col in ["image", "file_name", "filename", "path"]:
        if col in df.columns:
            image_col = col
            break
    if image_col is None:
        raise ValueError(f"No image/file_name column found in {path}. Columns: {df.columns.tolist()}")

    mask = pd.Series([True] * len(df))
    level_cols = [col for col in ["level"] if col in df.columns]
    if level_cols:
        mask = mask & df["level"].astype(str).str.strip().str.lower().isin(["drop", "bad", "remove", "1", "true"])
    flag_cols = [c for c in ["LabelError_PredError", "LabelError_PredTrue"] if c in df.columns]
    if flag_cols:
        flag_mask = pd.Series([False] * len(df))
        for col in flag_cols:
            flag_mask = flag_mask | as_bool_series(df[col])
        mask = mask | flag_mask

    bad = {Path(str(x)).name for x in df.loc[mask, image_col].dropna().tolist() if str(x).strip()}
    return bad, {"manifest_path": str(path), "rows": int(len(df)), "bad_images": int(len(bad)), "image_col": image_col}


def image_basename(file_name):
    return Path(str(file_name or "")).name


def load_clean_train_records():
    root = find_dataset_root()
    train_meta = root / "train" / "metadata.jsonl"
    records = read_jsonl(train_meta)
    bad_images, manifest_info = load_bad_image_basenames()
    clean = [r for r in records if image_basename(r.get("file_name")) not in bad_images]
    dropped = len(records) - len(clean)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_meta_path = OUTPUT_DIR / "clean_train_metadata_filtered.jsonl"
    with open(clean_meta_path, "w", encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "dataset_root": str(root),
        "train_metadata": str(train_meta),
        "records_total": len(records),
        "records_dropped_by_manifest": dropped,
        "records_clean": len(clean),
        "clean_metadata_path": str(clean_meta_path),
        **manifest_info,
    }
    (OUTPUT_DIR / "clean_filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return root, "train", clean, bad_images, report


def resolve_image_path(root, split, file_name):
    root = Path(root)
    raw = Path(str(file_name))
    name = raw.name
    stem = raw.stem
    candidate_names = [name] + [stem + ext for ext in IMAGE_EXTENSIONS if stem + ext != name]
    candidates = [
        root / split / raw,
        root / raw,
    ]
    for candidate_name in candidate_names:
        candidates.extend([
            root / split / "images" / candidate_name,
            root / split / candidate_name,
            root / "images" / candidate_name,
        ])
    for p in candidates:
        if p.exists():
            return str(p)
    return str(root / split / "images" / candidate_names[0])


def clamp_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def resize_to_pixel_budget(image, max_pixels=MAX_PIXELS_CROP):
    image = image.convert("RGB")
    total = max(1, image.size[0] * image.size[1])
    if total <= max_pixels:
        return image
    scale = (max_pixels / total) ** 0.5
    new_size = (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_with_padding(image_path, bbox, pad_ratio=CROP_PAD_RATIO):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        return resize_to_pixel_budget(img.crop((x1, y1, x2, y2)))


def has_cyrillic(text):
    return any(0x0400 <= ord(ch) <= 0x04FF for ch in str(text or ""))


def make_region_samples(records, root, split, allowed_types=None, max_text_chars=MAX_TEXT_CHARS, repeat_by_type=None):
    allowed_types = set(allowed_types or TEXT_TYPES)
    repeat_by_type = repeat_by_type or {}
    samples = []
    skipped = Counter()
    for record in records:
        image_path = resolve_image_path(root, split, record.get("file_name", ""))
        if not Path(image_path).exists():
            skipped["missing_image"] += 1
            continue
        width = int(record.get("image_width") or 0)
        height = int(record.get("image_height") or 0)
        if width <= 0 or height <= 0:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                skipped["bad_image"] += 1
                continue
        for region_idx, region in enumerate(record.get("regions") or []):
            rtype = normalize_type(region.get("type"))
            if rtype not in allowed_types:
                skipped["filtered_type"] += 1
                continue
            text = str(region.get("text") or "").strip()
            if not text:
                skipped["empty_text"] += 1
                continue
            if len(text) > max_text_chars:
                skipped["long_text"] += 1
                continue
            box = clamp_box(region.get("bbox"), width, height)
            if box is None:
                skipped["bad_box"] += 1
                continue
            prompt = CROP_PROMPTS.get(rtype, CROP_PROMPTS["default"])
            repeat = max(1, int(repeat_by_type.get(rtype, 1)))
            for rep in range(repeat):
                samples.append({
                    "sample_kind": "region_crop",
                    "image_path": image_path,
                    "bbox": box,
                    "region_type": rtype,
                    "prompt": prompt,
                    "answer": text,
                    "source": record.get("source", "unknown"),
                    "file_name": record.get("file_name", ""),
                    "region_index": region_idx,
                    "repeat_index": rep,
                })
    return samples, skipped


def cap_samples_by_type(samples, limits, seed=RANDOM_SEED):
    if not limits:
        return samples, {}
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for item in samples:
        key = (item.get("file_name"), item.get("region_index"), item.get("sample_kind"), item.get("answer"))
        grouped[(item.get("region_type", "unknown"), key)].append(item)

    by_type = defaultdict(list)
    for (rtype, key), items in grouped.items():
        by_type[rtype].append(items)

    selected = []
    report = {}
    for rtype, groups in sorted(by_type.items()):
        rng.shuffle(groups)
        limit = limits.get(rtype, len(groups))
        kept = groups[: min(limit, len(groups))]
        selected.extend(item for group in kept for item in group)
        report[rtype] = {
            "available_groups": len(groups),
            "kept_groups": len(kept),
            "kept_samples": sum(len(group) for group in kept),
            "limit": limit,
        }
    rng.shuffle(selected)
    return selected, report


def split_samples(samples, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for item in samples:
        by_source[item.get("source", "unknown")].append(item)
    train, val = [], []
    for source, items in sorted(by_source.items()):
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 20 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
        print(f"source={source}: train={len(items[n_val:])} val={len(items[:n_val])}")
    rng.shuffle(train)
    rng.shuffle(val)
    if MAX_VAL_SAMPLES is not None:
        val = val[:MAX_VAL_SAMPLES]
    return train, val


def load_sample_image(sample):
    if sample.get("sample_kind") == "synthetic_render":
        return render_text_image(sample["render_text"], int(sample["render_seed"]))
    return crop_with_padding(sample["image_path"], sample["bbox"])


SAMPLE_SCHEMA_DEFAULTS = {
    "sample_kind": "region_crop",
    "render_text": "",
    "render_seed": 0,
    "image_path": "",
    "bbox": [0, 0, 0, 0],
    "region_type": "handwritten",
    "prompt": "",
    "answer": "",
    "source": "unknown",
    "file_name": "",
    "region_index": -1,
    "repeat_index": 0,
}


def normalize_sample_schema(sample):
    row = dict(SAMPLE_SCHEMA_DEFAULTS)
    row.update(sample)
    if row["bbox"] is None:
        row["bbox"] = [0, 0, 0, 0]
    row["bbox"] = [int(v) for v in row["bbox"]]
    row["render_seed"] = int(row.get("render_seed") or 0)
    row["region_index"] = int(row.get("region_index") or 0)
    row["repeat_index"] = int(row.get("repeat_index") or 0)
    for key in ["sample_kind", "render_text", "image_path", "region_type", "prompt", "answer", "source", "file_name"]:
        row[key] = str(row.get(key) or "")
    return row


def normalize_samples_schema(samples):
    return [normalize_sample_schema(sample) for sample in samples]


def summarize_samples(samples):
    return {
        "count": len(samples),
        "by_kind": dict(Counter(item.get("sample_kind", "unknown") for item in samples)),
        "by_type": dict(Counter(item.get("region_type", "unknown") for item in samples)),
        "by_source": dict(Counter(item.get("source", "unknown") for item in samples)),
    }
'''


STAGE0_CONFIG = r'''# Stage 0: Ukrainian glyph/script warm-up.
STAGE_NAME = "stage0_ukrainian_glyph"
OUTPUT_DIR = Path("/kaggle/working/qwen3vl_ukrainian_stage0")
FINAL_ADAPTER_NAME = "qwen3vl_ukrainian_stage0_lora_final"
START_LORA_CANDIDATES = []
START_LORA_REQUIRED = False
START_LORA_HINTS = []

RUN_TRAINING = True
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 2
LEARNING_RATE = 5e-5
MAX_STEPS = 250
SAVE_STEPS = 125
LOGGING_STEPS = 1
SAVE_TOTAL_LIMIT = 2
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
USE_GRADIENT_CHECKPOINTING = False

STAGE0_SYNTHETIC_SAMPLES = 3000
STAGE0_REAL_CROP_LIMITS = {
    "handwritten": 900,
    "printed": 250,
    "annotation": 250,
}
STAGE0_REAL_MAX_TEXT_CHARS = 100
STAGE0_ALLOWED_REAL_TYPES = {"handwritten", "printed", "annotation"}
STAGE0_PROMPT = (
    "Transcribe this Ukrainian/Cyrillic glyph, word, or short line exactly. "
    "Do not correct spelling or replace Latin/Cyrillic lookalikes unless that is what is visibly written. "
    + OUTPUT_NOTATION_RULES
)
'''


STAGE0_DATA = r'''from PIL import ImageDraw, ImageFilter, ImageFont, ImageOps

UKRAINIAN_ALPHABET = list("АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя")
HARD_UKRAINIAN_EXAMPLES = [
    "і ї є ґ",
    "І Ї Є Ґ",
    "Україна український Київ",
    "пір'я м'ята обʼєднує",
    "ґанок ґрунт ґудзик",
    "їжак їдальня країна",
    "єдність Європа",
    "ш щ и й г ґ",
    "с о р х а е і",
    "c o p x a e i",
    "село поле хата ріка",
]


def clean_text_unit(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def collect_clean_text_units(records):
    lines = []
    words = []
    for record in records:
        for region in record.get("regions") or []:
            rtype = normalize_type(region.get("type"))
            if rtype not in {"handwritten", "printed", "annotation"}:
                continue
            text = clean_text_unit(region.get("text"))
            if not text or not has_cyrillic(text):
                continue
            if 1 <= len(text) <= 140:
                lines.append(text)
            words.extend(re.findall(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґʼ'’\\-]{2,}", text))
    words = [w for w in words if has_cyrillic(w) and len(w) <= 28]
    return sorted(set(lines)), sorted(set(words))


def find_font_paths():
    roots = [Path("/usr/share/fonts"), Path("/kaggle/input")]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.ttf", "*.otf"):
            for path in root.rglob(ext):
                low = path.name.lower()
                if any(token in low for token in ["emoji", "symbol", "awesome"]):
                    continue
                hits.append(str(path))
    preferred = [p for p in hits if any(token in Path(p).name.lower() for token in ["dejavu", "liberation", "noto", "free"])]
    return (preferred or hits)[:80]


FONT_PATHS = find_font_paths()
print("Fonts found:", len(FONT_PATHS), FONT_PATHS[:5])


def wrap_render_text(text, max_chars):
    words = str(text).split()
    if not words:
        return [str(text)]
    lines = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 <= max_chars:
            cur = (cur + " " + word).strip()
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines[:3]


def render_text_image(text, seed):
    rng = random.Random(seed)
    font_size = rng.randint(30, 62)
    font_path = rng.choice(FONT_PATHS) if FONT_PATHS else None
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    lines = wrap_render_text(text, rng.randint(18, 42))
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    margin_x = rng.randint(18, 42)
    margin_y = rng.randint(14, 34)
    line_gap = rng.randint(4, 12)
    width = max(64, max(widths or [20]) + margin_x * 2)
    height = max(48, sum(heights or [20]) + line_gap * max(0, len(lines) - 1) + margin_y * 2)
    bg = rng.randint(230, 255)
    img = Image.new("RGB", (width, height), (bg, bg, max(220, bg - rng.randint(0, 10))))
    draw = ImageDraw.Draw(img)
    ink = rng.randint(15, 70)
    y = margin_y
    for line, box, line_h in zip(lines, boxes, heights):
        x = margin_x + rng.randint(-4, 4)
        draw.text((x, y - box[1]), line, font=font, fill=(ink, ink, ink + rng.randint(0, 30)))
        y += line_h + line_gap
    if rng.random() < 0.65:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.55)))
    if rng.random() < 0.80:
        noise = Image.effect_noise(img.size, rng.uniform(3.0, 10.0)).convert("L")
        noise = ImageOps.colorize(noise, (225, 225, 225), (255, 255, 255))
        img = Image.blend(img, noise, rng.uniform(0.02, 0.08))
    if rng.random() < 0.65:
        img = img.rotate(rng.uniform(-2.0, 2.0), expand=True, fillcolor=(bg, bg, bg))
    return resize_to_pixel_budget(img, MAX_PIXELS_CROP)


def build_synthetic_texts(lines, words, n, seed):
    rng = random.Random(seed)
    texts = []
    base_lines = lines or HARD_UKRAINIAN_EXAMPLES
    base_words = words or re.findall(r"[А-Яа-яІіЇїЄєҐґʼ'’\\-]{2,}", " ".join(HARD_UKRAINIAN_EXAMPLES))
    for item in HARD_UKRAINIAN_EXAMPLES:
        texts.append(item)
    while len(texts) < n:
        mode = rng.random()
        if mode < 0.20:
            k = rng.randint(1, 12)
            texts.append(" ".join(rng.choice(UKRAINIAN_ALPHABET) for _ in range(k)))
        elif mode < 0.55:
            k = rng.randint(1, 5)
            texts.append(" ".join(rng.choice(base_words) for _ in range(k)))
        elif mode < 0.85:
            texts.append(rng.choice(base_lines))
        else:
            left = rng.choice(["c", "o", "p", "x", "a", "e", "i"])
            right = rng.choice(["с", "о", "р", "х", "а", "е", "і"])
            texts.append(f"{left} {right} {rng.choice(base_words)}")
    rng.shuffle(texts)
    return texts[:n]


root, split, clean_records, bad_images, filter_report = load_clean_train_records()
lines, words = collect_clean_text_units(clean_records)
print("Clean Ukrainian text pool:", {"lines": len(lines), "words": len(words)})

synthetic_texts = build_synthetic_texts(lines, words, STAGE0_SYNTHETIC_SAMPLES, RANDOM_SEED)
synthetic_samples = [
    {
        "sample_kind": "synthetic_render",
        "render_text": text,
        "render_seed": RANDOM_SEED * 1_000_003 + i,
        "region_type": "ukrainian_glyph",
        "prompt": STAGE0_PROMPT,
        "answer": text,
        "source": "stage0_synthetic_clean_text",
        "file_name": "",
        "region_index": i,
    }
    for i, text in enumerate(synthetic_texts)
]

real_samples, skipped = make_region_samples(
    clean_records,
    root,
    split,
    allowed_types=STAGE0_ALLOWED_REAL_TYPES,
    max_text_chars=STAGE0_REAL_MAX_TEXT_CHARS,
)
real_samples, cap_report = cap_samples_by_type(real_samples, STAGE0_REAL_CROP_LIMITS, RANDOM_SEED)
samples = synthetic_samples + real_samples
samples = normalize_samples_schema(samples)
train_rows, val_rows = split_samples(samples)

summary = {
    "stage": STAGE_NAME,
    "synthetic": len(synthetic_samples),
    "real": len(real_samples),
    "train": len(train_rows),
    "val": len(val_rows),
    "real_skipped": dict(skipped),
    "real_cap_report": cap_report,
    "sample_summary": summarize_samples(samples),
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "stage0_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(OUTPUT_DIR / "stage0_preview_samples.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in samples[:200]),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
train_ds = Dataset.from_list(train_rows)
val_ds = Dataset.from_list(val_rows)
print(train_ds[0])
'''


STAGE1_CONFIG = r'''# Stage 1: clean ground-truth region OCR fine-tune.
STAGE_NAME = "stage1_clean_region"
OUTPUT_DIR = Path("/kaggle/working/qwen3vl_ukrainian_stage1_clean_region")
FINAL_ADAPTER_NAME = "qwen3vl_ukrainian_stage1_clean_region_lora_final"
START_LORA_CANDIDATES = [
    "/kaggle/input/qwen3vl-ukrainian-stage0/qwen3vl_ukrainian_stage0_lora_final",
    "/kaggle/input/ukrainian-curriculum-stage0/qwen3vl_ukrainian_stage0_lora_final",
    "/kaggle/working/qwen3vl_ukrainian_stage0/qwen3vl_ukrainian_stage0_lora_final",
]
START_LORA_REQUIRED = True
START_LORA_HINTS = ["stage0", "ukrainian"]

RUN_TRAINING = True
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-5
MAX_STEPS = 900
SAVE_STEPS = 450
LOGGING_STEPS = 1
SAVE_TOTAL_LIMIT = 2
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

STAGE1_ALLOWED_TYPES = {"handwritten", "printed", "formula", "table", "annotation"}
STAGE1_TYPE_LIMITS = {
    "handwritten": 5000,
    "printed": 900,
    "formula": 1200,
    "table": 900,
    "annotation": 900,
}
STAGE1_REPEAT_BY_TYPE = {}
'''


STAGE1_DATA = r'''root, split, clean_records, bad_images, filter_report = load_clean_train_records()
samples, skipped = make_region_samples(
    clean_records,
    root,
    split,
    allowed_types=STAGE1_ALLOWED_TYPES,
    max_text_chars=MAX_TEXT_CHARS,
    repeat_by_type=STAGE1_REPEAT_BY_TYPE,
)
samples, cap_report = cap_samples_by_type(samples, STAGE1_TYPE_LIMITS, RANDOM_SEED)
samples = normalize_samples_schema(samples)
train_rows, val_rows = split_samples(samples)

summary = {
    "stage": STAGE_NAME,
    "train": len(train_rows),
    "val": len(val_rows),
    "skipped": dict(skipped),
    "cap_report": cap_report,
    "sample_summary": summarize_samples(samples),
    "uses_yolo_boxes": False,
    "bbox_source": "train metadata regions[].bbox",
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "stage1_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(OUTPUT_DIR / "stage1_preview_samples.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in samples[:200]),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not train_rows:
    raise RuntimeError("No Stage 1 training rows. Check dataset root, image layout, and manifest filter.")
train_ds = Dataset.from_list(train_rows)
val_ds = Dataset.from_list(val_rows)
print(train_ds[0])
'''


STAGE2_CONFIG = r'''# Stage 2: final OCR prompt/format alignment on clean GT bboxes.
STAGE_NAME = "stage2_final_format"
OUTPUT_DIR = Path("/kaggle/working/qwen3vl_ukrainian_stage2_final_format")
FINAL_ADAPTER_NAME = "qwen3vl_ukrainian_stage2_final_lora_final"
START_LORA_CANDIDATES = [
    "/kaggle/input/qwen3vl-ukrainian-stage1-clean-region/qwen3vl_ukrainian_stage1_clean_region_lora_final",
    "/kaggle/input/ukrainian-curriculum-stage1/qwen3vl_ukrainian_stage1_clean_region_lora_final",
    "/kaggle/working/qwen3vl_ukrainian_stage1_clean_region/qwen3vl_ukrainian_stage1_clean_region_lora_final",
]
START_LORA_REQUIRED = True
START_LORA_HINTS = ["stage1", "clean", "region", "ukrainian"]

RUN_TRAINING = True
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1e-5
MAX_STEPS = 500
SAVE_STEPS = 250
LOGGING_STEPS = 1
SAVE_TOTAL_LIMIT = 2
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

STAGE2_ALLOWED_TYPES = {"handwritten", "printed", "formula", "table", "annotation"}
STAGE2_TYPE_LIMITS = {
    "handwritten": 3500,
    "printed": 800,
    "formula": 1200,
    "table": 1200,
    "annotation": 1000,
}
STAGE2_REPEAT_BY_TYPE = {
    "formula": 2,
    "table": 2,
    "annotation": 2,
}
'''


STAGE2_DATA = r'''root, split, clean_records, bad_images, filter_report = load_clean_train_records()
samples, skipped = make_region_samples(
    clean_records,
    root,
    split,
    allowed_types=STAGE2_ALLOWED_TYPES,
    max_text_chars=MAX_TEXT_CHARS,
    repeat_by_type=STAGE2_REPEAT_BY_TYPE,
)
samples, cap_report = cap_samples_by_type(samples, STAGE2_TYPE_LIMITS, RANDOM_SEED)
samples = normalize_samples_schema(samples)
train_rows, val_rows = split_samples(samples)

summary = {
    "stage": STAGE_NAME,
    "train": len(train_rows),
    "val": len(val_rows),
    "skipped": dict(skipped),
    "cap_report": cap_report,
    "sample_summary": summarize_samples(samples),
    "uses_yolo_boxes": False,
    "bbox_source": "train metadata regions[].bbox",
    "purpose": "low-LR final prompt and output-format alignment",
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "stage2_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(OUTPUT_DIR / "stage2_preview_samples.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in samples[:200]),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not train_rows:
    raise RuntimeError("No Stage 2 training rows. Check dataset root, image layout, and manifest filter.")
train_ds = Dataset.from_list(train_rows)
val_ds = Dataset.from_list(val_rows)
print(train_ds[0])
'''


TRAINING = r'''from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer


def configure_processor(processor):
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    return processor


def force_fp16_config(model):
    model.config.torch_dtype = torch.float16
    for attr in ("text_config", "vision_config"):
        cfg = getattr(model.config, attr, None)
        if cfg is not None:
            cfg.torch_dtype = torch.float16
            if hasattr(cfg, "dtype"):
                cfg.dtype = "float16"


def make_lora_config():
    kwargs = dict(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        return LoraConfig(**kwargs, use_rslora=True)
    except TypeError:
        return LoraConfig(**kwargs)


def find_lora_dir(candidates, required=False):
    for item in candidates:
        p = Path(item)
        if (p / "adapter_config.json").exists():
            return p
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    for root in search_roots:
        if not root.exists():
            continue
        hits = sorted(root.rglob("adapter_config.json"))
        scored = []
        for hit in hits:
            parent = hit.parent
            low = str(parent).lower()
            score = sum(token in low for token in START_LORA_HINTS)
            score += sum(token in low for token in ["ukrainian", "rukopys", "qwen3vl"])
            scored.append((score, parent))
        if scored:
            scored.sort(key=lambda x: (-x[0], str(x[1])))
            if scored[0][0] > 0:
                return scored[0][1]
    if required:
        raise FileNotFoundError(
            "Could not find required previous-stage LoRA adapter. Add the previous notebook output as a Kaggle input "
            "or update START_LORA_CANDIDATES."
        )
    return None


class OCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.calls = 0
        try:
            self.assistant_prefix = processor.tokenizer.encode(
                "<|im_start|>assistant\n", allowed_special="all", add_special_tokens=False
            )
        except TypeError:
            self.assistant_prefix = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def should_log(self):
        return self.calls <= COLLATOR_LOG_FIRST_N or (
            COLLATOR_LOG_EVERY and self.calls % COLLATOR_LOG_EVERY == 0
        )

    def build_messages(self, sample):
        image = load_sample_image(sample)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "max_pixels": MAX_PIXELS_CROP},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]

    def __call__(self, examples):
        self.calls += 1
        t0 = time.time()
        should_log = self.should_log()
        if should_log:
            kinds = Counter(str(ex.get("sample_kind", "unknown")) for ex in examples)
            types = Counter(str(ex.get("region_type", "unknown")) for ex in examples)
            print(f"[{STAGE_NAME}] collator batch {self.calls} start: kinds={dict(kinds)} types={dict(types)}", flush=True)
        messages = [self.build_messages(ex) for ex in examples]
        if should_log:
            print(f"[{STAGE_NAME}] collator batch {self.calls} images/messages ready in {time.time() - t0:.1f}s", flush=True)
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        try:
            batch = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                text_kwargs={"padding": True, "truncation": True, "max_length": MAX_SEQ_LENGTH, "return_tensors": "pt"},
                images_kwargs={"return_tensors": "pt"},
                videos_kwargs={"return_tensors": "pt"},
            )
        except TypeError:
            batch = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
        if should_log:
            shape = tuple(batch["input_ids"].shape) if "input_ids" in batch else None
            print(f"[{STAGE_NAME}] collator batch {self.calls} encoded shape={shape} elapsed={time.time() - t0:.1f}s", flush=True)
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        prefix = torch.tensor(self.assistant_prefix, device=labels.device)
        prefix_len = len(self.assistant_prefix)
        eos_id = self.processor.tokenizer.eos_token_id
        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()
            start = -1
            for j in range(0, len(ids) - prefix_len + 1):
                if ids[j:j + prefix_len] == self.assistant_prefix:
                    start = j + prefix_len
                    break
            actual_len = int(batch["attention_mask"][i].sum().item())
            truncated = actual_len >= MAX_SEQ_LENGTH and (eos_id is None or ids[actual_len - 1] != eos_id)
            if start >= 0 and not truncated:
                labels[i, :start] = -100
            else:
                labels[i, :] = -100

        batch["labels"] = labels
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.float16)
        return batch


class PrintProgressCallback(TrainerCallback):
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.micro_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"[{self.name}] start: max_steps={state.max_steps}, grad_accum={args.gradient_accumulation_steps}", flush=True)

    def heartbeat(self, args, state, suffix):
        elapsed = time.time() - (self.start_time or time.time())
        msg = (
            f"[{self.name}] heartbeat step={state.global_step}/{state.max_steps} "
            f"micro={self.micro_steps} elapsed={elapsed/60:.1f}m {suffix}"
        )
        if torch.cuda.is_available():
            msg += " vram=" + ",".join(
                f"{i}:{torch.cuda.memory_allocated(i)/1024**3:.1f}GB"
                for i in range(torch.cuda.device_count())
            )
        print(msg, flush=True)

    def on_substep_end(self, args, state, control, **kwargs):
        self.micro_steps += 1
        if self.micro_steps <= 3 or (HEARTBEAT_SUBSTEPS and self.micro_steps % HEARTBEAT_SUBSTEPS == 0):
            self.heartbeat(args, state, "substep_end")

    def on_step_end(self, args, state, control, **kwargs):
        self.heartbeat(args, state, "optimizer_step_end")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        elapsed = time.time() - (self.start_time or time.time())
        step = max(1, state.global_step)
        eta = (elapsed / step) * max(0, state.max_steps - step)
        msg = f"[{self.name}] step {state.global_step}/{state.max_steps}"
        if "loss" in logs:
            msg += f" loss={logs['loss']:.4f}"
        if "learning_rate" in logs:
            msg += f" lr={logs['learning_rate']:.2e}"
        msg += f" elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        if torch.cuda.is_available():
            msg += " vram=" + ",".join(
                f"{i}:{torch.cuda.memory_allocated(i)/1024**3:.1f}GB"
                for i in range(torch.cuda.device_count())
            )
        print(msg, flush=True)


class OOMRecoverySFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            try:
                loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            except TypeError:
                loss = super().training_step(model, inputs)
            if self.args.device != loss.device:
                loss = loss.to(self.args.device)
            return loss
        except torch.cuda.OutOfMemoryError:
            print("OOM during training step; clearing cache and skipping this batch.", flush=True)
            for p in model.parameters():
                p.grad = None
            torch.cuda.empty_cache()
            gc.collect()
            return torch.tensor(0.0, device=self.args.device)


if not RUN_TRAINING:
    print("RUN_TRAINING=False, dataset was built but training is skipped.")
else:
    model_id = find_model_id()
    start_lora_dir = find_lora_dir(START_LORA_CANDIDATES, required=START_LORA_REQUIRED)
    print("Base model:", model_id)
    print("Start LoRA:", start_lora_dir)
    processor = configure_processor(AutoProcessor.from_pretrained(model_id, trust_remote_code=True))

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model_kwargs = dict(
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16, **model_kwargs)
    except TypeError:
        base_model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16, **model_kwargs)
    force_fp16_config(base_model)
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )
    if not USE_GRADIENT_CHECKPOINTING and hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()

    if start_lora_dir is not None:
        model = PeftModel.from_pretrained(base_model, start_lora_dir, is_trainable=True)
    else:
        model = get_peft_model(base_model, make_lora_config())
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / STAGE_NAME),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=False,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(MAX_STEPS * 0.03)),
        max_grad_norm=0.3,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = OOMRecoverySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=OCRDataCollator(processor),
        callbacks=[PrintProgressCallback(STAGE_NAME)],
    )
    trainer.train()

    final_dir = OUTPUT_DIR / FINAL_ADAPTER_NAME
    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    (final_dir / "rukopys_prompt_config.json").write_text(
        json.dumps(
            {
                "stage_name": STAGE_NAME,
                "base_model": str(model_id),
                "start_lora_dir": str(start_lora_dir) if start_lora_dir is not None else None,
                "crop_prompts": CROP_PROMPTS,
                "max_pixels_crop": MAX_PIXELS_CROP,
                "max_seq_length": MAX_SEQ_LENGTH,
                "bad_manifest_filename": BAD_MANIFEST_FILENAME,
                "uses_yolo_boxes": False,
                "bbox_source": "train metadata regions[].bbox or synthetic render",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved LoRA adapter to", final_dir)
'''


NOTEBOOKS = [
    (
        "00_stage0_ukrainian_glyph_pretrain.ipynb",
        "# Stage 0: Ukrainian Glyph/Script Warm-Up\n\nBuilds a clean Stage 0 dataset from filtered train labels plus synthetic Ukrainian/Cyrillic renderings, then trains a Qwen3-VL QLoRA adapter. It excludes images listed in the bad manifest and does not use YOLO boxes.",
        STAGE0_CONFIG,
        STAGE0_DATA,
    ),
    (
        "01_stage1_clean_region_finetune.ipynb",
        "# Stage 1: Clean GT Region OCR Fine-Tune\n\nContinues from the Stage 0 adapter and trains on ground-truth `regions[].bbox` crops from clean train metadata. YOLO boxes are intentionally not used.",
        STAGE1_CONFIG,
        STAGE1_DATA,
    ),
    (
        "02_stage2_final_format_finetune.ipynb",
        "# Stage 2: Final Prompt/Format Fine-Tune\n\nContinues from the Stage 1 adapter with a lower learning rate and extra repeats for formulas, tables, and annotations. It still uses only clean ground-truth metadata boxes.",
        STAGE2_CONFIG,
        STAGE2_DATA,
    ),
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for filename, title, config, data in NOTEBOOKS:
        cells = [
            nb_cell("markdown", title),
            nb_cell("code", DEPS),
            nb_cell("code", COMMON_CONFIG + "\n" + config),
            nb_cell("code", COMMON_HELPERS),
            nb_cell("code", data),
            nb_cell("code", TRAINING),
        ]
        path = ROOT / filename
        write_notebook(path, cells)
        print("Wrote", path)


if __name__ == "__main__":
    main()
