from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageEnhance, ImageFilter


Image.MAX_IMAGE_PIXELS = None

TARGET_TYPES = ("handwritten", "printed", "annotation")
CROP_PAD_RATIO = 0.0
HARD_MIN_CER = 0.25
HARD_MIN_WER = 0.50
AUG_COPIES_BY_TYPE = {
    "handwritten": 1,
    "printed": 2,
    "annotation": 4,
}
ALWAYS_AUGMENT_TYPES = {"annotation"}
JPEG_QUALITY = 95
MANIFEST_NAME = "stage2b_hardtype_aug_samples.jsonl"
PROMPT_VERSION = "wrong_sample_hybrid_prompt_no_pad_v1"

SOURCE_HINTS = {
    "dictation": "Ukrainian dictation handwriting. Do not complete from canonical text; read only visible characters.",
    "archive": "Historical Ukrainian/Cyrillic document. Preserve old spelling; do not modernize.",
    "school": "School homework. It may contain corrections, teacher marks, formulas, and mixed handwriting/print.",
    "university": "University exam/coursework. It may contain formulas, tables, chemistry notation, and technical symbols.",
}
DEFAULT_SOURCE_HINT = "Read only visible characters from this crop."

SPECIAL_TEXT_MARKER_RULES = (
    "Use [illegible] only for unreadable words inside an otherwise legible text region. "
    "Use ~~word~~ for visible strikethrough and ~~old~~{new} for visible correction."
)

STAGE_B_GUARDRAILS = (
    "The final transcription must be supported by the crop. "
    "Do not complete missing words from source hint, language prior, or canonical dictation text. "
    "Do not translate, correct grammar, normalize spelling, expand abbreviations, summarize, "
    "or infer hidden/missing text. No JSON, no Markdown, no explanation."
)

CROP_PROMPTS = {
    "handwritten": (
        "Transcribe the visible handwritten text exactly. Preserve punctuation, line content, "
        "corrections, spelling mistakes, capitalization, digits, abbreviations, quotes, hyphens, "
        "line-final dashes, visible spacing, and strikethrough markers. Return only text."
    ),
    "printed": (
        "Transcribe the visible printed or typed text exactly. Preserve punctuation, line content, "
        "corrections, spelling mistakes, capitalization, digits, abbreviations, quotes, hyphens, "
        "line-final dashes, visible spacing, and strikethrough markers. Return only text."
    ),
    "annotation": "Read this short annotation or teacher mark. Return only the exact visible text.",
}

AUG_ROTATE_DEGREES = {
    "handwritten": 1.0,
    "printed": 0.6,
    "annotation": 1.5,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_type(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_source(value: Any) -> str:
    value = str(value or "").strip().lower()
    return value if value in SOURCE_HINTS else "default"


def build_crop_prompt(region_type: str, source: Any = None) -> str:
    source_hint = SOURCE_HINTS.get(normalize_source(source), DEFAULT_SOURCE_HINT)
    type_prompt = CROP_PROMPTS.get(region_type, CROP_PROMPTS["handwritten"])
    return "\n".join([source_hint, type_prompt, SPECIAL_TEXT_MARKER_RULES, STAGE_B_GUARDRAILS])


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def path_variants(file_name: str) -> list[str]:
    normalized = file_name.replace("\\", "/").lstrip("/")
    variants = {
        normalized,
        normalized.replace("/images/", "/"),
        re.sub(r"^images/", "", normalized),
    }
    if normalized.startswith("silver/"):
        variants.add("sliver/" + normalized[len("silver/") :])
    if normalized.startswith("sliver/"):
        variants.add("silver/" + normalized[len("sliver/") :])
    return [value for value in variants if value]


def default_image_roots(metadata_path: Path) -> list[Path]:
    dataset_root = metadata_path.parent
    roots = [
        dataset_root,
        dataset_root / "train",
        dataset_root / "silver",
        dataset_root / "sliver",
        Path(r"C:\Users\HP\source\rukopys_data"),
    ]
    return [path for path in roots if path.exists()]


def candidate_image_paths(file_name: str, image_roots: Iterable[Path]) -> list[Path]:
    normalized = file_name.replace("\\", "/")
    basename = Path(normalized).name
    candidates = []
    for root in image_roots:
        for variant in path_variants(normalized):
            candidates.append(root / variant)
        candidates.append(root / basename)
        candidates.append(root / "images" / basename)

    deduped = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def resolve_image_path(file_name: str, image_roots: Iterable[Path]) -> Path | None:
    for path in candidate_image_paths(file_name, image_roots):
        if path.exists():
            return path
    return None


def metadata_keys(file_name: str) -> list[str]:
    normalized = file_name.replace("\\", "/").lstrip("/")
    return [normalized, Path(normalized).name]


def load_metadata_index(metadata_path: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for row in read_jsonl(metadata_path):
        file_name = str(row.get("file_name") or "")
        for key in metadata_keys(file_name):
            index.setdefault(key, row)
    return index


def clamp_bbox(value: Any, image_width: int, image_height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(image_width, round(x1)))
    x2 = max(0, min(image_width, round(x2)))
    y1 = max(0, min(image_height, round(y1)))
    y2 = max(0, min(image_height, round(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def float_metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def make_base_sample(
    row: dict[str, Any],
    metadata_index: dict[str, dict[str, Any]],
    image_roots: list[Path],
) -> tuple[dict[str, Any] | None, str | None]:
    region_type = normalize_type(row.get("type"))
    if region_type not in TARGET_TYPES:
        return None, "non_target_type"

    answer = str(row.get("ground_truth") or "").strip()
    if not answer:
        return None, "empty_ground_truth"

    source_file_name = str(row.get("source_file_name") or "")
    source_image_path = resolve_image_path(source_file_name, image_roots)
    if source_image_path is None:
        return None, "missing_image"

    meta = None
    for key in metadata_keys(source_file_name):
        meta = metadata_index.get(key)
        if meta is not None:
            break

    if meta is not None:
        image_width = int(meta.get("image_width") or 1)
        image_height = int(meta.get("image_height") or 1)
        source = meta.get("source", "unknown")
    else:
        with Image.open(source_image_path) as image:
            image_width, image_height = image.size
        source = "unknown"

    bbox = clamp_bbox(row.get("bbox"), image_width, image_height)
    if bbox is None:
        return None, "invalid_bbox"

    cer = float_metric(row, "cer")
    wer = float_metric(row, "wer")
    return {
        "task": "crop_ocr",
        "source_image_path": str(source_image_path),
        "source_file_name": source_file_name,
        "bbox": bbox,
        "region_type": region_type,
        "prompt": build_crop_prompt(region_type, source=source),
        "answer": answer,
        "source": source,
        "augment": False,
        "aug_id": 0,
        "error_cer": cer,
        "error_wer": wer,
        "error_prediction": "" if row.get("prediction") is None else str(row.get("prediction")),
        "error_crop_path": "" if row.get("crop_path") is None else str(row.get("crop_path")),
        "row_index": row.get("row_index"),
        "region_index": row.get("region_index"),
    }, None


def should_augment(sample: dict[str, Any]) -> bool:
    region_type = sample.get("region_type")
    if region_type in ALWAYS_AUGMENT_TYPES:
        return True
    return float(sample.get("error_cer") or 0.0) >= HARD_MIN_CER or float(sample.get("error_wer") or 0.0) >= HARD_MIN_WER


def build_augmented_mix(base_samples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    mixed = []
    for sample in base_samples:
        mixed.append(dict(sample, augment=False, aug_id=0))
        if not should_augment(sample):
            continue
        for aug_id in range(1, AUG_COPIES_BY_TYPE.get(sample.get("region_type"), 0) + 1):
            mixed.append(dict(sample, augment=True, aug_id=aug_id))
    rng.shuffle(mixed)
    return mixed


def stable_aug_rng(sample: dict[str, Any], seed: int) -> random.Random:
    key = (
        f"{sample.get('source_file_name')}|{sample.get('bbox')}|{sample.get('region_type')}|"
        f"{sample.get('aug_id', 0)}|{seed}"
    )
    return random.Random(key)


def apply_light_ocr_augment(image: Image.Image, region_type: str, rng: random.Random) -> Image.Image:
    degrees = AUG_ROTATE_DEGREES.get(region_type, 1.0)
    angle = rng.uniform(-degrees, degrees)
    if abs(angle) > 0.05:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))

    if rng.random() < 0.90:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
    if rng.random() < 0.90:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.14))
    if rng.random() < 0.30:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.90, 1.18))
    if rng.random() < (0.08 if region_type == "printed" else 0.14):
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.12, 0.35)))
    return image


def sample_cache_name(sample: dict[str, Any], index: int) -> str:
    key = json.dumps(
        {
            "file": sample.get("source_file_name"),
            "bbox": sample.get("bbox"),
            "type": sample.get("region_type"),
            "aug_id": sample.get("aug_id", 0),
            "crop_pad_ratio": CROP_PAD_RATIO,
            "error_cer": sample.get("error_cer"),
            "error_wer": sample.get("error_wer"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    region_type = safe_name(str(sample.get("region_type") or "crop"))
    return f"{index:06d}_{region_type}_aug{sample.get('aug_id', 0)}_{digest}.jpg"


def crop_exact(image: Image.Image, sample: dict[str, Any]) -> Image.Image | None:
    bbox = clamp_bbox(sample.get("bbox"), image.width, image.height)
    if bbox is None:
        return None
    return image.crop(tuple(bbox))


def materialize_crops(samples: list[dict[str, Any]], output_dir: Path, seed: int) -> tuple[list[dict[str, Any]], Counter]:
    manifest = []
    stats = Counter()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[str(sample["source_image_path"])].append((index, sample))

    processed = 0
    for source_image_path, items in grouped.items():
        try:
            with Image.open(source_image_path) as image:
                image = image.convert("RGB")
                for index, sample in items:
                    rel_path = Path("crops") / sample_cache_name(sample, index)
                    out_path = output_dir / rel_path
                    if not out_path.exists():
                        crop = crop_exact(image, sample)
                        if crop is None:
                            stats["invalid_crop_bbox"] += 1
                            continue
                        if sample.get("augment"):
                            crop = apply_light_ocr_augment(crop, sample.get("region_type", "handwritten"), stable_aug_rng(sample, seed))
                        crop.save(out_path, quality=JPEG_QUALITY, optimize=True)

                    row = dict(sample)
                    row["image_path"] = rel_path.as_posix()
                    row["relative_image_path"] = row["image_path"]
                    row["crop_pad_ratio"] = CROP_PAD_RATIO
                    row.pop("source_image_path", None)
                    manifest.append(row)
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"materialized {processed}/{len(samples)} crops", flush=True)
        except Exception as exc:
            stats["source_image_failed"] += len(items)
            print(f"failed source image {source_image_path}: {exc}", flush=True)
    return manifest, stats


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = Path(args.metadata)
    error_report_path = Path(args.error_report)
    image_roots = [Path(path) for path in args.image_root] if args.image_root else default_image_roots(metadata_path)
    metadata_index = load_metadata_index(metadata_path)

    base_samples = []
    skipped = Counter()
    for row in read_jsonl(error_report_path):
        sample, reason = make_base_sample(row, metadata_index, image_roots)
        if sample is None:
            skipped[reason or "unknown"] += 1
            continue
        base_samples.append(sample)

    base_samples = sorted(
        base_samples,
        key=lambda item: (
            item.get("region_type", ""),
            -float(item.get("error_cer") or 0.0),
            -float(item.get("error_wer") or 0.0),
            item.get("source_file_name", ""),
            item.get("region_index") or 0,
        ),
    )
    if args.max_base_samples is not None:
        base_samples = base_samples[: args.max_base_samples]

    mixed = build_augmented_mix(base_samples, args.seed)
    base_counts = Counter(sample["region_type"] for sample in base_samples)
    aug_counts = Counter(sample["region_type"] for sample in mixed if sample.get("augment"))
    manifest_counts = Counter(sample["region_type"] for sample in mixed)

    stats = {
        "stage": "wrong_sample_aug_dataset",
        "prompt_version": PROMPT_VERSION,
        "error_report_path": str(error_report_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(Path(args.output_dir)),
        "image_roots": [str(path) for path in image_roots],
        "target_types": list(TARGET_TYPES),
        "crop_pad_ratio": CROP_PAD_RATIO,
        "hard_min_cer": HARD_MIN_CER,
        "hard_min_wer": HARD_MIN_WER,
        "aug_copies_by_type": AUG_COPIES_BY_TYPE,
        "always_augment_types": sorted(ALWAYS_AUGMENT_TYPES),
        "skipped": dict(sorted(skipped.items())),
        "base_counts": dict(sorted(base_counts.items())),
        "augmented_counts": dict(sorted(aug_counts.items())),
        "manifest_counts": dict(sorted(manifest_counts.items())),
        "base_sample_count": len(base_samples),
        "manifest_sample_count": len(mixed),
        "seed": args.seed,
    }
    return mixed, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-padding OCR augmentation data from wrong-sample JSONL.")
    parser.add_argument("--error-report", default="dataset/ocr_error_analysis_report.jsonl")
    parser.add_argument("--metadata", default="dataset/metadata_hybrid.jsonl")
    parser.add_argument("--output-dir", default="artifacts/stage2b_wrong_sample_aug_data")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--max-base-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    samples, stats = build_dataset(args)

    if args.dry_run:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, materialize_stats = materialize_crops(samples, output_dir, args.seed)
    stats["materialize_stats"] = dict(sorted(materialize_stats.items()))
    stats["written_manifest_count"] = len(manifest)

    write_jsonl(output_dir / MANIFEST_NAME, manifest)
    write_json(output_dir / "prompt_config.json", {
        "prompt_version": PROMPT_VERSION,
        "stage": "wrong_sample_aug_dataset",
        "source_hints": SOURCE_HINTS,
        "default_source_hint": DEFAULT_SOURCE_HINT,
        "crop_prompts": CROP_PROMPTS,
        "special_text_marker_rules": SPECIAL_TEXT_MARKER_RULES,
        "stage_b_guardrails": STAGE_B_GUARDRAILS,
        "crop_pad_ratio": CROP_PAD_RATIO,
        "target_types": list(TARGET_TYPES),
        "aug_copies_by_type": AUG_COPIES_BY_TYPE,
        "hard_min_cer": HARD_MIN_CER,
        "hard_min_wer": HARD_MIN_WER,
        "always_augment_types": sorted(ALWAYS_AUGMENT_TYPES),
        "seed": args.seed,
        "sample_count": len(manifest),
    })
    write_json(output_dir / "stats.json", stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Wrote manifest:", output_dir / MANIFEST_NAME)
    print("Wrote prompt config:", output_dir / "prompt_config.json")
    print("Crop dir:", output_dir / "crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
