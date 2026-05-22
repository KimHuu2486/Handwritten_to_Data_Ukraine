from __future__ import annotations

import argparse
import json
import random
from typing import Any

from MainPipeline.src.common.io import load_config, read_jsonl, resolve_path, write_json, write_jsonl
from MainPipeline.src.phase1.dataset_common import image_keys


def manifest_exclude_keys(manifest_path: str | None) -> set[str]:
    """Load image identifiers from a frozen validation manifest so train rows can exclude them."""
    if not manifest_path:
        return set()
    keys: set[str] = set()
    for row in read_jsonl(resolve_path(manifest_path)):
        keys.update(image_keys(row))
    return keys


def row_is_excluded(row: dict[str, Any], exclude_keys: set[str]) -> bool:
    """Check whether a built SFT example belongs to an excluded validation image."""
    meta = row.get("meta", {})
    row_keys = set(meta.get("image_keys", []))
    return bool(row_keys & exclude_keys)


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample without replacement when possible, with replacement when the target is larger."""
    if count <= len(rows):
        return rng.sample(rows, count)
    sampled = list(rows)
    while len(sampled) < count:
        sampled.append(rng.choice(rows))
    return sampled


def load_weighted_inputs(cfg: dict[str, Any], rng: random.Random, exclude_keys: set[str]) -> list[dict[str, Any]]:
    """Load task JSONLs and apply optional weighting/capping for the final mixed dataset."""
    input_specs = cfg["inputs"]
    loaded: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for spec in input_specs:
        rows = [row for row in read_jsonl(resolve_path(spec["path"])) if not row_is_excluded(row, exclude_keys)]
        if spec.get("max_samples") is not None:
            rows = sample_rows(rows, min(int(spec["max_samples"]), len(rows)), rng)
        if not rows:
            continue
        loaded.append((spec, rows))

    total_samples = cfg.get("total_samples")
    if not total_samples:
        mixed = [row for _, rows in loaded for row in rows]
        rng.shuffle(mixed)
        return mixed

    total_weight = sum(float(spec.get("weight", 1.0)) for spec, _ in loaded)
    mixed: list[dict[str, Any]] = []
    for spec, rows in loaded:
        target = max(1, round(int(total_samples) * float(spec.get("weight", 1.0)) / total_weight))
        mixed.extend(sample_rows(rows, target, rng))
    rng.shuffle(mixed)
    return mixed


def split_train_val(rows: list[dict[str, Any]], val_ratio: float, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split mixed examples into train/val sets with a deterministic shuffle."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    val_count = max(1, round(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled[val_count:], shuffled[:val_count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mix Phase 1 A1/B1 task datasets.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    rng = random.Random(int(cfg.get("seed", 42)))
    exclude_keys = manifest_exclude_keys(cfg.get("exclude_manifest_path"))
    mixed_rows = load_weighted_inputs(cfg, rng, exclude_keys)
    if not mixed_rows:
        raise ValueError("No Phase 1 rows available after loading inputs and applying validation exclusions.")
    train_rows, val_rows = split_train_val(mixed_rows, float(cfg.get("val_ratio", 0.03)), rng)

    write_jsonl(resolve_path(cfg["output_train_jsonl"]), train_rows)
    write_jsonl(resolve_path(cfg["output_val_jsonl"]), val_rows)
    stats = {
        "total_rows": len(mixed_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "exclude_key_count": len(exclude_keys),
    }
    if cfg.get("stats_json"):
        write_json(resolve_path(cfg["stats_json"]), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
