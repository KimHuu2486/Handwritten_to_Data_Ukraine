from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from MainPipeline.src.common.io import load_config, resolve_path


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a CSV and keep the original field order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Atomically write CSV rows with the given field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp_path.replace(path)


def backup_file(path: Path, timestamp: str) -> Path | None:
    """Copy an existing file to a timestamped backup path."""
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def is_error_row(row: dict[str, Any], drop_parse_fail: bool) -> bool:
    """Return True when a prediction row should be removed before resume."""
    if str(row.get("error_type", "")).strip():
        return True
    if drop_parse_fail and str(row.get("parse_ok", "")).strip().lower() in {"false", "0", "no"}:
        return True
    return False


def rebuild_submission(submission_csv: Path, prediction_rows: list[dict[str, Any]], timestamp: str, dry_run: bool) -> Path | None:
    """Rewrite submission CSV from cleaned prediction rows."""
    if dry_run:
        return None
    backup_path = backup_file(submission_csv, timestamp)
    submission_rows = [{"image": row.get("image", ""), "regions": row.get("regions", "[]")} for row in prediction_rows]
    write_csv_rows(submission_csv, ["image", "regions"], submission_rows)
    return backup_path


def clean_predictions(cfg: dict[str, Any], dry_run: bool, drop_parse_fail: bool) -> dict[str, Any]:
    """Remove failed prediction rows and keep submission CSV in sync."""
    output_cfg = cfg["output"]
    predictions_csv = resolve_path(output_cfg["predictions_csv"])
    if not predictions_csv.exists():
        raise FileNotFoundError(f"predictions CSV not found: {predictions_csv}")

    fieldnames, rows = read_csv_rows(predictions_csv)
    if "image" not in fieldnames or "regions" not in fieldnames:
        raise ValueError(f"{predictions_csv} is missing required columns: image, regions")

    failed_rows = [row for row in rows if is_error_row(row, drop_parse_fail)]
    kept_rows = [row for row in rows if not is_error_row(row, drop_parse_fail)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prediction_backup = None if dry_run else backup_file(predictions_csv, timestamp)
    submission_backup = None
    if not dry_run:
        write_csv_rows(predictions_csv, fieldnames, kept_rows)

    submission_csv_value = output_cfg.get("submission_csv")
    if submission_csv_value:
        submission_csv = resolve_path(submission_csv_value)
        submission_backup = rebuild_submission(submission_csv, kept_rows, timestamp, dry_run)

    return {
        "predictions_csv": str(predictions_csv),
        "prediction_backup": str(prediction_backup) if prediction_backup else "",
        "submission_backup": str(submission_backup) if submission_backup else "",
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(failed_rows),
        "removed_images": [row.get("image", "") for row in failed_rows],
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove failed prediction rows so infer_phase1 --resume reruns them.")
    parser.add_argument("--config", required=True, help="Inference config with output.predictions_csv and optional output.submission_csv.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be removed without writing files.")
    parser.add_argument("--drop-parse-fail", action="store_true", help="Also remove rows where parse_ok is false and error_type is empty.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    summary = clean_predictions(cfg, dry_run=args.dry_run, drop_parse_fail=args.drop_parse_fail)
    print(
        "cleaned prediction errors: "
        f"input_rows={summary['input_rows']} kept_rows={summary['kept_rows']} "
        f"removed_rows={summary['removed_rows']} dry_run={summary['dry_run']}",
        flush=True,
    )
    if summary["prediction_backup"]:
        print(f"prediction_backup={summary['prediction_backup']}", flush=True)
    if summary["submission_backup"]:
        print(f"submission_backup={summary['submission_backup']}", flush=True)
    removed_images = summary["removed_images"]
    if removed_images:
        preview = ", ".join(removed_images[:20])
        suffix = " ..." if len(removed_images) > 20 else ""
        print(f"removed_images={preview}{suffix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
