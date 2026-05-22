from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    """Return the repository root from any module inside MainPipeline/src."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a config path relative to the repo root unless it is absolute."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    base_path = Path(base).expanduser() if base is not None else repo_root()
    return (base_path if base_path.is_absolute() else repo_root() / base_path) / path


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a path and return the normalized Path."""
    normalized = Path(path)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    return normalized


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a UTF-8 JSON object with stable formatting."""
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream JSONL rows and fail fast with a line number on invalid JSON."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no} is not a JSON object")
            yield row


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL rows with compact JSON to keep generated datasets small."""
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL file, creating parent directories as needed."""
    output_path = ensure_parent(path)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write text by replacing the destination after a temp write."""
    output_path = ensure_parent(path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, output_path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON configs, and YAML configs when PyYAML is available on the VM."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return read_json(config_path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{config_path} is YAML, but PyYAML is not installed. "
                "Use a JSON config or install PyYAML in the VM environment."
            ) from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data or {}
    raise ValueError(f"Unsupported config extension: {config_path.suffix}")

