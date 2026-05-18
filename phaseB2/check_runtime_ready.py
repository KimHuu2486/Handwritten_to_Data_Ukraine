from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from b2_common import (
    load_json,
    load_prompt_config,
    load_runtime_config,
    read_jsonl,
    repo_root,
    resolve_image_path,
    resolve_project_path,
    sha256_lf,
    validate_manifest_rows,
)


def add_result(results: list[tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check readiness for B2 baseline cache runtime.")
    parser.add_argument("--config", default="phaseB2/b2_runtime_config.json")
    parser.add_argument("--check-images", action="store_true", help="Resolve every manifest image path.")
    parser.add_argument("--sample-images", type=int, default=10, help="Image sample count when --check-images is not set.")
    args = parser.parse_args()

    root = repo_root()
    config_path = resolve_project_path(args.config, root)
    results: list[tuple[str, bool, str]] = []

    if not config_path.exists():
        add_result(results, "config_exists", False, f"missing {config_path}")
        print_results(results)
        return 1

    cfg = load_runtime_config(config_path)
    add_result(results, "config_exists", True, str(config_path))

    manifest_path = resolve_project_path(cfg["manifest_path"], root)
    gate_path = resolve_project_path(cfg["gate_report_path"], root)
    adapter_path = resolve_project_path(cfg["lora_adapter_path"], root)
    base_model_path = resolve_project_path(cfg["base_model_path"], root)

    add_result(results, "manifest_exists", manifest_path.exists(), str(manifest_path))
    manifest_rows = []
    if manifest_path.exists():
        try:
            manifest_rows = read_jsonl(manifest_path)
            errors = validate_manifest_rows(manifest_rows)
            add_result(results, "manifest_schema", not errors, "; ".join(errors[:3]) or f"{len(manifest_rows)} rows")
        except Exception as exc:
            add_result(results, "manifest_schema", False, str(exc))

    add_result(results, "gate_report_exists", gate_path.exists(), str(gate_path))
    if gate_path.exists():
        try:
            gate = load_json(gate_path)
            add_result(results, "gate_report_pass", bool(gate.get("pass")), gate.get("status", ""))
            expected_sha = gate.get("outputs_sha256", {}).get("manifest")
            if expected_sha and manifest_path.exists():
                actual_sha = sha256_lf(manifest_path)
                add_result(results, "manifest_checksum_lf", actual_sha == expected_sha, actual_sha)
        except Exception as exc:
            add_result(results, "gate_report_parse", False, str(exc))

    add_result(results, "base_model_path_exists", base_model_path.exists(), str(base_model_path))
    add_result(results, "lora_adapter_exists", adapter_path.exists(), str(adapter_path))
    add_result(results, "lora_adapter_config", (adapter_path / "adapter_config.json").exists(), str(adapter_path / "adapter_config.json"))
    has_weights = any((adapter_path / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    add_result(results, "lora_adapter_weights", has_weights, str(adapter_path))

    try:
        prompt_cfg = load_prompt_config(adapter_path)
        page_prompt = str(prompt_cfg.get("page_prompt", ""))
        add_result(results, "prompt_config", True, f"page_prompt chars={len(page_prompt)}")
        max_pixels_page = prompt_cfg.get("max_pixels_page")
        configured_pixels = cfg.get("generation_params", {}).get("max_pixels_page")
        add_result(
            results,
            "max_pixels_page_consistent",
            max_pixels_page in (None, configured_pixels),
            f"prompt={max_pixels_page}, config={configured_pixels}",
        )
    except Exception as exc:
        add_result(results, "prompt_config", False, str(exc))

    if manifest_rows:
        rows_to_check = manifest_rows if args.check_images else manifest_rows[: args.sample_images]
        missing = []
        for row in rows_to_check:
            try:
                resolve_image_path(row, cfg.get("image_roots", []))
            except FileNotFoundError as exc:
                missing.append(str(exc))
        label = "image_paths_all" if args.check_images else "image_paths_sample"
        add_result(results, label, not missing, "; ".join(missing[:2]) or f"checked {len(rows_to_check)}")

    for module in ("torch", "transformers", "peft", "qwen_vl_utils", "PIL", "pandas"):
        add_result(results, f"python_module_{module}", module_available(module), module)

    metric_notebook = resolve_project_path(cfg["official_metric_notebook_path"], root)
    add_result(results, "official_metric_notebook", metric_notebook.exists(), str(metric_notebook))

    print_results(results)
    return 0 if all(ok for _, ok, _ in results) else 1


def print_results(results: list[tuple[str, bool, str]]) -> None:
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
