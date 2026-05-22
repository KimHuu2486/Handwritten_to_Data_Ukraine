from __future__ import annotations

import argparse
import json

from MainPipeline.src.common.io import load_config, read_jsonl, resolve_path, write_json
from MainPipeline.src.common.scoring import score_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Phase 1 validation predictions with the official-compatible metric.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML validation config.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    manifest_rows = read_jsonl(resolve_path(cfg["manifest_path"]))
    score = score_validation(
        manifest_rows=manifest_rows,
        prediction_csv=resolve_path(cfg["predictions_csv"]),
        metric_notebook_path=resolve_path(cfg["metric_notebook_path"]),
        metric_path=resolve_path(cfg["metric_module_path"]),
    )
    write_json(resolve_path(cfg["score_json"]), score)
    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

