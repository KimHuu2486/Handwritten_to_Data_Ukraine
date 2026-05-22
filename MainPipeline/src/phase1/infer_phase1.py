from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

from MainPipeline.src.common.bbox import crop_region
from MainPipeline.src.common.io import append_jsonl, load_config, read_jsonl, resolve_path, write_json
from MainPipeline.src.common.json_parse import extract_json_array, normalize_layout_regions
from MainPipeline.src.common.prompts import stage_a_prompt, stage_b_prompt
from MainPipeline.src.common.schema import STRUCTURAL_TYPES, enforce_submission_policy, normalize_type
from MainPipeline.src.phase1.dataset_common import resolve_image_path


def import_runtime_modules():
    """Import VLM runtime dependencies only when inference is launched."""
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    return torch, PeftModel, process_vision_info, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def dtype_from_name(torch, name: str):
    """Map config dtype strings to torch dtype objects."""
    lowered = str(name).lower()
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_model(cfg: dict[str, Any]):
    """Load the base model plus the Phase 1 LoRA adapter for inference."""
    torch, PeftModel, process_vision_info, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig = import_runtime_modules()
    model_cfg = cfg["model"]
    load_cfg = cfg.get("model_load", {})
    base_model_path = str(resolve_path(model_cfg["base_model_path"]))
    adapter_path = str(resolve_path(model_cfg["adapter_path"]))
    device = load_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = dtype_from_name(torch, load_cfg.get("torch_dtype", "float16"))

    quantization_config = None
    if bool(load_cfg.get("load_in_4bit", False)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(torch, load_cfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=bool(load_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=load_cfg.get("bnb_4bit_quant_type", "nf4"),
        )

    model_kwargs = {
        "device_map": {"": device} if device != "auto" else "auto",
        "trust_remote_code": bool(load_cfg.get("trust_remote_code", True)),
        "attn_implementation": load_cfg.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": bool(load_cfg.get("low_cpu_mem_usage", True)),
        "quantization_config": quantization_config,
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, dtype=torch_dtype, **model_kwargs)
    except TypeError:
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, torch_dtype=torch_dtype, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(adapter_path if (Path(adapter_path) / "processor_config.json").exists() else base_model_path)
    return torch, process_vision_info, model, processor, device


def generate_text(torch, process_vision_info, model, processor, device: str, image_path: Path, prompt: str, max_pixels: int, max_new_tokens: int) -> str:
    """Run one deterministic VLM generation for a single image/prompt pair."""
    return generate_text_batch(torch, process_vision_info, model, processor, device, [(image_path, prompt)], max_pixels, max_new_tokens)[0]


def generate_text_batch(
    torch,
    process_vision_info,
    model,
    processor,
    device: str,
    items: list[tuple[Path, str]],
    max_pixels: int,
    max_new_tokens: int,
) -> list[str]:
    """Run deterministic VLM generation for a batch of image/prompt pairs."""
    messages_list = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path), "max_pixels": int(max_pixels)},
                {"type": "text", "text": prompt},
            ],
        }
        for image_path, prompt in items
    ]
    chat_messages = [[message] for message in messages_list]
    text_prompts: list[str] = []
    for messages in chat_messages:
        try:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_prompts.append(text_prompt)
    image_inputs, video_inputs = process_vision_info(chat_messages)
    inputs = processor(text=text_prompts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if device != "auto":
        inputs = inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split pending Stage B crop jobs into fixed-size generation batches."""
    safe_size = max(int(size), 1)
    return [items[index : index + safe_size] for index in range(0, len(items), safe_size)]


def clean_ocr_text(text: str) -> str:
    """Strip common formatting wrappers from OCR output while preserving actual content."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`").strip()
        if clean.startswith("text"):
            clean = clean[4:].strip()
    if clean in {'""', "''"}:
        return ""
    return clean


def existing_images(csv_path: Path) -> set[str]:
    """Read existing prediction rows for resume support."""
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["image"] for row in csv.DictReader(handle) if row.get("image")}


def read_inference_rows(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    """Load validation JSONL metadata or test rows from sample_submission.csv."""
    input_cfg = cfg["input"]
    if input_cfg.get("metadata_path"):
        metadata_path = resolve_path(input_cfg["metadata_path"])
        return read_jsonl(metadata_path), metadata_path

    sample_submission_path = resolve_path(input_cfg["sample_submission_csv"])
    with sample_submission_path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for csv_row in csv.DictReader(handle):
            image_name = str(csv_row["image"])
            rows.append(
                {
                    "submission_image": image_name,
                    "file_name": image_name,
                    "image_id": Path(image_name).stem,
                    "source": input_cfg.get("default_source", "unknown"),
                }
            )
    return rows, sample_submission_path


def write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write Phase 1 predictions in a Kaggle-compatible shape plus debug columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fields = ["image", "image_id", "file_name", "regions", "parse_ok", "error_type", "runtime_sec"]
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp_path.replace(path)


def write_submission_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the competition submission shape: image,regions only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "regions"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image": row.get("image", ""), "regions": row.get("regions", "[]")})
    tmp_path.replace(path)


def ensure_image_size(row: dict[str, Any], image_path: Path) -> dict[str, Any]:
    """Fill image_width/image_height for test CSV rows that do not have metadata."""
    if row.get("image_width") and row.get("image_height"):
        return row
    from PIL import Image

    with Image.open(image_path) as image:
        enriched = dict(row)
        enriched["image_width"] = image.width
        enriched["image_height"] = image.height
        return enriched


def run_page(cfg: dict[str, Any], runtime: tuple[Any, ...], row: dict[str, Any], image_path: Path, crops_dir: Path, raw_outputs_path: Path) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Run Phase 1 inference for one page: Stage A layout, Stage B crop OCR, Stage E guardrail."""
    torch, process_vision_info, model, processor, device = runtime
    generation_cfg = cfg["generation"]
    source = str(row.get("source", "unknown"))
    image_width = int(row.get("image_width", 1))
    image_height = int(row.get("image_height", 1))

    stage_a_raw = generate_text(
        torch,
        process_vision_info,
        model,
        processor,
        device,
        image_path,
        stage_a_prompt(source),
        int(generation_cfg.get("max_pixels_page", 850000)),
        int(generation_cfg.get("max_new_tokens_stage_a", 2048)),
    )
    append_jsonl(raw_outputs_path, {"stage": "A", "file_name": row.get("file_name"), "raw_text": stage_a_raw})
    raw_layout, parse_error = extract_json_array(stage_a_raw)
    if parse_error:
        return [], False, parse_error

    regions = normalize_layout_regions(raw_layout, image_width, image_height)
    final_regions: list[dict[str, Any] | None] = [None] * len(regions)
    pending_stage_b: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        region_type = normalize_type(region.get("type"))
        if region_type in STRUCTURAL_TYPES:
            final_regions[region_index] = {"bbox": region["bbox"], "type": region_type, "text": ""}
            continue

        crop_path = crops_dir / f"{Path(str(row.get('file_name', 'image'))).stem}__pred_{region_index:04d}__{region_type}.jpg"
        crop_region(image_path, region["bbox"], crop_path, float(generation_cfg.get("crop_pad_ratio", 0.02)))
        pending_stage_b.append(
            {
                "region_index": region_index,
                "region": region,
                "type": region_type,
                "crop_path": crop_path,
                "prompt": stage_b_prompt(source, region_type),
            }
        )

    stage_b_batch_size = int(generation_cfg.get("stage_b_batch_size", 1))
    for batch in chunks(pending_stage_b, stage_b_batch_size):
        batch_outputs = generate_text_batch(
            torch,
            process_vision_info,
            model,
            processor,
            device,
            [(item["crop_path"], item["prompt"]) for item in batch],
            int(generation_cfg.get("max_pixels_crop", 262144)),
            int(generation_cfg.get("max_new_tokens_stage_b", 1024)),
        )
        for item, stage_b_raw in zip(batch, batch_outputs):
            region = item["region"]
            append_jsonl(
                raw_outputs_path,
                {
                    "stage": "B",
                    "file_name": row.get("file_name"),
                    "region_index": item["region_index"],
                    "type": item["type"],
                    "bbox": region["bbox"],
                    "batch_size": len(batch),
                    "raw_text": stage_b_raw,
                },
            )
            final_regions[item["region_index"]] = {"bbox": region["bbox"], "type": item["type"], "text": clean_ocr_text(stage_b_raw)}
    return enforce_submission_policy([region for region in final_regions if region is not None]), True, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 inference: Stage A layout-only + Stage B crop OCR + Stage E assembly.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--resume", action="store_true", help="Skip images already present in output CSV.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    image_roots = [str(root) for root in cfg["input"].get("image_roots", [])]
    output_csv = resolve_path(cfg["output"]["predictions_csv"])
    raw_outputs_path = resolve_path(cfg["output"]["raw_outputs_jsonl"])
    crops_dir = resolve_path(cfg["output"].get("crops_dir", "artifacts/main_pipeline/phase1/predictions/crops_tmp"))
    submission_csv = resolve_path(cfg["output"]["submission_csv"]) if cfg["output"].get("submission_csv") else None
    runtime = build_model(cfg)
    if not args.resume and raw_outputs_path.exists():
        raw_outputs_path.unlink()
        print(f"reset raw outputs: {raw_outputs_path}", flush=True)

    rows, input_path = read_inference_rows(cfg)
    if args.limit is not None:
        rows = rows[: args.limit]
    done = existing_images(output_csv) if args.resume else set()
    prediction_rows: list[dict[str, Any]] = []
    if args.resume and output_csv.exists():
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))

    checkpoint_every = int(cfg["output"].get("checkpoint_every", 5))
    for row_index, row in enumerate(rows, start=1):
        image_name = str(row.get("submission_image") or Path(str(row.get("file_name", ""))).name)
        if image_name in done:
            print(f"[{row_index}/{len(rows)}] skip cached image={image_name}", flush=True)
            continue
        started = time.perf_counter()
        image_path = resolve_image_path(row, input_path, image_roots)
        if image_path is None:
            regions, parse_ok, error_type = [], False, "image_not_found"
        else:
            stage_b_batch_size = int(cfg.get("generation", {}).get("stage_b_batch_size", 1))
            print(f"[{row_index}/{len(rows)}] infer image={image_name} stage_b_batch={stage_b_batch_size}", flush=True)
            try:
                enriched_row = ensure_image_size(row, image_path)
                regions, parse_ok, error_type = run_page(cfg, runtime, enriched_row, image_path, crops_dir, raw_outputs_path)
            except Exception as exc:  # Keep long validation runs moving and preserve debug context.
                regions, parse_ok = [], False
                error_type = f"inference_exception:{exc.__class__.__name__}"
                append_jsonl(
                    raw_outputs_path,
                    {
                        "stage": "page_error",
                        "file_name": row.get("file_name"),
                        "image": image_name,
                        "error_type": error_type,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"[{row_index}/{len(rows)}] error image={image_name} type={error_type}: {exc}", flush=True)
        runtime_sec = time.perf_counter() - started
        prediction_rows.append(
            {
                "image": image_name,
                "image_id": row.get("image_id", row.get("file_name")),
                "file_name": row.get("file_name"),
                "regions": json.dumps(regions, ensure_ascii=False),
                "parse_ok": parse_ok,
                "error_type": error_type or "",
                "runtime_sec": f"{runtime_sec:.6f}",
            }
        )
        if len(prediction_rows) % checkpoint_every == 0:
            write_prediction_csv(output_csv, prediction_rows)
            if submission_csv:
                write_submission_csv(submission_csv, prediction_rows)
    write_prediction_csv(output_csv, prediction_rows)
    if submission_csv:
        write_submission_csv(submission_csv, prediction_rows)
    write_json(resolve_path(cfg["output"]["runtime_summary_json"]), {"row_count": len(prediction_rows), "finished_at": time.time()})
    print(f"wrote predictions: {output_csv}", flush=True)
    if submission_csv:
        print(f"wrote submission: {submission_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
