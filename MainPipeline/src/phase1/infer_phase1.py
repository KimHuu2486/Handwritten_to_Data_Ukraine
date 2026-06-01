from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

from MainPipeline.src.common.bbox import as_float_bbox, clamp_bbox, crop_region, is_valid_bbox
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


def configure_transformers_logging(cfg: dict[str, Any]) -> None:
    """Apply optional Transformers verbosity from config before model loading."""
    verbosity = str(cfg.get("logging", {}).get("transformers_verbosity", "")).strip().lower()
    if not verbosity:
        return
    from transformers.utils import logging as transformers_logging

    setters = {
        "debug": transformers_logging.set_verbosity_debug,
        "info": transformers_logging.set_verbosity_info,
        "warning": transformers_logging.set_verbosity_warning,
        "error": transformers_logging.set_verbosity_error,
        "critical": transformers_logging.set_verbosity_error,
    }
    setter = setters.get(verbosity)
    if setter is None:
        raise ValueError(f"Unsupported logging.transformers_verbosity: {verbosity}")
    setter()


def diagnostic_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when verbose inference diagnostics are enabled."""
    return bool(cfg.get("logging", {}).get("diagnostic_progress", False))


def diagnostic_log(cfg: dict[str, Any], message: str) -> None:
    """Print optional low-volume diagnostics for long-running generation calls."""
    if diagnostic_enabled(cfg):
        print(message, flush=True)


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
    configure_generation_padding(model, processor)
    return torch, process_vision_info, model, processor, device


def configure_generation_padding(model: Any, processor: Any) -> None:
    """Use left padding for decoder-only batched generation."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return
    for config in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        if config is not None and getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = pad_token_id


def apply_generation_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    """Format chat prompts while keeping Qwen thinking disabled across Transformers versions."""
    base_kwargs = {"tokenize": False, "add_generation_prompt": True}
    for extra_kwargs in (
        {"enable_thinking": False},
        {},
    ):
        try:
            return processor.apply_chat_template(messages, **base_kwargs, **extra_kwargs)
        except TypeError:
            continue
    return processor.apply_chat_template(messages, **base_kwargs)


def build_processor_inputs(processor: Any, text_prompts: list[str], image_inputs: Any, video_inputs: Any):
    """Tokenize multimodal prompts with left padding and a compatibility fallback."""
    try:
        return processor(
            text=text_prompts,
            images=image_inputs,
            videos=video_inputs,
            text_kwargs={"padding": True, "padding_side": "left", "return_tensors": "pt"},
            images_kwargs={"return_tensors": "pt"},
            videos_kwargs={"return_tensors": "pt"},
        )
    except TypeError:
        return processor(text=text_prompts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


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
        text_prompts.append(apply_generation_chat_template(processor, messages))
    image_inputs, video_inputs = process_vision_info(chat_messages)
    inputs = build_processor_inputs(processor, text_prompts, image_inputs, video_inputs)
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


def read_external_layout_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read precomputed bbox/type regions from a Kaggle-shaped CSV."""
    layouts: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=2):
            image_name = str(row.get("image") or "").strip()
            if not image_name:
                raise ValueError(f"external layout CSV row {row_index} is missing image")
            try:
                regions = json.loads(row.get("regions") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"external layout CSV row {row_index} has invalid regions JSON") from exc
            if isinstance(regions, dict):
                regions = regions.get("regions", [])
            if not isinstance(regions, list):
                raise ValueError(f"external layout CSV row {row_index} regions must be a list")
            layouts[Path(image_name).name] = regions
    return layouts


def normalize_external_layout_regions(raw_regions: list[Any], image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Normalize external detector output that is already in pixel coordinates."""
    regions: list[dict[str, Any]] = []
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        bbox = as_float_bbox(item.get("bbox"))
        if bbox is None:
            continue
        pixel_bbox = clamp_bbox(bbox, image_width, image_height)
        if not is_valid_bbox(pixel_bbox):
            continue
        regions.append({"bbox": pixel_bbox, "type": normalize_type(item.get("type")), "text": item.get("text", "")})
    return enforce_submission_policy(regions)


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


def build_page_state_from_regions(
    cfg: dict[str, Any],
    row: dict[str, Any],
    image_path: Path,
    crops_dir: Path,
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create page state and crop OCR jobs from normalized layout regions."""
    generation_cfg = cfg["generation"]
    source = str(row.get("source", "unknown"))
    page_state: dict[str, Any] = {
        "row": row,
        "final_regions": [None] * len(regions),
        "pending_stage_b": [],
    }
    final_regions: list[dict[str, Any] | None] = [None] * len(regions)
    pending_stage_b: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        region_type = normalize_type(region.get("type"))
        if region_type in STRUCTURAL_TYPES:
            final_regions[region_index] = {"bbox": region["bbox"], "type": region_type, "text": ""}
            continue

        crop_path = crops_dir / f"{Path(str(row.get('file_name', 'image'))).stem}__pred_{region_index:04d}__{region_type}.jpg"
        crop_region(image_path, region["bbox"], crop_path, float(generation_cfg.get("crop_pad_ratio", 0.0)))
        pending_stage_b.append(
            {
                "region_index": region_index,
                "region": region,
                "page_state": page_state,
                "type": region_type,
                "crop_path": crop_path,
                "prompt": stage_b_prompt(source, region_type),
            }
        )
    page_state["final_regions"] = final_regions
    page_state["pending_stage_b"] = pending_stage_b
    return page_state


def prepare_page_ocr_jobs(
    cfg: dict[str, Any],
    runtime: tuple[Any, ...],
    row: dict[str, Any],
    image_path: Path,
    crops_dir: Path,
    raw_outputs_path: Path,
    external_layout_regions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Build one page layout/crop state and defer Stage B OCR to a shared batch."""
    torch, process_vision_info, model, processor, device = runtime
    generation_cfg = cfg["generation"]
    source = str(row.get("source", "unknown"))
    image_width = int(row.get("image_width", 1))
    image_height = int(row.get("image_height", 1))

    if external_layout_regions is None:
        image_name = Path(str(row.get("file_name", "image"))).name
        max_pixels_page = int(generation_cfg.get("max_pixels_page", 850000))
        max_new_tokens_stage_a = int(generation_cfg.get("max_new_tokens_stage_a", 2048))
        diagnostic_log(
            cfg,
            f"[stage A start] image={image_name} source={source} "
            f"size={image_width}x{image_height} max_pixels={max_pixels_page} "
            f"max_new_tokens={max_new_tokens_stage_a}",
        )
        stage_a_started = time.perf_counter()
        stage_a_raw = generate_text(
            torch,
            process_vision_info,
            model,
            processor,
            device,
            image_path,
            stage_a_prompt(source),
            max_pixels_page,
            max_new_tokens_stage_a,
        )
        diagnostic_log(
            cfg,
            f"[stage A done] image={image_name} sec={time.perf_counter() - stage_a_started:.2f} "
            f"chars={len(stage_a_raw)}",
        )
        append_jsonl(raw_outputs_path, {"stage": "A", "file_name": row.get("file_name"), "raw_text": stage_a_raw})
        raw_layout, parse_error = extract_json_array(stage_a_raw)
        if parse_error:
            diagnostic_log(cfg, f"[stage A parse_failed] image={image_name} error={parse_error}")
            return None, False, parse_error
        regions = normalize_layout_regions(raw_layout, image_width, image_height)
        diagnostic_log(cfg, f"[stage A parsed] image={image_name} regions={len(regions)}")
    else:
        regions = normalize_external_layout_regions(external_layout_regions, image_width, image_height)
        append_jsonl(
            raw_outputs_path,
            {
                "stage": "external_layout",
                "file_name": row.get("file_name"),
                "source": "external_layout_csv",
                "raw_region_count": len(external_layout_regions),
                "normalized_region_count": len(regions),
            },
        )
    return build_page_state_from_regions(cfg, row, image_path, crops_dir, regions), True, None


def run_stage_a_batch_jobs(
    cfg: dict[str, Any],
    runtime: tuple[Any, ...],
    stage_a_results: list[dict[str, Any]],
    crops_dir: Path,
    raw_outputs_path: Path,
) -> list[dict[str, Any]]:
    """Run Stage A layout generation for multiple full pages, then build OCR jobs."""
    if not stage_a_results:
        return []
    torch, process_vision_info, model, processor, device = runtime
    generation_cfg = cfg["generation"]
    max_pixels_page = int(generation_cfg.get("max_pixels_page", 850000))
    max_new_tokens_stage_a = int(generation_cfg.get("max_new_tokens_stage_a", 2048))
    stage_a_batch_size = max(int(generation_cfg.get("stage_a_batch_size", 1)), 1)
    page_states: list[dict[str, Any]] = []

    for batch in chunks(stage_a_results, stage_a_batch_size):
        image_names = [str(item["image_name"]) for item in batch]
        diagnostic_log(
            cfg,
            f"[stage A batch start] count={len(batch)} images={','.join(image_names)} "
            f"max_pixels={max_pixels_page} max_new_tokens={max_new_tokens_stage_a}",
        )
        batch_started = time.perf_counter()
        try:
            batch_outputs = generate_text_batch(
                torch,
                process_vision_info,
                model,
                processor,
                device,
                [(item["image_path"], stage_a_prompt(str(item["row"].get("source", "unknown")))) for item in batch],
                max_pixels_page,
                max_new_tokens_stage_a,
            )
        except Exception as exc:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) > 1 and bool(generation_cfg.get("stage_a_fallback_single", True)):
                diagnostic_log(cfg, f"[stage A batch fallback] count={len(batch)} error={exc.__class__.__name__}")
                for item in batch:
                    page_states.extend(run_stage_a_batch_jobs(cfg, runtime, [item], crops_dir, raw_outputs_path))
                continue
            error_type = f"inference_exception:{exc.__class__.__name__}"
            for result in batch:
                result["error_type"] = error_type
                result["parse_ok"] = False
                append_jsonl(
                    raw_outputs_path,
                    {
                        "stage": "page_error",
                        "file_name": result["row"].get("file_name"),
                        "image": result["image_name"],
                        "error_type": error_type,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            continue

        diagnostic_log(
            cfg,
            f"[stage A batch done] count={len(batch)} sec={time.perf_counter() - batch_started:.2f} "
            f"chars={','.join(str(len(text)) for text in batch_outputs)}",
        )
        for result, stage_a_raw in zip(batch, batch_outputs):
            row = result["row"]
            image_name = result["image_name"]
            image_width = int(row.get("image_width", 1))
            image_height = int(row.get("image_height", 1))
            append_jsonl(raw_outputs_path, {"stage": "A", "file_name": row.get("file_name"), "raw_text": stage_a_raw})
            raw_layout, parse_error = extract_json_array(stage_a_raw)
            if parse_error:
                result["error_type"] = parse_error
                result["parse_ok"] = False
                diagnostic_log(cfg, f"[stage A parse_failed] image={image_name} error={parse_error}")
                continue
            try:
                regions = normalize_layout_regions(raw_layout, image_width, image_height)
                page_state = build_page_state_from_regions(cfg, row, result["image_path"], crops_dir, regions)
            except Exception as exc:
                error_type = f"inference_exception:{exc.__class__.__name__}"
                result["error_type"] = error_type
                result["parse_ok"] = False
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
                continue
            result["page_state"] = page_state
            result["parse_ok"] = True
            page_states.append(page_state)
            diagnostic_log(cfg, f"[stage A parsed] image={image_name} regions={len(regions)}")
    return page_states


def run_stage_b_jobs(cfg: dict[str, Any], runtime: tuple[Any, ...], pending_stage_b: list[dict[str, Any]], raw_outputs_path: Path) -> None:
    """Run OCR for all pending crop jobs, potentially spanning multiple pages."""
    if not pending_stage_b:
        return
    torch, process_vision_info, model, processor, device = runtime
    generation_cfg = cfg["generation"]

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
            page_state = item["page_state"]
            row = page_state["row"]
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
            page_state["final_regions"][item["region_index"]] = {"bbox": region["bbox"], "type": item["type"], "text": clean_ocr_text(stage_b_raw)}


def finalize_page_state(page_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Assemble one page after Stage B OCR has filled final regions."""
    return enforce_submission_policy([region for region in page_state["final_regions"] if region is not None])


def run_page(
    cfg: dict[str, Any],
    runtime: tuple[Any, ...],
    row: dict[str, Any],
    image_path: Path,
    crops_dir: Path,
    raw_outputs_path: Path,
    external_layout_regions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Run Phase 1 inference for one page: Stage A/external layout, Stage B crop OCR, Stage E guardrail."""
    page_state, parse_ok, error_type = prepare_page_ocr_jobs(cfg, runtime, row, image_path, crops_dir, raw_outputs_path, external_layout_regions)
    if not parse_ok or page_state is None:
        return [], parse_ok, error_type
    run_stage_b_jobs(cfg, runtime, page_state["pending_stage_b"], raw_outputs_path)
    return finalize_page_state(page_state), True, None


def build_prediction_row(row: dict[str, Any], image_name: str, regions: list[dict[str, Any]], parse_ok: bool, error_type: str | None, runtime_sec: float) -> dict[str, Any]:
    """Create one debug/submission row."""
    return {
        "image": image_name,
        "image_id": row.get("image_id", row.get("file_name")),
        "file_name": row.get("file_name"),
        "regions": json.dumps(regions, ensure_ascii=False),
        "parse_ok": parse_ok,
        "error_type": error_type or "",
        "runtime_sec": f"{runtime_sec:.6f}",
    }


def maybe_write_checkpoint(output_csv: Path, submission_csv: Path | None, prediction_rows: list[dict[str, Any]], checkpoint_every: int) -> None:
    """Persist predictions at the configured checkpoint cadence."""
    if checkpoint_every <= 0:
        return
    if len(prediction_rows) % checkpoint_every == 0:
        write_prediction_csv(output_csv, prediction_rows)
        if submission_csv:
            write_submission_csv(submission_csv, prediction_rows)


def log_progress(done_count: int, total_count: int, error_count: int, started_at: float) -> None:
    """Print compact progress instead of per-image logs."""
    elapsed = time.perf_counter() - started_at
    avg = elapsed / max(done_count, 1)
    print(f"[progress] done={done_count}/{total_count} errors={error_count} elapsed_sec={elapsed:.1f} avg_sec_per_image={avg:.2f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 inference: Stage A/external layout + Stage B crop OCR + Stage E assembly.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--resume", action="store_true", help="Skip images already present in output CSV.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    configure_transformers_logging(cfg)
    image_roots = [str(root) for root in cfg["input"].get("image_roots", [])]
    output_csv = resolve_path(cfg["output"]["predictions_csv"])
    raw_outputs_path = resolve_path(cfg["output"]["raw_outputs_jsonl"])
    crops_dir = resolve_path(cfg["output"].get("crops_dir", "artifacts/main_pipeline/phase1/predictions/crops_tmp"))
    submission_csv = resolve_path(cfg["output"]["submission_csv"]) if cfg["output"].get("submission_csv") else None
    external_layouts = None
    if cfg["input"].get("external_layout_csv"):
        external_layout_csv = resolve_path(cfg["input"]["external_layout_csv"])
        external_layouts = read_external_layout_csv(external_layout_csv)
        print(f"loaded external layout CSV: {external_layout_csv} rows={len(external_layouts)}", flush=True)
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

    generation_cfg = cfg.get("generation", {})
    checkpoint_every = int(cfg["output"].get("checkpoint_every", 5))
    image_batch_size = max(int(generation_cfg.get("image_batch_size", 1)), 1)
    stage_a_batch_size = max(int(generation_cfg.get("stage_a_batch_size", 1)), 1)
    progress_every = max(int(cfg["output"].get("progress_every", generation_cfg.get("progress_every", 10))), 1)
    run_started = time.perf_counter()
    error_count = sum(1 for row in prediction_rows if row.get("error_type"))
    last_progress_count = len(prediction_rows)

    row_items = [{"row_index": row_index, "row": row} for row_index, row in enumerate(rows, start=1)]
    for row_batch in chunks(row_items, image_batch_size):
        batch_results: list[dict[str, Any]] = []
        page_states: list[dict[str, Any]] = []
        stage_a_results: list[dict[str, Any]] = []

        for item in row_batch:
            row = item["row"]
            image_name = str(row.get("submission_image") or Path(str(row.get("file_name", ""))).name)
            layout_key = Path(image_name).name
            if image_name in done:
                continue

            result: dict[str, Any] = {
                "row": row,
                "image_name": image_name,
                "started": time.perf_counter(),
                "regions": [],
                "parse_ok": False,
                "error_type": None,
            }
            batch_results.append(result)

            image_path = resolve_image_path(row, input_path, image_roots)
            if image_path is None:
                result["error_type"] = "image_not_found"
                continue
            if external_layouts is not None and layout_key not in external_layouts:
                result["error_type"] = "external_layout_missing"
                continue

            try:
                enriched_row = ensure_image_size(row, image_path)
                result["row"] = enriched_row
                result["image_path"] = image_path
                if external_layouts is None:
                    stage_a_results.append(result)
                else:
                    page_state, parse_ok, error_type = prepare_page_ocr_jobs(
                        cfg,
                        runtime,
                        enriched_row,
                        image_path,
                        crops_dir,
                        raw_outputs_path,
                        external_layouts[layout_key],
                    )
                    result["parse_ok"] = parse_ok
                    result["error_type"] = error_type
                    if page_state is not None:
                        result["page_state"] = page_state
                        page_states.append(page_state)
            except Exception as exc:  # Keep long runs moving and preserve debug context.
                error_type = f"inference_exception:{exc.__class__.__name__}"
                result["error_type"] = error_type
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

        try:
            page_states.extend(run_stage_a_batch_jobs(cfg, runtime, stage_a_results, crops_dir, raw_outputs_path))
        except Exception as exc:  # Keep long runs moving if shared Stage A preparation fails unexpectedly.
            error_type = f"inference_exception:{exc.__class__.__name__}"
            for result in stage_a_results:
                if result.get("page_state") is not None or result.get("error_type"):
                    continue
                result["error_type"] = error_type
                result["parse_ok"] = False
                append_jsonl(
                    raw_outputs_path,
                    {
                        "stage": "page_error",
                        "file_name": result["row"].get("file_name"),
                        "image": result["image_name"],
                        "error_type": error_type,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

        pending_stage_b = [job for page_state in page_states for job in page_state["pending_stage_b"]]
        try:
            run_stage_b_jobs(cfg, runtime, pending_stage_b, raw_outputs_path)
        except Exception as exc:  # A failed OCR batch affects the pages whose crop jobs were grouped together.
            error_type = f"inference_exception:{exc.__class__.__name__}"
            for result in batch_results:
                page_state = result.get("page_state")
                if page_state is None or not page_state["pending_stage_b"]:
                    continue
                result["error_type"] = error_type
                result["parse_ok"] = False
                result["regions"] = []
                append_jsonl(
                    raw_outputs_path,
                    {
                        "stage": "page_error",
                        "file_name": result["row"].get("file_name"),
                        "image": result["image_name"],
                        "error_type": error_type,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

        for result in batch_results:
            page_state = result.get("page_state")
            if page_state is not None and not result.get("error_type"):
                result["regions"] = finalize_page_state(page_state)
                result["parse_ok"] = True
            elif page_state is not None and not page_state["pending_stage_b"]:
                result["regions"] = finalize_page_state(page_state)

            runtime_sec = time.perf_counter() - result["started"]
            prediction_rows.append(
                build_prediction_row(
                    result["row"],
                    result["image_name"],
                    result["regions"],
                    bool(result["parse_ok"]),
                    result["error_type"],
                    runtime_sec,
                )
            )
            done.add(result["image_name"])
            if result["error_type"]:
                error_count += 1
            maybe_write_checkpoint(output_csv, submission_csv, prediction_rows, checkpoint_every)

            if len(prediction_rows) - last_progress_count >= progress_every:
                log_progress(len(prediction_rows), len(rows), error_count, run_started)
                last_progress_count = len(prediction_rows)

    if len(prediction_rows) != last_progress_count:
        log_progress(len(prediction_rows), len(rows), error_count, run_started)
    write_prediction_csv(output_csv, prediction_rows)
    if submission_csv:
        write_submission_csv(submission_csv, prediction_rows)
    write_json(
        resolve_path(cfg["output"]["runtime_summary_json"]),
        {
            "row_count": len(prediction_rows),
            "error_count": error_count,
            "image_batch_size": image_batch_size,
            "stage_a_batch_size": stage_a_batch_size,
            "stage_b_batch_size": int(generation_cfg.get("stage_b_batch_size", 1)),
            "finished_at": time.time(),
        },
    )
    print(f"wrote predictions: {output_csv}", flush=True)
    if submission_csv:
        print(f"wrote submission: {submission_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
