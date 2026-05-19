from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from b2_common import (
    append_jsonl,
    extract_json_from_response,
    load_json,
    load_prompt_config,
    load_runtime_config,
    normalize_regions,
    read_existing_predictions,
    read_jsonl,
    repo_root,
    resolve_image_path,
    resolve_project_path,
    score_predictions,
    utc_now_iso,
    validate_manifest_rows,
    write_json,
    write_predictions_csv,
)


def import_runtime_modules():
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    return torch, PeftModel, process_vision_info, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def dtype_from_name(torch, name: str):
    lowered = str(name).lower()
    if lowered in ("float16", "fp16", "half"):
        return torch.float16
    if lowered in ("bfloat16", "bf16"):
        return torch.bfloat16
    if lowered in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_model_and_processor(cfg: dict[str, Any]):
    torch, PeftModel, process_vision_info, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig = import_runtime_modules()
    load_cfg = cfg.get("model_load", {})
    base_model_path = str(resolve_project_path(cfg["base_model_path"]))
    adapter_path = str(resolve_project_path(cfg["lora_adapter_path"]))
    device = load_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = dtype_from_name(torch, load_cfg.get("torch_dtype", "float16"))

    quantization_config = None
    if load_cfg.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(torch, load_cfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=bool(load_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=load_cfg.get("bnb_4bit_quant_type", "nf4"),
        )

    model_kwargs = {
        "device_map": {"": device} if device != "auto" else "auto",
        "quantization_config": quantization_config,
        "trust_remote_code": bool(load_cfg.get("trust_remote_code", True)),
        "attn_implementation": load_cfg.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": bool(load_cfg.get("low_cpu_mem_usage", True)),
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

    load_mode = "4bit" if quantization_config is not None else str(load_cfg.get("torch_dtype", "float16"))
    print(f"loading model base={base_model_path} adapter={adapter_path} device={device} mode={load_mode}", flush=True)
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, dtype=torch_dtype, **model_kwargs)
    except TypeError:
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, torch_dtype=torch_dtype, **model_kwargs)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    processor_source = adapter_path if (Path(adapter_path) / "processor_config.json").exists() else base_model_path
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    return torch, process_vision_info, model, processor, device


def run_single_image(
    torch,
    process_vision_info,
    model,
    processor,
    device: str,
    image_path: Path,
    prompt_text: str,
    generation_params: dict[str, Any],
) -> str:
    return run_image_batch(
        torch=torch,
        process_vision_info=process_vision_info,
        model=model,
        processor=processor,
        device=device,
        image_paths=[image_path],
        prompt_text=prompt_text,
        generation_params=generation_params,
    )[0]


def run_image_batch(
    torch,
    process_vision_info,
    model,
    processor,
    device: str,
    image_paths: list[Path],
    prompt_text: str,
    generation_params: dict[str, Any],
) -> list[str]:
    print(f"  build messages batch_size={len(image_paths)}", flush=True)
    messages_batch = []
    for image_path in image_paths:
        messages_batch.append(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path),
                            "max_pixels": int(generation_params["max_pixels_page"]),
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        )

    text_prompts = []
    for messages in messages_batch:
        try:
            text_prompts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            )
        except TypeError:
            text_prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    print("  process vision info", flush=True)
    image_inputs, video_inputs = process_vision_info(messages_batch)
    print("  processor encode", flush=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=text_prompts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if device != "auto":
        print(f"  move tensors to {device}", flush=True)
        inputs = inputs.to(device)

    generate_kwargs = {
        "max_new_tokens": int(generation_params["max_new_tokens_page"]),
        "do_sample": bool(generation_params["do_sample"]),
        "num_beams": int(generation_params["num_beams"]),
    }
    input_token_count = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
    print(
        "  generate start "
        f"batch_size={len(image_paths)} "
        f"input_tokens={input_token_count} "
        f"max_new_tokens={generate_kwargs['max_new_tokens']} "
        f"do_sample={generate_kwargs['do_sample']} "
        f"num_beams={generate_kwargs['num_beams']}",
        flush=True,
    )
    generate_start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)
    generate_sec = time.perf_counter() - generate_start
    output_token_count = int(generated_ids.shape[-1] - inputs.input_ids.shape[-1])
    print(f"  generate done runtime_sec={generate_sec:.2f} output_tokens_max={output_token_count}", flush=True)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print("  decode output", flush=True)
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_texts


def existing_done_ids(predictions_path: Path) -> set[str]:
    rows = read_existing_predictions(predictions_path)
    return {str(row.get("image_id")) for row in rows if row.get("image_id")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B2 baseline cache on frozen validation manifest.")
    parser.add_argument("--config", default="phaseB2/b2_runtime_config.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in validation_predictions.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in the selected output dir.")
    parser.add_argument("--skip-score", action="store_true", help="Write predictions without validation_score.json.")
    args = parser.parse_args()

    root = repo_root()
    cfg_path = resolve_project_path(args.config, root)
    cfg = load_runtime_config(cfg_path)

    manifest_path = resolve_project_path(cfg["manifest_path"], root)
    gate_path = resolve_project_path(cfg["gate_report_path"], root)
    adapter_path = resolve_project_path(cfg["lora_adapter_path"], root)
    output_dir = resolve_project_path(args.output_dir or cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "validation_predictions.csv"
    raw_outputs_path = output_dir / "validation_raw_outputs.jsonl"
    failed_rows_path = output_dir / "failed_rows.jsonl"
    runtime_summary_path = output_dir / "runtime_summary.json"
    score_path = output_dir / "validation_score.json"
    run_config_path = output_dir / "config.json"

    output_files = [
        predictions_path,
        raw_outputs_path,
        failed_rows_path,
        runtime_summary_path,
        score_path,
        run_config_path,
    ]
    if args.overwrite:
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()
    elif predictions_path.exists() and not args.resume:
        raise RuntimeError(
            f"{predictions_path} already exists. Use --resume, --overwrite, or a different --output-dir."
        )

    gate = load_json(gate_path)
    if not gate.get("pass"):
        raise RuntimeError(f"Gate report is not pass: {gate_path}")

    manifest_rows = read_jsonl(manifest_path)
    manifest_errors = validate_manifest_rows(manifest_rows)
    if manifest_errors:
        raise RuntimeError("Manifest schema failed: " + "; ".join(manifest_errors[:5]))
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    prompt_config = load_prompt_config(adapter_path)
    prompt_text = prompt_config["page_prompt"]
    if prompt_config.get("max_pixels_page") and cfg["generation_params"].get("max_pixels_page") == 850000:
        cfg["generation_params"]["max_pixels_page"] = int(prompt_config["max_pixels_page"])

    run_config = {
        **cfg,
        "config_path": str(cfg_path),
        "source_manifest": cfg["artifact_version"],
        "manifest_path": str(manifest_path),
        "gate_report_path": str(gate_path),
        "lora_adapter_path": str(adapter_path),
        "resolved_output_dir": str(output_dir),
        "prompt_text": prompt_text,
        "prompt_config": prompt_config,
        "created_at": utc_now_iso(),
    }
    write_json(run_config_path, run_config)

    prediction_rows = read_existing_predictions(predictions_path) if args.resume else []
    done_ids = existing_done_ids(predictions_path) if args.resume else set()

    torch, process_vision_info, model, processor, device = build_model_and_processor(cfg)

    started_at = utc_now_iso()
    start_time = time.perf_counter()
    parse_fail_count = sum(1 for row in prediction_rows if str(row.get("parse_ok")).lower() != "true")
    checkpoint_every = int(cfg.get("runtime", {}).get("checkpoint_every", 5))
    inference_batch_size = max(1, int(cfg.get("runtime", {}).get("inference_batch_size", 1)))
    flush_every_row = args.limit is not None and args.limit <= checkpoint_every

    pending_rows = []
    for index, row in enumerate(manifest_rows):
        if str(row["image_id"]) in done_ids:
            print(f"[{index + 1}/{len(manifest_rows)}] skip cached image_id={row['image_id']}", flush=True)
        else:
            pending_rows.append((index, row))

    for batch_start in range(0, len(pending_rows), inference_batch_size):
        batch_items = pending_rows[batch_start : batch_start + inference_batch_size]
        valid_items = []
        prepared_results: dict[int, dict[str, Any]] = {}

        for index, row in batch_items:
            raw_output_id = f"raw_{index:06d}"
            row_start = time.perf_counter()
            try:
                print(f"[{index + 1}/{len(manifest_rows)}] start image_id={row['image_id']}", flush=True)
                image_path = resolve_image_path(row, cfg.get("image_roots", []))
                print(f"[{index + 1}/{len(manifest_rows)}] resolved image_path={image_path}", flush=True)
                valid_items.append((index, row, image_path, raw_output_id, row_start))
            except Exception as exc:
                prepared_results[index] = {
                    "row": row,
                    "raw_output_id": raw_output_id,
                    "row_start": row_start,
                    "raw_text": traceback.format_exc(),
                    "regions": [],
                    "parse_ok": False,
                    "error_type": type(exc).__name__,
                }

        if valid_items:
            image_paths = [item[2] for item in valid_items]
            try:
                print(f"batch generate: size={len(valid_items)}", flush=True)
                raw_texts = run_image_batch(
                    torch=torch,
                    process_vision_info=process_vision_info,
                    model=model,
                    processor=processor,
                    device=device,
                    image_paths=image_paths,
                    prompt_text=prompt_text,
                    generation_params=cfg["generation_params"],
                )
                for item, raw_text in zip(valid_items, raw_texts):
                    index, row, _, raw_output_id, row_start = item
                    prepared_results[index] = {
                        "row": row,
                        "raw_output_id": raw_output_id,
                        "row_start": row_start,
                        "raw_text": raw_text,
                        "regions": [],
                        "parse_ok": True,
                        "error_type": "",
                    }
            except Exception:
                if len(valid_items) > 1:
                    print("batch generate failed; falling back to single-image generate", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for index, row, image_path, raw_output_id, row_start in valid_items:
                    try:
                        raw_text = run_single_image(
                            torch=torch,
                            process_vision_info=process_vision_info,
                            model=model,
                            processor=processor,
                            device=device,
                            image_path=image_path,
                            prompt_text=prompt_text,
                            generation_params=cfg["generation_params"],
                        )
                        prepared_results[index] = {
                            "row": row,
                            "raw_output_id": raw_output_id,
                            "row_start": row_start,
                            "raw_text": raw_text,
                            "regions": [],
                            "parse_ok": True,
                            "error_type": "",
                        }
                    except Exception as exc:
                        prepared_results[index] = {
                            "row": row,
                            "raw_output_id": raw_output_id,
                            "row_start": row_start,
                            "raw_text": traceback.format_exc(),
                            "regions": [],
                            "parse_ok": False,
                            "error_type": type(exc).__name__,
                        }

        for index, result in sorted(prepared_results.items()):
            row = result["row"]
            raw_output_id = result["raw_output_id"]
            row_start = result["row_start"]
            raw_text = result["raw_text"]
            regions: list[dict[str, Any]] = result["regions"]
            parse_ok = bool(result["parse_ok"])
            error_type = str(result["error_type"])

            if parse_ok:
                print(f"[{index + 1}/{len(manifest_rows)}] parse raw output chars={len(raw_text)}", flush=True)
                raw_regions, parse_error = extract_json_from_response(raw_text)
                if parse_error:
                    parse_ok = False
                    error_type = parse_error
                    print(f"[{index + 1}/{len(manifest_rows)}] parse fail error_type={error_type}", flush=True)
                else:
                    regions, normalize_error = normalize_regions(
                        raw_regions,
                        int(row["image_width"]),
                        int(row["image_height"]),
                        int(cfg["generation_params"]["max_pixels_page"]),
                    )
                    if normalize_error:
                        parse_ok = False
                        error_type = normalize_error
                        print(f"[{index + 1}/{len(manifest_rows)}] normalize fail error_type={error_type}", flush=True)
                    else:
                        print(f"[{index + 1}/{len(manifest_rows)}] parse ok regions={len(regions)}", flush=True)
            else:
                print(f"[{index + 1}/{len(manifest_rows)}] exception error_type={error_type}", flush=True)

            runtime_sec = time.perf_counter() - row_start
            print(f"[{index + 1}/{len(manifest_rows)}] row done runtime_sec={runtime_sec:.2f} parse_ok={parse_ok}", flush=True)
            append_jsonl(
                raw_outputs_path,
                {
                    "raw_output_id": raw_output_id,
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "prompt_version": cfg["prompt_version"],
                    "generation_params": cfg["generation_params"],
                    "raw_text": raw_text,
                    "created_at": utc_now_iso(),
                },
            )

            if not parse_ok:
                parse_fail_count += 1
                append_jsonl(
                    failed_rows_path,
                    {
                        "image_id": row["image_id"],
                        "file_name": row["file_name"],
                        "raw_output_id": raw_output_id,
                        "error_type": error_type,
                        "created_at": utc_now_iso(),
                    },
                )

            prediction_rows.append(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "submission_image": row["submission_image"],
                    "regions": json.dumps(regions, ensure_ascii=False),
                    "parse_ok": parse_ok,
                    "error_type": "" if parse_ok else error_type,
                    "raw_output_id": raw_output_id,
                    "checkpoint_id": cfg["checkpoint_id"],
                    "prompt_version": cfg["prompt_version"],
                    "runtime_sec": f"{runtime_sec:.6f}",
                }
            )

            if len(prediction_rows) % checkpoint_every == 0:
                write_predictions_csv(predictions_path, prediction_rows)
                print(f"checkpoint: {len(prediction_rows)}/{len(manifest_rows)} rows")
            elif flush_every_row:
                write_predictions_csv(predictions_path, prediction_rows)
                print(f"smoke flush: {len(prediction_rows)}/{len(manifest_rows)} rows", flush=True)

    write_predictions_csv(predictions_path, prediction_rows)

    runtime_total_sec = time.perf_counter() - start_time
    runtime_per_page_sec = runtime_total_sec / max(len(prediction_rows), 1)
    runtime_summary = {
        "run_id": cfg["run_id"],
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "row_count": len(prediction_rows),
        "manifest_row_count": len(manifest_rows),
        "parse_fail_count": parse_fail_count,
        "runtime_total_sec": runtime_total_sec,
        "runtime_per_page_sec": runtime_per_page_sec,
        "device": device,
        "inference_batch_size": inference_batch_size,
    }
    write_json(runtime_summary_path, runtime_summary)

    if not args.skip_score:
        metric_notebook_path = resolve_project_path(cfg["official_metric_notebook_path"], root)
        metric_path = Path(__file__).resolve().parent / "kaggle_metric.py"
        score = score_predictions(
            manifest_rows,
            prediction_rows,
            runtime_total_sec,
            runtime_per_page_sec,
            parse_fail_count,
            cfg["metric_version"],
            metric_notebook_path,
            metric_path,
        )
        write_json(score_path, score)

    print(f"wrote cache: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
