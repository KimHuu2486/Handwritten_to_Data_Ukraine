from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from MainPipeline.src.common.io import load_config, resolve_path, write_json


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base so --help works on machines without the ML stack installed."""

        pass


class Phase1ProgressCallback(TrainerCallback):
    """Print compact training progress with VRAM stats so long VM runs are observable."""

    def __init__(self) -> None:
        self.started_at = 0.0
        self.last_log_time = 0.0
        self.last_log_step = 0

    @staticmethod
    def _fmt_float(value: Any, digits: int = 4) -> str:
        """Format noisy Trainer floats into short, readable log values."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:.{digits}g}"

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        """Format durations as HH:MM:SS for VM logs."""
        total = max(int(seconds), 0)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.time()
        self.last_log_time = self.started_at
        self.last_log_step = 0
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print(
            "[train] start | "
            f"steps={state.max_steps} | epochs={args.num_train_epochs} | "
            f"batch={args.per_device_train_batch_size} | grad_accum={args.gradient_accumulation_steps} | "
            f"effective_batch={effective_batch}",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        now = time.time()
        elapsed = now - self.started_at
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)
        lr = logs.get("learning_rate", "n/a")
        current_step = max(int(state.global_step), 0)
        total_steps = max(int(state.max_steps), 1)
        steps_since_last_log = max(current_step - self.last_log_step, 0)
        seconds_since_last_log = max(now - self.last_log_time, 1e-6)
        steps_per_sec = steps_since_last_log / seconds_since_last_log if steps_since_last_log else 0.0
        sec_per_step = elapsed / max(current_step, 1)
        eta_sec = sec_per_step * max(total_steps - current_step, 0)
        vram = ""
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            vram = f" | vram={allocated:.1f}/{reserved:.1f}GB"
        progress = current_step / total_steps * 100
        log_kind = "eval" if eval_loss is not None else "train"
        metric_text = []
        if train_loss is not None:
            metric_text.append(f"loss={self._fmt_float(train_loss)}")
        if eval_loss is not None:
            metric_text.append(f"eval_loss={self._fmt_float(eval_loss)}")
        if logs.get("mean_token_accuracy") is not None:
            metric_text.append(f"token_acc={self._fmt_float(logs['mean_token_accuracy'])}")
        if logs.get("grad_norm") is not None:
            metric_text.append(f"grad_norm={self._fmt_float(logs['grad_norm'])}")
        if not metric_text:
            metric_text.append("loss=n/a")
        print(
            f"[{log_kind}] step={current_step}/{total_steps} ({progress:.1f}%) | "
            f"epoch={state.epoch:.2f} | {' | '.join(metric_text)} | "
            f"lr={self._fmt_float(lr)} | elapsed={self._fmt_seconds(elapsed)} | "
            f"eta={self._fmt_seconds(eta_sec)} | sec/step={sec_per_step:.2f} | "
            f"step/s={steps_per_sec:.3f}{vram}",
            flush=True,
        )
        self.last_log_time = now
        self.last_log_step = current_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Print eval_loss as its own event so validation checkpoints are obvious in logs."""
        if metrics and "eval_loss" in metrics:
            print(f"[eval] complete | step={state.global_step} | eval_loss={self._fmt_float(metrics['eval_loss'])}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        """Print the exact checkpoint directory saved by Trainer."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"[save] checkpoint | step={state.global_step} | path={checkpoint_path}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        """Print best checkpoint metadata after Trainer optionally reloads the best model."""
        print(
            "[train] end | "
            f"steps={state.global_step} | best_metric={self._fmt_float(state.best_metric)} | "
            f"best_model_checkpoint={state.best_model_checkpoint}",
            flush=True,
        )


def import_training_modules():
    """Import heavy training dependencies only when the train entrypoint is used."""
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    return (
        load_dataset,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        process_vision_info,
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    )


def dtype_from_name(name: str):
    """Map config dtype strings to torch dtype objects."""
    import torch

    lowered = str(name).lower()
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_model_and_processor(cfg: dict[str, Any], modules: tuple[Any, ...]):
    """Load Qwen3-VL and attach a trainable LoRA adapter for Phase 1."""
    import torch

    (
        _load_dataset,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        _process_vision_info,
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        _SFTConfig,
        _SFTTrainer,
    ) = modules

    model_cfg = cfg["model"]
    load_cfg = cfg.get("model_load", {})
    base_model_path = str(resolve_path(model_cfg["base_model_path"]))
    adapter_path = model_cfg.get("resume_adapter_path")
    load_in_4bit = bool(load_cfg.get("load_in_4bit", False))
    torch_dtype = dtype_from_name(load_cfg.get("torch_dtype", "float16"))

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(load_cfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=bool(load_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=load_cfg.get("bnb_4bit_quant_type", "nf4"),
        )

    model_kwargs = {
        "device_map": load_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(load_cfg.get("trust_remote_code", True)),
        "attn_implementation": load_cfg.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": bool(load_cfg.get("low_cpu_mem_usage", True)),
        "quantization_config": quantization_config,
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    print(f"loading base model={base_model_path} load_in_4bit={load_in_4bit}", flush=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(base_model_path, dtype=torch_dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(base_model_path, torch_dtype=torch_dtype, **model_kwargs)

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if bool(load_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if adapter_path:
        resolved_adapter = str(resolve_path(adapter_path))
        print(f"loading trainable adapter={resolved_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, resolved_adapter, is_trainable=True)
    else:
        lora_cfg = cfg["lora"]
        lora_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=list(lora_cfg["target_modules"]),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)

    if bool(load_cfg.get("trainable_params_float32", True)):
        for _name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
    return model, processor


def make_data_collator(processor, process_vision_info, cfg: dict[str, Any]):
    """Create a VLM collator that masks user tokens and trains only on assistant answers."""
    import torch

    max_length = int(cfg.get("data", {}).get("max_length", 4096))
    mask_truncated = bool(cfg.get("data", {}).get("mask_truncated_samples", True))
    fail_on_empty_label_batch = bool(cfg.get("data", {}).get("fail_on_empty_label_batch", True))
    truncation_report_every = int(cfg.get("data", {}).get("truncation_report_every", 100))
    assistant_marker = cfg.get("data", {}).get("assistant_marker", "<|im_start|>assistant\n")
    label_stats = {
        "seen": 0,
        "masked_truncated": 0,
        "missing_assistant_marker": 0,
        "empty_label_batches": 0,
        "last_report": 0,
    }

    def data_collator(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages_list = [example["messages"] for example in examples]
        texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) for messages in messages_list]
        image_inputs, video_inputs = process_vision_info(messages_list)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "right"
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        marker_ids = processor.tokenizer.encode(assistant_marker, allowed_special="all", add_special_tokens=False)
        marker_len = len(marker_ids)
        for row_index in range(labels.shape[0]):
            token_ids = input_ids[row_index].tolist()
            if attention_mask is not None:
                real_token_count = int(attention_mask[row_index].sum().item())
            else:
                real_token_count = len(token_ids)
            is_truncated = real_token_count >= max_length
            answer_start = -1
            for idx in range(0, len(token_ids) - marker_len + 1):
                if token_ids[idx : idx + marker_len] == marker_ids:
                    answer_start = idx + marker_len
                    break
            if answer_start >= 0 and mask_truncated and is_truncated:
                labels[row_index, :] = -100
                label_stats["masked_truncated"] += 1
            elif answer_start >= 0:
                labels[row_index, :answer_start] = -100
            else:
                labels[row_index, :] = -100
                label_stats["missing_assistant_marker"] += 1
        batch["labels"] = labels
        label_stats["seen"] += int(labels.shape[0])
        if int((labels != -100).sum().item()) == 0:
            label_stats["empty_label_batches"] += 1
            if fail_on_empty_label_batch:
                raise RuntimeError(
                    "All labels in this batch are masked. "
                    "Increase data.max_length, reduce max_pixels, or set fail_on_empty_label_batch=false for diagnostics."
                )
        if truncation_report_every > 0 and label_stats["seen"] >= label_stats["last_report"] + truncation_report_every:
            print(
                "phase1 collator stats "
                f"seen={label_stats['seen']} "
                f"masked_truncated={label_stats['masked_truncated']} "
                f"missing_assistant_marker={label_stats['missing_assistant_marker']} "
                f"empty_label_batches={label_stats['empty_label_batches']}",
                flush=True,
            )
            label_stats["last_report"] = label_stats["seen"]

        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.float16)
        return batch

    return data_collator


def build_training_args(cfg: dict[str, Any], SFTConfig):
    """Translate the JSON config training block into TRL SFTConfig."""
    train_cfg = cfg["training"]
    output_dir = str(resolve_path(train_cfg["output_dir"]))
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),
        optim=train_cfg.get("optim", "adamw_torch"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        disable_tqdm=bool(train_cfg.get("disable_tqdm", True)),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps"),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps"),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", False)),
        metric_for_best_model=train_cfg.get("metric_for_best_model"),
        greater_is_better=train_cfg.get("greater_is_better"),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        use_liger_kernel=bool(train_cfg.get("use_liger_kernel", False)),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def remove_default_trainer_loggers(trainer: Any) -> None:
    """Remove Hugging Face default loggers so Phase1ProgressCallback is the only console logger."""
    try:
        from transformers import PrinterCallback, ProgressCallback
    except ImportError:
        return
    for callback_cls in (PrinterCallback, ProgressCallback):
        try:
            trainer.remove_callback(callback_cls)
        except ValueError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 1 Qwen3-VL LoRA on mixed A1/B1 tasks.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    modules = import_training_modules()
    (
        load_dataset,
        _LoraConfig,
        _PeftModel,
        _get_peft_model,
        _prepare_model_for_kbit_training,
        process_vision_info,
        _AutoModelForImageTextToText,
        _AutoProcessor,
        _BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    ) = modules

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(resolve_path(cfg["data"]["train_jsonl"])),
            "validation": str(resolve_path(cfg["data"]["validation_jsonl"])),
        },
    )
    model, processor = build_model_and_processor(cfg, modules)
    trainer = SFTTrainer(
        model=model,
        args=build_training_args(cfg, SFTConfig),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=make_data_collator(processor, process_vision_info, cfg),
        callbacks=[Phase1ProgressCallback()],
    )
    remove_default_trainer_loggers(trainer)

    resume_checkpoint = cfg["training"].get("resume_checkpoint")
    trainer.train(resume_from_checkpoint=str(resolve_path(resume_checkpoint)) if resume_checkpoint else None)

    final_adapter_dir = resolve_path(cfg["training"]["final_adapter_dir"])
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    write_json(final_adapter_dir / "phase1_train_config.json", cfg)
    print(f"saved final adapter: {final_adapter_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
