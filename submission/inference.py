#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


VALID_TYPES = {"handwritten", "printed", "formula", "table", "annotation", "image", "graph"}
BAOHA_TYPES = {"handwritten", "printed", "annotation"}
QWEN_TYPES = {"formula", "table"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

APP_DIR = Path(__file__).resolve().parent


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
    "formula": (
        "Read this standalone math, logic, vector, matrix, determinant, set/relation, statistics, physics, "
        "or chemistry expression exactly as written. Return only formula text, using LaTeX when it is the "
        "clearest representation and plain Unicode when it better matches the handwriting. Preserve visible "
        "symbols, indices, superscripts, subscripts, arrows, fractions, matrix/determinant structure, punctuation, "
        "numbering, and strikethrough/correction markers. Do not solve, simplify, normalize, explain, or convert "
        "old notation into a different style."
    ),
    "table": (
        "Read this table region exactly. Return only pipe-separated table text. Use one output line per visual row "
        "and | between cells. Preserve empty cells with empty fields, for example A||C. Preserve row order, "
        "column order, multi-word cell text, wrapped cell text, numbers, units, punctuation, dashes, and visible "
        "spelling mistakes. Do not infer missing cells, do not rebalance columns, do not summarize, and do not explain."
    ),
    "default": "Transcribe the visible content exactly. Preserve punctuation, corrections, and visible spacing. Return only text.",
}


@dataclass
class RuntimeConfig:
    input_dir: Path
    output_dir: Path
    hpa_model_dir: Path
    qwen_base_dir: Path
    qwen_lora_dir: Path
    yolo_weights: Path
    yolo_extra_weights: list[Path]
    device: str
    yolo_backend: str
    yolo_img_size: int
    yolo_conf: float
    yolo_max_det: int
    yolo_iou_nms: float
    yolo_agnostic_nms: bool
    yolo_pad_scale_x: float
    yolo_pad_scale_y: float
    yolo_dedup_iou: float
    crop_batch_size: int
    max_pixels_crop: int
    max_new_tokens_qwen: int
    hpa_crop_pad_ratio: float
    qwen_crop_pad_ratio: float
    lazy_load: bool


def log(message: str) -> None:
    print(message, flush=True)


def normalize_type(value: Any) -> str:
    value = str(value or "handwritten").strip().lower()
    return value if value in VALID_TYPES else "handwritten"


def normalize_source(value: Any) -> str:
    value = str(value or "").strip().lower()
    return value if value in SOURCE_HINTS else "default"


def get_source_hint(source: Any) -> str:
    return SOURCE_HINTS.get(normalize_source(source), DEFAULT_SOURCE_HINT)


def build_qwen_prompt(rtype: str, source: Any = None) -> str:
    rtype = normalize_type(rtype)
    type_prompt = CROP_PROMPTS.get(rtype, CROP_PROMPTS["default"])
    return "\n".join([get_source_hint(source), type_prompt, SPECIAL_TEXT_MARKER_RULES, STAGE_B_GUARDRAILS])


def clean_crop_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(text|transcription|answer)\s*:\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "text" in obj:
                text = str(obj["text"])
            else:
                return ""
        except Exception:
            return ""
    return text[:500]


def clamp_xyxy(box: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def clamp_xyxy_float(box: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter) / max(1.0, float(area_a + area_b - inter))


def sort_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def strip_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": [round(float(v), 2) for v in region["bbox"]],
        "type": normalize_type(region.get("type")),
        "text": str(region.get("text") or ""),
    }


def nms_indices(indices: list[int], boxes: list[list[float]], scores: list[float], threshold: float) -> list[int]:
    pending = sorted(indices, key=lambda idx: scores[idx], reverse=True)
    kept: list[int] = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        pending = [idx for idx in pending if iou(boxes[current], boxes[idx]) <= threshold]
    return kept


def resize_to_pixel_budget(img: Any, max_pixels: int) -> Any:
    from PIL import Image

    w, h = img.size
    total = max(1, w * h)
    if total <= max_pixels:
        return img
    scale = (max_pixels / total) ** 0.5
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def crop_image(image_path: Path, bbox: list[int], pad_ratio: float = 0.0, max_pixels: int | None = None) -> Any:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = bbox
        pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
        x1 = int(round(max(0, x1 - pad)))
        y1 = int(round(max(0, y1 - pad)))
        x2 = int(round(min(w, x2 + pad)))
        y2 = int(round(min(h, y2 + pad)))
        crop = img.crop((x1, y1, x2, y2))
    if max_pixels is not None:
        crop = resize_to_pixel_budget(crop, max_pixels)
    return crop


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
    return records


def discover_records(input_dir: Path) -> list[dict[str, Any]]:
    metadata_path = input_dir / "metadata.jsonl"
    if metadata_path.exists():
        records = read_jsonl(metadata_path)
        if records:
            return records

    search_root = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    paths = sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    records = []
    for path in paths:
        rel = path.relative_to(input_dir).as_posix()
        records.append({"file_name": rel})
    return records


def resolve_image_path(input_dir: Path, file_name: str) -> Path:
    raw = Path(file_name)
    name = raw.name
    stem = raw.stem
    candidates: list[Path] = [
        input_dir / file_name,
        input_dir / name,
        input_dir / "images" / name,
    ]
    for ext in IMAGE_EXTENSIONS:
        candidates.extend([input_dir / f"{stem}{ext}", input_dir / "images" / f"{stem}{ext}"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return input_dir / file_name


def output_image_name(record: dict[str, Any]) -> str:
    return Path(str(record.get("file_name") or record.get("image") or "")).name


def write_submission(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "regions"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(output_csv)


def validate_rows(rows: list[dict[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f"submission row count mismatch: {len(rows)} != {expected_count}")
    for row in rows:
        if set(row.keys()) != {"image", "regions"}:
            raise RuntimeError(f"bad row columns: {row.keys()}")
        parsed = json.loads(row["regions"])
        if not isinstance(parsed, list):
            raise RuntimeError(f"regions is not a list for {row['image']}")
        for region in parsed:
            if not isinstance(region, dict):
                raise RuntimeError(f"region is not an object for {row['image']}")
            if set(region.keys()) != {"bbox", "type", "text"}:
                raise RuntimeError(f"bad region keys for {row['image']}: {region.keys()}")
            if not isinstance(region["bbox"], list) or len(region["bbox"]) != 4:
                raise RuntimeError(f"bad bbox for {row['image']}: {region.get('bbox')}")
            if normalize_type(region["type"]) not in VALID_TYPES:
                raise RuntimeError(f"bad type for {row['image']}: {region.get('type')}")


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_model_paths(cfg: RuntimeConfig) -> None:
    require_path(cfg.hpa_model_dir / "config.json", "Baoha TrOCR config")
    require_path(cfg.hpa_model_dir / "model.safetensors", "Baoha TrOCR model")
    require_path(cfg.qwen_lora_dir / "adapter_config.json", "Kimhuu LoRA adapter config")
    require_path(cfg.qwen_lora_dir / "adapter_model.safetensors", "Kimhuu LoRA adapter model")
    require_path(cfg.yolo_weights, "YOLO weights")
    for path in cfg.yolo_extra_weights:
        require_path(path, "Extra YOLO weights")
    require_path(cfg.qwen_base_dir / "config.json", "Qwen3-VL base model config")


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def yolo_predict_device(device: str) -> int | str:
    if device.startswith("cuda"):
        parts = device.split(":", 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
    return "cpu"


def parse_path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    parts: list[str] = []
    for chunk in value.split(os.pathsep):
        parts.extend(chunk.split(","))
    return [Path(part.strip()).resolve() for part in parts if part.strip()]


def auto_yolo_extra_weights(primary: Path, explicit_value: str | None) -> list[Path]:
    paths = parse_path_list(explicit_value)
    if paths:
        return paths
    candidate = primary.with_name("DoclayoutYoloV4.2.pt")
    return [candidate] if candidate.exists() else []


class YoloDetector:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.models: list[Any] = []

    def load(self) -> None:
        if self.models:
            return
        weights = [self.cfg.yolo_weights] + list(self.cfg.yolo_extra_weights)
        for path in weights:
            self.models.append(self._load_one(path))
        log("YOLO weights: " + ", ".join(str(path) for path in weights))

    def _load_one(self, weights_path: Path) -> Any:
        errors: list[str] = []

        if self.cfg.yolo_backend in {"auto", "doclayout_yolo"}:
            try:
                if "doclayout_yolo.utils.callbacks.hub" not in sys.modules:
                    dummy_hub = ModuleType("doclayout_yolo.utils.callbacks.hub")
                    dummy_hub.callbacks = {}
                    sys.modules["doclayout_yolo.utils.callbacks.hub"] = dummy_hub
                from doclayout_yolo import YOLOv10

                model = YOLOv10(str(weights_path))
                self._try_to_device(model)
                log("YOLO backend: doclayout_yolo")
                return model
            except Exception as exc:
                errors.append(f"doclayout_yolo: {exc}")

        if self.cfg.yolo_backend in {"auto", "ultralytics"}:
            try:
                from ultralytics import YOLO

                model = YOLO(str(weights_path))
                self._try_to_device(model)
                log("YOLO backend: ultralytics")
                return model
            except Exception as exc:
                errors.append(f"ultralytics: {exc}")

        raise RuntimeError(f"Could not load YOLO model {weights_path}. " + " | ".join(errors))

    def _try_to_device(self, model: Any) -> None:
        try:
            model.to(self.cfg.device)
        except Exception as exc:
            log(f"YOLO .to({self.cfg.device}) skipped: {exc}")

    def names(self) -> Any:
        model = self.models[0]
        names = getattr(model, "names", None)
        if names is None and hasattr(model, "model"):
            names = getattr(model.model, "names", None)
        return names or {}

    @staticmethod
    def class_name(names: Any, cls_id: int) -> str:
        if isinstance(names, dict):
            return names.get(cls_id, names.get(str(cls_id), "handwritten"))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
        return "handwritten"

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        self.load()
        result_list: list[Any] = []
        for model in self.models:
            results = model.predict(
                source=str(image_path),
                imgsz=self.cfg.yolo_img_size,
                conf=self.cfg.yolo_conf,
                max_det=self.cfg.yolo_max_det,
                verbose=False,
                device=yolo_predict_device(self.cfg.device),
            )
            if results:
                result_list.append(results[0])
        if not result_list:
            return []

        has_boxes = any(getattr(result, "boxes", None) is not None and len(result.boxes) > 0 for result in result_list)
        if not has_boxes:
            return []
        result = result_list[0]
        if hasattr(result, "orig_shape") and result.orig_shape:
            img_h, img_w = result.orig_shape
        else:
            from PIL import Image

            with Image.open(image_path) as img:
                img_w, img_h = img.size
        return self._postprocess(result_list, int(img_w), int(img_h))

    def _postprocess(self, result_list: list[Any], img_w: int, img_h: int) -> list[dict[str, Any]]:
        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[int] = []
        for result in result_list:
            boxes_obj = getattr(result, "boxes", None)
            if boxes_obj is None:
                continue
            for box in boxes_obj:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(float(box.conf[0]))
                labels.append(int(box.cls[0]))

        if not boxes:
            return []

        if self.cfg.yolo_agnostic_nms:
            keep_indices = nms_indices(list(range(len(boxes))), boxes, scores, self.cfg.yolo_iou_nms)
        else:
            keep_indices: list[int] = []
            for cls_id in sorted(set(labels)):
                cls_indices = [i for i, label in enumerate(labels) if label == cls_id]
                keep_indices.extend(nms_indices(cls_indices, boxes, scores, self.cfg.yolo_iou_nms))

        regions: list[dict[str, Any]] = []
        names = self.names()
        for idx in sorted(set(keep_indices)):
            x1, y1, x2, y2 = boxes[idx]
            height = max(1.0, y2 - y1)
            pad_x = self.cfg.yolo_pad_scale_x * height
            pad_y = self.cfg.yolo_pad_scale_y * height
            box = clamp_xyxy_float([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], img_w, img_h)
            if box is None:
                continue
            rtype = normalize_type(self.class_name(names, labels[idx]))
            regions.append({"bbox": box, "type": rtype, "text": "", "_score": scores[idx]})

        kept: list[dict[str, Any]] = []
        for region in sorted(regions, key=lambda r: float(r.get("_score", 0.0)), reverse=True):
            if not any(iou(region["bbox"], old["bbox"]) > self.cfg.yolo_dedup_iou for old in kept):
                kept.append(region)
        return sort_regions([strip_region(r) for r in kept])


class HpaOcr:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.generation_by_type = self._load_generation_config()

    def _load_generation_config(self) -> dict[str, dict[str, Any]]:
        path = self.cfg.hpa_model_dir / "generation_config_by_type.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        log(f"Loading Baoha TrOCR: {self.cfg.hpa_model_dir}")
        self.processor = TrOCRProcessor.from_pretrained(str(self.cfg.hpa_model_dir))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.cfg.hpa_model_dir)).to(self.cfg.device)
        self.model.eval()

    def ocr_regions(self, image_path: Path, regions: list[dict[str, Any]]) -> None:
        self.load()
        for region in regions:
            if normalize_type(region.get("type")) not in BAOHA_TYPES:
                continue
            try:
                crop = crop_image(image_path, region["bbox"], pad_ratio=self.cfg.hpa_crop_pad_ratio)
                pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.cfg.device)
                kwargs = dict(self.generation_by_type.get(normalize_type(region.get("type")), {}))
                if not kwargs:
                    kwargs = {"max_length": 128}
                with self.torch.no_grad():
                    try:
                        generated_ids = self.model.generate(pixel_values, **kwargs)
                    except TypeError:
                        if "max_new_tokens" in kwargs:
                            kwargs["max_length"] = kwargs.pop("max_new_tokens")
                        generated_ids = self.model.generate(pixel_values, **kwargs)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                region["text"] = text
            except Exception as exc:
                log(f"Baoha OCR failed for {image_path.name} bbox={region.get('bbox')}: {exc}")


class QwenOcr:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.processor: Any = None
        self.process_vision_info: Any = None
        self.torch: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if not self.cfg.device.startswith("cuda"):
            raise RuntimeError("Qwen3-VL branch requires CUDA. Use --smoke-csv only for format checks on CPU hosts.")

        self.torch = torch
        self.process_vision_info = process_vision_info
        log(f"Loading Qwen3-VL base: {self.cfg.qwen_base_dir}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        common_kwargs = dict(
            device_map={"": self.cfg.device},
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        try:
            base = AutoModelForImageTextToText.from_pretrained(
                str(self.cfg.qwen_base_dir), dtype=torch.float16, **common_kwargs
            )
        except TypeError:
            base = AutoModelForImageTextToText.from_pretrained(
                str(self.cfg.qwen_base_dir), torch_dtype=torch.float16, **common_kwargs
            )

        log(f"Loading Kimhuu LoRA: {self.cfg.qwen_lora_dir}")
        self.model = PeftModel.from_pretrained(base, str(self.cfg.qwen_lora_dir))
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(str(self.cfg.qwen_base_dir), trust_remote_code=True)
        self._configure_processor()

    def _configure_processor(self) -> None:
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is not None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        candidates = [
            {"tokenize": False, "add_generation_prompt": True, "template_kwargs": {"enable_thinking": False}},
            {"tokenize": False, "add_generation_prompt": True, "processor_kwargs": {"enable_thinking": False}},
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
        ]
        for kwargs in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict.*",
                    )
                    return self.processor.apply_chat_template(messages, **kwargs)
            except TypeError:
                continue
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def crop_messages(self, image_path: Path, region: dict[str, Any], source: Any = None) -> list[dict[str, Any]]:
        prompt = build_qwen_prompt(normalize_type(region.get("type")), source=source)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": crop_image(
                            image_path,
                            region["bbox"],
                            pad_ratio=self.cfg.qwen_crop_pad_ratio,
                            max_pixels=self.cfg.max_pixels_crop,
                        ),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        self.processor.tokenizer.padding_side = "left"
        texts = [self.apply_chat_template(messages) for messages in messages_batch]
        image_inputs, video_inputs = self.process_vision_info(messages_batch)
        try:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                text_kwargs={"padding": True, "return_tensors": "pt"},
                images_kwargs={"return_tensors": "pt"},
                videos_kwargs={"return_tensors": "pt"},
            )
        except TypeError:
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.cfg.device)

        autocast = (
            self.torch.amp.autocast("cuda", dtype=self.torch.float16)
            if self.cfg.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with self.torch.no_grad(), autocast:
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens_qwen,
                do_sample=False,
                num_beams=1,
            )
        trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, out)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        del inputs, out, trimmed
        self.torch.cuda.empty_cache()
        return decoded

    def ocr_regions(self, image_path: Path, regions: list[dict[str, Any]], source: Any = None) -> None:
        indices = [idx for idx, region in enumerate(regions) if normalize_type(region.get("type")) in QWEN_TYPES]
        if not indices:
            return
        self.load()
        for start in range(0, len(indices), self.cfg.crop_batch_size):
            batch_indices = indices[start : start + self.cfg.crop_batch_size]
            messages = [self.crop_messages(image_path, regions[idx], source=source) for idx in batch_indices]
            try:
                outs = self.generate_batch(messages)
            except Exception as exc:
                log(f"Qwen OCR batch failed for {image_path.name}: {exc}")
                self.torch.cuda.empty_cache()
                gc.collect()
                continue
            for idx, out in zip(batch_indices, outs):
                text = clean_crop_text(out)
                if text:
                    regions[idx]["text"] = text


class InferencePipeline:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.yolo = YoloDetector(cfg)
        self.hpa = HpaOcr(cfg)
        self.qwen = QwenOcr(cfg)

    def load_models(self) -> None:
        self.yolo.load()
        if not self.cfg.lazy_load:
            self.hpa.load()
            self.qwen.load()

    def infer_one(self, image_path: Path, source: Any = None) -> list[dict[str, Any]]:
        regions = self.yolo.detect(image_path)
        if not regions:
            return []
        self.hpa.ocr_regions(image_path, regions)
        self.qwen.ocr_regions(image_path, regions, source=source)
        return sort_regions([strip_region(region) for region in regions])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline final inference for Handwritten to Data.")
    parser.add_argument("input_dir", nargs="?", default="/data/input", help="Directory containing test images or metadata.jsonl.")
    parser.add_argument("output_dir", nargs="?", default="/data/output", help="Directory where submission.csv will be written.")
    parser.add_argument("--hpa-model-dir", default=os.environ.get("HPA_MODEL_DIR", str(APP_DIR / "trocr_model")))
    parser.add_argument("--qwen-base-dir", default=os.environ.get("QWEN_BASE_MODEL_DIR", str(APP_DIR / "qwen3vl_8b_instruct")))
    parser.add_argument("--qwen-lora-dir", default=os.environ.get("QWEN_LORA_DIR", str(APP_DIR / "qwen3vl_lora_adapter")))
    parser.add_argument("--yolo-weights", default=os.environ.get("YOLO_WEIGHTS", str(APP_DIR / "DoclayoutYoloV4.1.pt")))
    parser.add_argument(
        "--yolo-extra-weights",
        default=os.environ.get("YOLO_EXTRA_WEIGHTS", ""),
        help="Optional extra YOLO checkpoint(s), comma-separated or pathsep-separated. Auto-detects DoclayoutYoloV4.2.pt if present.",
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--yolo-backend", choices=["auto", "doclayout_yolo", "ultralytics"], default="auto")
    parser.add_argument("--yolo-img-size", type=int, default=1280)
    parser.add_argument("--yolo-conf", type=float, default=0.26)
    parser.add_argument("--yolo-max-det", type=int, default=200)
    parser.add_argument("--yolo-iou-nms", type=float, default=0.60)
    parser.add_argument("--yolo-per-class-nms", action="store_true", help="Use per-class NMS instead of Quang Minh's agnostic NMS.")
    parser.add_argument("--yolo-pad-scale-x", type=float, default=0.0)
    parser.add_argument("--yolo-pad-scale-y", type=float, default=0.0)
    parser.add_argument("--yolo-dedup-iou", type=float, default=0.90)
    parser.add_argument("--crop-batch-size", type=int, default=2)
    parser.add_argument("--max-pixels-crop", type=int, default=262_144)
    parser.add_argument("--max-new-tokens-qwen", type=int, default=192)
    parser.add_argument("--hpa-crop-pad-ratio", type=float, default=0.0)
    parser.add_argument("--qwen-crop-pad-ratio", type=float, default=0.04)
    parser.add_argument("--lazy-load", action="store_true", help="Load OCR models on first use instead of at startup.")
    parser.add_argument("--smoke-csv", action="store_true", help="Only discover images and write empty-region CSV for format checks.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    device = "cpu" if args.smoke_csv else pick_device(args.device)
    yolo_weights = Path(args.yolo_weights).resolve()
    return RuntimeConfig(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        hpa_model_dir=Path(args.hpa_model_dir).resolve(),
        qwen_base_dir=Path(args.qwen_base_dir).resolve(),
        qwen_lora_dir=Path(args.qwen_lora_dir).resolve(),
        yolo_weights=yolo_weights,
        yolo_extra_weights=auto_yolo_extra_weights(yolo_weights, args.yolo_extra_weights),
        device=device,
        yolo_backend=args.yolo_backend,
        yolo_img_size=args.yolo_img_size,
        yolo_conf=args.yolo_conf,
        yolo_max_det=args.yolo_max_det,
        yolo_iou_nms=args.yolo_iou_nms,
        yolo_agnostic_nms=not args.yolo_per_class_nms,
        yolo_pad_scale_x=args.yolo_pad_scale_x,
        yolo_pad_scale_y=args.yolo_pad_scale_y,
        yolo_dedup_iou=args.yolo_dedup_iou,
        crop_batch_size=args.crop_batch_size,
        max_pixels_crop=args.max_pixels_crop,
        max_new_tokens_qwen=args.max_new_tokens_qwen,
        hpa_crop_pad_ratio=args.hpa_crop_pad_ratio,
        qwen_crop_pad_ratio=args.qwen_crop_pad_ratio,
        lazy_load=args.lazy_load,
    )


def configure_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    warnings.filterwarnings("ignore", message=r".*Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict.*")
    logging.getLogger("transformers").setLevel(logging.ERROR)


def main() -> int:
    args = parse_args()
    configure_runtime()
    cfg = build_config(args)

    require_path(cfg.input_dir, "Input directory")
    records = discover_records(cfg.input_dir)
    if not records:
        raise RuntimeError(f"No input images found under {cfg.input_dir}")
    log(f"Input: {cfg.input_dir}")
    log(f"Output: {cfg.output_dir}")
    log(f"Images: {len(records)}")

    output_csv = cfg.output_dir / "submission.csv"
    if args.smoke_csv:
        rows = [{"image": output_image_name(record), "regions": "[]"} for record in records]
        validate_rows(rows, len(records))
        write_submission(rows, output_csv)
        log(f"Smoke CSV written: {output_csv}")
        return 0

    validate_model_paths(cfg)
    log(f"Device: {cfg.device}")
    pipeline = InferencePipeline(cfg)
    pipeline.load_models()

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        image_name = output_image_name(record)
        image_path = resolve_image_path(cfg.input_dir, str(record.get("file_name") or image_name))
        if not image_path.exists():
            log(f"[{index}/{len(records)}] missing image: {image_name}")
            regions: list[dict[str, Any]] = []
        else:
            log(f"[{index}/{len(records)}] {image_name}")
            try:
                regions = pipeline.infer_one(image_path, source=record.get("source"))
            except Exception as exc:
                log(f"Image failed: {image_name}: {exc}")
                regions = []
        rows.append({"image": image_name, "regions": json.dumps(regions, ensure_ascii=False)})

    validate_rows(rows, len(records))
    write_submission(rows, output_csv)
    total_regions = sum(len(json.loads(row["regions"])) for row in rows)
    log(f"Wrote {output_csv} rows={len(rows)} total_regions={total_regions}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
