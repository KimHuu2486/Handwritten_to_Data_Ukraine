#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mã nguồn huấn luyện OCR tinh gọn, hiệu năng cao cho mô hình Kansallisarkisto/cyrillic-htr-model.
Hỗ trợ đa miền vùng văn bản (handwritten, printed, annotation) thuộc nhánh RUKOPYS HPA.
Tối ưu hóa toàn diện cho phần cứng GPU NVIDIA A6000 (Thunder Compute).
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from jiwer import cer, wer
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    TrainerCallback,
    VisionEncoderDecoderModel,
)

ImageFile.LOAD_TRUNCATED_IMAGES = False

# ============================================================================
# CẤU HÌNH THAM SỐ HỆ THỐNG (Dễ dàng tùy chỉnh trực tiếp)
# ============================================================================
DATA_ROOT = Path(os.getenv("HPA_DATA_ROOT", os.getenv("DATA_ROOT", "/home/ubuntu/dataset")))
LABELS_JSONL = Path(os.getenv("HPA_LABELS_JSONL", str(DATA_ROOT / "labels.jsonl")))
IMAGE_ROOT = Path(os.getenv("HPA_IMAGE_ROOT", str(DATA_ROOT)))
OUTPUT_DIR = Path(os.getenv("HPA_OUTPUT_DIR", os.getenv("OUTPUT_DIR", "/home/ubuntu/second-try/outputs")))
MODEL_NAME = "Kansallisarkisto/cyrillic-htr-model"
BASE_PROCESSOR_MODEL = "microsoft/trocr-base-handwritten"

# Siêu tham số huấn luyện phối bộ cho GPU A6000
BATCH_SIZE = 64
NUM_EPOCHS = 8
LEARNING_RATE = 4e-5
MAX_TARGET_LENGTH = 192
MAX_EVAL_SAMPLES = 1000          # Giới hạn số mẫu kiểm định trong mỗi kết thúc epoch để giảm bottleneck
TYPED_EVAL_BATCH_SIZE = 32       # Batch size an toàn, tránh OOM trong lúc sinh chuỗi tự hồi quy
EARLY_STOPPING_PATIENCE = 3      # Dừng sớm nếu CER không cải thiện sau 3 epoch liên tiếp

SEED = 42
VAL_RATIO = 0.10
DATALOADER_NUM_WORKERS = 4
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 50
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"
OPTIM = "adamw_torch"

TARGET_LABELS = ("handwritten", "printed", "annotation")
UKRAINIAN_EXTRA_TOKENS = ["і", "ї", "є", "ґ", "І", "Ї", "Є", "Ґ"]

# Cấu hình sinh chuỗi tối ưu theo đặc trưng độ dài vùng văn bản khi chạy Full Eval
GEN_CONFIG_BY_TYPE = {
    "handwritten": {"num_beams": 3, "max_new_tokens": 192, "length_penalty": 1.0, "early_stopping": True},
    "printed": {"num_beams": 3, "max_new_tokens": 128, "length_penalty": 1.0, "early_stopping": True},
    "annotation": {"num_beams": 1, "max_new_tokens": 16, "length_penalty": 0.8, "early_stopping": True},
}

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return text.replace("\u00a0", " ").strip()

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def open_rgb_or_raise(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {path}")
    return Image.open(path).convert("RGB")

def load_metadata_records(jsonl_path: Path, image_root: Path) -> List[Dict[str, Any]]:
    raw_data = read_jsonl(jsonl_path)
    records = []
    for item in raw_data:
        # Hỗ trợ phân tách linh hoạt cấu trúc trường nhãn của file dữ liệu đầu vào
        crop_path = item.get("crop_path") or item.get("image") or item.get("file_name") or item.get("path")
        label_type = item.get("type") or item.get("label") or ""
        text = normalize_text(item.get("text", ""))
        
        if not crop_path or not label_type or not text:
            continue
        if label_type not in TARGET_LABELS:
            continue
            
        records.append({
            "image_path": image_root / crop_path,
            "label": label_type,
            "text": text
        })
    return records

def stratified_split(records: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
        
    train_rows, val_rows = [], []
    for label in TARGET_LABELS:
        rows = list(by_label[label])
        rng.shuffle(rows)
        if len(rows) <= 1:
            train_rows.extend(rows)
            continue
        n_val = max(1, round(len(rows) * val_ratio))
        n_val = min(n_val, len(rows) - 1)
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
        
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows

class HpaDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], processor: TrOCRProcessor, max_target_length: int):
        self.records = list(records)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = open_rgb_or_raise(Path(record["image_path"]))
        
        # Xử lý ma trận ảnh chuẩn hóa sang định dạng Tensor đầu vào của Encoder
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Chỉ Tokenize nhãn văn bản mục tiêu, giữ nguyên cơ chế đệm tự động cho Bộ thu gom (Collator)
        labels = self.processor.tokenizer(
            record["text"],
            padding=False,
            truncation=True,
            max_length=self.max_target_length,
        )["input_ids"]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# ============================================================================
# CUSTOM DATA COLLATOR DÀNH RIÊNG CHO BÀI TOÁN OCR VĂN BẢN
# ============================================================================
class OcrDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Gom tụ tập hợp các ma trận pixel ảnh đầu vào của khối cấu trúc Vision Encoder
        pixel_values = torch.stack([
            f["pixel_values"] if isinstance(f["pixel_values"], torch.Tensor)
            else torch.tensor(f["pixel_values"])
            for f in features
        ])
        
        # Thu thập tập hợp danh sách nhãn văn bản của Decoder
        labels = [f["labels"] for f in features]
        
        # Thực hiện đệm động (Dynamic Padding) nhãn văn bản theo chuỗi dài nhất trong batch hiện tại
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )
        
        labels_tensor = batch_labels["input_ids"]
        # Chuyển đổi toàn bộ các Token đệm thành giá trị -100 để loại bỏ khỏi hàm loss của Decoder
        labels_tensor = labels_tensor.masked_fill(labels_tensor == self.tokenizer.pad_token_id, -100)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels_tensor
        }

def configure_tokenizer_and_model(processor: TrOCRProcessor, model: VisionEncoderDecoderModel):
    tokenizer = processor.tokenizer
    changed_vocab = False

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        changed_vocab = True

    # Đồng bộ hóa và bổ sung dải ký tự chữ Cyrillic tiếng Ukraina vào Tokenizer nền
    vocab = tokenizer.get_vocab()
    missing_tokens = [t for t in UKRAINIAN_EXTRA_TOKENS if t not in vocab]
    if missing_tokens:
        tokenizer.add_tokens(missing_tokens)
        changed_vocab = True

    if changed_vocab:
        new_size = len(tokenizer)
        model.decoder.resize_token_embeddings(new_size)
        model.config.vocab_size = new_size
        model.config.decoder.vocab_size = new_size

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    model.config.decoder_start_token_id = bos_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.decoder_start_token_id = bos_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id
    
    model.generation_config = GenerationConfig(
        decoder_start_token_id=bos_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

@torch.inference_mode()
def evaluate_typed_generation(
    model: VisionEncoderDecoderModel, 
    processor: TrOCRProcessor, 
    records: Sequence[Dict[str, Any]], 
    device: torch.device, 
    batch_size: int, 
    output_jsonl: Path,
    use_fixed_beams: Optional[int] = None
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    
    # Sao lưu trạng thái use_cache ban đầu để bảo vệ luồng Gradient Checkpointing
    old_use_cache = getattr(model.config, "use_cache", None)
    old_decoder_use_cache = getattr(model.decoder.config, "use_cache", None) if hasattr(model, "decoder") else None
    
    try:
        model.config.use_cache = True
        if hasattr(model, "decoder"):
            model.decoder.config.use_cache = True
            
        rows = []
        by_label = defaultdict(list)
        for r in records:
            by_label[r["label"]].append(r)

        for label in TARGET_LABELS:
            label_records = by_label[label]
            if not label_records:
                continue
                
            # Thiết lập tham số sinh chuỗi động dựa trên loại vùng văn bản
            gen_kwargs = GEN_CONFIG_BY_TYPE[label].copy()
            if use_fixed_beams is not None:
                gen_kwargs["num_beams"] = use_fixed_beams
                if use_fixed_beams == 1:
                    gen_kwargs["early_stopping"] = False
            
            for start in tqdm(range(0, len(label_records), batch_size), desc=f"[Inference Kiểm Định] Nhóm loại vùng: {label}"):
                batch = label_records[start : start + batch_size]
                images = [open_rgb_or_raise(Path(r["image_path"])) for r in batch]
                pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
                
                generated_ids = model.generate(pixel_values, **gen_kwargs)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                for record, pred in zip(batch, preds):
                    rows.append({
                        "image_path": str(record["image_path"]),
                        "label": record["label"],
                        "reference": record["text"],
                        "prediction": normalize_text(pred),
                    })
    finally:
        # KHÓA CỐT LÕI: Đảm bảo khôi phục lại trạng thái use_cache ban đầu trong mọi tình huống kết thúc luồng sinh
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if old_decoder_use_cache is not None:
            model.decoder.config.use_cache = old_decoder_use_cache
        if was_training:
            model.train()
    
    # Tính toán các chỉ số độ lỗi OCR đặc hiệu (CER / WER / Exact Match)
    def compute_metrics_for_subset(sub_rows):
        if not sub_rows: 
            return {"cer": 1.0, "wer": 1.0, "exact_match": 0.0}
        refs = [r["reference"] for r in sub_rows]
        preds = [r["prediction"] for r in sub_rows]
        em = sum(r == p for r, p in zip(refs, preds)) / len(sub_rows)
        return {"cer": round(cer(refs, preds), 6), "wer": round(wer(refs, preds), 6), "exact_match": round(em, 4)}

    metrics = {"overall": compute_metrics_for_subset(rows), "by_label": {}}
    for label in TARGET_LABELS:
        metrics["by_label"][label] = compute_metrics_for_subset([r for r in rows if r["label"] == label])
        
    write_jsonl(output_jsonl, rows)
    return metrics

def save_model_artifact(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)

# ============================================================================
# CUSTOM TRAINING CALLBACK: QUẢN LÝ KIỂM ĐỊNH TỐI ƯU VÀ DỪNG SỚM (EARLY STOPPING)
# ============================================================================
class TypedEvalCheckpointCallback(TrainerCallback):
    def __init__(self, processor: TrOCRProcessor, val_records: Sequence[Dict[str, Any]], device: torch.device, batch_size: int, output_dir: Path, patience: int, max_eval_samples: int):
        self.processor = processor
        self.val_records = list(val_records)
        self.device = device
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.eval_dir = output_dir / "epoch_evaluations"
        self.best_model_dir = output_dir / "best_model"
        
        self.best_cer = float("inf")
        self.patience = patience
        self.max_eval_samples = max_eval_samples
        self.epochs_no_improve = 0  

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None: 
            return control

        epoch_label = f"{state.epoch:.1f}"
        prediction_path = self.eval_dir / f"epoch_{epoch_label}_predictions.jsonl"
        metrics_path = self.eval_dir / f"epoch_{epoch_label}_metrics.json"

        # Tối ưu hóa: Trích xuất tập nhỏ mẫu kiểm định để tránh làm nghẽn thời gian giữa các Epoch
        eval_records = self.val_records
        if len(eval_records) > self.max_eval_samples:
            eval_records = eval_records[:self.max_eval_samples]
            print(f"\n[Hệ Thống] Giới hạn tập Validation xuống {self.max_eval_samples} mẫu để tăng tốc độ Epoch Evaluation.")

        print(f"\n--- Bắt đầu chạy quét đánh giá nhanh (num_beams=1) tại cuối Epoch {epoch_label} ---")
        typed_metrics = evaluate_typed_generation(
            model=model, processor=self.processor, records=eval_records,
            device=self.device, batch_size=self.batch_size, output_jsonl=prediction_path,
            use_fixed_beams=1  # Sử dụng num_beams=1 để xử lý nhanh trong quá trình huấn luyện
        )
        save_json(metrics_path, typed_metrics)

        current_cer = typed_metrics["overall"]["cer"]
        print(f"[Kết quả Epoch {epoch_label}] CER Tổng thể: {current_cer:.5f} | WER: {typed_metrics['overall']['wer']:.5f}")
        
        # Kiểm tra điều kiện lưu mô hình tối ưu dựa trên CER
        if current_cer < self.best_cer:
            self.best_cer = current_cer
            self.epochs_no_improve = 0  
            save_model_artifact(model, self.processor, self.best_model_dir)
            save_json(self.best_model_dir / "val_metrics.json", typed_metrics)
            print(f"--> [LƯU MODEL] Tìm thấy trạng thái tối ưu mới. Đã lưu Checkpoint vào thư mục: {self.best_model_dir}")
        else:
            self.epochs_no_improve += 1
            print(f"--> [Early Stopping] Chỉ số CER không cải thiện liên tiếp: {self.epochs_no_improve}/{self.patience} epoch.")
            
            if self.epochs_no_improve >= self.patience:
                print(f"--> [Early Stopping Triggered] Kích hoạt dừng huấn luyện sớm do vượt ngưỡng kiên nhẫn.")
                control.should_training_stop = True
            
        return control

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị tính toán được chỉ định: {device}")

    # Khởi tạo nạp bộ xử lý Processor và Tokenizer đồng bộ
    print("\n[Hệ Thống] Đang tiến hành tải mô hình và bộ cấu trúc xử lý ký tự nền tảng...")
    processor = TrOCRProcessor.from_pretrained(BASE_PROCESSOR_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor.tokenizer = tokenizer

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    configure_tokenizer_and_model(processor, model)
    model.to(device)

    # Đọc nạp toàn bộ cấu trúc dữ liệu từ tệp nhãn JSONL duy nhất đã qua Augment
    print(f"\n[Dữ Liệu] Đang đọc tệp nhãn chính từ cấu trúc: {LABELS_JSONL}")
    all_records = load_metadata_records(LABELS_JSONL, IMAGE_ROOT)
    print(f"[Dữ Liệu] Tổng số lượng mẫu hợp lệ tìm thấy: {len(all_records):,}")
    
    # Phân tách tập huấn luyện và tập kiểm định theo tỷ lệ phân tầng cố định
    train_records, val_records = stratified_split(all_records, VAL_RATIO, SEED)
    print(f"[Dữ Liệu] Tập Huấn Luyện: {len(train_records):,} | Tập Kiểm Định: {len(val_records):,}")

    train_dataset = HpaDataset(train_records, processor, MAX_TARGET_LENGTH)

    # Khởi tạo Custom Data Collator chuyên biệt cho OCR hình ảnh văn bản
    data_collator = OcrDataCollator(tokenizer=processor.tokenizer)

    # Tự động cấu hình toán hạng tính toán bf16 hoặc fp16 dựa trên năng lực nhân lõi phần cứng của mô hình GPU
    bf16_status = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_status = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    print(f"[Cấu Hình] Chế độ AMP tự động - Kích hoạt bfloat16: {bf16_status} | Kích hoạt fp16: {fp16_status}")

    # Chuyển đổi tham số cấu hình sang dạng tương thích cao "evaluation_strategy"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "trainer_workspace"),
        run_name="rukopys_ocr_hpa_single_stage",
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIM,
        bf16=bf16_status,
        fp16=fp16_status,
        gradient_checkpointing=True,
        save_strategy="no",                 # Vô hiệu hóa lưu mặc định để tránh tốn không gian đĩa cứng vô ích
        evaluation_strategy="no",           # Vô hiệu hóa tính toán lặp mặc định, chuyển giao cho luồng Custom Callback
        logging_steps=LOGGING_STEPS,
        report_to="none",
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,          # Tối ưu hóa lưu trữ ánh xạ vùng nhớ cho phần cứng A6000
        persistent_workers=True,             # Giữ các luồng nạp dữ liệu hoạt động liên tục giữa các Epoch
        remove_unused_columns=True,
    )

    eval_callback = TypedEvalCheckpointCallback(
        processor=processor, val_records=val_records, device=device,
        batch_size=TYPED_EVAL_BATCH_SIZE, output_dir=OUTPUT_DIR,
        patience=EARLY_STOPPING_PATIENCE, max_eval_samples=MAX_EVAL_SAMPLES
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback],
    )

    # Tắt tính năng lưu cấu trúc cache để đảm bảo không xung đột với cơ chế Gradient Checkpointing trong lúc Train
    model.config.use_cache = False
    
    print("\n[Hành Trình] Bắt đầu kích hoạt luồng huấn luyện OCR đơn tầng trực tiếp...")
    trainer.train()

    # ============================================================================
    # GIAI ĐOẠN KIỂM ĐỊNH FULL TOÀN DIỆN VÀ XUẤT THÀNH PHẨM SAU CÙNG
    # ============================================================================
    print("\n" + "="*90)
    print("HOÀN THÀNH TIẾN TRÌNH TRAIN. KHỞI CHẠY GIAI ĐOẠN KIỂM ĐỊNH FULL (BEAM SEARCH = 3)")
    print("="*90)
    
    # Nạp lại trọng số tốt nhất ghi nhận được từ luồng Custom Callback kiểm định trước đó
    if eval_callback.best_model_dir.exists():
        print(f"[Hệ Thống] Tải lại mô hình có chỉ số CER tối ưu nhất từ: {eval_callback.best_model_dir}")
        model = VisionEncoderDecoderModel.from_pretrained(eval_callback.best_model_dir)
        model.to(device)

    final_predictions_path = OUTPUT_DIR / "final_full_val_predictions.jsonl"
    final_metrics_path = OUTPUT_DIR / "final_full_val_metrics.json"

    # Chạy sinh chuỗi đầy đủ 100% mẫu Validation với tham số chuẩn Beam Search của từng vùng miền văn bản
    final_metrics = evaluate_typed_generation(
        model=model, processor=processor, records=val_records,
        device=device, batch_size=TYPED_EVAL_BATCH_SIZE, output_jsonl=final_predictions_path,
        use_fixed_beams=None  # Áp dụng chính xác tham số num_beams=3 cấu hình riêng của từng vùng
    )
    
    save_json(final_metrics_path, final_metrics)
    print(f"\n[KẾT QUẢ CUỐI CÙNG] Chỉ số Full Validation Metrics:\n{json.dumps(final_metrics, indent=2, ensure_ascii=False)}")

    final_model_dir = OUTPUT_DIR / "final_cyrillic_htr_model"
    print(f"\n[Hệ Thống] Tiến hành xuất mô hình OCR thành phẩm sau cùng tại: {final_model_dir}")
    save_model_artifact(model, processor, final_model_dir)

if __name__ == "__main__":
    main()