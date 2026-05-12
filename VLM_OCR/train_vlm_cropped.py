import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import TrainerCallback
from PIL import Image

class PrintLossCallback(TrainerCallback):
    """Callback tùy chỉnh để in log sạch đẹp và dễ nhìn trên Kaggle mỗi 10 steps"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs.get("loss", 0)
            lr = logs.get("learning_rate", 0)
            epoch = logs.get("epoch", 0)
            print(f"📊 [Step {step}] Epoch {epoch:.2f} | Loss: {loss:.4f} | LR: {lr:.2e}")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
MODEL_ID = '/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1'
TRAIN_DATA_PATH = '/kaggle/input/datasets/trankimhuu/rukopys-vlm-dataset/train_vlm.jsonl'
OUTPUT_DIR = 'qwen3_vl_lora_crop_output'

# ==========================================
# DATA COLLATOR SIÊU TỐC
# ==========================================
class QwenDataCollator:
    def __init__(self, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length
        # Chuẩn bị sẵn pattern để mask
        self.search_seq = processor.tokenizer.encode("<|im_start|>assistant\n", allowed_special="all", add_special_tokens=False)
        self.search_len = len(self.search_seq)

    def __call__(self, examples):
        messages_list = [example["messages"] for example in examples]
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages_list]
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        # [TỐI ƯU RESIZE] Giới hạn độ phân giải của các ảnh crop quá to
        # Giúp tiết kiệm VRAM và tăng tốc độ Train giống hệt như bên Inference
        if image_inputs is not None:
            optimized_images = []
            for img in image_inputs:
                if isinstance(img, list):
                    opt_list = []
                    for i in img:
                        if hasattr(i, "thumbnail"):
                            i = i.copy()
                            i.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        opt_list.append(i)
                    optimized_images.append(opt_list)
                else:
                    if hasattr(img, "thumbnail"):
                        img = img.copy()
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    optimized_images.append(img)
            image_inputs = optimized_images
            
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=False, # BẮT BUỘC TẮT TRUNCATION RÕ RÀNG ĐỂ CHỐNG LỖI CẮT ẢNH
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Masking siêu tốc bằng Vectorized Tensor (nhanh hơn vòng lặp gấp nhiều lần)
        search_tensor = torch.tensor(self.search_seq, device=labels.device)
        for i in range(len(labels)):
            # Chống crash nếu sequence quá ngắn (vài token)
            if labels[i].size(0) < self.search_len:
                labels[i, :] = -100
                continue
                
            windows = labels[i].unfold(0, self.search_len, 1)
            matches = (windows == search_tensor).all(dim=1)
            match_indices = matches.nonzero(as_tuple=True)[0]
            if len(match_indices) > 0:
                match_idx = match_indices[0].item() + self.search_len
                labels[i, :match_idx] = -100
            else:
                labels[i, :] = -100

        batch["labels"] = labels
        
        # Ép kiểu ảnh và input sang float16 để bật Tensor Cores triệt để
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float32 or value.dtype == torch.bfloat16:
                    batch[key] = value.to(torch.float16)
                
        return batch

def main():
    print("🚀 Bắt đầu Fine-tuning Qwen-VL trên dữ liệu CROPPED siêu nhỏ")
    
    # 1. Load Dataset & Chia tập Validation
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA_PATH})
    
    # Chia 5% dữ liệu (khoảng ~1260 mẫu) để làm tập Validation đánh giá model
    split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    
    # 2. Cấu hình 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 3. Load Model (KHÔNG BỊ ÉP BỎ max_memory NỮA)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Tắt cảnh báo Right-padding của HuggingFace (An toàn cho cả Train)
    processor.tokenizer.padding_side = "left"
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    )
    
    # Fix lỗi BFloat16 trên Kaggle T4
    model.config.torch_dtype = torch.float16
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = torch.float16
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # 4. LORA Config (Bảo toàn sức mạnh r=16, target=7 lớp)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # ÉP KIỂU TUYỆT ĐỐI (TRẢM BFLOAT16)
    # Card T4 của Kaggle không hỗ trợ BFloat16. Phải đảm bảo không còn 1 mầm mống bfloat16 nào
    # tồn tại trong cả weights, bias và buffers sau khi gắn LoRA.
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)

    model.print_trainable_parameters()
    
    # Khởi tạo Collator siêu tốc ở ngoài
    data_collator = QwenDataCollator(processor=processor, max_length=1024)

    # 6. Cấu hình Train Cởi trói (Batch lớn hơn, Load đa luồng)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      # Chuyển về 1 để ôm trọn mọi kích thước ảnh (kể cả 4000 tokens)
        gradient_accumulation_steps=8,      # Tăng lên 8 để giữ nguyên Effective Batch Size = 8
        learning_rate=2e-5,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",              # Đánh giá model trên tập Val sau mỗi epoch
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=2,           # Nạp ảnh dưới background để chống nghẽn CPU
    )
    
    # 7. Khởi chạy
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,             # Dữ liệu Train
        eval_dataset=val_data,                # Dữ liệu Validation
        data_collator=data_collator,
        callbacks=[PrintLossCallback()],      # Gắn thêm bộ in log 10 steps
    )
    
    print("Bắt đầu huấn luyện...")
    trainer.train()
    
    final_path = os.path.join(OUTPUT_DIR, "qwen_lora_final")
    trainer.model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"✅ Hoàn thành! Model lưu tại {final_path}")

if __name__ == "__main__":
    main()
