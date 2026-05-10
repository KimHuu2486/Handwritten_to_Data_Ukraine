import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
MODEL_ID = '/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1'
TRAIN_DATA_PATH = '/kaggle/input/datasets/trankimhuu/rukopys-vlm-dataset/train_vlm.jsonl'
VAL_DATA_PATH = '/kaggle/input/datasets/trankimhuu/rukopys-vlm-dataset/val_vlm.jsonl'
OUTPUT_DIR = 'qwen3_vl_lora_output'

# Đường dẫn đến Checkpoint cũ (nếu có Kaggle Timeout). 
# VD: '/kaggle/input/rukopys-checkpoint/checkpoint-123'
RESUME_CHECKPOINT_DIR = ''

def main():
    print(f"Bắt đầu quá trình cấu hình Fine-tuning cho Qwen3-VL-8B")
    
    # 1. Load Dataset
    print(f"Đang tải dữ liệu từ {TRAIN_DATA_PATH} và {VAL_DATA_PATH}...")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_DATA_PATH, "validation": VAL_DATA_PATH}
    )
    
    # 2. Cấu hình Quantization 4-bit (QLoRA)
    print("Đang cấu hình 4-bit quantization để tránh OOM...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 3. Load Model & Processor
    print("Đang load model Qwen3-VL-8B...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        max_memory={0: "14GB", 1: "6GB"},  # Ép GPU 1 xuống 6GB để lấy tới 9GB VRAM trống cho Loss
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="sdpa", # Quan trọng để tiết kiệm VRAM
        low_cpu_mem_usage=True
    )
    
    # Kích hoạt gradient checkpointing để tiết kiệm memory
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # 4. Cấu hình LoRA
    print("Đang áp dụng LoRA adapters...")
    lora_config = LoraConfig(
        r=8,                 # Giảm Rank để cứu vớt VRAM lúc Backward
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Data Collator Tùy Chỉnh Cho VLM
    def data_collator(examples):
        messages_list = [example["messages"] for example in examples]
        
        # Tạo text prompt hoàn chỉnh
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
            for msg in messages_list
        ]
        
        # Lấy thông tin vision (pixel values)
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        # Pass qua processor
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Cấu hình Labels cho Loss function
        labels = batch["input_ids"].clone()
        # Không tính loss cho phần padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Masking phần User Prompt (Chỉ tính Loss cho câu trả lời của Assistant)
        try:
            search_seq = processor.tokenizer.encode("<|im_start|>assistant\n", allowed_special="all")
            search_len = len(search_seq)
            for i in range(len(labels)):
                label_list = labels[i].tolist()
                match_idx = -1
                for j in range(len(label_list) - search_len + 1):
                    if label_list[j:j+search_len] == search_seq:
                        match_idx = j + search_len
                        break
                
                if match_idx != -1:
                    labels[i, :match_idx] = -100
                else:
                    # Rất hiếm khi xảy ra, nhưng nếu không tìm thấy thì cứ mask phần đầu
                    labels[i, :len(label_list)//2] = -100
        except Exception as e:
            print(f"Warning: Masking user prompt failed: {e}")

        batch["labels"] = labels
        
        return batch

    # 6. Training Arguments
    print("Đang thiết lập Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      # Bắt buộc = 1 trên Kaggle T4
        gradient_accumulation_steps=8,      # Tạo effective batch size = 8
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,                          # Mixed precision training
        optim="paged_adamw_8bit",           # Tiết kiệm tối đa VRAM cho Optimizer
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",              # Đánh giá sau mỗi epoch
        save_strategy="epoch",              # Lưu checkpoint sau mỗi epoch
        save_total_limit=2,                 # Chỉ giữ 2 checkpoint gần nhất để đỡ tốn dung lượng
        report_to="none",                   # Tắt WandB trên Kaggle
        remove_unused_columns=False,        # QUAN TRỌNG: False vì ta cần cột "messages" trong collator
        gradient_checkpointing=True,        # Giảm OOM
        use_liger_kernel=True               # Bật Liger Kernel để tối ưu OOM (giảm 50% VRAM ở Loss)
    )
    
    # 7. Trainer Khởi Tạo
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # 8. Bắt đầu Train
    print("Bắt đầu Fine-tuning...")
    
    # Logic kiểm tra thông minh: Chạy tiếp (Resume) hay Train Mới
    resume_path = None
    if RESUME_CHECKPOINT_DIR and os.path.exists(RESUME_CHECKPOINT_DIR):
        print(f"🔄 Tìm thấy thư mục Checkpoint tại {RESUME_CHECKPOINT_DIR}.")
        print("Tiến trình sẽ tiếp tục chạy từ vị trí đã dừng!")
        resume_path = RESUME_CHECKPOINT_DIR
    else:
        print("🆕 Không có cấu hình Checkpoint (hoặc đường dẫn không tồn tại). Train mới từ đầu!")

    trainer.train(resume_from_checkpoint=resume_path)
    
    # 9. Lưu model cuối cùng
    final_path = os.path.join(OUTPUT_DIR, "qwen_lora_final")
    print(f"Đang lưu model LoRA tại {final_path}...")
    trainer.model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print("Hoàn thành Fine-tuning!")

if __name__ == "__main__":
    main()