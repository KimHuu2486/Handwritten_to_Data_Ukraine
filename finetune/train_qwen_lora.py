import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import sys
import time
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ==========================================
# CẤU HÌNH LOGGING - ĐẢM BẢO LOG HIỆN RA
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Force HuggingFace Transformers logger hiện INFO
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("transformers.trainer").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# CUSTOM CALLBACK - IN TIẾN TRÌNH CHI TIẾT
# ==========================================
class PrintProgressCallback(TrainerCallback):
    """Callback in tiến trình training chi tiết ra stdout với flush=True."""
    
    def __init__(self):
        self.train_start_time = None
        self.step_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        total_steps = state.max_steps
        print(f"\n{'='*70}", flush=True)
        print(f"🚀 BẮT ĐẦU TRAINING", flush=True)
        print(f"   Total steps: {total_steps}", flush=True)
        print(f"   Epochs: {args.num_train_epochs}", flush=True)
        print(f"   Batch size (per device): {args.per_device_train_batch_size}", flush=True)
        print(f"   Gradient accumulation: {args.gradient_accumulation_steps}", flush=True)
        print(f"   Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}", flush=True)
        print(f"{'='*70}\n", flush=True)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Được gọi mỗi logging_steps - in thông tin chi tiết."""
        if logs is None:
            return
        
        current_step = state.global_step
        max_steps = state.max_steps
        epoch = state.epoch or 0
        
        # Lấy loss
        loss = logs.get("loss", logs.get("eval_loss", None))
        lr = logs.get("learning_rate", None)
        
        # Tính tốc độ và ETA
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        speed = elapsed / max(current_step, 1)
        remaining_steps = max_steps - current_step
        eta_seconds = speed * remaining_steps
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        
        # Lấy thông tin VRAM
        vram_info = ""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                vram_info += f" | GPU{i}: {allocated:.1f}/{reserved:.1f}GB"
        
        # Format và print
        progress_pct = (current_step / max_steps * 100) if max_steps > 0 else 0
        msg = f"📊 [Step {current_step}/{max_steps} ({progress_pct:.1f}%) | Epoch {epoch:.2f}]"
        if loss is not None:
            msg += f" Loss: {loss:.4f}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        msg += f" | Speed: {speed:.1f}s/step | Elapsed: {elapsed_str} | ETA: {eta_str}"
        msg += vram_info
        
        print(msg, flush=True)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"\n🏁 EPOCH {epoch:.0f} HOÀN THÀNH | Elapsed: {elapsed_str}", flush=True)
        print(f"-"*70, flush=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", "N/A")
            print(f"📝 EVAL RESULT: eval_loss = {eval_loss}", flush=True)
    
    def on_save(self, args, state, control, **kwargs):
        print(f"💾 Checkpoint đã lưu tại step {state.global_step}", flush=True)
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time if self.train_start_time else 0
        total_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f"\n{'='*70}", flush=True)
        print(f"✅ TRAINING HOÀN TẤT", flush=True)
        print(f"   Tổng thời gian: {total_str}", flush=True)
        print(f"   Tổng steps: {state.global_step}", flush=True)
        print(f"   Best metric: {state.best_metric}", flush=True)
        print(f"{'='*70}\n", flush=True)

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

class OOMRecoverySFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            import inspect
            sig = inspect.signature(super().training_step)
            if 'num_items_in_batch' in sig.parameters:
                loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            else:
                loss = super().training_step(model, inputs)
            
            # FIX: Ép loss về device của Trainer để tránh ValueError trên 2 GPU (device_map="auto")
            if self.args.device != loss.device:
                loss = loss.to(self.args.device)
                
            return loss
        except torch.cuda.OutOfMemoryError:
            import gc
            print("\n🚨 [OOM WARNING] Hết VRAM tại một batch! Đang dọn dẹp và skip batch...", flush=True)
            for p in model.parameters():
                p.grad = None  # Xóa sạch các gradient đang tính dở dang một cách an toàn
            torch.cuda.empty_cache()
            gc.collect()
            
            # Trả về tensor loss = 0. Không dùng requires_grad=True vì Trainer đã lo backward bên trong training_step rồi.
            return torch.tensor(0.0, device=self.args.device)

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
        torch_dtype=torch.float16,          # FIX: Ép kiểu float16 thay vì bfloat16 mặc định của model để chạy được trên T4
        trust_remote_code=True,
        attn_implementation="sdpa", # Quan trọng để tiết kiệm VRAM
        low_cpu_mem_usage=True
    )
    
    # FIX TRIỆT ĐỂ LỖI BFLOAT16: Các mô hình đa phương thức (VLM) thường lồng config bên trong.
    # Dù đã truyền torch_dtype=float16 ở trên, text_config bên trong vẫn lén giữ bfloat16 gây crash GradScaler!
    model.config.torch_dtype = torch.float16
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = torch.float16
        if hasattr(model.config.text_config, "dtype"):
            model.config.text_config.dtype = "float16"
    if hasattr(model.config, "vision_config"):
        model.config.vision_config.torch_dtype = torch.float16
        if hasattr(model.config.vision_config, "dtype"):
            model.config.vision_config.dtype = "float16"
            
    # Ép toàn bộ tham số sót lại (như lm_head) về float16
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    
    # Kích hoạt gradient checkpointing để tiết kiệm memory
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # 4. Cấu hình LoRA
    print("Đang áp dụng LoRA adapters...")
    lora_config = LoraConfig(
        r=16,                 # Tăng rank từ 8 lên 16
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # FIX CUỐI CÙNG VÀ CHẮC CHẮN NHẤT CHO LỖI BFLOAT16 GRADSCALER:
    # Mặc dù ta đã ép model về float16, thư viện PEFT khi khởi tạo LoRA có thể vô tình kế thừa lại bfloat16 
    # từ một ngóc ngách nào đó của Qwen. GradScaler của T4 sẽ crash nếu Gradient là bfloat16.
    # GIẢI PHÁP: Ép toàn bộ trọng số đang được train (LoRA adapters) sang float32.
    # float32 không chỉ được GradScaler hỗ trợ 100% mà còn giúp hội tụ tốt hơn (chống underflow).
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
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
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Cấu hình Labels cho Loss function
        labels = batch["input_ids"].clone()
        # Không tính loss cho phần padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Masking phần User Prompt (Chỉ tính Loss cho câu trả lời của Assistant)
        try:
            # FIX: Thêm add_special_tokens=False để không bị dính BOS token
            search_seq = processor.tokenizer.encode("<|im_start|>assistant\n", allowed_special="all", add_special_tokens=False)
            search_len = len(search_seq)
            for i in range(len(labels)):
                label_list = labels[i].tolist()
                
                # Logic skip dữ liệu theo yêu cầu:
                # Vì padding=True và batch_size=1, nếu len == 2048 tức là mẫu này đã bị cắt cụt do vượt quá max_length
                is_truncated = (len(label_list) == 2048)
                
                match_idx = -1
                for j in range(len(label_list) - search_len + 1):
                    if label_list[j:j+search_len] == search_seq:
                        match_idx = j + search_len
                        break
                
                if match_idx != -1 and not is_truncated:
                    labels[i, :match_idx] = -100
                else:
                    # Nếu không tìm thấy marker HOẶC mẫu bị cắt bớt, mask TOÀN BỘ (-100) để bỏ qua sample này
                    labels[i, :] = -100
                    if is_truncated:
                        print(f"\n⚠️ Bỏ qua sample (Mask = -100) vì vượt quá giới hạn 2048 token.", flush=True)
        except Exception as e:
            print(f"Warning: Masking user prompt failed: {e}")

        batch["labels"] = labels
        
        # FIX TỐI ƯU HÓA VRAM & TỐC ĐỘ: Ép kiểu float32 -> float16 cho ảnh
        # Vì đã tắt fp16=False ở Trainer để tránh lỗi conflict, tính năng autocast bị vô hiệu.
        # Processor mặc định xuất pixel_values dưới dạng float32, gây chậm và ngốn gấp đôi VRAM.
        # Chuyển đổi thủ công về float16 giúp kích hoạt Tensor Cores của thẻ T4 và tiết kiệm VRAM!
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.float16)
        
        return batch

    # 6. Training Arguments
    print("Đang thiết lập Training Arguments (SFTConfig)...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      # Bắt buộc = 1 trên Kaggle T4
        gradient_accumulation_steps=8,      # Tạo effective batch size = 8
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=False,                         # TẮT fp16 CỦA TRAINER ĐỂ TẮT GRADSCALER (Fix lỗi BFloat16 trên T4)
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
        use_liger_kernel=False,             # TẮT Liger Kernel (Nguyên nhân gây lỗi Triton device mismatch trên multi-GPU)
        dataset_text_field="",              # SFTTrainer field (bỏ trống vì ta dùng custom collator)
        dataset_kwargs={"skip_prepare_dataset": True}, # Bỏ qua logic auto-format của TRL
    )
    
    # 7. Trainer Khởi Tạo (với Custom Callback và OOM Recovery SFT)
    trainer = OOMRecoverySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[PrintProgressCallback()],
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