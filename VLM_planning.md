# 🧠 M4 - VLM Planning: Kế Hoạch Chi Tiết Cho Vai Trò Generative & VLM

**Vai trò:** M4 - Chuyên gia Mô hình Lớn (Vision-Language Models)
**Phạm vi trách nhiệm:** Fine-tune VLM End-to-End, Prompt Engineering, Spell Check, LLM-based Post-processing.
**Cuộc thi:** [Handwritten to Data](https://www.kaggle.com/competitions/handwritten-to-data) — Hạn chót: 15/06/2026.

---

## 📊 Hiểu Rõ Bài Toán (Trước Khi Code)

### Công thức chấm điểm (Composite Score)
```
Score = 0.15 × Det-F1 + 0.05 × ClassAcc + 0.30 × (1 - CER) + 0.50 × (1 - PageCER)
```
> **Insight cho M4:** PageCER chiếm **50%** tổng điểm. VLM End-to-End có lợi thế cực lớn ở đây vì nó đọc toàn bộ trang theo đúng thứ tự (top-to-bottom, left-to-right), giảm lỗi ghép nối giữa Detection và OCR.

### 7 Region Types cần xử lý
| Type | VLM cần làm gì | Lưu ý |
|------|----------------|-------|
| `handwritten` | Đọc chữ viết tay tiếng Ukraina | Chiếm đa số regions |
| `printed` | Đọc chữ in | Dễ hơn handwritten |
| `formula` | Xuất ra LaTeX hoặc Unicode tương đương | Normalizer sẽ quy đổi `\frac{a}{b}` ↔ `a/b` |
| `table` | Xuất Pipe-separated values (`a\|b\|c`) | Normalizer strip whitespace quanh pipes |
| `annotation` | Đọc điểm số, ghi chú giáo viên | Thường rất ngắn |
| `image` | Trả về text rỗng `""` | Không cần OCR |
| `graph` | Trả về text rỗng `""` | Không cần OCR |

### Text Normalization (Quan trọng cho Prompt Design)
VLM không cần phải xuất ra đúng 100% format vì Normalizer sẽ xử lý:
- Latin `c`, `o`, `e` → Cyrillic `с`, `о`, `е` (tự động)
- `\frac{1}{2}` ↔ `1/2` (cả hai đều được chấp nhận)
- `\sqrt{169}` ↔ `√169` (tương đương)
- `x²` ↔ `x^2` (tương đương)
- Dấu gạch ngang (`—`, `–`) → `-` (tự động)

---

## 🗓️ Lộ Trình 5 Tuần Chi Tiết

---

### 📚 TUẦN 1: Nghiên Cứu & Thiết Lập Môi Trường (Ngày 1-7)

#### Giai đoạn 1A: Đọc Paper & Documentation (Ngày 1-3)

**Papers bắt buộc đọc:**
1. **Qwen2.5-VL Technical Report** — Hiểu kiến trúc ViT + LLM, cách model xử lý ảnh có độ phân giải cao (dynamic resolution).
2. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) — Nền tảng cho toàn bộ quá trình fine-tune.
3. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) — Kỹ thuật quantize 4-bit để fit model 8B vào GPU 24GB.

**Documentation bắt buộc đọc:**
- [HuggingFace PEFT (LoRA/QLoRA)](https://huggingface.co/docs/peft)
- [Qwen3-VL-8B-Instruct Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Unsloth Fine-tuning Guide](https://github.com/unslothai/unsloth) — Tăng tốc fine-tune 2x
- [Google Gemma 4 Collection](https://huggingface.co/collections/google/gemma-4)

**Kết quả cần đạt (DoD):**
- [ ] Tóm tắt 1 trang A4 về kiến trúc Qwen3-VL và cách LoRA hoạt động.
- [ ] Danh sách ưu/nhược điểm: Qwen3-VL-8B vs Gemma-4-E4B vs Gemma-4-26B.

#### Giai đoạn 1B: Setup Môi Trường (Ngày 4-5)

**Cài đặt dependencies:**
```bash
pip install transformers accelerate peft bitsandbytes
pip install qwen-vl-utils  # Xử lý ảnh cho Qwen3-VL
pip install datasets pillow
```

**Kiểm tra phần cứng:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
# Yêu cầu tối thiểu: 1x GPU 24GB (RTX 4090 / A100 / L4)
# Khuyến nghị: Kaggle T4x2 hoặc Colab A100
```

**Kết quả cần đạt:**
- [ ] Load được Qwen3-VL-8B-Instruct ở chế độ 4-bit quantization.
- [ ] Chạy thử inference trên 1 ảnh test → nhận được text output (dù chưa chính xác).

#### Giai đoạn 1C: Zero-shot Baseline (Ngày 6-7)

**Mục tiêu:** Chạy VLM trên toàn bộ 386 ảnh test mà KHÔNG cần fine-tune để xem điểm "miễn phí" là bao nhiêu.

**Prompt Zero-shot cơ bản:**
```python
ZERO_SHOT_PROMPT = """You are a document understanding model for Ukrainian handwritten text.
Analyze this image and extract all text regions.

For each region, output a JSON object with:
- "bbox": [x1, y1, x2, y2] pixel coordinates
- "type": one of "handwritten", "printed", "formula", "table", "annotation", "image", "graph"
- "text": the transcribed text (empty string for image/graph types)

Output format: a JSON list of region objects.
Important: For tables, use pipe-separated values (|). For formulas, use LaTeX or plain Unicode.
"""
```

**Kết quả cần đạt:**
- [ ] File `submission_zero_shot.csv` submit thành công lên Kaggle.
- [ ] Ghi nhận điểm Zero-shot làm mốc so sánh (Baseline Score).

---

### 🔧 TUẦN 2: Fine-tune VLM Lần 1 (Ngày 8-14)

#### Giai đoạn 2A: Chuẩn Bị Dataset Fine-tune (Ngày 8-9)

**Chuyển đổi metadata.jsonl → conversation format cho VLM:**
```python
def create_vlm_training_sample(record):
    """Chuyển 1 record từ metadata.jsonl thành 1 conversation."""
    image_path = record["file_name"]
    regions = record["regions"]

    # Format đầu ra mong muốn
    output_regions = []
    for r in regions:
        output_regions.append({
            "bbox": r["bbox"],
            "type": r["type"],
            "text": r.get("text", "")
        })

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps(output_regions, ensure_ascii=False)
            }
        ]
    }
```

**Chiến lược chia dữ liệu:**
- Train: 80% của tập `train` (1,064 ảnh) — chia theo `source` để cân bằng.
- Validation: 20% (266 ảnh) — đảm bảo đại diện cả 4 nguồn (dictation, archive, university, school).
- **KHÔNG dùng `silver` ở tuần này** (để đánh giá sạch).

#### Giai đoạn 2B: Fine-tune với LoRA (Ngày 10-12)

**Cấu hình LoRA chống OOM:**
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # Rank thấp để tiết kiệm VRAM
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training args
training_args = {
    "per_device_train_batch_size": 1,      # Batch 1 để tránh OOM
    "gradient_accumulation_steps": 8,       # Effective batch = 8
    "gradient_checkpointing": True,         # BẮT BUỘC
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "fp16": True,                           # Mixed precision
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "save_strategy": "epoch",
    "logging_steps": 10,
}
```

**Script tự động giảm batch size khi OOM:**
```python
import torch

def safe_train_step(model, batch, optimizer):
    """Tự động retry với gradient accumulation nếu gặp OOM."""
    try:
        loss = model(**batch).loss
        loss.backward()
        return loss.item()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("⚠️ OOM detected! Skipping batch.")
        return None
```

#### Giai đoạn 2C: Đánh Giá Lần 1 (Ngày 13-14)

**Đánh giá trên tập Validation:**
```python
from kaggle_metric import score_detailed
# Chạy inference trên 266 ảnh validation
# So sánh với Zero-shot baseline
```

**Kết quả cần đạt:**
- [ ] Model LoRA weights: `vlm_v1_lora/`
- [ ] Báo cáo: Score fine-tuned vs Score zero-shot (kỳ vọng tăng ≥ 15%).
- [ ] Submit lên Kaggle, ghi nhận điểm.

---

### 🎯 TUẦN 3: Prompt Engineering & Table/Formula (Ngày 15-21)

#### Giai đoạn 3A: Tối Ưu Prompt (Ngày 15-17)

**Thử nghiệm các chiến lược prompt khác nhau:**

1. **Structured JSON Prompt** (Yêu cầu VLM xuất JSON trực tiếp)
2. **Two-pass Prompt** (Pass 1: Liệt kê bbox + type. Pass 2: Đọc text cho từng bbox)
3. **Region-specific Prompt** (Prompt riêng cho formula/table)

**Prompt chuyên biệt cho Table:**
```
For table regions, output the content as pipe-separated values.
Each row on a new line. Example:
Назва|Кількість|Ціна
Яблука|5|12.50
Груші|3|15.00
```

**Prompt chuyên biệt cho Formula:**
```
For formula regions, output using plain Unicode math notation.
Examples: a/b (not \frac{a}{b}), √169 (not \sqrt{169}), x^2, y_3
Use Unicode symbols: π, ∠, ⊥, ∥, →, ∴
```

#### Giai đoạn 3B: Fine-tune Lần 2 với Prompt Tối Ưu (Ngày 18-20)

- Dùng prompt tốt nhất từ 3A để tạo lại dataset.
- Fine-tune thêm 2 epochs trên model v1 (continual training).
- Nếu M1 đã có Pseudo-labels, tích hợp thêm vào tập train.

#### Giai đoạn 3C: Đánh Giá Chéo (Ngày 21)

- So sánh score giữa v1 và v2.
- Phân tích: VLM đang yếu ở component nào? (Detection F1 hay CER?)

**Kết quả cần đạt:**
- [ ] Model LoRA weights v2: `vlm_v2_lora/`
- [ ] Báo cáo chi tiết: Prompt nào cho kết quả tốt nhất.
- [ ] VLM xuất đúng format Table (`a|b|c`) và Formula.

---

### 🔍 TUẦN 4: Spell Check & Error Analysis (Ngày 22-28)

#### Giai đoạn 4A: Xây Dựng Spell Checker (Ngày 22-25)

**Phương án 1: SymSpell (Ưu tiên — Siêu nhanh)**
```python
import pkg_resources
from symspellpy import SymSpell

sym_spell = SymSpell(max_dictionary_edit_distance=2)
# Load từ điển tiếng Ukraina
sym_spell.load_dictionary("uk_50k.txt", term_index=0, count_index=1)

def correct_text(text: str) -> str:
    """Sửa lỗi chính tả cho text tiếng Ukraina."""
    words = text.split()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            corrected.append(word)
    return " ".join(corrected)
```

**Phương án 2: VLM Self-correction (Backup)**
```python
CORRECTION_PROMPT = """The following Ukrainian text was extracted by OCR and may contain errors.
Fix only obvious spelling mistakes. Do not change meaning or add words.
Text: {text}
Corrected:"""
```

**Quan trọng:** Spell Check chỉ áp dụng cho `handwritten` và `printed`. KHÔNG áp dụng cho `formula`, `table`, `annotation`.

#### Giai đoạn 4B: Error Analysis Chi Tiết (Ngày 26-28)

**Chạy `score_detailed()` và phân tích:**
```python
breakdown = score_detailed(solution_df, submission_df, "image")
print(f"Detection F1:  {breakdown['detection_f1']:.4f}")
print(f"ClassAcc:      {breakdown['classification_accuracy']:.4f}")
print(f"Region CER:    {breakdown['region_cer']:.4f}")
print(f"Page CER:      {breakdown['page_cer']:.4f}")
```

**Phân tích lỗi theo từng nguồn dữ liệu:**
- `dictation`: Chữ viết tay trên giấy trắng → VLM nên mạnh.
- `archive`: Mực pen & ink, chính tả cổ (1920s) → Khó nhất.
- `university`: Công thức toán, hóa học → Cần prompt đặc biệt.
- `school`: Ảnh chụp điện thoại, mờ, nghiêng → Cần augmentation.

**Kết quả cần đạt:**
- [ ] Module `spell_checker.py` chạy < 0.1s/ảnh.
- [ ] Báo cáo Error Analysis: "VLM yếu nhất ở nguồn X, ký tự Y".

---

### 🏆 TUẦN 5: Đóng Gói & Nộp Bài (Ngày 29-35)

#### Giai đoạn 5A: Xuất Predictions Cuối Cùng (Ngày 29-31)

**Pipeline inference hoàn chỉnh của M4:**
```
Ảnh test → VLM (Qwen3-VL + LoRA v2)
         → Parse JSON output
         → Spell Check (chỉ cho handwritten/printed)
         → Text Normalization
         → Xuất JSON theo Interface của M5
```

**Tối ưu tốc độ:**
- Dùng `torch.compile()` nếu PyTorch ≥ 2.0.
- Batch inference: nhóm các ảnh có kích thước gần nhau.
- `torch.cuda.empty_cache()` sau mỗi batch.

#### Giai đoạn 5B: Giao Cho M5 Ensembling (Ngày 32-33)

- Xuất file `predictions_vlm.json` theo đúng format Interface.
- Cung cấp cho M5: model weights, script inference, và confidence scores.

#### Giai đoạn 5C: Hỗ Trợ Chốt Submission (Ngày 34-35)

- Hỗ trợ M5 debug nếu pipeline bị lỗi trên Kaggle.
- Review code lần cuối, đảm bảo không có dependency thiếu.

**Kết quả cần đạt:**
- [ ] File `predictions_vlm.json` sẵn sàng cho Ensembling.
- [ ] VLM inference chạy ổn định trên Kaggle Notebook (không OOM, không timeout).

---

## ⚙️ Quyết Định Kỹ Thuật Quan Trọng

### Chọn Model: Qwen3-VL-8B vs Gemma-4

| Tiêu chí | Qwen3-VL-8B | Gemma-4-E4B (8B MoE) | Gemma-4-26B |
|-----------|-------------|----------------------|-------------|
| VRAM (4-bit) | ~6GB | ~5GB | ~16GB |
| Tốc độ infer | Nhanh | Rất nhanh (MoE) | Chậm |
| Hỗ trợ Cyrillic | Tốt | Tốt | Rất tốt |
| LoRA ecosystem | Rất tốt (Unsloth) | Đang phát triển | Tốt |
| **Khuyến nghị** | **Ưu tiên #1** | Backup nếu OOM | Chỉ dùng nếu có A100 |

### Chiến lược Training

```
Tuần 2: Fine-tune trên 1,064 ảnh train (gold) → Model v1
Tuần 3: Continual training + Pseudo-labels từ M1 → Model v2
Tuần 4: Không train thêm, tập trung post-processing
Tuần 5: Đóng gói, xuất predictions
```

---

## 📋 Checklist Tổng Hợp

- [ ] **Tuần 1:** Đọc 3 papers + setup môi trường + Zero-shot baseline
- [ ] **Tuần 2:** Fine-tune LoRA v1 + Submit Kaggle lần 2
- [ ] **Tuần 3:** Prompt Engineering + Fine-tune v2 + Table/Formula handling
- [ ] **Tuần 4:** Spell Check module + Error Analysis report
- [ ] **Tuần 5:** Xuất predictions + Hỗ trợ Ensembling + Chốt submission

---

## 🔗 Giao Tiếp Với Các Thành Viên Khác

| Nhận từ ai | Nội dung | Thời điểm |
|------------|----------|-----------|
| **M1 (Data)** | Pseudo-labels từ tập `silver` | Tuần 3 |
| **M1 (Data)** | Từ điển tiếng Ukraina cho Spell Check | Tuần 4 |
| **M5 (MLOps)** | Interface JSON chuẩn (`interfaces.py`) | Tuần 1 |

| Gửi cho ai | Nội dung | Thời điểm |
|------------|----------|-----------|
| **M5 (MLOps)** | File `predictions_vlm.json` | Tuần 5 |
| **M5 (MLOps)** | Model weights + inference script | Tuần 5 |
| **Cả team** | Báo cáo Error Analysis | Tuần 4 |
