# RUKOPYS HTR — Kế hoạch Toàn diện: Thi đấu & Nghiên cứu

> **Competition:** [Handwritten to Data — Kaggle](https://www.kaggle.com/competitions/handwritten-to-data) | Deadline: 15/06/2026  
> **Base model:** Qwen3-VL 8B Instruct  
> **Hardware:** RTX A6000 48 GB VRAM · 32 GB RAM  
> **Paper target:** ACCV 2026 / CSONet 2026

---

## 0. Phân tích cuộc thi & Chiến lược tổng quan

### 0.1 Công thức điểm

```
Score = 0.15 × Det-F1  +  0.05 × ClassAcc  +  0.30 × (1 − CER)  +  0.50 × (1 − PageCER)
```

| Component | Trọng số | Ghi chú |
|---|---|---|
| Detection F1 | 15% | IoU ≥ 0.5, type-agnostic |
| Classification Accuracy | 5% | Đúng `type` trong matched pair |
| Region CER | 30% | CER trung bình mỗi region |
| **Page CER** | **50%** | **Toàn trang, top-to-bottom → left-to-right** |

**Insight chiến lược:**
- PageCER chiếm **50%** → thứ tự các region trên trang và chất lượng transcription toàn cục là quyết định
- Detection F1 chỉ 15% → không cần detector hoàn hảo, nhưng phải đủ tốt để CER được tính
- Text normalization đã xử lý nhiều biến thể (em-dash, lookalikes, LaTeX↔Unicode) → pipeline không cần lo về stylistic variants

### 0.2 Chiến lược tổng

1. **End-to-end với Qwen3-VL 8B** — một model duy nhất làm cả detection + classification + OCR
2. **Curriculum training**: Silver (noisy) → Gold (clean)
3. **Uncertainty-Aware CoT** — điểm nhấn học thuật: khi model phân vân, tự kích hoạt chuỗi lý luận
4. **Post-processing mạnh** — LLM spell-check tiếng Ukraine + sắp xếp tọa độ tối ưu PageCER

---

## 1. Thiết lập Môi trường

### 1.1 Dependencies

```bash
# Môi trường Python
conda create -n rukopys python=3.11 -y
conda activate rukopys

# Core
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.0 accelerate peft bitsandbytes
pip install qwen-vl-utils

# Training
pip install trl deepspeed flash-attn --no-build-isolation
pip install liger-kernel  # memory-efficient kernels

# Data
pip install datasets huggingface_hub pillow opencv-python
pip install textrecognitiondatagenerator  # synthetic data

# Evaluation
pip install editdistance jiwer
```

### 1.2 Cấu hình A6000 48GB

```python
# LoRA config tối ưu cho 8B model trên 48GB VRAM
LORA_CONFIG = {
    "r": 64,                          # rank — cao hơn để bắt visual features
    "lora_alpha": 128,
    "target_modules": [               # cả vision và language components
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # effective batch = 32
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "max_seq_length": 4096,
}
```

---

## 2. Dữ liệu (Data Pipeline)

### 2.1 Tải và khám phá dữ liệu

```python
from datasets import load_dataset

# Gold annotations (1,330 ảnh, 25,523 regions)
train_ds = load_dataset(
    "UkrainianCatholicUniversity/rukopys",
    name="gt_only",
    split="train"
)

# Silver annotations (8,207 ảnh, 161,065 regions) — auto by Qwen3+Gemini
silver_ds = load_dataset(
    "UkrainianCatholicUniversity/rukopys",
    name="full",
    split="silver"
)

# Test (386 ảnh, no annotations)
test_ds = load_dataset(
    "UkrainianCatholicUniversity/rukopys",
    name="test",
    split="test"
)
```

**Phân bố dữ liệu Gold:**

| Source | Images | Đặc điểm |
|---|---|---|
| dictation | 359 | Phone camera, nhiều handwriting styles |
| school | 682 | Phone camera, grades 5-11, đa dạng môn |
| university | 162 | Scanner, math/chemistry/tables |
| archive | 127 | Scanner, 1919-1935, chữ viết cổ |

### 2.2 Quality Filtering Silver Data

Silver data được tạo tự động bởi AI nên cần lọc kỹ:

```python
def filter_silver_quality(example):
    regions = example["regions"]
    if not regions:
        return False
    
    filters = []
    for r in regions:
        text = r.get("text", "")
        
        # Loại bỏ text quá ngắn hoặc rỗng
        if len(text.strip()) < 2:
            continue
        
        # Loại bỏ hallucination patterns (aaaa..., repeated n-grams)
        if has_repetition_loop(text):
            continue
        
        # Loại bỏ bbox degenerate
        bbox = r["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 10 or h < 8 or w / h > 50 or h / w > 20:
            continue
        
        filters.append(r)
    
    return len(filters) >= 1

def has_repetition_loop(text, threshold=5):
    """Phát hiện pattern lặp lại (hallucination)."""
    for n in range(1, 6):
        for i in range(len(text) - n * threshold):
            substr = text[i:i+n]
            if text[i:i+n*threshold] == substr * threshold:
                return True
    return False

silver_filtered = silver_ds.filter(filter_silver_quality)
# Kỳ vọng giữ lại ~85-90% silver data
```

### 2.3 Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation cho handwritten documents (không quá aggressive)
train_transform = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.CLAHE(clip_limit=2.0),
    ], p=0.5),
    
    A.OneOf([
        A.GaussNoise(var_limit=(5, 25)),
        A.ISONoise(color_shift=(0.01, 0.03)),
    ], p=0.3),
    
    # Simulating phone camera artifacts
    A.OneOf([
        A.Blur(blur_limit=3),
        A.MotionBlur(blur_limit=5),
    ], p=0.2),
    
    # Perspective distortion (phone photos thường bị nghiêng)
    A.Perspective(scale=(0.02, 0.06), p=0.3),
    
    # Simulating different lighting conditions
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.2),
    
    # Random rotation nhỏ (chữ viết tay thường không thẳng hoàn toàn)
    A.Rotate(limit=3, p=0.3),
])
```

### 2.4 Synthetic Data Generation

Dùng **TextRecognitionDataGenerator (TRDG)** để sinh thêm dữ liệu tiếng Ukraine:

```bash
# Cài font tiếng Ukraine
# Download từ Google Fonts: các font có Ukrainian support

# Generate synthetic handwriting-style data
trdg -l uk \                          # Ukrainian language
     -c 10000 \                       # 10,000 samples
     -f 32 \                          # font size 32
     -t 8 \                           # 8 threads
     --distorsion 2 \                 # elastic distortion
     --random_distorsion \
     --handwriting_style \            # handwriting fonts
     --output_dir ./synthetic_data/
```

Kết hợp với corpus văn bản Ukraine phổ biến:
- **UberText** (300M từ Ukrainian web text)
- **Kobza** (thi ca, văn xuôi Ukraine)
- Dictation texts từ các năm trước (publicly available)

### 2.5 CoT Dataset Generation (Paper Contribution)

Đây là điểm mấu chốt để bài paper được accept. Cần tạo dataset **Chain-of-Thought reasoning cho OCR**:

```python
import anthropic  # hoặc OpenAI

def generate_cot_annotation(image_path, ground_truth_text, region_type):
    """
    Dùng GPT-4o / Gemini 2.5 Pro để tạo lời giải thích CoT
    cho các ảnh chữ viết tay khó.
    """
    
    prompt = f"""You are an expert handwriting analyst. 
    
Look at this handwritten text region (type: {region_type}).
The ground truth text is: "{ground_truth_text}"

Please analyze the handwriting stroke by stroke and explain:
1. Which characters were ambiguous and why (visual similarity, stroke quality)
2. What contextual clues helped resolve ambiguity
3. Your step-by-step reasoning process

Format your response as:
<ambiguous_chars>...</ambiguous_chars>
<visual_analysis>...</visual_analysis>
<context_clues>...</context_clues>
<reasoning>...</reasoning>
<conclusion>{ground_truth_text}</conclusion>"""
    
    # Call API với ảnh + prompt
    response = call_vision_api(image_path, prompt)
    return parse_cot_response(response)

# Target: 5,000 CoT samples từ gold train data
# Ưu tiên: regions có legibility = "illegible" hoặc "partially_legible"
```

**Output format mỗi CoT sample:**
```json
{
  "image_id": "uuid_xxx",
  "region_bbox": [x1, y1, x2, y2],
  "region_type": "handwritten",
  "ground_truth": "Сьогодні гарна погода",
  "cot_reasoning": "<ambiguous_chars>С, г</ambiguous_chars>...",
  "uncertainty_words": ["Сьогодні", "погода"],
  "uncertainty_scores": [0.73, 0.45]
}
```

---

## 3. Kiến trúc Model (Pipeline 4 Stage)

### Stage 1: Region Detection & Classification

**Input:** Ảnh tài liệu gốc (nguyên kích thước)  
**Output:** Danh sách `{bbox, type}` cho mỗi region

```python
DETECTION_PROMPT = """Analyze this document image and detect all text regions.
For each region, output a JSON object with:
- bbox: [x1, y1, x2, y2] in pixels
- type: one of [handwritten, printed, formula, table, annotation, image, graph]

Return a JSON array. Be thorough — detect every region including margins, 
annotations, and small text areas. Sort by top-to-bottom, left-to-right order."""

def detect_regions(image, model, processor):
    inputs = processor(
        text=DETECTION_PROMPT,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,  # low temp for detection consistency
        )
    
    regions = parse_json_output(processor.decode(output[0]))
    return regions
```

**Lưu ý quan trọng:** Qwen3-VL 8B đã được pretrain tốt cho detection. Fine-tune trên RUKOPYS train data là đủ.

### Stage 2: Per-Region OCR + Logit Capture

**Input:** Ảnh crop của từng region + context về `type`  
**Output:** Text transcription + **raw logits** từng token

```python
def transcribe_region_with_logits(image_crop, region_type, model, processor):
    """
    Transcribe một region VÀ capture logits để tính uncertainty.
    """
    
    type_hints = {
        "handwritten": "Transcribe this handwritten Ukrainian text exactly.",
        "printed": "Transcribe this printed text exactly.",
        "formula": "Transcribe this mathematical formula using LaTeX or Unicode notation.",
        "table": "Extract table content, separating cells with |",
        "annotation": "Transcribe this annotation/note.",
    }
    
    prompt = f"""Region type: {region_type}
{type_hints.get(region_type, 'Transcribe this text.')}
Output ONLY the transcribed text, nothing else."""
    
    inputs = processor(
        text=prompt,
        images=image_crop,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate với output_scores=True để lấy logits
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            output_scores=True,        # KEY: lấy logits
            return_dict_in_generate=True,
            temperature=0.3,
        )
    
    # Lấy text và scores
    generated_ids = output.sequences[0][inputs.input_ids.shape[1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    scores = output.scores  # list of (vocab_size,) tensors per token
    
    return text, scores
```

### Stage 3: Token Entropy & Uncertainty Detection

```python
import torch
import torch.nn.functional as F

def compute_token_entropy(logits_per_token: list[torch.Tensor]) -> list[float]:
    """
    Tính entropy cho từng token đã generate.
    H(X) = -Σ p(xᵢ) log p(xᵢ)
    """
    entropies = []
    for logits in logits_per_token:
        probs = F.softmax(logits, dim=-1)
        # Entropy cao = model phân vân nhiều
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        entropies.append(entropy)
    return entropies

def detect_uncertain_segments(
    text: str,
    entropies: list[float],
    threshold: float = 3.5,      # τ — tune trên validation set
    window: int = 3,             # số token liên tiếp để gộp thành 1 segment
) -> list[dict]:
    """
    Phát hiện các đoạn text không chắc chắn.
    Trả về list các {text_segment, position, avg_entropy}
    """
    tokens = text.split()  # simplified; dùng tokenizer thực tế
    uncertain_segments = []
    
    i = 0
    while i < len(entropies):
        if entropies[i] > threshold:
            # Gộp các token lân cận cũng uncertain
            j = i
            while j < len(entropies) and entropies[j] > threshold:
                j += 1
            
            segment_entropy = sum(entropies[i:j]) / (j - i)
            uncertain_segments.append({
                "start_token": i,
                "end_token": j,
                "avg_entropy": segment_entropy,
                "text": " ".join(tokens[i:j]) if i < len(tokens) else "",
            })
            i = j
        else:
            i += 1
    
    return uncertain_segments
```

### Stage 4: Multimodal Chain-of-Thought Fallback

Kích hoạt khi phát hiện uncertain segment:

```python
def cot_fallback(
    full_image,
    uncertain_bbox,       # tọa độ trong ảnh gốc
    preceding_context,    # text trước đó đã nhận dạng
    following_context,    # text sau (nếu có từ pass trước)
    region_type,
    model,
    processor,
    zoom_factor=2.0,
):
    """
    Chain-of-Thought reasoning cho vùng uncertain.
    
    Bước 1: Zoom vào vùng mờ
    Bước 2: Inject context
    Bước 3: Yêu cầu model suy luận từng bước trước khi output
    """
    
    # Zoom crop
    x1, y1, x2, y2 = uncertain_bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    w, h = x2-x1, y2-y1
    zoomed_bbox = [
        max(0, cx - int(w*zoom_factor/2)),
        max(0, cy - int(h*zoom_factor/2)),
        min(full_image.width, cx + int(w*zoom_factor/2)),
        min(full_image.height, cy + int(h*zoom_factor/2)),
    ]
    zoomed_crop = full_image.crop(zoomed_bbox)
    
    # CoT prompt
    cot_prompt = f"""You are analyzing an ambiguous handwritten {region_type} region.

Context BEFORE this region: "{preceding_context[-100:]}"
Context AFTER this region: "{following_context[:100]}"

This word/phrase appears ambiguous. Please:
1. Describe the visual stroke characteristics you observe
2. Consider what characters are visually similar in Ukrainian Cyrillic cursive
3. Use the surrounding context (semantic and syntactic) to resolve ambiguity
4. State your final answer

Think step by step before giving your final transcription.
End with: FINAL: <your transcription>"""
    
    inputs = processor(
        text=cot_prompt,
        images=zoomed_crop,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.5,   # slightly higher temp for reasoning
        )
    
    cot_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract final answer
    if "FINAL:" in cot_text:
        final_answer = cot_text.split("FINAL:")[-1].strip()
    else:
        final_answer = cot_text.split("\n")[-1].strip()
    
    # Log reasoning cho paper output
    reasoning_log = {
        "uncertain_region": uncertain_bbox,
        "cot_reasoning": cot_text,
        "resolved_text": final_answer,
        "method": "zoom_crop + context + cot"
    }
    
    return final_answer, reasoning_log
```

### Stage 5: Post-processing (Critical for PageCER)

```python
import re

def normalize_output_text(text: str, region_type: str) -> str:
    """
    Post-process text để khớp với evaluation normalization.
    Áp dụng cùng rules như kaggle_metric.py
    """
    
    # 1. Cyrillic/Latin lookalike unification
    lookalikes = {
        'c': 'с', 'o': 'о', 'p': 'р', 'a': 'а', 'e': 'е',
        'x': 'х', 'i': 'і', 'A': 'А', 'B': 'В', 'C': 'С',
        'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О',
        'P': 'Р', 'T': 'Т', 'X': 'Х',
    }
    result = ""
    for ch in text:
        result += lookalikes.get(ch, ch)
    text = result
    
    # 2. Dash unification
    text = re.sub(r'[–—−]', '-', text)
    
    # 3. Quote normalization
    text = re.sub(r'[«»""'']', '"', text)
    
    # 4. Whitespace collapse
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Strikethrough removal
    text = re.sub(r'~~[^~]*~~\{([^}]*)\}', r'\1', text)  # ~~old~~{new} → new
    text = re.sub(r'~~([^~]*)~~', r'\1', text)             # ~~text~~ → text
    
    if region_type in ["formula", "table"]:
        text = normalize_formula(text)
    
    return text


def sort_regions_for_page_cer(regions: list[dict]) -> list[dict]:
    """
    Sắp xếp regions theo thứ tự đọc: top-to-bottom, left-to-right.
    CRITICAL: PageCER tính theo thứ tự này — đây là 50% tổng điểm!
    
    Dùng column-aware sorting: phát hiện multi-column layout.
    """
    
    if not regions:
        return regions
    
    # Compute median bbox height để determine row grouping threshold
    heights = [r["bbox"][3] - r["bbox"][1] for r in regions]
    median_h = sorted(heights)[len(heights)//2]
    row_threshold = median_h * 0.7
    
    # Group regions into "rows" (overlapping y ranges)
    regions_sorted_by_y = sorted(regions, key=lambda r: r["bbox"][1])
    
    rows = []
    current_row = [regions_sorted_by_y[0]]
    current_row_y = regions_sorted_by_y[0]["bbox"][1]
    
    for region in regions_sorted_by_y[1:]:
        if region["bbox"][1] - current_row_y < row_threshold:
            current_row.append(region)
        else:
            rows.append(sorted(current_row, key=lambda r: r["bbox"][0]))
            current_row = [region]
            current_row_y = region["bbox"][1]
    rows.append(sorted(current_row, key=lambda r: r["bbox"][0]))
    
    return [r for row in rows for r in row]


def llm_spell_check(text: str, region_type: str) -> str:
    """
    Optional: dùng Ukrainian language model để sửa lỗi chính tả.
    Recommended: ua-gpt-neox hoặc BERT-based Ukrainian model.
    Chỉ áp dụng cho handwritten/printed, không áp dụng formula.
    """
    if region_type in ["formula", "table", "image", "graph"]:
        return text
    
    # Implement với Ukrainian BERT/GPT spell checker
    # Ví dụ: Hugging Face "benjamin/roberta-base-wechsel-ukrainian"
    corrected = ukrainian_spell_check(text)
    return corrected
```

---

## 4. Chiến lược Training

### 4.1 Phase 1: Pre-training trên Silver Data (Curriculum)

**Mục tiêu:** Làm model quen với đặc điểm chữ viết tay Ukraine trước khi học từ gold data.

```python
# Prompt template cho training
def format_training_sample(image, regions, task="full"):
    if task == "detection":
        # Task 1: Detect + classify all regions
        prompt = """Detect all text regions in this document. 
Output JSON array with bbox and type for each region."""
        target = json.dumps([{
            "bbox": r["bbox"],
            "type": r["type"]
        } for r in regions])
    
    elif task == "transcription":
        # Task 2: Transcribe given a single region crop
        r = random.choice(regions)
        image = image.crop(r["bbox"])
        prompt = f"Transcribe this {r['type']} text region exactly."
        target = r.get("text", "")
    
    elif task == "full":
        # Task 3: Full page → structured output (for PageCER optimization)
        prompt = """Analyze this document. For each text region:
1. Identify its bounding box
2. Classify its type  
3. Transcribe its content
Output as JSON array sorted top-to-bottom, left-to-right."""
        target = json.dumps([{
            "bbox": r["bbox"],
            "type": r["type"],
            "text": r.get("text", "")
        } for r in sort_regions_for_page_cer(regions)])
    
    return {"image": image, "prompt": prompt, "target": target}
```

**Training silver:**
```bash
python train.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --dataset silver_filtered \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --output_dir ./checkpoints/silver_phase \
    --use_lora True \
    --task_mix "detection:0.3,transcription:0.4,full:0.3"
```

### 4.2 Phase 2: Fine-tuning trên Gold Data

```bash
python train.py \
    --model_name ./checkpoints/silver_phase \  # resume từ silver
    --dataset gold_train \
    --num_epochs 5 \
    --learning_rate 5e-5 \             # lower LR cho fine-tuning
    --output_dir ./checkpoints/gold_phase \
    --use_lora True \
    --task_mix "full:0.6,transcription:0.3,cot:0.1" \
    --use_source_weights True \        # professional annotations > volunteer
    --scheduler cosine_with_restarts
```

### 4.3 Phase 3: CoT Fine-tuning (Paper Contribution)

```bash
python train.py \
    --model_name ./checkpoints/gold_phase \
    --dataset cot_dataset \            # 5,000 CoT samples từ bước 2.5
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --output_dir ./checkpoints/cot_phase \
    --loss_weights "ocr:0.7,reasoning:0.3"  # multi-task loss
```

### 4.4 Hyperparameter Cheat Sheet (A6000 48GB)

| Parameter | Silver Phase | Gold Phase | CoT Phase |
|---|---|---|---|
| LoRA rank | 32 | 64 | 64 |
| Batch size | 4 | 2 | 2 |
| Grad accum | 8 | 16 | 16 |
| LR | 2e-4 | 5e-5 | 1e-5 |
| Epochs | 2 | 5 | 3 |
| Max length | 2048 | 4096 | 4096 |
| VRAM usage | ~38GB | ~44GB | ~44GB |

---

## 5. Inference & Submission

### 5.1 Full Inference Pipeline

```python
class RUKOPYSPipeline:
    def __init__(self, model_path, threshold=3.5):
        self.model = load_model(model_path)
        self.processor = load_processor(model_path)
        self.entropy_threshold = threshold
        self.reasoning_logs = []
    
    def predict_image(self, image_path):
        image = Image.open(image_path)
        
        # Stage 1: Detect regions
        regions = detect_regions(image, self.model, self.processor)
        
        results = []
        for region in regions:
            # Stage 2: OCR với logit capture
            text, scores = transcribe_region_with_logits(
                image.crop(region["bbox"]),
                region["type"],
                self.model,
                self.processor
            )
            
            # Stage 3: Compute entropy
            entropies = compute_token_entropy(scores)
            uncertain_segs = detect_uncertain_segments(
                text, entropies, self.entropy_threshold
            )
            
            # Stage 4: CoT fallback nếu cần
            if uncertain_segs:
                for seg in uncertain_segs:
                    correction, log = cot_fallback(
                        image,
                        region["bbox"],
                        preceding_context=text[:seg["start_token"]],
                        following_context=text[seg["end_token"]:],
                        region_type=region["type"],
                        model=self.model,
                        processor=self.processor,
                    )
                    text = apply_correction(text, seg, correction)
                    self.reasoning_logs.append(log)
            
            # Stage 5: Post-processing
            text = normalize_output_text(text, region["type"])
            
            results.append({
                "bbox": region["bbox"],
                "type": region["type"],
                "text": text,
                "uncertainty_score": max(entropies) if entropies else 0.0,
            })
        
        # Sắp xếp theo thứ tự đọc (critical for PageCER!)
        results = sort_regions_for_page_cer(results)
        
        return results
    
    def generate_submission(self, test_dataset, output_path):
        rows = []
        for example in tqdm(test_dataset):
            image_path = example["image_path"]
            image_name = Path(image_path).name
            
            regions = self.predict_image(image_path)
            
            # Format submission
            regions_json = json.dumps([{
                "bbox": r["bbox"],
                "type": r["type"],
                "text": r["text"],
            } for r in regions])
            
            rows.append({
                "image": image_name,
                "regions": regions_json,
            })
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
```

### 5.2 Ensemble (Optional — nếu có thêm thời gian)

```python
def ensemble_predictions(predictions_list, strategy="voting"):
    """
    Ensemble nhiều model predictions.
    Dùng editdistance để vote consensus text.
    """
    # Chạy 2-3 checkpoints khác nhau, merge predictions
    pass
```

---

## 6. Đánh giá & Ablation

### 6.1 Local Evaluation

```python
# Dùng official kaggle_metric.py
from kaggle_metric import score_detailed

detailed = score_detailed(solution_df, submission_df)
print(f"Det-F1: {detailed['detection_f1']:.4f}")
print(f"ClassAcc: {detailed['classification_accuracy']:.4f}")
print(f"CER: {detailed['region_cer']:.4f}")
print(f"PageCER: {detailed['page_cer']:.4f}")
print(f"TOTAL SCORE: {detailed['score']:.4f}")
```

### 6.2 Ablation Study (cho Paper)

| Variant | Det-F1 | CER | PageCER | Score |
|---|---|---|---|---|
| Baseline (no fine-tune) | ? | ? | ? | ? |
| + Silver pre-training | ? | ? | ? | ? |
| + Gold fine-tuning | ? | ? | ? | ? |
| + Uncertainty detection | ? | ? | ? | ? |
| + CoT fallback | ? | ? | ? | ? |
| + Post-processing | ? | ? | ? | ? |
| **Full pipeline** | ? | ? | ? | ? |

### 6.3 Error Analysis

- Phân loại lỗi theo `source` (archive vs school vs dictation vs university)
- Phân loại lỗi theo `legibility` (legible / partially_legible / illegible)
- Tìm common character confusion pairs trong Ukrainian Cyrillic cursive
- Visualize entropy heatmap trên các ảnh khó

---

## 7. Đóng góp cho Paper

### 7.1 Tên phương pháp

**"UA-STAR: Uncertainty-guided Stepwise Transcription with Adaptive Reasoning for Ukrainian Handwritten Documents"**

### 7.2 Mạch truyện (Storyline)

**Problem:** Chữ viết tay Ukraine đặc biệt khó nhận dạng do: (1) đa dạng nguồn gốc từ 1920s đến 2025, (2) ambiguous cursive Cyrillic strokes, (3) không có large-scale dataset trước RUKOPYS.

**Gap:** Các VLM hiện tại xử lý OCR như một "black box" — không biết khi nào nên tin tưởng output của mình, dẫn đến confident errors trên vùng chữ mờ.

**Contribution:**
1. **Uncertainty-Guided CoT**: Cơ chế entropy-based phát hiện vùng không chắc chắn và tự động kích hoạt CoT reasoning — *dừng lại và suy nghĩ như con người*
2. **CoT-OCR Dataset**: 5,000 samples với lời giải thích stroke-level reasoning (có thể release public để benefit community)
3. **Multi-objective Training**: Loss kết hợp OCR accuracy + reasoning quality
4. **Curriculum Learning Strategy**: Silver → Gold giúp domain adaptation hiệu quả

### 7.3 Figures quan trọng cho Paper

- Figure 1: Pipeline overview
- Figure 2: Entropy heatmap trên ảnh chữ viết tay — visualization của uncertain regions
- Figure 3: CoT reasoning example — model "suy nghĩ" step-by-step để giải ambiguity
- Figure 4: Ablation table
- Figure 5: Error analysis by source/legibility

---

## 8. Timeline (6 tuần đến deadline 15/06/2026)

| Tuần | Task | Deliverable |
|---|---|---|
| **Tuần 1** | Setup env, load data, baseline với zero-shot Qwen3-VL | Submission baseline score |
| **Tuần 2** | Lọc silver data, format training data, bắt đầu silver pre-training | Silver model checkpoint |
| **Tuần 3** | Gold fine-tuning, implement post-processing, evaluation pipeline | Improved score |
| **Tuần 4** | Implement entropy + CoT pipeline, generate CoT dataset (GPT-4o) | CoT-enhanced model |
| **Tuần 5** | CoT fine-tuning, ensemble experiments, ablation study | Near-final model |
| **Tuần 6** | Final submission, paper draft, polishing | Submission + Paper draft |

---

## 9. Quick Start Checklist

- [ ] Clone repo, setup conda env
- [ ] Tải RUKOPYS dataset từ HuggingFace
- [ ] Chạy zero-shot baseline với Qwen3-VL 8B (no fine-tuning) → có baseline score
- [ ] Chạy silver quality filtering → `silver_filtered` dataset
- [ ] Format training data theo prompt templates
- [ ] Bắt đầu Phase 1 training (silver)
- [ ] Implement local evaluation với `kaggle_metric.py`
- [ ] Submit đầu tiên lên Kaggle trước tuần 2

---
