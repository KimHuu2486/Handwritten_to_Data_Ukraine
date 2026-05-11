"""
RUKOPYS — LLM Spell-Check Post-Processing
==========================================

Luồng hoạt động:
  1. Đọc file submission CSV (output của kaggle_finetuned_inference.py)
  2. Load model LLM text-only nhỏ (Qwen2.5-1.5B-Instruct)
  3. Với mỗi ảnh, gom text từ tất cả regions → gửi LLM sửa chính tả
  4. Áp text đã sửa ngược lại vào regions
  5. Ghi ra file submission mới đã được spell-check

Cách sử dụng trên Kaggle:
  - Chạy SAU khi inference xong (kaggle_finetuned_inference.py đã tạo submission.csv)
  - Thêm Kaggle Model "Qwen/Qwen2.5-1.5B-Instruct" vào Input
  - Chạy: python spell_check_submission.py

Cách sử dụng local (test thử):
  - python spell_check_submission.py --input submission.csv --output submission_fixed.csv --local

Yêu cầu:
  - transformers, torch, pandas
  - GPU với >=2GB VRAM trống (hoặc dùng CPU với --cpu flag)
"""

import os
import json
import csv
import re
import time
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# CẤU HÌNH
# ==========================================

# Đường dẫn model trên Kaggle (mount từ Kaggle Models)
KAGGLE_MODEL_PATHS = [
    "/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1",
]

# Đường dẫn model khi chạy local (tải từ HuggingFace)
LOCAL_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Đường dẫn file mặc định trên Kaggle
DEFAULT_INPUT_CSV = "/kaggle/input/datasets/trankimhuu/submission-raw/submission (1).csv"
DEFAULT_OUTPUT_CSV = "/kaggle/working/submission.csv"  # Ghi đè luôn để Kaggle submit

# Prompt cho spell-check — Viết bằng tiếng Ukraina để model không tự ý dịch
SPELL_CHECK_PROMPT = """\
Ти — коректор українського тексту, отриманого з OCR рукописних документів.

Правила:
1. Виправляй ТІЛЬКИ очевидні помилки розпізнавання символів (неправильні, пропущені або зайві літери)
2. НЕ змінюй порядок слів, значення, структуру речень або стиль пунктуації
3. Зберігай переноси рядків ТОЧНО як вони є — однакова кількість рядків на вході та виході
4. Якщо слово виглядає правильним або ти не впевнений — НЕ змінюй його
5. НЕ додавай жодних пояснень — виводь ТІЛЬКИ виправлений текст
6. Відповідай ТІЛЬКИ українською мовою. ЗАБОРОНЕНО перекладати на англійську, російську чи будь-яку іншу мову
7. ЗАБОРОНЕНО використовувати латинські літери (a-z, A-Z). Весь текст має бути кирилицею

Текст OCR для виправлення:
{text}"""

# Giới hạn an toàn: nếu text thay đổi quá nhiều, giữ nguyên bản gốc
MAX_LENGTH_CHANGE_RATIO = 0.35  # 35%


# ==========================================
# HÀM CHÍNH
# ==========================================

def find_model_path():
    """Tìm đường dẫn model trên Kaggle."""
    for path in KAGGLE_MODEL_PATHS:
        if os.path.exists(path):
            print(f"✅ Tìm thấy model tại: {path}", flush=True)
            return path
    
    # Fallback: quét /kaggle/input/ tìm config.json của Qwen2.5
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files:
                config_path = os.path.join(root, "config.json")
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    if "Qwen2" in cfg.get("architectures", [""])[0]:
                        print(f"✅ Tìm thấy Qwen2 model tại: {root}", flush=True)
                        return root
                except Exception:
                    pass
    
    return None


def load_spell_check_model(model_path, device="cuda:0", use_cpu=False):
    """Load model spell-check."""
    if use_cpu:
        device = "cpu"
        dtype = torch.float32
    else:
        dtype = torch.float16
    
    print(f"📦 Đang load spell-check model từ {model_path}...", flush=True)
    print(f"   Device: {device} | Dtype: {dtype}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if not use_cpu else "cpu",
        trust_remote_code=True
    )
    model.eval()
    
    # Kiểm tra VRAM
    if torch.cuda.is_available() and not use_cpu:
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   VRAM sử dụng: {allocated:.1f} GB", flush=True)
    
    print("✅ Model spell-check đã sẵn sàng!", flush=True)
    return model, tokenizer


def spell_check_text(text, model, tokenizer, max_new_tokens=2048):
    """
    Sửa lỗi chính tả cho đoạn text OCR.
    
    Args:
        text: Đoạn text OCR cần sửa
        model: LLM model
        tokenizer: Tokenizer tương ứng
        max_new_tokens: Số token tối đa cho output
    
    Returns:
        Text đã sửa chính tả
    """
    if not text or len(text.strip()) < 3:
        return text
    
    prompt = SPELL_CHECK_PROMPT.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
        )
    
    # Decode chỉ phần generated (bỏ phần prompt)
    result = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return result.strip()


def safe_apply_correction(original_text, corrected_text):
    """
    Kiểm tra an toàn trước khi áp dụng correction.
    Reject nếu thay đổi quá lớn hoặc bị dịch sang ngôn ngữ khác.
    
    Returns:
        Tuple (text_cuối_cùng, đã_sửa: bool)
    """
    if not corrected_text:
        return original_text, False
    
    orig_len = len(original_text)
    corr_len = len(corrected_text)
    
    # Reject nếu độ dài thay đổi quá ngưỡng cho phép
    if orig_len > 0:
        change_ratio = abs(orig_len - corr_len) / orig_len
        if change_ratio > MAX_LENGTH_CHANGE_RATIO:
            return original_text, False
    
    # Reject nếu corrected text rỗng nhưng original không rỗng
    if orig_len > 5 and corr_len < 3:
        return original_text, False
    
    # === BỘ LỌC CHỐNG DỊCH/TRANSLITERATION ===
    # Đếm số ký tự Latin (a-z, A-Z) trong output
    latin_chars = len(re.findall(r'[a-zA-Z]', corrected_text))
    # Đếm số ký tự Latin trong input gốc (để so sánh)
    orig_latin = len(re.findall(r'[a-zA-Z]', original_text))
    
    # Nếu output có NHIỀU ký tự Latin hơn input → model đã dịch/transliterate
    if latin_chars > orig_latin + 5:  # Cho phép sai lệch tối đa 5 ký tự
        return original_text, False
    
    # Nếu output có >10% ký tự Latin (trừ khi input cũng có) → reject
    alpha_chars = len(re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄґҐ]', corrected_text))
    if alpha_chars > 0 and latin_chars / alpha_chars > 0.10:
        if orig_latin / max(len(re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄґҐ]', original_text)), 1) < 0.10:
            return original_text, False
    
    return corrected_text, True


def process_regions(regions_json_str, model, tokenizer):
    """
    Spell-check toàn bộ regions của 1 ảnh.
    
    Chiến lược:
      1. Gom tất cả text thành 1 block (giữ dấu xuống dòng)
      2. Spell-check cả block cùng lúc (LLM có ngữ cảnh tốt hơn)
      3. Tách lại theo dòng và áp vào từng region
      4. Nếu số dòng không khớp → fallback spell-check từng region riêng
    
    Returns:
        Tuple (regions_json_str đã sửa, số regions đã sửa)
    """
    try:
        regions = json.loads(regions_json_str)
    except (json.JSONDecodeError, TypeError):
        return regions_json_str, 0
    
    if not regions:
        return regions_json_str, 0
    
    # Lọc các region có text đáng để sửa
    texts = [r.get("text", "") for r in regions]
    combined_text = "\n".join(texts)
    
    # Bỏ qua nếu quá ngắn (không đáng spell-check)
    if len(combined_text.strip()) < 10:
        return regions_json_str, 0
    
    # === Chiến lược 1: Spell-check cả block ===
    corrected_block = spell_check_text(combined_text, model, tokenizer)
    corrected_lines = corrected_block.split("\n")
    
    fixed_count = 0
    
    if len(corrected_lines) == len(texts):
        # Số dòng khớp → áp từng dòng
        for i, region in enumerate(regions):
            corrected_line, was_fixed = safe_apply_correction(
                texts[i], corrected_lines[i]
            )
            if was_fixed and corrected_line != texts[i]:
                region["text"] = corrected_line
                fixed_count += 1
    else:
        # === Fallback: Spell-check từng region riêng lẻ ===
        for i, region in enumerate(regions):
            original = texts[i]
            if len(original.strip()) < 5:
                continue
            
            corrected = spell_check_text(original, model, tokenizer)
            corrected, was_fixed = safe_apply_correction(original, corrected)
            if was_fixed and corrected != original:
                region["text"] = corrected
                fixed_count += 1
    
    return json.dumps(regions, ensure_ascii=False), fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="RUKOPYS — LLM Spell-Check Post-Processing"
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_CSV,
        help="Đường dẫn file submission CSV đầu vào"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_CSV,
        help="Đường dẫn file submission CSV đầu ra"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Đường dẫn model LLM (nếu không set, tự động tìm trên Kaggle)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Chạy ở chế độ local (tải model từ HuggingFace)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Chạy trên CPU (chậm hơn nhưng không cần GPU)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="GPU device (mặc định: cuda:0)"
    )
    args, _ = parser.parse_known_args()
    
    print("=" * 60, flush=True)
    print("🔤 RUKOPYS — LLM Spell-Check Post-Processing", flush=True)
    print("=" * 60, flush=True)
    
    # === 1. Xác định đường dẫn model ===
    if args.model:
        model_path = args.model
    elif args.local:
        model_path = LOCAL_MODEL_ID
        print(f"📥 Chế độ local: sẽ tải model từ HuggingFace ({model_path})", flush=True)
    else:
        model_path = find_model_path()
        if not model_path:
            print("❌ Không tìm thấy model spell-check trên Kaggle!", flush=True)
            print("   Hãy thêm 'Qwen/Qwen2.5-1.5B-Instruct' vào Kaggle Input.", flush=True)
            print("   Hoặc chạy với --local để tải từ HuggingFace.", flush=True)
            sys.exit(1)
    
    # === 2. Load model ===
    model, tokenizer = load_spell_check_model(
        model_path, device=args.device, use_cpu=args.cpu
    )
    
    # === 3. Đọc submission CSV ===
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file: {input_path}", flush=True)
        sys.exit(1)
    
    print(f"\n📄 Đang đọc: {input_path}", flush=True)
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"   Tổng số ảnh: {len(rows)}", flush=True)
    
    # === 4. Spell-check từng ảnh ===
    print(f"\n🔄 Bắt đầu spell-check...\n", flush=True)
    
    total_fixed = 0
    total_regions = 0
    start_time = time.time()
    
    for idx, row in enumerate(rows):
        image_name = row["image"]
        regions_str = row.get("regions", "[]")
        
        # Đếm regions
        try:
            num_regions = len(json.loads(regions_str))
        except Exception:
            num_regions = 0
        total_regions += num_regions
        
        if num_regions == 0:
            continue
        
        # Spell-check
        corrected_str, fixed_count = process_regions(
            regions_str, model, tokenizer
        )
        row["regions"] = corrected_str
        total_fixed += fixed_count
        
        # Progress log (mỗi 10 ảnh hoặc ảnh cuối)
        if (idx + 1) % 10 == 0 or idx == len(rows) - 1:
            elapsed = time.time() - start_time
            speed = elapsed / (idx + 1)
            eta = speed * (len(rows) - idx - 1)
            print(
                f"  [{idx+1}/{len(rows)}] "
                f"Đã sửa: {total_fixed} regions | "
                f"Speed: {speed:.1f}s/ảnh | "
                f"ETA: {time.strftime('%M:%S', time.gmtime(eta))}",
                flush=True
            )
    
    # === 5. Ghi ra file output ===
    output_path = args.output
    
    # Ghi file tạm trước, rename sau (atomic write)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "regions"])
        writer.writeheader()
        writer.writerows(rows)
    
    os.replace(tmp_path, output_path)
    
    # === 6. Thống kê ===
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}", flush=True)
    print(f"✅ HOÀN THÀNH SPELL-CHECK!", flush=True)
    print(f"   Tổng ảnh xử lý:    {len(rows)}", flush=True)
    print(f"   Tổng regions:       {total_regions}", flush=True)
    print(f"   Regions đã sửa:    {total_fixed}", flush=True)
    print(f"   Thời gian:          {time.strftime('%H:%M:%S', time.gmtime(elapsed))}", flush=True)
    print(f"   Output:             {output_path}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()