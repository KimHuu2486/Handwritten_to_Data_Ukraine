# ==============================================================================
# Fine-tuned Inference - RUKOPYS Kaggle Competition
# Model: Qwen 3 VL (8b-instruct) + LoRA Adapter
# Method: Multi-processing on multiple GPUs
#
# Submission format (from official evaluation notebook):
#   CSV with columns: image, regions
#   regions = JSON list of {"bbox": [x1,y1,x2,y2], "type": "...", "text": "..."}
#   bbox = absolute pixel coordinates, top-left origin
#
# HƯỚNG DẪN CHẠY TRÊN KAGGLE:
#   1. Tạo notebook mới trên Kaggle
#   2. Add Input:
#      - Model: qwen-lm/qwen-3-vl (8b-instruct)
#      - Dataset: quii29/rukopys-dataset
#      - Output notebook fine-tune: trankimhuu/finetune-vlm-lora
#   3. Bật GPU T4 x2, Internet ON
#   4. Paste toàn bộ script này vào 1 cell và chạy
# ==============================================================================

# ==========================================
# CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# ==========================================
import subprocess, sys

def install_packages():
    """Cài đặt các thư viện cần thiết trên Kaggle."""
    packages = [
        'peft',
        'bitsandbytes',
        'qwen-vl-utils',
    ]
    print("📦 Đang cài đặt thư viện cần thiết...", flush=True)
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"  ✅ {pkg}", flush=True)
    
    # Cài transformers nightly (cần cho Qwen3-VL)
    print("  ⏳ transformers (nightly)...", flush=True)
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'git+https://github.com/huggingface/transformers.git'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print("  ✅ transformers (nightly)", flush=True)
    print("📦 Hoàn tất cài đặt!\n", flush=True)

install_packages()

import os
import shutil
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import multiprocessing as mp
import math

# Tắt giới hạn DecompressionBomb cho ảnh lớn (94MP+ trong dataset)
Image.MAX_IMAGE_PIXELS = None

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN KAGGLE
# ==========================================
INPUT_DIR = '/kaggle/input/datasets/quii29/rukopys-dataset'
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, 'test/images')
TEST_METADATA_PATH = os.path.join(INPUT_DIR, 'test/metadata.jsonl')
OUTPUT_CSV_PATH = 'submission.csv'

# Đường dẫn đến model Qwen 3 VL
MODEL_PATH = '/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1'

# Đường dẫn đến thư mục chứa trọng số LoRA sau khi đã train xong
# Script sẽ tự động tìm LoRA weights theo thứ tự ưu tiên:
#   1. /kaggle/working/... (nếu chạy cùng notebook fine-tune)
#   2. /kaggle/input/finetune-vlm-lora/... (nếu mount output fine-tune như Input)
LORA_CANDIDATES = [
    '/kaggle/input/datasets/trankimhuu/qwen-lora/qwen3_vl_lora_output/qwen_lora_final',
]

def find_lora_weights():
    """Tự động tìm đường dẫn LoRA weights từ các vị trí có thể."""
    for path in LORA_CANDIDATES:
        if os.path.exists(path) and os.path.isdir(path):
            # Kiểm tra có file adapter_config.json (dấu hiệu của LoRA adapter)
            if os.path.exists(os.path.join(path, 'adapter_config.json')):
                print(f"✅ Tìm thấy LoRA weights tại: {path}", flush=True)
                return path
    
    # Fallback: tìm bất kỳ adapter_config.json nào trong /kaggle/input/
    print("⚠️ Không tìm thấy LoRA weights ở các vị trí mặc định. Đang quét /kaggle/input/...", flush=True)
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'adapter_config.json' in files:
            print(f"✅ Tìm thấy LoRA weights tại: {root}", flush=True)
            return root
    
    raise FileNotFoundError(
        "❌ Không tìm thấy LoRA weights! Hãy mount output notebook fine-tune như Input.\n"
        "   Vào Kaggle notebook → Add Input → Notebook Output → chọn 'finetune-vlm-lora'"
    )

# LORA_WEIGHTS_PATH sẽ được set trong main() để tránh crash khi import module
LORA_WEIGHTS_PATH = None

# Giới hạn pixel đầu vào để tránh OOM trên T4 và giảm thời gian inference (~1024x768)
MAX_PIXELS = 786432

# CHẾ ĐỘ TEST NHANH: Nếu True, chỉ chạy 1 ảnh trên mỗi GPU rồi xuất kết quả
TEST_MODE = False

# ==========================================
# PROMPT - Phải khớp CHÍNH XÁC với prompt dùng lúc fine-tune
# (xem prepare_vlm_dataset.py)
# ==========================================
INFERENCE_PROMPT = """You are a document understanding model for Ukrainian handwritten text.
Analyze this image and extract all text regions LINE BY LINE. 
It is critical that EACH INDIVIDUAL LINE of text is returned as a SEPARATE region. Do not group multiple lines into a single bounding box.
Transcribe all legible text exactly as it appears, including crossed-out or strikethrough text.

For each line region, output a JSON object with:
- "bbox": [x1, y1, x2, y2] relative coordinates from 0 to 1000 (where 0 is top/left and 1000 is bottom/right)
- "type": one of "handwritten", "printed", "formula", "table", "annotation", "image", "graph"
- "text": the transcribed text of that line (empty string for image/graph types)

Output format: a JSON list of region objects.
Important: For tables, use pipe-separated values (|). For formulas, use LaTeX or plain Unicode.
Only output the JSON list, nothing else.
"""


def extract_json_from_response(text):
    text = text.strip()
    
    clean_text = text
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()
    
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
        if isinstance(parsed, dict):
            for key in ("regions", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return json.dumps(parsed[key], ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    
    # Greedy match: lấy từ [ đầu tiên đến ] cuối cùng trong toàn bộ text.
    # Điều này cố ý để bắt được JSON array hoàn chỉnh ngay cả khi có nested arrays.
    # Kết quả sẽ được validate bằng json.loads() bên dưới.
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
            
    objects = []
    dict_pattern = r'\{[^{}]*"bbox"[^{}]*"type"[^{}]*"text"[^{}]*\}'
    
    for match in re.finditer(dict_pattern, text):
        obj_str = match.group(0)
        try:
            obj = json.loads(obj_str)
            if "bbox" in obj and "text" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            pass
            
    if objects:
        return json.dumps(objects, ensure_ascii=False)
        
    return "[]"


def rescale_bboxes_to_original(json_str, img_path):
    try:
        parsed = json.loads(json_str)
        if not parsed:
            return json_str
            
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
        
        total_pixels = orig_w * orig_h
        if total_pixels > MAX_PIXELS:
            scale_down = (MAX_PIXELS / total_pixels) ** 0.5
            resized_w = int(orig_w * scale_down)
            resized_h = int(orig_h * scale_down)
        else:
            resized_w, resized_h = orig_w, orig_h
        
        scale_x = orig_w / max(resized_w, 1)
        scale_y = orig_h / max(resized_h, 1)
        
        for item in parsed:
            if "bbox" not in item or len(item["bbox"]) != 4:
                continue
            box = item["bbox"]
            
            max_val = max(box) if box else 0
            
            if max_val > 1005:
                # Fallback: Model trả pixel của ảnh đã resize → scale ngược
                x1 = int(max(0, box[0]) * scale_x)
                y1 = int(max(0, box[1]) * scale_y)
                x2 = int(min(resized_w, box[2]) * scale_x)
                y2 = int(min(resized_h, box[3]) * scale_y)
            elif all(isinstance(v, float) for v in box) and max_val <= 1.0:
                # Fallback: Model trả normalized [0.0-1.0] (float)
                x1 = int(max(0, min(1, box[0])) * orig_w)
                y1 = int(max(0, min(1, box[1])) * orig_h)
                x2 = int(max(0, min(1, box[2])) * orig_w)
                y2 = int(max(0, min(1, box[3])) * orig_h)
            else:
                # Default: [0-1000] normalized (như prompt yêu cầu)
                x1 = int(max(0, min(1000, box[0])) / 1000 * orig_w)
                y1 = int(max(0, min(1000, box[1])) / 1000 * orig_h)
                x2 = int(max(0, min(1000, box[2])) / 1000 * orig_w)
                y2 = int(max(0, min(1000, box[3])) / 1000 * orig_h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            item["bbox"] = [x1, y1, x2, y2]
        
        return json.dumps(parsed, ensure_ascii=False)
        
    except Exception as e:
        print(f"⚠️ Lỗi rescale bbox cho {img_path}: {e}")
        return json_str


def run_inference_single_image(model, processor, img_path, device):
    if not os.path.exists(img_path):
        return "[]", f"⚠️ Không tìm thấy ảnh {img_path}"

    # Mô hình đã được Fine-tune → chỉ cần 1 message duy nhất (không cần few-shot)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": img_path,
                    "max_pixels": MAX_PIXELS
                },
                {"type": "text", "text": INFERENCE_PROMPT}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    raw_json_str = extract_json_from_response(output_text)
    final_json_str = rescale_bboxes_to_original(raw_json_str, img_path)
    
    # Sort regions theo y1 (từ trên xuống dưới) để đảm bảo thứ tự nhất quán
    try:
        regions = json.loads(final_json_str)
        if regions:
            regions.sort(key=lambda r: r.get('bbox', [0, 0, 0, 0])[1])
            final_json_str = json.dumps(regions, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    
    torch.cuda.empty_cache()
    
    return final_json_str, output_text


def worker_process(gpu_id, image_filenames, output_csv):
    """Tiến trình xử lý độc lập trên từng GPU."""
    device = f"cuda:{gpu_id}"
    print(f"🚀 [Worker {gpu_id}] Bắt đầu trên {device} với {len(image_filenames)} ảnh...")
    
    # Cấu hình 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Ép toàn bộ base model load vào riêng GPU này
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map={"": device},
        dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter lên trên base model rồi merge để tăng tốc inference ~10-15%
    print(f"🔧 [Worker {gpu_id}] Đang load LoRA weights từ {LORA_WEIGHTS_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    model = model.merge_and_unload()
    print(f"🔧 [Worker {gpu_id}] Đã merge LoRA → base model thành công.")
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    already_done = set()
    results = []
    
    # Kiểm tra checkpoint riêng của tiến trình này (với validation + dedup)
    if os.path.exists(output_csv):
        try:
            partial_df = pd.read_csv(output_csv, encoding='utf-8')
            partial_df = partial_df.drop_duplicates(subset=['image'], keep='last')
            already_done = set(partial_df['image'].tolist())
            results = partial_df.to_dict('records')
            print(f"♻️ [Worker {gpu_id}] Tìm thấy checkpoint: đã xử lý {len(already_done)} ảnh.")
        except Exception as e:
            print(f"⚠️ [Worker {gpu_id}] Checkpoint CSV bị lỗi, bỏ qua và chạy lại: {e}")
            already_done = set()
            results = []
        
    for img_name in tqdm(image_filenames, desc=f"GPU {gpu_id}", position=gpu_id):
        if img_name in already_done:
            continue
            
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        try:
            final_json_str, _ = run_inference_single_image(model, processor, img_path, device)
        except Exception as e:
            print(f"⚠️ [Worker {gpu_id}] Lỗi xử lý {img_name}: {e}")
            final_json_str = "[]"
            torch.cuda.empty_cache()
        
        results.append({
            "image": img_name,
            "regions": final_json_str
        })
        
        # Lưu checkpoint thường xuyên (atomic write để tránh corrupt)
        if len(results) % 5 == 0:
            tmp_csv = output_csv + '.tmp'
            pd.DataFrame(results).to_csv(tmp_csv, index=False, encoding='utf-8')
            os.replace(tmp_csv, output_csv)
            
    # Lưu toàn bộ sau khi xong (atomic write)
    tmp_csv = output_csv + '.tmp'
    pd.DataFrame(results).to_csv(tmp_csv, index=False, encoding='utf-8')
    os.replace(tmp_csv, output_csv)
    print(f"✅ [Worker {gpu_id}] Đã hoàn thành.")


def main():
    global LORA_WEIGHTS_PATH
    
    # Tìm LoRA weights (thực hiện ở đây thay vì module-level để tránh crash khi import)
    LORA_WEIGHTS_PATH = find_lora_weights()
    
    # Sử dụng start_method 'fork' để chạy được trực tiếp trong Notebook
    mp.set_start_method('fork', force=True)
    
    print("🚀 Đọc metadata test...")
    if not os.path.exists(TEST_METADATA_PATH):
        print(f"❌ Lỗi: Không tìm thấy {TEST_METADATA_PATH}")
        return

    test_records = []
    with open(TEST_METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line))
            
    image_filenames = [record['file_name'].split('/')[-1] for record in test_records]
    print(f"Tìm thấy tổng cộng {len(image_filenames)} ảnh test.")
    
    # KHÔNG dùng torch.cuda.device_count() ở đây để tránh khởi tạo CUDA trước khi fork
    # Mặc định Kaggle T4 x2 là 2 GPU. Ta đếm qua lệnh hệ thống nvidia-smi
    try:
        num_gpus = len(subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
    except Exception:
        num_gpus = 2  # Fallback cho Kaggle T4 x2
        
    if num_gpus < 1:
        print("❌ Lỗi: Không tìm thấy GPU nào! Code cần GPU để chạy.")
        return
        
    print(f"🔥 Phát hiện {num_gpus} GPU. Chuẩn bị chạy Multi-processing (Fork)...")
    
    # -----------------------------------------------------------------
    # KIỂM TRA VÀ KHÔI PHỤC CHECKPOINT TỪ KAGGLE DATASET (NẾU CÓ)
    # -----------------------------------------------------------------
    CHECKPOINT_DIR = '/kaggle/input/rukopys-checkpoints'
    if os.path.exists(CHECKPOINT_DIR):
        print(f"\n🔄 TÌM THẤY CHECKPOINT DATASET TẠI: {CHECKPOINT_DIR}")
        for i in range(num_gpus):
            chk_file = f"partial_results_gpu{i}.csv"
            src = os.path.join(CHECKPOINT_DIR, chk_file)
            dst = chk_file
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  ✅ Đã copy {chk_file} vào working directory.")
            else:
                print(f"  ⚠️ Không tìm thấy {chk_file} trong dataset checkpoint.")
        print("========================================================\n")
    else:
        print(f"ℹ️ Không tìm thấy checkpoint tại {CHECKPOINT_DIR}. Sẽ chạy từ đầu.")
    # -----------------------------------------------------------------
    
    if TEST_MODE:
        print("🛠️ ĐANG CHẠY TRONG CHẾ ĐỘ TEST MODE: Chỉ lấy số lượng ảnh bằng số GPU.")
        image_filenames = image_filenames[:num_gpus]
        print(f"Danh sách test: {image_filenames}")
    
    # Chia danh sách ảnh thành `num_gpus` phần tương đối bằng nhau
    chunk_size = math.ceil(len(image_filenames) / num_gpus)
    chunks = [image_filenames[i:i + chunk_size] for i in range(0, len(image_filenames), chunk_size)]
    
    processes = []
    worker_csvs = []
    
    for i in range(num_gpus):
        worker_csv = f"partial_results_gpu{i}.csv"
        worker_csvs.append(worker_csv)
        
        if i < len(chunks):
            p = mp.Process(target=worker_process, args=(i, chunks[i], worker_csv))
            p.start()
            processes.append(p)
            
    # Đợi tất cả tiến trình hoàn tất
    for p in processes:
        p.join()
        
    print("\n🚀 Gộp kết quả từ các GPU...")
    all_results = []
    for csv_file in worker_csvs:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_results.append(df)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"✅ Đã lưu kết quả cuối cùng tại {OUTPUT_CSV_PATH}")
        
        # Chỉ xóa checkpoint sau khi xác nhận file output hợp lệ
        if os.path.exists(OUTPUT_CSV_PATH) and os.path.getsize(OUTPUT_CSV_PATH) > 0:
            for csv_file in worker_csvs:
                if os.path.exists(csv_file):
                    os.remove(csv_file)
            print("🗑️ Đã xóa các file checkpoint tạm.")
    else:
        print("❌ Không có kết quả nào được tạo ra từ các worker!")

if __name__ == "__main__":
    main()