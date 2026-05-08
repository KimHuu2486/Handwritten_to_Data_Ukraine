# ==============================================================================
# Zero-shot Baseline - RUKOPYS Kaggle Competition
# Model: Qwen 3 VL (8b-instruct)
#
# Submission format (from official evaluation notebook):
#   CSV with columns: image, regions
#   regions = JSON list of {"bbox": [x1,y1,x2,y2], "type": "...", "text": "..."}
#   bbox = absolute pixel coordinates, top-left origin
# ==============================================================================

import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN KAGGLE
# ==========================================
INPUT_DIR = '/kaggle/input/datasets/quii29/rukopys-dataset'
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, 'test/images')
TEST_METADATA_PATH = os.path.join(INPUT_DIR, 'test/metadata.jsonl')
OUTPUT_CSV_PATH = 'submission.csv'
PARTIAL_CSV_PATH = 'partial_results.csv'  # Checkpoint file để resume nếu bị crash

# Đường dẫn đến model Qwen 3 VL
MODEL_PATH = '/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1'

# Giới hạn pixel đầu vào để tránh OOM trên T4 (~1024x980)
MAX_PIXELS = 1003520

# ==========================================
# PROMPT ZERO-SHOT
# ==========================================
# Yêu cầu model trả bbox theo pixel coordinates của ảnh GỐC.
# Lưu ý: Model nhận ảnh đã resize, nên bbox sẽ theo ảnh đã resize.
# Ta sẽ scale lại về kích thước gốc trong post-processing.
ZERO_SHOT_PROMPT = """You are a document understanding model for Ukrainian handwritten text.
Analyze this image and extract all text regions.

For each region, output a JSON object with:
- "bbox": [x1, y1, x2, y2] pixel coordinates of the bounding box
- "type": one of "handwritten", "printed", "formula", "table", "annotation", "image", "graph"
- "text": the transcribed text (empty string for image/graph types)

Output format: a JSON list of region objects.
Important: For tables, use pipe-separated values (|). For formulas, use LaTeX or plain Unicode.
Only output the JSON list, nothing else.
"""


def extract_json_from_response(text):
    """Trích xuất chuỗi JSON array từ output của model.
    
    Xử lý nhiều trường hợp:
    - Model trả về JSON thuần
    - Model bọc trong ```json ... ```
    - Model chèn text thừa trước/sau JSON
    - Model trả dict thay vì list
    """
    text = text.strip()
    
    # Bỏ markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Thử parse trực tiếp
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
        # Nếu model trả dict (ví dụ {"regions": [...]})
        if isinstance(parsed, dict):
            for key in ("regions", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return json.dumps(parsed[key], ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    
    # Fallback: tìm JSON array đầu tiên bằng regex
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    
    return "[]"


def rescale_bboxes_to_original(json_str, img_path):
    """Scale bbox từ tọa độ ảnh đã resize về tọa độ ảnh gốc.
    
    Qwen-VL nhận ảnh đã bị resize bởi max_pixels, nên bbox output
    tương ứng với kích thước ảnh nhỏ. Hàm này tính tỉ lệ scale
    và nhân ngược lại để bbox khớp với ảnh gốc.
    
    Args:
        json_str: JSON string chứa list các region với bbox
        img_path: Đường dẫn đến ảnh gốc
        
    Returns:
        JSON string đã được rescale bbox
    """
    try:
        parsed = json.loads(json_str)
        if not parsed:
            return json_str
            
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
        
        # Tính kích thước ảnh sau khi bị resize bởi max_pixels
        # (Thuật toán giống qwen_vl_utils: giữ tỉ lệ, ép tổng pixel <= MAX_PIXELS)
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
            
            # Phát hiện hệ tọa độ model đang dùng
            max_val = max(box) if box else 0
            
            if max_val <= 1.0:
                # Model trả normalized [0.0 - 1.0] → nhân trực tiếp với kích thước gốc
                x1 = int(max(0, min(1, box[0])) * orig_w)
                y1 = int(max(0, min(1, box[1])) * orig_h)
                x2 = int(max(0, min(1, box[2])) * orig_w)
                y2 = int(max(0, min(1, box[3])) * orig_h)
            elif max_val <= 1005:
                # Qwen-VL thường trả tọa độ [0-1000] → normalize rồi nhân với gốc
                x1 = int(max(0, min(1000, box[0])) / 1000 * orig_w)
                y1 = int(max(0, min(1000, box[1])) / 1000 * orig_h)
                x2 = int(max(0, min(1000, box[2])) / 1000 * orig_w)
                y2 = int(max(0, min(1000, box[3])) / 1000 * orig_h)
            else:
                # Model trả pixel của ảnh đã resize → scale ngược lại
                x1 = int(max(0, box[0]) * scale_x)
                y1 = int(max(0, box[1]) * scale_y)
                x2 = int(min(resized_w, box[2]) * scale_x)
                y2 = int(min(resized_h, box[3]) * scale_y)
            
            # Clamp để bbox không vượt ra ngoài ảnh gốc
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            item["bbox"] = [x1, y1, x2, y2]
        
        # KHÔNG dùng indent — giữ JSON trên 1 dòng để CSV không bị lỗi parse
        return json.dumps(parsed, ensure_ascii=False)
        
    except Exception as e:
        print(f"⚠️ Lỗi rescale bbox cho {img_path}: {e}")
        return json_str


def run_inference_single_image(model, processor, img_path):
    """Chạy inference trên 1 ảnh và trả về kết quả đã rescale bbox.
    
    Args:
        model: Model Qwen3-VL đã load
        processor: AutoProcessor tương ứng
        img_path: Đường dẫn tuyệt đối đến file ảnh
        
    Returns:
        tuple: (json_str đã rescale, raw_output_text từ model)
    """
    if not os.path.exists(img_path):
        return "[]", f"⚠️ Không tìm thấy ảnh {img_path}"
        
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": img_path,
                    "max_pixels": MAX_PIXELS
                },
                {"type": "text", "text": ZERO_SHOT_PROMPT}
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
    
    inputs = inputs.to("cuda:0")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,  # Tăng lên để tránh bị cắt JSON ở trang dày đặc chữ
            do_sample=False
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Trích xuất JSON → rescale bbox về kích thước gốc
    raw_json_str = extract_json_from_response(output_text)
    final_json_str = rescale_bboxes_to_original(raw_json_str, img_path)
    
    torch.cuda.empty_cache()
    
    return final_json_str, output_text


def main():
    print("🚀 Bắt đầu Zero-shot Baseline với Qwen 3 VL (Kaggle Model)...")
    print(f"Loading model từ: {MODEL_PATH}")
    
    # Cấu hình 4-bit NF4 + Double Quant để tiết kiệm VRAM tối đa
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Khống chế VRAM trên mỗi GPU — chừa vùng đệm cho inference
    num_gpus = torch.cuda.device_count()
    max_memory_mapping = {i: "12GB" for i in range(num_gpus)}
    max_memory_mapping["cpu"] = "25GB"
    print(f"Phát hiện {num_gpus} GPU. Max memory mapping: {max_memory_mapping}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
        max_memory=max_memory_mapping,
        low_cpu_mem_usage=True
    )
    
    # In ra device map để debug
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Đọc danh sách ảnh test
    print(f"Đọc metadata từ {TEST_METADATA_PATH}")
    
    if not os.path.exists(TEST_METADATA_PATH):
        print(f"❌ Lỗi: Không tìm thấy {TEST_METADATA_PATH}")
        return

    test_records = []
    with open(TEST_METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line))
            
    image_filenames = [record['file_name'].split('/')[-1] for record in test_records]
    print(f"Tìm thấy {len(image_filenames)} ảnh test.")
    
    # Kiểm tra checkpoint — nếu đã chạy dở thì resume
    already_done = set()
    results = []
    if os.path.exists(PARTIAL_CSV_PATH):
        partial_df = pd.read_csv(PARTIAL_CSV_PATH, encoding='utf-8')
        already_done = set(partial_df['image'].tolist())
        results = partial_df.to_dict('records')
        print(f"♻️ Tìm thấy checkpoint: đã xử lý {len(already_done)} ảnh trước đó. Tiếp tục từ đây...")
    
    print("🚀 BẮT ĐẦU CHẠY TOÀN BỘ DATASET...")
    
    for idx, img_name in enumerate(tqdm(image_filenames, desc="Đang phân tích")):
        if img_name in already_done:
            continue
            
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        final_json_str, _ = run_inference_single_image(model, processor, img_path)
        
        results.append({
            "image": img_name,
            "regions": final_json_str
        })
        
        # Lưu checkpoint mỗi khi tổng số kết quả đạt bội số của 10
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(PARTIAL_CSV_PATH, index=False, encoding='utf-8')
            print(f"\n[INFO] Đã tích lũy {len(results)} / {len(image_filenames)} ảnh. (Checkpoint saved)")

    # Lưu kết quả cuối cùng
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"\n✅ Đã hoàn thành và lưu kết quả tại {OUTPUT_CSV_PATH}")
    
    # Dọn file checkpoint
    if os.path.exists(PARTIAL_CSV_PATH):
        os.remove(PARTIAL_CSV_PATH)
        print("🗑️ Đã xóa file checkpoint.")

if __name__ == "__main__":
    main()