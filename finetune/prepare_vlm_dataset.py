import os
import json
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import posixpath

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN KAGGLE & SIÊU THAM SỐ
# ==========================================
METADATA_PATH = '/kaggle/input/datasets/quii29/rukopys-dataset/train/metadata.jsonl'
OUTPUT_TRAIN = 'train_vlm.jsonl'
OUTPUT_VAL = 'val_vlm.jsonl'
# Thư mục chứa ảnh trên Kaggle
KAGGLE_PREFIX = '/kaggle/input/datasets/quii29/rukopys-dataset/train/'

TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Phải khớp hoàn toàn với Zero-shot baseline để đảm bảo Model học đúng format
ZERO_SHOT_PROMPT = """You are a document understanding model for Ukrainian handwritten text.
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

def process_record(record):
    """Chuyển đổi 1 record metadata thành format Conversation của Qwen."""
    image_path = posixpath.join(KAGGLE_PREFIX, record["file_name"])
    w = max(1, record.get("image_width", 1))
    h = max(1, record.get("image_height", 1))
    
    out_regions = []
    for r in record.get("regions", []):
        box = r.get("bbox", [])
        if len(box) == 4:
            # Chuẩn hóa về [0, 1000] theo đúng yêu cầu prompt
            nx1 = int(max(0, min(w, box[0])) / w * 1000)
            ny1 = int(max(0, min(h, box[1])) / h * 1000)
            nx2 = int(max(0, min(w, box[2])) / w * 1000)
            ny2 = int(max(0, min(h, box[3])) / h * 1000)
            
            out_regions.append({
                "bbox": [nx1, ny1, nx2, ny2],
                "type": r.get("type", "handwritten"),
                "text": r.get("text", "")
            })
            
    # Build format hội thoại chuẩn HuggingFace
    message = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "max_pixels": 262144
                    },
                    {"type": "text", "text": ZERO_SHOT_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(out_regions, ensure_ascii=False)}
                ]
            }
        ]
    }
    return message

def main():
    metadata_file = METADATA_PATH
    if not os.path.exists(METADATA_PATH):
        # Fallback local path (chạy thử nghiệm trên máy tính cá nhân)
        local_path = "dataset/train/metadata.jsonl"
        if os.path.exists(local_path):
            print(f"ℹ️ Sử dụng local metadata để test: {local_path}")
            metadata_file = local_path
        else:
            print(f"❌ Lỗi: Không tìm thấy file {METADATA_PATH}")
            return
            
    source_dict = defaultdict(list)
    
    print("🚀 Đang đọc và xử lý dữ liệu metadata...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            processed = process_record(record)
            src = record.get("source", "unknown")
            source_dict[src].append(processed)
            
    train_data = []
    val_data = []
    
    # Chia Train/Val dựa trên Nguồn (source) để cân bằng phân bố
    random.seed(RANDOM_SEED)
    for src, items in source_dict.items():
        random.shuffle(items)
        split_idx = int(len(items) * TRAIN_SPLIT_RATIO)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])
        print(f"  - Nguồn '{src}': {len(items[:split_idx])} train, {len(items[split_idx:])} val")
        
    # Trộn ngẫu nhiên lần cuối
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # Ghi ra file
    print("\n💾 Đang ghi ra file JSONL...")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n✅ Hoàn thành! Tổng cộng: {len(train_data)} train, {len(val_data)} validation.")
    print(f"📁 Output files: {OUTPUT_TRAIN}, {OUTPUT_VAL}")

if __name__ == "__main__":
    main()
