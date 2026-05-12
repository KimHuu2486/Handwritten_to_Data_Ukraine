import json
import torch
import os
import csv
import itertools
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = "You are an expert OCR assistant. You ONLY output the transcribed text exactly as written in the image. Never explain, never refuse, never add commentary."

def generate_prompt_for_rukopys(box_class):
    if box_class in ["handwritten", "printed", "annotation"]:
        return "Transcribe the text in this image exactly as written. Output ONLY the raw text, nothing else. /no_think"
    elif box_class == "formula":
        return "Transcribe the formula in this image into LaTeX. Output ONLY the LaTeX expression, nothing else. /no_think"
    elif box_class == "table":
        return "Transcribe the table in this image into pipe-separated values (e.g., cell1|cell2|cell3). Output ONLY the table, nothing else. /no_think"
    return "Transcribe the content of this image exactly as written. Output ONLY the text, nothing else. /no_think"

def main():
    # 1. Khởi tạo BASE MODEL
    MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1" 
    print(f"Đang tải Base model {MODEL_PATH} để test Zero-shot...")
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Bắt buộc padding bên trái cho Inference Batching để mô hình không sinh chữ rác
    processor.tokenizer.padding_side = "left"
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto",
        attn_implementation="sdpa" # TĂNG TỐC ĐỘ ATTENTION CALCULATION
    )
    
    # 2. Cấu hình I/O
    INPUT_JSONL = "/kaggle/input/datasets/trankimhuu/metadata-rukopys/metadata-1.jsonl"
    OUTPUT_CSV = "baseline_submission.csv" # Đã sửa thành CSV để nộp Kaggle
    IMAGE_BASE_DIR = "/kaggle/input/datasets/quii29/rukopys-dataset/test" 
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out_f = open(OUTPUT_CSV, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(out_f)
    
    # Ghi header chuẩn Kaggle
    csv_writer.writerow(["image", "regions"])
    
    # ==============================================================
    # 3. LUỒNG SẢN XUẤT (PRODUCER) - TỐI ƯU DISK I/O & BẢO TOÀN THỨ TỰ
    # ==============================================================
    results = {}
    
    # Quét nhanh 1 vòng cấu trúc file JSONL để bảo toàn thứ tự kết quả 100%
    for line in lines:
        line = line.strip()
        if not line: continue
        data = json.loads(line)
        basename = os.path.basename(data.get("file_name", ""))
        regions = data.get("regions", [])
        # Tạo sẵn mảng trống chứa None, kích thước bằng đúng số lượng box
        results[basename] = [None] * len(regions)
        
    def region_generator():
        """Hàm Generator mở mỗi ảnh ĐÚNG 1 LẦN, cắt tất cả các box và nhả (yield) liên tục vào luồng"""
        for line in lines:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            basename = os.path.basename(data.get("file_name", ""))
            img_path = os.path.join(IMAGE_BASE_DIR, data.get("file_name", ""))
            regions = data.get("regions", [])
            
            try:
                # Mở file ảnh ĐÚNG 1 LẦN
                full_img = Image.open(img_path).convert("RGB")
                img_w, img_h = full_img.size
            except Exception as e:
                print(f"⚠️ Lỗi mở ảnh {img_path}: {e}")
                # Nếu ảnh lỗi, điền rỗng cho toàn bộ box của ảnh này
                for r_idx in range(len(regions)):
                    results[basename][r_idx] = {"bbox": regions[r_idx]["bbox"], "type": regions[r_idx].get("type", ""), "text": ""}
                continue
                
            for r_idx, region in enumerate(regions):
                box_type = region.get("type", "")
                bbox = region["bbox"]
                x_min, y_min, w, h = bbox
                
                # Chống Out of Bounds
                c_xmin = max(0, int(round(x_min)))
                c_ymin = max(0, int(round(y_min)))
                c_xmax = min(img_w, int(round(x_min + w)))
                c_ymax = min(img_h, int(round(y_min + h)))
                kaggle_bbox = [c_xmin, c_ymin, c_xmax, c_ymax]
                
                # Các box rỗng hoặc vô nghĩa -> Điền trực tiếp vào kết quả, bỏ qua GPU
                if box_type in ["image", "graph"] or c_xmax - c_xmin < 10 or c_ymax - c_ymin < 10:
                    results[basename][r_idx] = {"bbox": kaggle_bbox, "type": box_type, "text": ""}
                    continue
                    
                cropped_img = full_img.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                
                # [VŨ KHÍ TỐI THƯỢNG CHỐNG OOM] 
                # Khống chế độ phân giải của box: Nếu box bị cắt lỗi và quá to (>1024x1024),
                # nó sẽ tự động bóp nhỏ lại giữ nguyên tỉ lệ. Đảm bảo GPU không bao giờ bị nổ VRAM vì Token Explosion.
                # Những box nhỏ (chữ viết tay bình thường) sẽ không bị ảnh hưởng.
                cropped_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                prompt_text = generate_prompt_for_rukopys(box_type)
                
                # Nhả data "ngon" vào luồng cho GPU xử lý
                yield (basename, r_idx, kaggle_bbox, box_type, cropped_img, prompt_text)

    # ==============================================================
    # 4. LUỒNG TIÊU THỤ (CONSUMER) - TỐI ƯU GPU GLOBAL BATCHING
    # ==============================================================
    BATCH_SIZE = 8
    valid_generator = region_generator()
    processed_count = 0
    
    print("\n🚀 Bắt đầu quá trình Inference Siêu Tốc...")
    
    while True:
        # Hút ĐÚNG 8 vùng ảnh từ luồng (bất kể đến từ 1 hay 8 bức ảnh gốc khác nhau)
        batch = list(itertools.islice(valid_generator, BATCH_SIZE))
        if not batch:
            break
            
        # Gom tin nhắn cho Batch
        messages_list = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
            ]
            for (_, _, _, _, img, prompt) in batch
        ]
        
        batch_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in messages_list]
        batch_img, batch_vid = process_vision_info(messages_list)
        
        inputs = processor(
            text=batch_texts,
            images=batch_img,
            videos=batch_vid,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # GPU chạy đồng loạt (Tắt sampling để sinh chữ cực nhanh)
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Lắp ráp kết quả text sinh ra nhét lại đúng vị trí của file ảnh gốc
        for (basename, r_idx, bbox, box_type, _, _), out_text in zip(batch, output_texts):
            results[basename][r_idx] = {
                "bbox": bbox,
                "type": box_type,
                "text": out_text.strip()
            }
            
        processed_count += len(batch)
        print(f"⚡ Đã dự đoán xong {processed_count} đoạn text...")
        
        # Dọn dẹp rác GPU sau mỗi lô để chống phân mảnh VRAM tích tụ theo thời gian
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    # ==============================================================
    # 5. LƯU KẾT QUẢ CUỐI CÙNG RA CSV
    # ==============================================================
    print("\nĐang đóng gói và ghi kết quả ra file CSV...")
    for basename, regions_list in results.items():
        # Lọc an toàn các box bị rớt (nếu có lỗi hệ thống)
        clean_regions = [r for r in regions_list if r is not None]
        csv_writer.writerow([basename, json.dumps(clean_regions, ensure_ascii=False)])
        
    out_f.close()
    print(f"🎉 Hoàn tất! Cỗ máy cày đã xử lý xong. File {OUTPUT_CSV} sẵn sàng nộp!")

if __name__ == "__main__":
    main()