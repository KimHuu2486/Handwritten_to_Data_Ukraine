import json
import os

def generate_prompt_for_rukopys(box_class):
    """Định tuyến prompt dựa theo đúng 7 class yêu cầu của cuộc thi RUKOPYS"""
    if box_class in ["handwritten", "printed", "annotation"]:
        return "Transcribe the text in this image precisely in Ukrainian/Russian:"
    elif box_class == "formula":
        return "Transcribe the mathematical formula in this image into LaTeX format:"
    elif box_class == "table":
        return "Transcribe the table in this image into pipe-separated values format (e.g., cell1|cell2|cell3):"
    return "Transcribe the content of this image:"

def main():
    # File đầu vào ở local
    INPUT_JSON = "metadata.jsonl"  
    
    # Gắn cứng đường dẫn của Kaggle để file sinh ra dùng được luôn trên Kaggle
    IMAGE_DIR = "/kaggle/input/datasets/quii29/cropped-rukopys-dataset/train" 
    
    # File đầu ra để nạp vào Qwen
    OUTPUT_JSONL = "train_vlm.jsonl"
    
    print(f"Đang đọc dữ liệu từ {INPUT_JSON}...")
    try:
        data = []
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️ Chưa tìm thấy file {INPUT_JSON}. Hãy yêu cầu đồng đội (DE) cung cấp file này hoặc kiểm tra lại đường dẫn!")
        return
    except json.JSONDecodeError as e:
        print(f"⚠️ Lỗi định dạng JSON tại file {INPUT_JSON}: {e}")
        return
        
    count = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        for item in data:
            box_class = item.get('label', '')
            ground_truth = item.get('text', '')
            
            # Bỏ qua các ảnh rác hoặc rỗng không cần VLM học
            if box_class in ["image", "graph"] or not ground_truth:
                continue
                
            # Nối chuỗi thủ công dùng '/' để đảm bảo đường dẫn chuẩn Linux (Kaggle), 
            # tránh bị Windows thay bằng dấu backslash (\)
            img_path = f"{IMAGE_DIR.rstrip('/')}/{item['image']}"
            prompt = generate_prompt_for_rukopys(box_class)
            
            # Cấu trúc chuẩn ChatML
            qwen_msg = {
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": img_path}, 
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "text", "text": ground_truth}
                        ]
                    }
                ]
            }
            f_out.write(json.dumps(qwen_msg, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"✅ Xong! Đã tạo thành công file {OUTPUT_JSONL} với {count} mẫu dữ liệu chất lượng cao.")

if __name__ == "__main__":
    main()
