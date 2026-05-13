import json
import csv
import os

input_file = "c:/Users/HP/source/Handwritten_to_Data_Ukraine/VLM_OCR/source_test/metadata-1.jsonl"
output_file = "c:/Users/HP/source/Handwritten_to_Data_Ukraine/VLM_OCR/source_test/submission_format.csv"

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['image', 'regions'])
    
    for line in f_in:
        if not line.strip():
            continue
        data = json.loads(line)
        
        image_name = os.path.basename(data['file_name'])
        
        new_regions = []
        for region in data.get('regions', []):
            bbox = region['bbox']
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            new_bbox = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            
            new_region = {
                "bbox": new_bbox,
                "type": region.get("type", "handwritten"),
                "text": ""
            }
            new_regions.append(new_region)
        
        regions_str = json.dumps(new_regions, ensure_ascii=False)
        writer.writerow([image_name, regions_str])

print("Done converting!")
