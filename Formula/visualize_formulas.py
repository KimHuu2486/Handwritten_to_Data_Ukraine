import pandas as pd
import json
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
csv_path = Path("Formula/hybrid_gold_train_formula.csv")
images_dir = Path("Dataset/train/images")
output_dir = Path("Formula/visualizations")

# Create output dir if not exists
output_dir.mkdir(parents=True, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

count = 0
for idx, row in df.iterrows():
    image_name = row['image']
    regions_str = row['regions']
    
    try:
        regions = json.loads(regions_str)
    except json.JSONDecodeError:
        continue
    
    # Check if there is any formula region
    has_formula = any(r.get('type') == 'formula' for r in regions)
    if not has_formula:
        continue
        
    img_path = images_dir / image_name
    if not img_path.exists():
        print(f"Warning: Image not found -> {img_path}")
        continue
        
    try:
        img = Image.open(img_path)
    except IOError:
        print(f"Warning: Could not read image -> {img_path}")
        continue
        
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img)
    ax.axis('off')
    
    for r in regions:
        if r.get('type') == 'formula':
            bbox = r.get('bbox')
            text = r.get('text', '')
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                     fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                
                # Draw text above bounding box
                # Wrap text to avoid it going out of bounds if it's too long
                display_text = text if len(text) < 50 else text[:50] + "..."
                if not display_text.strip():
                    display_text = "<EMPTY TEXT>"
                    
                ax.text(x1, y1 - 5, display_text, color='red', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=2))

    output_path = output_dir / image_name
    
    # Ensure subdirectory structure is maintained if image_name contains slashes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    count += 1
    print(f"Processed {count}: {image_name}")

print(f"Done! {count} images with formulas visualized in '{output_dir}'.")
