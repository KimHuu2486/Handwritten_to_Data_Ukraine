import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# 1. Configuration
# =============================================================================
INPUT_DIR = Path("outputs/oof_label_audit")
OUTPUT_DIR = Path("outputs/oof_official_image_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_ANALYSIS_PATH = INPUT_DIR / "oof_region_analysis.csv"
FALSE_POSITIVE_PATH = INPUT_DIR / "oof_false_positive_regions.csv"

FOLD_META = {
    "fold_3": {"model_name": "M12", "metadata_path": "metadata_part3.jsonl", "submission_path": "submission1.csv"},
    "fold_2": {"model_name": "M13", "metadata_path": "metadata_part2.jsonl", "submission_path": "submission2.csv"},
    "fold_1": {"model_name": "M23", "metadata_path": "metadata_part1.jsonl", "submission_path": "submission3.csv"},
}

W_DET = 0.15
W_CLS = 0.05
W_REGION_CER = 0.30
W_PAGE_CER = 0.50

REVIEW_THRESHOLD = 0.20
DROP_THRESHOLD = 0.40

# =============================================================================
# 2. Helper Functions
# =============================================================================
def canonical_image_id(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name

def scorable_region(region: Dict[str, Any]) -> bool:
    if region.get("type", "handwritten") in ("image", "graph"): return False
    if region.get("language", "uk") == "other": return False
    if region.get("legibility", "legible") == "illegible": return False
    return True

def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists(): return [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []

def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def parse_bbox(value: str) -> List[float] | None:
    try: return [float(x) for x in ast.literal_eval(value)] if value else None
    except Exception: return None

def reading_order_key(bbox: List[float]) -> Tuple[float, float]:
    return (((bbox[1] + bbox[3]) / 2) // 15, (bbox[0] + bbox[2]) / 2)

def levenshtein_distance(a: str, b: str) -> int:
    a, b = str(a), str(b)
    if len(a) < len(b): a, b = b, a
    if len(b) == 0: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def level_from_suspicious(score: float) -> str:
    if score < 0.10: return "ok"
    if score < REVIEW_THRESHOLD: return "low"
    if score < DROP_THRESHOLD: return "review"
    return "drop"

# =============================================================================
# 3. Main Logic
# =============================================================================
def main():
    print(f"Reading from: {REGION_ANALYSIS_PATH}")
    
    if not REGION_ANALYSIS_PATH.exists():
        print(f"Error: Required input {REGION_ANALYSIS_PATH} not found. Run the region audit first.")
        return

    scorable_gt = {}
    for fold_name, cfg in FOLD_META.items():
        meta_path = Path(cfg["metadata_path"])
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    image = canonical_image_id(obj.get("image") or obj.get("file_name"))
                    for i, r in enumerate(obj.get("regions", [])):
                        scorable_gt[(fold_name, image, i)] = scorable_region(r)

    region_rows, _ = read_csv_rows(REGION_ANALYSIS_PATH)
    fp_rows, _ = read_csv_rows(FALSE_POSITIVE_PATH)

    regions_by_image = defaultdict(list)
    fp_by_image = defaultdict(list)

    for r in region_rows: regions_by_image[(r["fold_name"], r["model_name"], r["image"])].append(r)
    for r in fp_rows: fp_by_image[(r["fold_name"], r["model_name"], r["image"])].append(r)

    image_score_rows = []
    
    for key, rows in sorted(regions_by_image.items()):
        fold_name, model_name, image = key
        f_rows = fp_by_image.get(key, [])

        n_gt_regions = len(rows)
        n_fp = len(f_rows)
        n_missed = sum(1 for r in rows if r.get("error_type") == "missed_region")
        n_matched = sum(1 for r in rows if str(r.get("matched")).lower() == "true")
        
        det_tp, det_fp, det_fn = n_matched, n_fp, n_missed
        det_prec = det_tp / max(det_tp + det_fp, 1)
        det_rec = det_tp / max(det_tp + det_fn, 1)
        det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-9)

        class_total = n_matched
        class_correct = sum(1 for r in rows if str(r.get("matched")).lower() == "true" and str(r.get("type_correct")).lower() == "true")
        class_acc = class_correct / max(class_total, 1)

        region_cers, gt_items, pred_items = [], [], []
        
        for r in rows:
            try: gt_idx = int(float(r.get("gt_region_idx", -1)))
            except Exception: gt_idx = -1
            
            is_scorable = scorable_gt.get((fold_name, image, gt_idx), r.get("gt_type") not in ("image", "graph"))
            if not is_scorable: continue

            gt_bbox = parse_bbox(r.get("gt_bbox", "")) or [0.0]*4
            gt_items.append((reading_order_key(gt_bbox), r.get("normalized_gt_text", "")))
            
            if str(r.get("matched")).lower() == "true":
                if r.get("cer", "") != "": region_cers.append(float(r["cer"]))
                pred_bbox = parse_bbox(r.get("pred_bbox", "")) or gt_bbox
                pred_items.append((reading_order_key(pred_bbox), r.get("normalized_pred_text", "")))

        for r in f_rows:
            if r.get("pred_type") in ("image", "graph"): continue
            pred_bbox = parse_bbox(r.get("pred_bbox", "")) or [0.0]*4
            pred_items.append((reading_order_key(pred_bbox), r.get("normalized_pred_text", "")))

        region_cer = sum(region_cers) / len(region_cers) if region_cers else 1.0
        gt_page = "\n".join(t for _, t in sorted(gt_items, key=lambda x: x[0]))
        pred_page = "\n".join(t for _, t in sorted(pred_items, key=lambda x: x[0]))
        
        page_cer = levenshtein_distance(pred_page, gt_page) / len(gt_page) if gt_page else 1.0

        image_score = (W_DET * det_f1) + (W_CLS * class_acc) + (W_REGION_CER * max(0.0, 1.0 - region_cer)) + (W_PAGE_CER * max(0.0, 1.0 - page_cer))
        suspicious_score = 1.0 - image_score

        image_score_rows.append({
            "fold_name": fold_name, "model_name": model_name, "image": image,
            "detection_f1": det_f1, "classification_accuracy": class_acc,
            "region_cer": region_cer, "page_cer": page_cer,
            "official_like_image_score": image_score, "official_like_suspicious_score": suspicious_score,
            "official_like_level": level_from_suspicious(suspicious_score)
        })

    IMAGE_COLS = ["fold_name", "model_name", "image", "detection_f1", "classification_accuracy", "region_cer", "page_cer", "official_like_image_score", "official_like_suspicious_score", "official_like_level"]
    
    write_csv_rows(OUTPUT_DIR / "official_like_image_scores.csv", image_score_rows, IMAGE_COLS)
    print(f"Computed {len(image_score_rows)} image rows. Output saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
