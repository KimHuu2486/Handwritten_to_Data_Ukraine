import ast
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# 1. Configuration
# =============================================================================
OUTPUT_DIR = Path("outputs/oof_label_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROW_ID_COL = "image"
REGIONS_COL = "regions"
ROW_ID_ALIASES = ("image", "file_name")
NORMALIZE_IMAGE_ID_TO_BASENAME = True

IOU_THRESHOLD = 0.5
CER_LOW = 0.10
CER_MEDIUM = 0.30
CER_HIGH = 0.50

SUSPICIOUS_THRESHOLD_REVIEW = 0.20
SUSPICIOUS_THRESHOLD_DROP_REGION = 0.60
SUSPICIOUS_THRESHOLD_DROP_IMAGE = 0.60

EXPORT_CLEANED_METADATA = True
DROP_FULL_IMAGES = False

IMAGE_ROOT = None

metadata_1_path = "metadata_part1.jsonl"
metadata_2_path = "metadata_part2.jsonl"
metadata_3_path = "metadata_part3.jsonl"

submission_12_path = "submission1.csv"
submission_13_path = "submission2.csv"
submission_23_path = "submission3.csv"

folds = [
    {
        "fold_name": "fold_3",
        "model_name": "M12",
        "train_on": "metadata_1 + metadata_2",
        "gt_path": metadata_3_path,
        "pred_path": submission_12_path,
        "cleaned_output": "cleaned_metadata_3.csv",
    },
    {
        "fold_name": "fold_2",
        "model_name": "M13",
        "train_on": "metadata_1 + metadata_3",
        "gt_path": metadata_2_path,
        "pred_path": submission_13_path,
        "cleaned_output": "cleaned_metadata_2.csv",
    },
    {
        "fold_name": "fold_1",
        "model_name": "M23",
        "train_on": "metadata_2 + metadata_3",
        "gt_path": metadata_1_path,
        "pred_path": submission_23_path,
        "cleaned_output": "cleaned_metadata_1.csv",
    },
]

KAGGLE_METRIC_PATH = Path("kaggle_metric.py")
OFFICIAL_METRIC_NOTEBOOK_CANDIDATES = [
    Path("official-evaluation-metric-text-normalization.ipynb"),
    Path("Handwritten_to_Data_Ukraine/official-evaluation-metric-text-normalization.ipynb"),
]

pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 180)


# =============================================================================
# 2. Utility Functions
# =============================================================================
def canonical_image_id(value: Any) -> str:
    text = str(value).strip()
    if NORMALIZE_IMAGE_ID_TO_BASENAME:
        text = text.replace("\\", "/")
        text = Path(text).name
    return text


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported format for {path}. Expected .csv, .jsonl, .ndjson, or .json.")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = _read_table(path)
    original_columns = list(df.columns)
    row_id_source_col = next((col for col in ROW_ID_ALIASES if col in df.columns), None)
    
    if not row_id_source_col:
        raise ValueError(f"{path}: missing row id column. Available: {list(df.columns)}")
    
    df[ROW_ID_COL] = df[row_id_source_col].apply(canonical_image_id)
    df = df.sort_values(ROW_ID_COL).reset_index(drop=True)
    df.attrs["source_path"] = str(path)
    df.attrs["original_columns"] = original_columns
    return df


def parse_regions(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        parsed = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        parsed = []
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "null"}:
            parsed = []
        else:
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception as e:
                    raise ValueError(f"Failed to parse regions JSON/literal.") from e
    else:
        parsed = []
    return parsed if isinstance(parsed, list) else []


def validate_region(region: Dict[str, Any], image: str, region_idx: int, source: str) -> Dict[str, Any]:
    if not isinstance(region, dict):
        raise ValueError("Region must be dict")
    bbox = region.get("bbox", [])
    if len(bbox) != 4:
        raise ValueError("bbox must have length 4")
    out = dict(region)
    out["bbox"] = [float(x) for x in bbox]
    out["type"] = str(out.get("type", ""))
    out["text"] = str(out.get("text", ""))
    out["_region_idx"] = int(region_idx)
    return out


def dataframe_to_region_map(df: pd.DataFrame, source: str) -> Dict[str, List[Dict[str, Any]]]:
    region_map = {}
    for _, row in df.iterrows():
        image = str(row[ROW_ID_COL])
        parsed = parse_regions(row[REGIONS_COL])
        region_map[image] = [validate_region(r, image, i, source) for i, r in enumerate(parsed)]
    return region_map


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
        bx1, by1, bx2, by2 = [float(x) for x in box_b]
    except Exception:
        return 0.0

    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


# =============================================================================
# 3. Metric Loading
# =============================================================================
def find_official_metric_notebook() -> Optional[Path]:
    for candidate in OFFICIAL_METRIC_NOTEBOOK_CANDIDATES:
        if Path(candidate).exists():
            return Path(candidate)
    return None

def generate_kaggle_metric_py(notebook_path: Path, output_path: Path = KAGGLE_METRIC_PATH):
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        lines = source.splitlines()
        if lines and lines[0].strip().startswith("%%writefile") and "kaggle_metric.py" in lines[0]:
            metric_code = "\n".join(lines[1:]).rstrip() + "\n"
            output_path.write_text(metric_code, encoding="utf-8")
            return output_path
    raise RuntimeError("Could not find %%writefile kaggle_metric.py cell.")

if not KAGGLE_METRIC_PATH.exists():
    metric_notebook = find_official_metric_notebook()
    if metric_notebook:
        generate_kaggle_metric_py(metric_notebook)

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    import kaggle_metric
    importlib.reload(kaggle_metric)
    from kaggle_metric import score, score_detailed
    try:
        from kaggle_metric import _normalize_text
    except Exception:
        _normalize_text = None
except ImportError:
    print("Warning: kaggle_metric.py missing. Some metric calculations might fail.")
    score = score_detailed = lambda *a, **k: {}
    _normalize_text = None

def normalize_for_metric(text: str, region_type: str) -> str:
    if _normalize_text:
        return _normalize_text(str(text), str(region_type))
    return str(text).strip()

def levenshtein_distance(a: str, b: str) -> int:
    a, b = str(a), str(b)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return int(previous[-1])

def character_error_rate(gt_text: str, pred_text: str, region_type: str):
    normalized_gt = normalize_for_metric(gt_text, region_type)
    normalized_pred = normalize_for_metric(pred_text, region_type)
    if not normalized_gt and not normalized_pred:
        cer = 0.0
    elif not normalized_gt and normalized_pred:
        cer = 1.0
    else:
        cer = levenshtein_distance(normalized_gt, normalized_pred) / max(len(normalized_gt), 1)
    return float(cer), normalized_gt, normalized_pred


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value) if not np.isnan(value) else None
    return value

def regions_to_json_string(value: Any) -> str:
    return json.dumps(to_jsonable(parse_regions(value)), ensure_ascii=False)

def make_metric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[ROW_ID_COL, REGIONS_COL]].copy()
    out[REGIONS_COL] = out[REGIONS_COL].apply(regions_to_json_string)
    return out

# =============================================================================
# 4. Region Matching Logic
# =============================================================================
def classify_matched_error(type_correct: bool, cer: float) -> str:
    if type_correct and cer <= CER_LOW: return "ok"
    if not type_correct and cer > CER_HIGH: return "type_mismatch_and_high_cer"
    if not type_correct: return "type_mismatch"
    if cer > CER_HIGH: return "high_cer"
    if cer > CER_MEDIUM: return "medium_cer"
    if cer > CER_LOW: return "low_cer"
    return "ok"

def match_regions_for_image(image: str, gt_regions: list, pred_regions: list):
    pairs = []
    for gi, gt_region in enumerate(gt_regions):
        for pi, pred_region in enumerate(pred_regions):
            iou = bbox_iou(gt_region.get("bbox", []), pred_region.get("bbox", []))
            if iou >= IOU_THRESHOLD:
                pairs.append((gi, pi, iou))
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    used_gt, used_pred, matches = set(), set(), []
    for gi, pi, iou in pairs:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append((gi, pi, iou))

    analysis_rows, fp_rows = [], []
    for gi, pi, iou in matches:
        gt_region, pred_region = gt_regions[gi], pred_regions[pi]
        gt_type, pred_type = str(gt_region.get("type", "")), str(pred_region.get("type", ""))
        gt_text, pred_text = str(gt_region.get("text", "")), str(pred_region.get("text", ""))
        
        cer, n_gt, n_pred = character_error_rate(gt_text, pred_text, gt_type)
        type_correct = gt_type == pred_type
        
        analysis_rows.append({
            "image": image, "gt_region_idx": gt_region.get("_region_idx", gi),
            "pred_region_idx": pred_region.get("_region_idx", pi),
            "gt_bbox": gt_region.get("bbox"), "pred_bbox": pred_region.get("bbox"),
            "iou": float(iou), "gt_type": gt_type, "pred_type": pred_type,
            "type_correct": bool(type_correct), "gt_text": gt_text, "pred_text": pred_text,
            "normalized_gt_text": n_gt, "normalized_pred_text": n_pred, "cer": float(cer),
            "matched": True, "error_type": classify_matched_error(type_correct, cer)
        })

    for gi, gt_region in enumerate(gt_regions):
        if gi not in used_gt:
            gt_type = str(gt_region.get("type", ""))
            gt_text = str(gt_region.get("text", ""))
            n_gt = normalize_for_metric(gt_text, gt_type)
            analysis_rows.append({
                "image": image, "gt_region_idx": gt_region.get("_region_idx", gi),
                "pred_region_idx": None, "gt_bbox": gt_region.get("bbox"), "pred_bbox": None,
                "iou": 0.0, "gt_type": gt_type, "pred_type": None, "type_correct": None,
                "gt_text": gt_text, "pred_text": None, "normalized_gt_text": n_gt,
                "normalized_pred_text": "", "cer": 1.0 if n_gt else 0.0,
                "matched": False, "error_type": "missed_region"
            })

    for pi, pred_region in enumerate(pred_regions):
        if pi not in used_pred:
            fp_rows.append({
                "image": image, "gt_region_idx": None,
                "pred_region_idx": pred_region.get("_region_idx", pi),
                "gt_bbox": None, "pred_bbox": pred_region.get("bbox"), "iou": 0.0,
                "gt_type": None, "pred_type": str(pred_region.get("type", "")), "type_correct": None,
                "gt_text": None, "pred_text": str(pred_region.get("text", "")),
                "normalized_gt_text": "", "normalized_pred_text": normalize_for_metric(str(pred_region.get("text", "")), str(pred_region.get("type", ""))),
                "cer": None, "matched": False, "error_type": "false_positive"
            })
    return analysis_rows, fp_rows

# =============================================================================
# 5. Core Pipeline execution
# =============================================================================
def main():
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    
    fold_dataframes = {}
    for fold in folds:
        try:
            gt_df = load_dataframe(fold["gt_path"])
            pred_df = load_dataframe(fold["pred_path"])
            fold_dataframes[fold["fold_name"]] = {"gt": gt_df, "pred": pred_df}
            print(f"Loaded {fold['fold_name']}: {len(gt_df)} GT, {len(pred_df)} Pred")
        except Exception as e:
            print(f"Warning: Failed to load data for {fold['fold_name']}: {e}")

    if not fold_dataframes:
        print("No valid data loaded. Exiting.")
        return

    all_region_rows, all_false_positive_rows, image_base_rows = [], [], []

    for fold in folds:
        fold_name = fold["fold_name"]
        if fold_name not in fold_dataframes:
            continue
        
        gt_df, pred_df = fold_dataframes[fold_name]["gt"], fold_dataframes[fold_name]["pred"]
        gt_map = dataframe_to_region_map(gt_df, str(fold["gt_path"]))
        pred_map = dataframe_to_region_map(pred_df, str(fold["pred_path"]))
        
        for image in sorted(gt_map.keys()):
            gt_regions = gt_map.get(image, [])
            pred_regions = pred_map.get(image, [])
            
            image_base_rows.append({
                "fold_name": fold["fold_name"], "model_name": fold["model_name"],
                "image": image, "n_gt_regions": len(gt_regions), "n_pred_regions": len(pred_regions),
            })
            
            region_rows, fp_rows = match_regions_for_image(image, gt_regions, pred_regions)
            
            for r in region_rows: r.update({"fold_name": fold_name, "model_name": fold["model_name"]})
            for r in fp_rows: r.update({"fold_name": fold_name, "model_name": fold["model_name"]})
            
            all_region_rows.extend(region_rows)
            all_false_positive_rows.extend(fp_rows)

    all_region_analysis_df = pd.DataFrame(all_region_rows)
    all_false_positive_df = pd.DataFrame(all_false_positive_rows)

    if not all_region_analysis_df.empty:
        all_region_analysis_df.to_csv(OUTPUT_DIR / "oof_region_analysis.csv", index=False)
        print(f"Saved region analysis: {len(all_region_analysis_df)} rows")
    if not all_false_positive_df.empty:
        all_false_positive_df.to_csv(OUTPUT_DIR / "oof_false_positive_regions.csv", index=False)

if __name__ == "__main__":
    main()
