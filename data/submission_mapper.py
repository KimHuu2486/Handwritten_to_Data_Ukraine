from __future__ import annotations

import copy
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Import nội bộ trong hệ thống của bạn (cần đảm bảo module này tồn tại khi chạy)
try:
    from train_preprocess_pipeline import PipelineConfig, process_record
except ImportError:
    print("Cảnh báo: Không tìm thấy module 'train_preprocess_pipeline'. Một số tính năng rebuild có thể không hoạt động.")
    PipelineConfig = Any
    process_record = Any

JsonDict = Dict[str, Any]


def read_jsonl(path: Path) -> List[JsonDict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def image_key(value: Any) -> str:
    return Path(str(value)).name


def parse_regions(value: Any) -> List[JsonDict]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("regions must be a JSON list")
    return parsed


def read_submission_csv(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in (reader.fieldnames or []) or "regions" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain image,regions columns")
        for row in reader:
            rows.append({"image": row["image"], "regions": parse_regions(row.get("regions", "[]"))})
    return rows


def write_submission_csv(rows: Sequence[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "regions"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "regions": json.dumps(row.get("regions", []), ensure_ascii=False),
                }
            )


def load_test_records_from_bbox_csv(
    bbox_csv: Path = Path("old_bbox_bbox_only.csv"),
    dataset_root: Path = Path("dataset"),
) -> List[JsonDict]:
    records: List[JsonDict] = []
    with Path(bbox_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = image_key(row["image"])
            image_path = Path(dataset_root) / "test" / "images" / name
            with Image.open(image_path) as image:
                width, height = image.size
            records.append(
                {
                    "file_name": f"test/images/{name}",
                    "image_width": int(width),
                    "image_height": int(height),
                    "source": "test",
                    "regions": parse_regions(row.get("regions", "[]")),
                }
            )
    return records


def pipeline_config_from_preprocessed_summary(
    summary_path: Path = Path("dataset/preprocessed_scanner_v1/summary.json"),
) -> PipelineConfig:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    values = dict(summary.get("config", {}))
    allowed = set(PipelineConfig.__dataclass_fields__) if hasattr(PipelineConfig, "__dataclass_fields__") else values.keys()
    values = {key: value for key, value in values.items() if key in allowed}
    for key in ("train_root", "output_root", "preview_root"):
        if key in values:
            values[key] = Path(values[key])
    return PipelineConfig(**values)


def _matrix_worker(payload: Tuple[JsonDict, PipelineConfig]) -> JsonDict:
    record, config = payload
    result = process_record(record, config)
    matrix = np.asarray(result.matrix, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    name = image_key(record["file_name"])
    return {
        "image": name,
        "scanned_image": image_key(result.metadata["file_name"]),
        "original_file_name": record["file_name"],
        "scanned_file_name": result.metadata["file_name"],
        "mode": result.report.get("mode"),
        "original_width": int(result.report["exif"]["actual_width"]),
        "original_height": int(result.report["exif"]["actual_height"]),
        "scanned_width": int(result.report["output_width"]),
        "scanned_height": int(result.report["output_height"]),
        "forward_matrix": matrix.tolist(),
        "inverse_matrix": inverse.tolist(),
        "transforms": result.report.get("transforms", []),
        "validation": result.report.get("validation", {}),
    }


def build_matrix_cache(
    records: Sequence[JsonDict],
    config: PipelineConfig,
    output_path: Path,
    workers: int = 1,
    limit: Optional[int] = None,
) -> List[JsonDict]:
    selected = list(records[: int(limit)]) if limit is not None else list(records)
    if int(workers) <= 1:
        matrices = [_matrix_worker((record, config)) for record in selected]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            matrices = list(executor.map(_matrix_worker, [(record, config) for record in selected], chunksize=4))
    matrices.sort(key=lambda item: item["image"])
    write_jsonl(matrices, output_path)
    return matrices


def load_matrix_cache(path: Path) -> Dict[str, JsonDict]:
    return {image_key(record["image"]): record for record in read_jsonl(path)}


def bbox_corners(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denom = transformed[:, 2:3].copy()
    denom[np.abs(denom) < 1e-9] = 1e-9
    return transformed[:, :2] / denom


def clip_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]


def map_bbox_to_original(
    bbox: Sequence[float],
    matrix_record: JsonDict,
    clip_input: bool = True,
    min_size: int = 1,
) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    scanned_width = int(matrix_record["scanned_width"])
    scanned_height = int(matrix_record["scanned_height"])
    original_width = int(matrix_record["original_width"])
    original_height = int(matrix_record["original_height"])
    box = [float(value) for value in bbox]
    if clip_input:
        box = clip_box(box, scanned_width, scanned_height)
    points = transform_points(bbox_corners(box), np.asarray(matrix_record["inverse_matrix"], dtype=np.float64))
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    clipped = clip_box([math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)], original_width, original_height)
    out = [int(round(value)) for value in clipped]
    if out[2] <= out[0]:
        out[2] = min(original_width, out[0] + int(min_size))
    if out[3] <= out[1]:
        out[3] = min(original_height, out[1] + int(min_size))
    return out


def map_regions_to_original(
    regions: Sequence[JsonDict],
    matrix_record: JsonDict,
    clip_input: bool = True,
) -> Tuple[List[JsonDict], List[JsonDict]]:
    mapped: List[JsonDict] = []
    issues: List[JsonDict] = []
    for region_index, region in enumerate(regions):
        new_region = copy.deepcopy(region)
        try:
            new_region["bbox"] = map_bbox_to_original(new_region["bbox"], matrix_record, clip_input=clip_input)
            mapped.append(new_region)
        except Exception as exc:
            issues.append({"region_index": region_index, "error": str(exc), "region": region})
    return mapped, issues


def map_submission_to_original(
    scanned_submission_csv: Path,
    matrix_cache_jsonl: Path,
    output_csv: Path,
    output_image_column: str = "basename",
    clip_input_boxes: bool = True,
) -> JsonDict:
    matrix_by_image = load_matrix_cache(matrix_cache_jsonl)
    rows_in = read_submission_csv(scanned_submission_csv)
    rows_out: List[JsonDict] = []
    issues: List[JsonDict] = []

    for row in rows_in:
        key = image_key(row["image"])
        matrix_record = matrix_by_image.get(key)
        if matrix_record is None:
            issues.append({"image": row["image"], "issue": "missing matrix"})
            rows_out.append({"image": key if output_image_column == "basename" else row["image"], "regions": []})
            continue
        mapped_regions, region_issues = map_regions_to_original(
            row.get("regions", []),
            matrix_record,
            clip_input=clip_input_boxes,
        )
        for issue in region_issues:
            issue["image"] = row["image"]
        issues.extend(region_issues)
        image_value = key if output_image_column == "basename" else matrix_record["original_file_name"]
        rows_out.append({"image": image_value, "regions": mapped_regions})

    write_submission_csv(rows_out, output_csv)
    report = {
        "input_csv": str(scanned_submission_csv),
        "output_csv": str(output_csv),
        "matrix_cache": str(matrix_cache_jsonl),
        "rows_in": len(rows_in),
        "rows_out": len(rows_out),
        "regions_in": sum(len(row.get("regions", [])) for row in rows_in),
        "regions_out": sum(len(row.get("regions", [])) for row in rows_out),
        "issues_count": len(issues),
        "issues": issues[:100],
    }
    Path(output_csv).with_suffix(".mapping_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def audit_cache_against_processing_report(
    matrix_records: Sequence[JsonDict],
    processing_report_path: Path = Path("dataset/preprocessed_scanner_v1/test/processing_report.jsonl"),
) -> JsonDict:
    report_by_image = {image_key(record["output_file_name"]): record for record in read_jsonl(processing_report_path)}
    issues: List[JsonDict] = []
    mode_counts: Dict[str, int] = {}
    for item in matrix_records:
        key = image_key(item["image"])
        expected = report_by_image.get(key)
        mode_counts[str(item.get("mode"))] = mode_counts.get(str(item.get("mode")), 0) + 1
        if expected is None:
            issues.append({"image": key, "issue": "missing original processing report"})
            continue
        if int(item["scanned_width"]) != int(expected["output_width"]) or int(item["scanned_height"]) != int(expected["output_height"]):
            issues.append(
                {
                    "image": key,
                    "issue": "output size mismatch",
                    "cache_size": [item["scanned_width"], item["scanned_height"]],
                    "report_size": [expected["output_width"], expected["output_height"]],
                }
            )
        if str(item.get("mode")) != str(expected.get("mode")):
            issues.append({"image": key, "issue": "mode mismatch", "cache": item.get("mode"), "report": expected.get("mode")})
    return {
        "records_checked": len(matrix_records),
        "mode_counts": mode_counts,
        "issues_count": len(issues),
        "issues": issues[:100],
    }


def default_output_config() -> JsonDict:
    return {
        "bbox_csv": "old_bbox_bbox_only.csv",
        "dataset_root": "dataset",
        "preprocessed_summary": "dataset/preprocessed_scanner_v1/summary.json",
        "processing_report": "dataset/preprocessed_scanner_v1/test/processing_report.jsonl",
        "matrix_cache": "outputs/test_submission_inverse_mapping/test_inverse_matrices.jsonl",
        "mapped_submission": "outputs/test_submission_inverse_mapping/submission_mapped_to_original.csv",
    }


def main():
    print("=" * 60)
    print("PIPELINE: MAP SUBMISSION TỪ TEST SCAN VỀ TEST GỐC")
    print("=" * 60)

    # 1. Load metadata test gốc và config preprocess
    print("\n[1] Đang nạp cấu hình và thông tin file test...")
    PATHS = default_output_config()
    
    BBOX_CSV = Path(PATHS["bbox_csv"])
    DATASET_ROOT = Path(PATHS["dataset_root"])
    SUMMARY_PATH = Path(PATHS["preprocessed_summary"])
    PROCESSING_REPORT = Path(PATHS["processing_report"])
    MATRIX_CACHE = Path(PATHS["matrix_cache"])
    
    records = []
    config = None
    try:
        records = load_test_records_from_bbox_csv(BBOX_CSV, DATASET_ROOT)
        config = pipeline_config_from_preprocessed_summary(SUMMARY_PATH)
        print(f" -> Đã nạp {len(records)} test records.")
        if hasattr(config, 'train_root'):
            print(f" -> Train root sử dụng: {config.train_root}")
    except FileNotFoundError as e:
        print(f" [!] Bỏ qua bước nạp file test do không tìm thấy file gốc: {e}")
    except Exception as e:
        print(f" [!] Lỗi khi nạp dữ liệu: {e}")

    # 2. Tạo hoặc load cache ma trận
    print("\n[2] Đang xử lý matrix cache...")
    REBUILD_MATRIX_CACHE = False
    WORKERS = 1
    matrices = []
    
    if MATRIX_CACHE.exists() and not REBUILD_MATRIX_CACHE:
        print(f" -> Load cache từ: {MATRIX_CACHE}")
        matrices = list(load_matrix_cache(MATRIX_CACHE).values())
        print(f" -> Đã load {len(matrices)} records ma trận.")
    elif records and config:
        print(f" -> Đang rebuild matrix cache tới: {MATRIX_CACHE}")
        try:
            matrices = build_matrix_cache(
                records=records,
                config=config,
                output_path=MATRIX_CACHE,
                workers=WORKERS,
            )
            print(f" -> Rebuild hoàn tất ({len(matrices)} records).")
        except Exception as e:
            print(f" [!] Lỗi khi rebuild cache: {e}")
    else:
        print(" [!] Không thể load hoặc build matrix cache do thiếu dữ liệu hoặc file cache.")
        
    if matrices:
        sample = {k: matrices[0].get(k) for k in ["image", "mode", "original_width", "original_height", "scanned_width", "scanned_height"]}
        print(f" -> Dữ liệu mẫu (ảnh đầu tiên): {sample}")

    # 3. Audit cache với report preprocess cũ
    if matrices and PROCESSING_REPORT.exists():
        print("\n[3] Đang audit cache so với bản báo cáo xử lý cũ...")
        try:
            audit = audit_cache_against_processing_report(matrices, PROCESSING_REPORT)
            CACHE_AUDIT_PATH = MATRIX_CACHE.parent / "cache_audit.json"
            CACHE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f" -> Audit hoàn tất! Đã kiểm tra {audit.get('records_checked', 0)} records.")
            print(f" -> Số lỗi (issues) phát hiện: {audit.get('issues_count', 0)}")
            print(f" -> Báo cáo audit đã được lưu tại: {CACHE_AUDIT_PATH}")
        except Exception as e:
            print(f" [!] Lỗi khi audit cache: {e}")
    else:
         print("\n[3] Bỏ qua bước audit do không tìm thấy Processing Report.")

    # 4. Map submission scan về submission gốc
    print("\n[4] Đang thực hiện map submission (chuyển đổi tọa độ)...")
    SCANNED_SUBMISSION_CSV = Path("submission_on_scanned_test.csv")
    OUTPUT_CSV = Path(PATHS["mapped_submission"])
    
    if SCANNED_SUBMISSION_CSV.exists() and MATRIX_CACHE.exists():
        try:
            report = map_submission_to_original(
                scanned_submission_csv=SCANNED_SUBMISSION_CSV,
                matrix_cache_jsonl=MATRIX_CACHE,
                output_csv=OUTPUT_CSV,
                output_image_column="basename",
                clip_input_boxes=True,
            )
            print(f" -> Quá trình map thành công!")
            print(f" -> Đã xử lý {report.get('rows_in')} ảnh (tổng cộng {report.get('regions_in')} regions).")
            print(f" -> Đã xuất ra {report.get('rows_out')} ảnh (tổng cộng {report.get('regions_out')} regions).")
            print(f" -> Lỗi trong lúc map: {report.get('issues_count')} issues.")
            print(f" -> File submission đã map thành công và lưu tại: {OUTPUT_CSV}")
            
            # 5. Kiểm tra nhanh output
            print("\n[5] Preview kết quả Output:")
            with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in zip(range(3), reader):
                    preview_regions = str(row['regions'])
                    if len(preview_regions) > 100:
                        preview_regions = preview_regions[:100] + "... (truncated)"
                    print(f"  - Ảnh: {row['image']}")
                    print(f"    Regions: {preview_regions}")
        except Exception as e:
            print(f" [!] Đã xảy ra lỗi khi map submission: {e}")
    else:
        print(f" [!] Bỏ qua mapping: Cần phải có file prediction '{SCANNED_SUBMISSION_CSV}' và matrix cache.")
        print(" Vui lòng chạy dự đoán mô hình của bạn trên tập test đã scan, đổi tên file CSV thành 'submission_on_scanned_test.csv', và chạy lại script này.")

if __name__ == "__main__":
    main()
