#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script cắt ảnh dòng (Text Line Crops) từ tập dữ liệu trang quét (Scanned Page Dataset)
dựa trên thông tin Bounding Box (bbox) trong tệp metadata.jsonl.

Đầu ra của script được chuẩn hóa hoàn toàn theo định dạng yêu cầu của script huấn luyện
train_kansallisarkisto_hpa_single_train.py:
  OUTPUT_DIR/
    train/
      metadata.jsonl
      images/
        ... (các tệp ảnh dòng đã cắt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Danh sách các nhãn vùng mặc định cần trích xuất cho huấn luyện TrOCR HPA
DEFAULT_TARGET_TYPES = ("handwritten", "printed", "annotation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cắt các vùng ảnh chữ dòng theo bbox từ file metadata.jsonl trang quét."
    )
    parser.add_argument(
        "--input-metadata",
        type=Path,
        default=Path("metadata.jsonl"),
        help="Đường dẫn tới file metadata.jsonl đầu vào của tập trang quét (Mặc định: metadata.jsonl).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Thư mục gốc chứa ảnh trang quét gốc (Mặc định: thư mục hiện tại).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Thư mục gốc lưu tập dữ liệu đầu ra sau khi crop (Mặc định: dataset).",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="train",
        help="Tên thư mục split con trong output-dir (Mặc định: train).",
    )
    parser.add_argument(
        "--target-types",
        type=str,
        default="handwritten,printed,annotation",
        help="Danh sách các loại vùng văn bản cần trích xuất, phân tách bằng dấu phẩy (Mặc định: handwritten,printed,annotation).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Lượng margin (pixel) mở rộng thêm ra ngoài viền bbox (Mặc định: 0).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=3,
        help="Kích thước chiều rộng/chiều cao tối thiểu của ảnh crop (Mặc định: 3px).",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Định dạng ảnh lưu ra (jpg hoặc png) (Mặc định: jpg).",
    )
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return text.replace("\u00a0", " ").strip()


def resolve_page_image_path(base_dir: Path, file_name: str) -> Optional[Path]:
    """Tìm kiếm vị trí thực tế của tệp ảnh trang quét gốc."""
    raw_path = Path(file_name)
    candidates = [
        base_dir / raw_path,
        base_dir / raw_path.name,
        base_dir / "images" / raw_path.name,
    ]
    if raw_path.parts and raw_path.parts[0] != "images":
        candidates.append(base_dir / "images" / raw_path)

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def crop_lines_from_metadata(
    input_metadata: Path,
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    target_types: Tuple[str, ...],
    margin: int = 0,
    min_size: int = 3,
    image_format: str = "jpg",
) -> Dict[str, Any]:
    if not input_metadata.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp metadata đầu vào: {input_metadata}")

    # Chuẩn bị cấu trúc thư mục đầu ra theo đúng quy chuẩn train script
    split_dir = output_dir / split_name
    images_output_dir = split_dir / "images"
    out_metadata_path = split_dir / "metadata.jsonl"

    images_output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "total_pages": 0,
        "missing_pages": 0,
        "total_regions_found": 0,
        "total_crops_saved": 0,
        "skipped_non_target_type": 0,
        "skipped_empty_text": 0,
        "skipped_invalid_bbox": 0,
        "label_counts": Counter(),
    }

    out_records: List[Dict[str, Any]] = []

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with input_metadata.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    stats["total_pages"] = len(lines)
    iterator = tqdm(lines, desc="Đang xử lý cắt dòng ảnh") if has_tqdm else lines

    for line_idx, line in enumerate(iterator):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[CẢNH BÁO] Lỗi định dạng JSON ở dòng {line_idx + 1}: {exc}", file=sys.stderr)
            continue

        raw_file_name = item.get("file_name") or item.get("image") or item.get("path")
        if not raw_file_name:
            continue

        page_img_path = resolve_page_image_path(input_dir, raw_file_name)
        if not page_img_path:
            stats["missing_pages"] += 1
            continue

        try:
            page_img = Image.open(page_img_path).convert("RGB")
        except Exception as exc:
            print(f"[CẢNH BÁO] Không thể mở ảnh {page_img_path}: {exc}", file=sys.stderr)
            stats["missing_pages"] += 1
            continue

        img_w, img_h = page_img.size
        regions = item.get("regions", [])
        page_stem = Path(raw_file_name).stem

        for r_idx, region in enumerate(regions):
            stats["total_regions_found"] += 1
            region_type = str(region.get("type", region.get("label", "")))
            if region_type not in target_types:
                stats["skipped_non_target_type"] += 1
                continue

            text = normalize_text(region.get("text", ""))
            if not text:
                stats["skipped_empty_text"] += 1
                continue

            bbox = region.get("bbox")
            if not bbox or len(bbox) != 4:
                stats["skipped_invalid_bbox"] += 1
                continue

            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Áp dụng margin điều chỉnh nếu có
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(img_w, xmax + margin)
            ymax = min(img_h, ymax + margin)

            crop_w = xmax - xmin
            crop_h = ymax - ymin

            if crop_w < min_size or crop_h < min_size:
                stats["skipped_invalid_bbox"] += 1
                continue

            # Cắt ảnh dòng
            cropped_img = page_img.crop((xmin, ymin, xmax, ymax))

            crop_filename = f"{page_stem}_crop_{r_idx}_{region_type}.{image_format}"
            crop_rel_path = f"images/{crop_filename}"
            save_path = images_output_dir / crop_filename

            if image_format.lower() in ("jpg", "jpeg"):
                cropped_img.save(save_path, quality=95)
            else:
                cropped_img.save(save_path)

            record = {
                "file_name": crop_rel_path,
                "image": crop_rel_path,
                "type": region_type,
                "label": region_type,
                "text": text,
                "parent_image": raw_file_name,
                "bbox": [xmin, ymin, xmax, ymax],
            }

            out_records.append(record)
            stats["total_crops_saved"] += 1
            stats["label_counts"][region_type] += 1

    # Ghi tệp metadata.jsonl đầu ra
    with out_metadata_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def main():
    args = parse_args()
    target_types = tuple(t.strip() for t in args.target_types.split(",") if t.strip())

    print("=" * 70)
    print(" BẮT ĐẦU CẮT ẢNH DÒNG VĂN BẢN (LINE CROPPER FOR SCANNED DATASET)")
    print("=" * 70)
    print(f" File metadata đầu vào : {args.input_metadata}")
    print(f" Thư mục ảnh trang quét  : {args.input_dir}")
    print(f" Thư mục tập dữ liệu xuất: {args.output_dir / args.split_name}")
    print(f" Các loại vùng chọn     : {target_types}")
    print("=" * 70)

    stats = crop_lines_from_metadata(
        input_metadata=args.input_metadata,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name=args.split_name,
        target_types=target_types,
        margin=args.margin,
        min_size=args.min_size,
        image_format=args.image_format,
    )

    print("\n" + "=" * 70)
    print(" HOÀN THÀNH TIẾN TRÌNH CẮT ẢNH")
    print("=" * 70)
    print(f" Tổng số trang quét đã đọc  : {stats['total_pages']:,}")
    print(f" Số trang quét không tìm thấy: {stats['missing_pages']:,}")
    print(f" Tổng số vùng tìm thấy      : {stats['total_regions_found']:,}")
    print(f" Tổng số ảnh dòng đã lưu     : {stats['total_crops_saved']:,}")
    print(" Chi tiết số ảnh crop theo nhãn:")
    for label, count in stats["label_counts"].items():
        print(f"   - {label}: {count:,} ảnh")
    print(f" File metadata kết quả      : {args.output_dir / args.split_name / 'metadata.jsonl'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
