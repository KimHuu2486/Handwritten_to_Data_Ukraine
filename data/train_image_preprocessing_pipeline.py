"""
# Train Image Preprocessing Pipeline

Notebook này chỉ đọc dữ liệu trong `dataset/train`: `dataset/train/metadata.jsonl` và `dataset/train/images/`. Pipeline không ghi đè dữ liệu gốc; ảnh và metadata sau xử lý sẽ được ghi ra `outputs/train_preprocessed/` khi bật cell chạy toàn bộ.

Mục tiêu chính là cải thiện chất lượng ảnh theo hướng scanner-like có kiểm soát, đồng thời cập nhật chính xác metadata bbox sau mọi biến đổi hình học.

## Nguyên tắc thiết kế

- Chỉ dùng page warp khi detect trang giấy đủ tin cậy và bbox text nằm hợp lý bên trong trang.
- Nếu không detect được trang giấy, crop theo union bbox text và thêm margin/padding để không mất chữ.
- Mọi biến đổi hình học đều được gom vào ma trận `3x3`; bbox/polygon được biến đổi bằng cùng ma trận đó.
- Page detector thử nhiều candidate theo thứ tự score, nhưng chỉ accept candidate khi toàn bộ bbox còn hợp lệ.
- Enhancement ảnh được áp sau geometry: giảm bóng, ổn định sáng, CLAHE nhẹ, denoise nhẹ, sharpen nhẹ.
- Morphology chỉ dùng cho detection/deskew phụ trợ, không biến đổi trực tiếp nét chữ trong ảnh train cuối.
- Các cell chạy thử và chạy full dataset đều bị tắt mặc định để tránh chạy khi thiếu dependency.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Kiểm tra dependency trước khi import các thư viện bên ngoài
required = ['cv2', 'numpy', 'PIL']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(
        'Missing dependencies: ' + ', '.join(missing) +
        '. Install at least: opencv-python numpy pillow.'
    )

import cv2
import numpy as np
from PIL import Image, ImageOps

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class PipelineConfig:
    train_root: Path = Path("dataset/train")
    output_root: Path = Path("outputs/train_preprocessed")
    preview_root: Path = Path("outputs/train_preprocess_preview")

    detection_max_side: int = 1200
    page_min_area_ratio: float = 0.20
    page_max_area_ratio: float = 0.995
    page_min_confidence: float = 0.62
    page_min_text_center_inside: float = 0.85
    page_min_text_corner_inside: float = 0.35
    page_expand_ratio: float = 0.005
    reject_page_warp_if_bbox_clipped: bool = True
    page_max_dark_fraction: float = 0.18
    page_line_detection_enabled: bool = True
    page_min_edge_support: float = 0.035
    page_candidate_try_limit: int = 4
    allow_region_drop: bool = False
    max_region_drop_ratio: float = 0.0
    drop_region_visible_ratio: float = 0.98

    crop_margin_ratio: float = 0.055
    crop_min_margin_px: int = 24
    crop_allow_padding: bool = False
    keep_if_crop_area_ratio_gt: float = 0.92
    include_region_types_for_crop: Optional[Tuple[str, ...]] = None

    final_max_side: int = 2200
    target_median_bbox_height: int = 80
    max_upscale: float = 1.45

    shadow_strength: float = 0.75
    clahe_clip_limit: float = 1.8
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    denoise: str = "median"  # none | median | nlm_luminance
    sharpen_amount: float = 0.35
    sharpen_sigma: float = 1.0

    deskew_enabled: bool = False
    deskew_max_abs_angle: float = 8.0
    deskew_min_segments: int = 10

    add_polygon_to_regions: bool = True
    min_bbox_visible_ratio: float = 0.98
    jpg_quality: int = 95
    jpg_optimize: bool = False


@dataclass
class ProcessResult:
    image: np.ndarray
    metadata: JsonDict
    report: JsonDict
    matrix: np.ndarray


def read_jsonl(path: Path) -> List[JsonDict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_image_path(record: JsonDict, train_root: Path) -> Path:
    return Path(train_root) / record["file_name"]


def bbox_or_none(region: JsonDict) -> Optional[np.ndarray]:
    box = region.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    arr = np.asarray(box, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    x1, y1, x2, y2 = arr.tolist()
    if x2 <= x1 or y2 <= y1:
        return None
    return arr


def extract_boxes(
    record: JsonDict,
    region_types: Optional[Sequence[str]] = None,
) -> List[Tuple[int, np.ndarray]]:
    allowed = set(region_types) if region_types is not None else None
    boxes: List[Tuple[int, np.ndarray]] = []
    for idx, region in enumerate(record.get("regions", [])):
        if allowed is not None and region.get("type") not in allowed:
            continue
        box = bbox_or_none(region)
        if box is not None:
            boxes.append((idx, box))
    return boxes


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clip_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]


def bbox_corners(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    out = pts_h @ np.asarray(matrix, dtype=np.float64).T
    denom = out[:, 2:3].copy()
    denom[np.abs(denom) < 1e-9] = 1e-9
    return out[:, :2] / denom


def transformed_region_geometries(record: JsonDict, matrix: np.ndarray) -> List[JsonDict]:
    transformed: List[JsonDict] = []
    for idx, box in extract_boxes(record):
        polygon = transform_points(bbox_corners(box), matrix)
        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)
        transformed.append(
            {
                "index": idx,
                "bbox": np.asarray([x1, y1, x2, y2], dtype=np.float64),
                "polygon": polygon,
                "original_bbox": box,
            }
        )
    return transformed


def exif_orientation_matrix(width: int, height: int, orientation: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    w, h = float(width), float(height)
    if orientation == 2:
        matrix = np.asarray([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        return matrix, (width, height)
    if orientation == 3:
        matrix = np.asarray([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64)
        return matrix, (width, height)
    if orientation == 4:
        matrix = np.asarray([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float64)
        return matrix, (width, height)
    if orientation == 5:
        matrix = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        return matrix, (height, width)
    if orientation == 6:
        matrix = np.asarray([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        return matrix, (height, width)
    if orientation == 7:
        matrix = np.asarray([[0, -1, h], [-1, 0, w], [0, 0, 1]], dtype=np.float64)
        return matrix, (height, width)
    if orientation == 8:
        matrix = np.asarray([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64)
        return matrix, (height, width)
    return np.eye(3, dtype=np.float64), (width, height)


def load_image_rgb_with_exif(record: JsonDict, config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, JsonDict]:
    path = resolve_image_path(record, config.train_root)
    with Image.open(path) as im:
        raw_width, raw_height = im.size
        orientation = int(im.getexif().get(274, 1))
        matrix, expected_size = exif_orientation_matrix(raw_width, raw_height, orientation)
        im = ImageOps.exif_transpose(im).convert("RGB")
        image = np.asarray(im)
    actual_height, actual_width = image.shape[:2]
    info = {
        "path": str(path),
        "raw_width": raw_width,
        "raw_height": raw_height,
        "exif_orientation": orientation,
        "expected_width": expected_size[0],
        "expected_height": expected_size[1],
        "actual_width": actual_width,
        "actual_height": actual_height,
    }
    return image, matrix, info


def resize_for_detection(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 0.999:
        return image, 1.0
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]
    ordered = np.zeros((4, 2), dtype=np.float64)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def expand_quad(quad: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
    if ratio <= 0:
        return quad.astype(np.float64)
    center = quad.mean(axis=0, keepdims=True)
    expanded = center + (quad - center) * (1.0 + float(ratio))
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype(np.float64)


def page_output_size(quad: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = quad
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    out_w = max(16, int(round(max(width_a, width_b))))
    out_h = max(16, int(round(max(height_a, height_b))))
    return out_w, out_h


def estimate_background_rgb(image: np.ndarray, border_frac: float = 0.04) -> Tuple[int, int, int]:
    height, width = image.shape[:2]
    by = max(4, int(round(height * border_frac)))
    bx = max(4, int(round(width * border_frac)))
    strips = [
        image[:by, :, :],
        image[-by:, :, :],
        image[:, :bx, :],
        image[:, -bx:, :],
    ]
    pixels = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    med = np.median(pixels, axis=0)
    return tuple(int(v) for v in med.tolist())


def shadow_normalized_luminance(image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    if sigma is None:
        sigma = max(18.0, float(max(image.shape[:2])) / 45.0)
    background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    background = np.maximum(background, 8).astype(np.uint8)
    return cv2.divide(l_channel, background, scale=255)


def dark_fraction(image: np.ndarray, threshold: int = 45) -> float:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    return float(np.mean(l_channel < int(threshold)))


def text_coverage_in_quad(boxes: Sequence[JsonDict], quad: np.ndarray) -> Tuple[float, float]:
    if not boxes:
        return 0.0, 0.0
    contour = np.asarray(quad, dtype=np.float32)
    center_hits = 0
    corner_hits = 0
    total_corners = 0
    for item in boxes:
        box = np.asarray(item["bbox"], dtype=np.float64)
        x1, y1, x2, y2 = box.tolist()
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            center_hits += 1
        for point in bbox_corners(box):
            total_corners += 1
            if cv2.pointPolygonTest(contour, tuple(point.tolist()), False) >= 0:
                corner_hits += 1
    return center_hits / len(boxes), corner_hits / max(1, total_corners)


def union_bbox_from_items(boxes: Sequence[JsonDict]) -> Optional[np.ndarray]:
    if not boxes:
        return None
    arr = np.asarray([item["bbox"] for item in boxes], dtype=np.float64)
    return np.asarray(
        [arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()],
        dtype=np.float64,
    )


def text_margin_score(boxes: Sequence[JsonDict], quad: np.ndarray) -> float:
    union = union_bbox_from_items(boxes)
    if union is None:
        return 0.0
    page_box = np.asarray(
        [quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()],
        dtype=np.float64,
    )
    union_area = box_area(union)
    page_area = box_area(page_box)
    if union_area <= 0 or page_area <= 0:
        return 0.0
    area_ratio = page_area / union_area
    area_score = float(np.clip((area_ratio - 1.03) / 1.4, 0.0, 1.0))
    margins = np.asarray(
        [
            union[0] - page_box[0],
            union[1] - page_box[1],
            page_box[2] - union[2],
            page_box[3] - union[3],
        ],
        dtype=np.float64,
    )
    short_side = max(1.0, min(page_box[2] - page_box[0], page_box[3] - page_box[1]))
    margin_score = float(np.clip(np.percentile(margins, 20) / (short_side * 0.035), 0.0, 1.0))
    return 0.55 * area_score + 0.45 * margin_score


def quad_edge_support(quad: np.ndarray, edges: np.ndarray) -> float:
    height, width = edges.shape[:2]
    line_mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(quad).astype(np.int32)
    for idx in range(4):
        p1 = tuple(pts[idx].tolist())
        p2 = tuple(pts[(idx + 1) % 4].tolist())
        cv2.line(line_mask, p1, p2, 255, 3, cv2.LINE_AA)
    if cv2.countNonZero(line_mask) == 0:
        return 0.0
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_hits = cv2.dilate(edges, edge_kernel, iterations=1)
    return float(cv2.countNonZero(cv2.bitwise_and(edge_hits, line_mask)) / cv2.countNonZero(line_mask))


def quad_mask_support(quad: np.ndarray, mask: np.ndarray) -> float:
    height, width = mask.shape[:2]
    region_mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(region_mask, pts, 255)
    denom = cv2.countNonZero(region_mask)
    if denom == 0:
        return 0.0
    return float(cv2.countNonZero(cv2.bitwise_and(mask, region_mask)) / denom)


def line_coeff_from_segment(segment: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in segment]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm <= 1e-6:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray([a / norm, b / norm, c / norm], dtype=np.float64)


def intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> Optional[np.ndarray]:
    a1, b1, c1 = line_a.tolist()
    a2, b2, c2 = line_b.tolist()
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    if not np.isfinite([x, y]).all():
        return None
    return np.asarray([x, y], dtype=np.float64)


def line_quad_candidates(edges: np.ndarray, width: int, height: int) -> List[np.ndarray]:
    min_len = max(80, int(round(min(width, height) * 0.22)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(60, int(round(min(width, height) * 0.055))),
        minLineLength=min_len,
        maxLineGap=max(18, int(round(min(width, height) * 0.035))),
    )
    if lines is None:
        return []

    horizontal: List[JsonDict] = []
    vertical: List[JsonDict] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in raw]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        item = {
            "segment": np.asarray([x1, y1, x2, y2], dtype=np.float64),
            "line": line_coeff_from_segment([x1, y1, x2, y2]),
            "length": length,
            "x_mid": (x1 + x2) * 0.5,
            "y_mid": (y1 + y2) * 0.5,
        }
        if angle <= 28 or angle >= 152:
            horizontal.append(item)
        elif 62 <= angle <= 118:
            vertical.append(item)

    if len(horizontal) < 2 or len(vertical) < 2:
        return []

    horizontal = sorted(horizontal, key=lambda item: (-item["length"], item["y_mid"]))
    vertical = sorted(vertical, key=lambda item: (-item["length"], item["x_mid"]))
    top_lines = sorted(horizontal[:24], key=lambda item: item["y_mid"])[:4]
    bottom_lines = sorted(horizontal[:24], key=lambda item: item["y_mid"])[-4:]
    left_lines = sorted(vertical[:24], key=lambda item: item["x_mid"])[:4]
    right_lines = sorted(vertical[:24], key=lambda item: item["x_mid"])[-4:]

    quads: List[np.ndarray] = []
    margin = max(width, height) * 0.25
    for top in top_lines:
        for bottom in bottom_lines:
            if bottom["y_mid"] - top["y_mid"] < height * 0.22:
                continue
            for left in left_lines:
                for right in right_lines:
                    if right["x_mid"] - left["x_mid"] < width * 0.22:
                        continue
                    tl = intersect_lines(top["line"], left["line"])
                    tr = intersect_lines(top["line"], right["line"])
                    br = intersect_lines(bottom["line"], right["line"])
                    bl = intersect_lines(bottom["line"], left["line"])
                    if tl is None or tr is None or br is None or bl is None:
                        continue
                    quad = order_quad(np.asarray([tl, tr, br, bl], dtype=np.float64))
                    if not np.isfinite(quad).all():
                        continue
                    if (quad[:, 0] < -margin).any() or (quad[:, 0] > width + margin).any():
                        continue
                    if (quad[:, 1] < -margin).any() or (quad[:, 1] > height + margin).any():
                        continue
                    if not cv2.isContourConvex(quad.astype(np.float32)):
                        continue
                    quads.append(quad)
    return quads[:48]


def detect_page_quad(
    image: np.ndarray,
    boxes_current: Sequence[JsonDict],
    config: PipelineConfig,
) -> Optional[JsonDict]:
    small, scale = resize_for_detection(image, config.detection_max_side)
    small_h, small_w = small.shape[:2]
    gray = shadow_normalized_luminance(small)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidate_sources: List[Tuple[str, np.ndarray, bool]] = []

    med = float(np.median(gray))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med + 20))
    edges = cv2.Canny(gray, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate_sources.append(("edge", edges, False))

    _, paper_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    large_k = max(9, int(round(max(small_h, small_w) * 0.025)) | 1)
    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (large_k, large_k))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate_sources.append(("luminance_mask", paper_mask, True))

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    sat_limit = int(max(45, min(115, np.percentile(saturation, 60) + 20)))
    val_limit = int(max(70, np.percentile(value, 20)))
    paper_like = np.where((saturation <= sat_limit) & (value >= val_limit), 255, 0).astype(np.uint8)
    paper_like = cv2.morphologyEx(paper_like, cv2.MORPH_CLOSE, large_kernel, iterations=2)
    paper_like = cv2.morphologyEx(paper_like, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate_sources.append(("paper_like_mask", paper_like, True))

    scaled_boxes = []
    for item in boxes_current:
        scaled = dict(item)
        scaled["bbox"] = np.asarray(item["bbox"], dtype=np.float64) * scale
        scaled_boxes.append(scaled)

    image_area = float(small_w * small_h)
    candidates: List[JsonDict] = []
    seen: set = set()

    def add_quad_candidate(
        source_name: str,
        quad_kind: str,
        raw_quad: np.ndarray,
        source_area: Optional[float] = None,
    ) -> None:
        quad = order_quad(raw_quad)
        if not np.isfinite(quad).all():
            return
        out_w, out_h = page_output_size(quad)
        if min(out_w, out_h) < 64:
            return
        aspect = max(float(out_w) / float(out_h), float(out_h) / float(out_w))
        if aspect > 3.2:
            return
        poly_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        area_ratio = poly_area / max(1.0, image_area)
        if area_ratio < config.page_min_area_ratio or area_ratio > config.page_max_area_ratio:
            return
        if not scaled_boxes and area_ratio < max(0.35, config.page_min_area_ratio):
            return
        key = tuple(np.round(quad.reshape(-1) / 8.0).astype(int).tolist())
        if key in seen:
            return
        seen.add(key)

        rect_area = float(out_w * out_h)
        if source_area is None:
            rectangularity = float(np.clip(poly_area / max(1.0, rect_area), 0.0, 1.0))
        else:
            rectangularity = float(np.clip(source_area / max(1.0, rect_area), 0.0, 1.0))
        if scaled_boxes:
            center_inside, corner_inside = text_coverage_in_quad(scaled_boxes, quad)
            if center_inside < config.page_min_text_center_inside:
                return
            if corner_inside < config.page_min_text_corner_inside:
                return
        else:
            center_inside, corner_inside = 1.0, 1.0

        edge_support = quad_edge_support(quad, edges)
        paper_support = max(
            quad_mask_support(quad, paper_mask),
            quad_mask_support(quad, paper_like),
        )
        area_score = float(np.clip(area_ratio / 0.70, 0.0, 1.0))
        text_score = 0.72 * center_inside + 0.28 * corner_inside
        edge_score = float(np.clip(edge_support / 0.22, 0.0, 1.0))
        paper_score = float(np.clip((paper_support - 0.45) / 0.45, 0.0, 1.0))
        margin_score = text_margin_score(scaled_boxes, quad)
        if scaled_boxes:
            legacy_score = 0.34 * area_score + 0.28 * rectangularity + 0.38 * text_score
        else:
            legacy_score = 0.46 * area_score + 0.30 * rectangularity + 0.14 * paper_score + 0.10 * edge_score
        if (
            edge_support < config.page_min_edge_support
            and paper_support < 0.55
            and legacy_score < config.page_min_confidence + 0.04
        ):
            return
        method_bonus = 0.0
        method_penalty = 0.0
        if source_name == "edge" and quad_kind == "contour_quad":
            method_bonus += 0.025
        if source_name == "line_intersections":
            method_bonus += 0.030
        if quad_kind == "min_area_rect":
            method_penalty += 0.035
        if scaled_boxes:
            multi_cue_score = (
                0.18 * area_score
                + 0.24 * rectangularity
                + 0.23 * text_score
                + 0.15 * edge_score
                + 0.12 * paper_score
                + 0.08 * margin_score
                + method_bonus
                - method_penalty
            )
        else:
            multi_cue_score = (
                0.45 * area_score
                + 0.25 * rectangularity
                + 0.18 * paper_score
                + 0.12 * edge_score
                + method_bonus
                - method_penalty
            )
        confidence = max(multi_cue_score, legacy_score + method_bonus - method_penalty)
        candidates.append(
            {
                "quad_small": quad,
                "quad": quad / scale,
                "confidence": float(confidence),
                "area_ratio": float(area_ratio),
                "rectangularity": rectangularity,
                "text_center_inside": float(center_inside),
                "text_corner_inside": float(corner_inside),
                "edge_support": float(edge_support),
                "paper_support": float(paper_support),
                "text_margin_score": float(margin_score),
                "source": source_name,
                "quad_kind": quad_kind,
            }
        )

    for source_name, source_mask, allow_min_area_rect in candidate_sources:
        contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:16]:
            area = float(cv2.contourArea(contour))
            area_ratio = area / max(1.0, image_area)
            if area_ratio < config.page_min_area_ratio or area_ratio > config.page_max_area_ratio:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                add_quad_candidate(source_name, "contour_quad", approx.reshape(4, 2), area)
            if allow_min_area_rect and area_ratio >= config.page_min_area_ratio:
                add_quad_candidate(source_name, "min_area_rect", cv2.boxPoints(cv2.minAreaRect(contour)), area)

    if config.page_line_detection_enabled:
        for quad in line_quad_candidates(edges, small_w, small_h):
            add_quad_candidate("line_intersections", "hough_quad", quad, None)

    if not candidates:
        return None
    qualified = sorted(
        [item for item in candidates if item["confidence"] >= config.page_min_confidence],
        key=lambda item: item["confidence"],
        reverse=True,
    )
    if not qualified:
        return None
    best = dict(qualified[0])
    max_candidates = max(1, int(config.page_candidate_try_limit))
    best["alternates"] = [dict(item) for item in qualified[1:max_candidates]]
    return best


def apply_page_warp(image: np.ndarray, quad: np.ndarray, config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, JsonDict]:
    height, width = image.shape[:2]
    quad = expand_quad(order_quad(quad), config.page_expand_ratio, width, height)
    out_w, out_h = page_output_size(quad)
    dst = np.asarray(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float64,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    bg = estimate_background_rgb(image)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )
    return warped, matrix, {"output_width": out_w, "output_height": out_h}


def compute_text_crop_rect(
    image: np.ndarray,
    boxes_current: Sequence[JsonDict],
    config: PipelineConfig,
) -> Optional[Tuple[int, int, int, int]]:
    if not boxes_current:
        return None
    height, width = image.shape[:2]
    boxes = np.asarray([item["bbox"] for item in boxes_current], dtype=np.float64)
    x1, y1 = boxes[:, :2].min(axis=0)
    x2, y2 = boxes[:, 2:].max(axis=0)
    union_w = max(1.0, x2 - x1)
    union_h = max(1.0, y2 - y1)
    margin_x = max(float(config.crop_min_margin_px), union_w * config.crop_margin_ratio)
    margin_y = max(float(config.crop_min_margin_px), union_h * config.crop_margin_ratio)
    desired = (
        int(math.floor(x1 - margin_x)),
        int(math.floor(y1 - margin_y)),
        int(math.ceil(x2 + margin_x)),
        int(math.ceil(y2 + margin_y)),
    )
    crop_w = max(1, desired[2] - desired[0])
    crop_h = max(1, desired[3] - desired[1])
    crop_area_ratio = float(crop_w * crop_h) / float(max(1, width * height))
    needs_padding = desired[0] < 0 or desired[1] < 0 or desired[2] > width or desired[3] > height
    if crop_area_ratio > config.keep_if_crop_area_ratio_gt and not needs_padding:
        return None
    return desired


def clip_crop_rect_to_image(
    rect: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = rect
    clipped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def crop_with_padding(image: np.ndarray, rect: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, JsonDict]:
    x1, y1, x2, y2 = rect
    height, width = image.shape[:2]
    out_w = max(1, x2 - x1)
    out_h = max(1, y2 - y1)
    bg = estimate_background_rgb(image)
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(width, x2)
    src_y2 = min(height, y2)
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[
            dst_y1 : dst_y1 + (src_y2 - src_y1),
            dst_x1 : dst_x1 + (src_x2 - src_x1),
        ] = image[src_y1:src_y2, src_x1:src_x2]

    matrix = np.asarray([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64)
    info = {
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "output_width": out_w,
        "output_height": out_h,
        "pad_left": max(0, -x1),
        "pad_top": max(0, -y1),
        "pad_right": max(0, x2 - width),
        "pad_bottom": max(0, y2 - height),
    }
    return canvas, matrix, info


def estimate_skew_angle(image: np.ndarray, config: PipelineConfig) -> Optional[JsonDict]:
    gray = shadow_normalized_luminance(image)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    height, width = binary.shape[:2]
    min_len = max(40, int(width * 0.12))
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=18,
    )
    if lines is None:
        return None
    angles: List[float] = []
    weights: List[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) <= config.deskew_max_abs_angle:
            angles.append(angle)
            weights.append(length)
    if len(angles) < config.deskew_min_segments:
        return None
    angles_np = np.asarray(angles, dtype=np.float64)
    median = float(np.median(angles_np))
    mad = float(np.median(np.abs(angles_np - median)))
    if mad > 2.5 or abs(median) < 0.35:
        return None
    return {"angle": median, "mad": mad, "segments": len(angles)}


def rotation_matrix_expand(image: np.ndarray, angle_degrees: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    m2 = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(m2[0, 0])
    sin = abs(m2[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    m2[0, 2] += new_w * 0.5 - center[0]
    m2[1, 2] += new_h * 0.5 - center[1]
    matrix = np.vstack([m2, [0, 0, 1]]).astype(np.float64)
    return matrix, (new_w, new_h)


def apply_deskew_if_enabled(
    image: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, JsonDict]:
    if not config.deskew_enabled:
        return image, np.eye(3, dtype=np.float64), {"applied": False}
    estimate = estimate_skew_angle(image, config)
    if estimate is None:
        return image, np.eye(3, dtype=np.float64), {"applied": False}
    matrix, (new_w, new_h) = rotation_matrix_expand(image, estimate["angle"])
    bg = estimate_background_rgb(image)
    rotated = cv2.warpPerspective(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )
    estimate = dict(estimate)
    estimate.update({"applied": True, "output_width": new_w, "output_height": new_h})
    return rotated, matrix, estimate


def smart_resize(
    image: np.ndarray,
    boxes_current: Sequence[JsonDict],
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, JsonDict]:
    height, width = image.shape[:2]
    max_side = max(height, width)
    scale = 1.0
    reason = "identity"
    if max_side > config.final_max_side:
        scale = float(config.final_max_side) / float(max_side)
        reason = "downscale_max_side"
    elif boxes_current:
        heights = [float(item["bbox"][3] - item["bbox"][1]) for item in boxes_current]
        median_h = float(np.median([h for h in heights if h > 0])) if heights else 0.0
        if 0 < median_h < config.target_median_bbox_height:
            wanted = float(config.target_median_bbox_height) / median_h
            side_limited = float(config.final_max_side) / float(max_side)
            scale = min(config.max_upscale, wanted, max(1.0, side_limited))
            if scale > 1.01:
                reason = "upscale_text_height"
    if abs(scale - 1.0) < 0.01:
        return image, np.eye(3, dtype=np.float64), {"scale": 1.0, "reason": "identity"}
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    matrix = np.asarray([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return resized, matrix, {"scale": float(scale), "reason": reason, "output_width": new_w, "output_height": new_h}


def enhance_image(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    sigma = max(18.0, float(max(image.shape[:2])) / 45.0)
    bg = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = np.maximum(bg, 8).astype(np.uint8)
    normalized = cv2.divide(l_channel, bg, scale=255)
    strength = float(np.clip(config.shadow_strength, 0.0, 1.0))
    l_channel = cv2.addWeighted(l_channel, 1.0 - strength, normalized, strength, 0)
    clahe = cv2.createCLAHE(
        clipLimit=float(config.clahe_clip_limit),
        tileGridSize=tuple(config.clahe_tile_grid),
    )
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    if config.denoise == "median":
        enhanced = cv2.medianBlur(enhanced, 3)
    elif config.denoise == "nlm_luminance":
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.fastNlMeansDenoising(l_channel, None, h=6, templateWindowSize=7, searchWindowSize=21)
        enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)

    amount = float(max(0.0, config.sharpen_amount))
    if amount > 0:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=float(config.sharpen_sigma), sigmaY=float(config.sharpen_sigma))
        enhanced = cv2.addWeighted(enhanced, 1.0 + amount, blur, -amount, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def current_boxes_for_crop(
    record: JsonDict,
    matrix: np.ndarray,
    config: PipelineConfig,
) -> List[JsonDict]:
    if config.include_region_types_for_crop is None:
        indices = None
    else:
        indices = {
            idx
            for idx, _box in extract_boxes(record, region_types=config.include_region_types_for_crop)
        }
    items = transformed_region_geometries(record, matrix)
    if indices is None:
        return items
    return [item for item in items if item["index"] in indices]


def build_transformed_metadata(
    record: JsonDict,
    matrix: np.ndarray,
    width: int,
    height: int,
    file_name: Optional[str],
    config: PipelineConfig,
    drop_region_indices: Optional[Sequence[int]] = None,
) -> JsonDict:
    out = copy.deepcopy(record)
    if file_name is not None:
        out["file_name"] = file_name
    out["image_width"] = int(width)
    out["image_height"] = int(height)

    drop_indices = set(drop_region_indices or [])
    new_regions = []
    for idx, region in enumerate(out.get("regions", [])):
        if idx in drop_indices:
            continue
        box = bbox_or_none(region)
        if box is None:
            new_regions.append(region)
            continue
        polygon = transform_points(bbox_corners(box), matrix)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height)
        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)
        clipped = clip_box([math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)], width, height)
        region["bbox"] = [int(round(v)) for v in clipped]
        if config.add_polygon_to_regions:
            region["polygon"] = [[round(float(x), 2), round(float(y), 2)] for x, y in polygon.tolist()]
        new_regions.append(region)
    out["regions"] = new_regions
    return out


def max_allowed_region_drops(num_boxes: int, config: PipelineConfig) -> int:
    if not config.allow_region_drop or num_boxes <= 0:
        return 0
    return int(math.floor(num_boxes * float(config.max_region_drop_ratio) + 1e-9))


def validate_transformed_boxes(
    record: JsonDict,
    matrix: np.ndarray,
    width: int,
    height: int,
    config: PipelineConfig,
    ignore_region_indices: Optional[Sequence[int]] = None,
) -> JsonDict:
    ignored = set(ignore_region_indices or [])
    original = [(idx, box) for idx, box in extract_boxes(record) if idx not in ignored]
    transformed = [item for item in transformed_region_geometries(record, matrix) if item["index"] not in ignored]
    warnings: List[str] = []
    visible_ratios: List[float] = []
    area_scales: List[float] = []
    per_region: List[JsonDict] = []
    out_of_bounds = 0

    for item in transformed:
        box = item["bbox"].tolist()
        clipped = clip_box(box, width, height)
        visible = box_area(clipped) / max(1e-6, box_area(box))
        visible_ratios.append(float(visible))
        if visible < config.min_bbox_visible_ratio:
            out_of_bounds += 1
        orig_area = box_area(item["original_bbox"])
        area_scales.append(float(box_area(box) / max(1e-6, orig_area)))
        per_region.append(
            {
                "index": int(item["index"]),
                "visible_ratio": float(visible),
                "area_scale": float(area_scales[-1]),
            }
        )

    if out_of_bounds:
        warnings.append(f"{out_of_bounds} transformed boxes are partially outside the output image")
    drop_candidate_indices = [
        int(item["index"])
        for item in per_region
        if item["visible_ratio"] < float(config.drop_region_visible_ratio)
    ]
    allowed_drops = max_allowed_region_drops(len(original), config)
    return {
        "num_boxes_before": len(original),
        "num_boxes_after": len(transformed),
        "num_ignored_boxes": len(ignored),
        "bbox_out_of_bounds": int(out_of_bounds),
        "drop_candidate_indices": drop_candidate_indices,
        "drop_candidate_count": len(drop_candidate_indices),
        "max_allowed_region_drops": allowed_drops,
        "bbox_visible_ratio_min": float(min(visible_ratios)) if visible_ratios else None,
        "bbox_visible_ratio_median": float(np.median(visible_ratios)) if visible_ratios else None,
        "bbox_area_scale_min": float(min(area_scales)) if area_scales else None,
        "bbox_area_scale_max": float(max(area_scales)) if area_scales else None,
        "warnings": warnings,
    }


def output_file_name(record: JsonDict) -> str:
    return f"images/{Path(record['file_name']).name}"


def process_record(record: JsonDict, config: PipelineConfig) -> ProcessResult:
    image, m_exif, exif_info = load_image_rgb_with_exif(record, config)
    matrix_total = m_exif.copy()
    report: JsonDict = {
        "file_name": record.get("file_name"),
        "mode": "keep",
        "transforms": [],
        "exif": exif_info,
        "page_detection": None,
        "dropped_region_indices": [],
        "warnings": [],
    }
    dropped_region_indices: set[int] = set()

    boxes_current = transformed_region_geometries(record, matrix_total)
    page = detect_page_quad(image, boxes_current, config)
    page_applied = False
    if page is not None:
        page_candidates = [page] + list(page.get("alternates", []))
        attempts: List[JsonDict] = []
        for candidate_rank, page_candidate in enumerate(page_candidates):
            warped, m_page, page_info = apply_page_warp(image, page_candidate["quad"], config)
            candidate_matrix = m_page @ matrix_total
            page_validation = validate_transformed_boxes(
                record,
                candidate_matrix,
                warped.shape[1],
                warped.shape[0],
                config,
            )
            page_validation["dark_fraction"] = dark_fraction(warped)
            candidate_report = {
                k: v
                for k, v in page_candidate.items()
                if k not in {"quad", "quad_small", "alternates"}
            }
            candidate_report["rank"] = candidate_rank
            candidate_report["validation_before_accept"] = page_validation
            reject_reasons = []
            if page_validation["bbox_out_of_bounds"] > 0:
                if config.reject_page_warp_if_bbox_clipped:
                    reject_reasons.append("bbox would be clipped")
            if page_validation["dark_fraction"] > config.page_max_dark_fraction:
                reject_reasons.append("warp keeps too much dark background")
            candidate_report["reject_reasons"] = reject_reasons
            attempts.append(candidate_report)
            if reject_reasons:
                continue

            image = warped
            matrix_total = candidate_matrix
            page_applied = True
            report["mode"] = "page_warp"
            report["page_detection"] = candidate_report
            report["transforms"].append(
                {
                    "name": "page_warp",
                    "rank": candidate_rank,
                    **page_info,
                }
            )
            break
        report["page_detection_attempts"] = attempts
        if not page_applied and attempts:
            report["page_detection"] = attempts[0]
            report["transforms"].append(
                {
                    "name": "page_warp_rejected",
                    "attempted_candidates": len(attempts),
                    "reason": "no candidate passed validation",
                }
            )

    if not page_applied:
        crop_boxes = current_boxes_for_crop(record, matrix_total, config)
        crop_rect = compute_text_crop_rect(image, crop_boxes, config)
        if crop_rect is not None:
            requested_crop_rect = crop_rect
            if not config.crop_allow_padding:
                height, width = image.shape[:2]
                crop_rect = clip_crop_rect_to_image(crop_rect, width, height)
                if crop_rect == (0, 0, width, height):
                    crop_rect = None
            if crop_rect is None:
                report["transforms"].append(
                    {
                        "name": "text_crop_skipped",
                        "reason": "clipped crop equals full image",
                        "requested_crop": list(requested_crop_rect),
                        "padding_allowed": bool(config.crop_allow_padding),
                    }
                )
            else:
                cropped, m_crop, crop_info = crop_with_padding(image, crop_rect)
                image = cropped
                matrix_total = m_crop @ matrix_total
                report["mode"] = "text_crop"
                report["transforms"].append(
                    {
                        "name": "text_crop",
                        "requested_crop": list(requested_crop_rect),
                        "padding_allowed": bool(config.crop_allow_padding),
                        **crop_info,
                    }
                )

    image, m_deskew, deskew_info = apply_deskew_if_enabled(image, config)
    if deskew_info.get("applied"):
        matrix_total = m_deskew @ matrix_total
        report["transforms"].append({"name": "deskew", **deskew_info})

    boxes_current = [
        item for item in transformed_region_geometries(record, matrix_total)
        if item["index"] not in dropped_region_indices
    ]
    image, m_resize, resize_info = smart_resize(image, boxes_current, config)
    if resize_info.get("reason") != "identity":
        matrix_total = m_resize @ matrix_total
        report["transforms"].append({"name": "smart_resize", **resize_info})

    image = enhance_image(image, config)
    out_h, out_w = image.shape[:2]
    metadata = build_transformed_metadata(
        record,
        matrix_total,
        out_w,
        out_h,
        output_file_name(record),
        config,
        drop_region_indices=sorted(dropped_region_indices),
    )
    validation = validate_transformed_boxes(
        record,
        matrix_total,
        out_w,
        out_h,
        config,
        ignore_region_indices=sorted(dropped_region_indices),
    )
    report["dropped_region_indices"] = sorted(dropped_region_indices)
    report["dropped_region_count"] = len(dropped_region_indices)
    report["validation"] = validation
    report["warnings"].extend(validation["warnings"])
    report["output_width"] = out_w
    report["output_height"] = out_h
    return ProcessResult(image=image, metadata=metadata, report=report, matrix=matrix_total)


def save_rgb_jpeg(image: np.ndarray, path: Path, quality: int = 95, optimize: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=int(quality), subsampling=0, optimize=bool(optimize))


def save_processed_result(result: ProcessResult, config: PipelineConfig) -> None:
    out_path = Path(config.output_root) / result.metadata["file_name"]
    save_rgb_jpeg(result.image, out_path, config.jpg_quality, config.jpg_optimize)


def _process_and_save_worker(payload: Tuple[JsonDict, PipelineConfig]) -> Tuple[JsonDict, JsonDict]:
    record, config = payload
    result = process_record(record, config)
    save_processed_result(result, config)
    return result.metadata, result.report


def process_train_dataset(
    config: PipelineConfig,
    limit: Optional[int] = None,
    workers: int = 1,
) -> JsonDict:
    records = read_jsonl(Path(config.train_root) / "metadata.jsonl")
    if limit is not None:
        records = records[: int(limit)]
    Path(config.output_root, "images").mkdir(parents=True, exist_ok=True)

    metadata_path = Path(config.output_root) / "metadata.jsonl"
    report_path = Path(config.output_root) / "processing_report.jsonl"
    summary_path = Path(config.output_root) / "summary.json"

    processed = 0
    mode_counts: Dict[str, int] = {}
    warning_count = 0
    dropped_region_count = 0

    with metadata_path.open("w", encoding="utf-8") as meta_f, report_path.open("w", encoding="utf-8") as report_f:
        if workers <= 1:
            iterator = (_process_and_save_worker((record, config)) for record in records)
        else:
            executor = ProcessPoolExecutor(max_workers=int(workers))
            iterator = executor.map(_process_and_save_worker, [(record, config) for record in records], chunksize=4)
        try:
            for metadata, report in iterator:
                meta_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                report_f.write(json.dumps(report, ensure_ascii=False) + "\n")
                processed += 1
                mode = str(report.get("mode", "unknown"))
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                warning_count += int(bool(report.get("warnings")))
                dropped_region_count += int(report.get("dropped_region_count", 0))
        finally:
            if workers > 1:
                executor.shutdown(wait=True)

    summary = {
        "train_root": str(config.train_root),
        "output_root": str(config.output_root),
        "processed": processed,
        "mode_counts": mode_counts,
        "records_with_warnings": warning_count,
        "dropped_region_count": dropped_region_count,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def draw_regions(
    image: np.ndarray,
    regions: Sequence[JsonDict],
    color: Tuple[int, int, int] = (50, 220, 80),
    thickness: int = 3,
) -> np.ndarray:
    canvas = image.copy()
    for region in regions:
        box = bbox_or_none(region)
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        polygon = region.get("polygon")
        if polygon:
            pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=(250, 120, 40), thickness=max(1, thickness - 1))
    return canvas


def resize_panel(image: np.ndarray, max_height: int = 1100, max_width: int = 1100) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_height) / height, float(max_width) / width)
    if scale >= 0.999:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    pad = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min(canvas.shape[:2]) / 900.0)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(canvas, (0, 0), (tw + pad * 2, th + pad * 2), (20, 20, 20), -1)
    cv2.putText(canvas, text, (pad, th + pad // 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def make_before_after_preview(
    record: JsonDict,
    config: PipelineConfig,
    out_path: Path,
) -> JsonDict:
    before_image, m_exif, _info = load_image_rgb_with_exif(record, config)
    before_h, before_w = before_image.shape[:2]
    before_meta = build_transformed_metadata(record, m_exif, before_w, before_h, None, config)
    result = process_record(record, config)

    before_overlay = draw_regions(before_image, before_meta.get("regions", []), color=(40, 210, 80))
    after_overlay = draw_regions(result.image, result.metadata.get("regions", []), color=(40, 210, 80))
    before_panel = label_panel(resize_panel(before_overlay), "Before")
    after_panel = label_panel(resize_panel(after_overlay), "After")

    target_h = max(before_panel.shape[0], after_panel.shape[0])
    bg = np.asarray([210, 210, 210], dtype=np.uint8)

    def pad_height(panel: np.ndarray) -> np.ndarray:
        if panel.shape[0] == target_h:
            return panel
        out = np.full((target_h, panel.shape[1], 3), bg, dtype=np.uint8)
        out[: panel.shape[0], : panel.shape[1]] = panel
        return out

    before_panel = pad_height(before_panel)
    after_panel = pad_height(after_panel)
    divider = np.full((target_h, 12, 3), 185, dtype=np.uint8)
    pair = np.concatenate([before_panel, divider, after_panel], axis=1)
    save_rgb_jpeg(pair, out_path, config.jpg_quality, config.jpg_optimize)
    return {
        "file_name": record.get("file_name"),
        "preview_path": str(out_path),
        "mode": result.report.get("mode"),
        "dropped_region_count": result.report.get("dropped_region_count", 0),
        "dropped_region_indices": result.report.get("dropped_region_indices", []),
        "warnings": result.report.get("warnings", []),
    }


def create_preview_pairs(
    records: Sequence[JsonDict],
    config: PipelineConfig,
    n: int = 20,
    seed: int = 2026,
) -> List[JsonDict]:
    out_dir = Path(config.preview_root) / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = list(records)
    chosen = rng.sample(records, k=min(int(n), len(records)))
    manifest: List[JsonDict] = []
    for idx, record in enumerate(chosen, start=1):
        stem = Path(record["file_name"]).stem
        out_path = out_dir / f"{idx:04d}_{stem}.jpg"
        manifest.append(make_before_after_preview(record, config, out_path))

    manifest_path = Path(config.preview_root) / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "preview_path",
                "mode",
                "dropped_region_count",
                "dropped_region_indices",
                "warnings",
            ],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "preview_path": row["preview_path"],
                    "mode": row["mode"],
                    "dropped_region_count": row.get("dropped_region_count", 0),
                    "dropped_region_indices": " ".join(str(v) for v in row.get("dropped_region_indices", [])),
                    "warnings": "; ".join(row.get("warnings", [])),
                }
            )
    return manifest


if __name__ == '__main__':
    # Biến điều khiển chạy từng chế độ
    RUN_SINGLE = False
    RUN_PREVIEW_20 = False
    RUN_FULL_TRAIN = False
    WORKERS = 4

    TRAIN_ROOT = Path('dataset/train')
    
    # Kiểm tra xem có cấu trúc dataset train chưa để code không lỗi
    # Chúng ta cho pass nếu đang chưa set up folder
    if TRAIN_ROOT.exists() and (TRAIN_ROOT / 'metadata.jsonl').exists() and (TRAIN_ROOT / 'images').exists():
        sys.path.append(str(Path.cwd()))
        print('Train root:', TRAIN_ROOT)

        """
        ## Đọc metadata train

        Cell này chỉ đọc `dataset/train/metadata.jsonl` để kiểm tra nhanh số record và số ảnh. Không chạm tới `silver`, `silver_rare_crops`, hay nguồn dữ liệu khác.
        """
        records = read_jsonl(TRAIN_ROOT / 'metadata.jsonl')
        image_files = [p for p in (TRAIN_ROOT / 'images').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

        print('metadata records:', len(records))
        print('image files:', len(image_files))
        if records:
            print('first record keys:', sorted(records[0].keys()))
            print('first file:', records[0]['file_name'])
            print('first image size from metadata:', records[0]['image_width'], records[0]['image_height'])
            print('first num regions:', len(records[0].get('regions', [])))

        """
        ## Cấu hình pipeline

        `deskew_enabled=False` ở mặc định vì deskew tự động có thể rủi ro với chữ viết tay. Crop hiện mặc định không thêm padding ngoài ảnh gốc; nếu bbox sát mép ảnh thì margin sẽ bị clip vào biên ảnh thay vì tạo nền trắng/kem.
        """
        config = PipelineConfig(
            train_root=TRAIN_ROOT,
            output_root=Path('outputs/train_preprocessed'),
            preview_root=Path('outputs/train_preprocess_preview'),
            detection_max_side=1200,
            page_min_area_ratio=0.20,
            page_min_confidence=0.62,
            page_min_text_center_inside=0.85,
            page_min_text_corner_inside=0.35,
            page_expand_ratio=0.005,
            reject_page_warp_if_bbox_clipped=True,
            page_max_dark_fraction=0.18,
            page_line_detection_enabled=True,
            page_min_edge_support=0.035,
            page_candidate_try_limit=4,
            allow_region_drop=False,
            max_region_drop_ratio=0.0,
            drop_region_visible_ratio=0.98,
            crop_margin_ratio=0.055,
            crop_min_margin_px=24,
            crop_allow_padding=False,
            final_max_side=2200,
            target_median_bbox_height=80,
            max_upscale=1.45,
            shadow_strength=0.75,
            clahe_clip_limit=1.8,
            denoise='median',
            sharpen_amount=0.35,
            jpg_optimize=False,
            deskew_enabled=False,
        )
        print(config)

        """
        ## Chạy thử một ảnh

        Bật `RUN_SINGLE=True` nếu muốn kiểm tra report của một ảnh. Report sẽ cho biết ảnh đi qua mode nào: `page_warp`, `text_crop`, hoặc `keep`, kèm validation bbox sau transform.
        """
        if RUN_SINGLE and records:
            result = process_record(records[0], config)
            print('mode:', result.report['mode'])
            print('output size:', result.report['output_width'], result.report['output_height'])
            print('validation:', result.report['validation'])
            print('warnings:', result.report['warnings'])

        """
        ## Preview 20 ảnh trước/sau

        Bật `RUN_PREVIEW_20=True` khi dependency đã sẵn sàng. Cell này tạo 20 ảnh ghép before/after, có overlay bbox để đánh giá bằng mắt cả chất lượng ảnh lẫn độ đúng của metadata. Output nằm ở `outputs/train_preprocess_preview/pairs/` và manifest ở `outputs/train_preprocess_preview/manifest.csv`.
        """
        if RUN_PREVIEW_20:
            manifest = create_preview_pairs(records, config, n=20, seed=2026)
            print('created preview pairs:', len(manifest))
            if manifest:
                print('first preview:', manifest[0]['preview_path'])
                try:
                    from IPython.display import Image, display
                    display(Image(filename=manifest[0]['preview_path']))
                except Exception as exc:
                    print('Preview saved, but inline display failed:', exc)

        """
        ## Chạy toàn bộ train

        Sau khi preview ổn, bật `RUN_FULL_TRAIN=True`. Pipeline sẽ ghi:

        - `outputs/train_preprocessed/images/*.jpg`
        - `outputs/train_preprocessed/metadata.jsonl`
        - `outputs/train_preprocessed/processing_report.jsonl`
        - `outputs/train_preprocessed/summary.json`

        Metadata đầu ra giữ nguyên các trường gốc, cập nhật `file_name`, `image_width`, `image_height`, `regions[*].bbox`, và thêm `regions[*].polygon` nếu `add_polygon_to_regions=True`.
        """
        if RUN_FULL_TRAIN:
            summary = process_train_dataset(config, workers=WORKERS)
            print(summary)

        """
        ## Cách đọc validation report

        - `mode`: nhánh xử lý hình học được chọn.
        - `bbox_out_of_bounds`: số bbox bị ra ngoài ảnh sau transform; nên bằng 0.
        - `bbox_visible_ratio_min`: tỉ lệ bbox còn nằm trong ảnh; nên gần 1.0.
        - `bbox_area_scale_min/max`: cảnh báo sơ bộ nếu bbox phóng/thu bất thường.
        - `dropped_region_count`: phải bằng 0; pipeline hiện giữ toàn bộ region trong metadata.
        - `warnings`: ảnh cần review thủ công.

        Nếu preview cho thấy page warp quá mạnh, hãy tăng `page_min_confidence`, tăng `page_min_text_corner_inside`, hoặc giảm `page_expand_ratio`. Nếu crop quá sát chữ, tăng `crop_margin_ratio` hoặc `crop_min_margin_px`.
        """
    else:
        print("Bỏ qua phần execution do không tìm thấy folder 'dataset/train'. Hãy setup môi trường trước khi chạy.")
