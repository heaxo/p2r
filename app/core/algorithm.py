#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plate_measure_yolo_sam2_console.py

控制台版：YOLO + SAM2 识别钢板和 A4 paper，并基于 A4 四角 Homography 计算钢板大概尺寸，输出 DXF 和调试图。

本版修正：
1. DXF 默认只输出钢板外轮廓，不再写入 A4 纸轮廓。
2. A4 四角默认优先从 paper 轮廓凸包拟合透视四边形，避免 minAreaRect 把透视梯形强行变成矩形导致尺寸偏差。

集成自 HTTP 版核心识别逻辑：
1. 原图先统一成 canonical 中间图，YOLO 和 SAM2 强制共用同一张图，避免 EXIF/旋转/尺寸导致坐标错位。
2. paper：YOLO 找到 paper -> 取 paper 内部点 -> SAM2 生成 paper mask；失败可退回 YOLO mask/box。
3. plate：YOLO 找到 plate -> 根据 YOLO mask/box 自动选多个内部点 -> SAM2 生成 plate mask；未识别 plate 可中心兜底。
4. final_plate_mask = plate_mask OR paper_mask，把 A4 纸遮挡区域补回钢板。
5. 用 paper mask 提取 A4 四角，建立 像素坐标 -> 毫米坐标 的透视矩阵，将钢板轮廓直接映射到毫米坐标后计算尺寸和 DXF。

依赖：
    pip install ultralytics opencv-python numpy pillow
    # 还需要你的 osam / SAM2 运行环境

示例：
    python plate_measure_yolo_sam2_console.py --image test.jpg --model best2.pt --out measure_out --a4-orientation landscape
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


# =========================
# 默认配置
# =========================
DEFAULT_MODEL_PATH = "best2.pt"
# sam2:large
# sam2:latest
# sam2:small
# sam2:tiny
# 其中 sam2:tiny 就是最轻、最快的版本；sam2:small 是速度和效果折中；sam2:latest 是默认值，但不一定是最快
# 如果 sam2:tiny 边缘不够准 换成sam2，不推荐默认用sam2:large（机器显存足够，而且特别追求 mask 细节）

DEFAULT_SAM_MODEL_NAME = "sam2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

YOLO_CONF = 0.35
YOLO_IMGSZ = 1280

# plate 自动点选择
AVOID_BY_TARGET = {
    "plate": ["paper", "hole"],
    "paper": [],
    "hole": [],
}
USE_AVOID_MASK_FOR_POINTS = True
AVOID_MASK_DILATE_KERNEL = 25
BOX_SHRINK_RATIO = 0.22
CANDIDATE_GRID_SIZE = 3
MAX_SAM_TRY_POINTS = 3
MULTI_POINT_COUNT = 3
ENABLE_SINGLE_POINT_FALLBACK = True
POINT_CLEAN_WINDOW_SIZE = 51

# plate 未被 YOLO 识别到时中心兜底
FALLBACK_TO_CENTER_WHEN_NO_PLATE = True
PLATE_FALLBACK_POINT_RATIO = (0.5, 0.5)
FALLBACK_AVOID_CLASS_NAMES = ["paper", "hole"]
FALLBACK_AVOID_DILATE_KERNEL = 35
FALLBACK_POINT_SEARCH_STEP_PX = 40
FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO = 0.45
FALLBACK_POINT_SEARCH_ANGLE_COUNT = 32

# mask 合理面积范围
MIN_SAM_MASK_AREA_RATIO_BY_CLASS = {
    "plate": 0.003,
    "paper": 0.0003,
    "hole": 0.0002,
}
MAX_SAM_MASK_AREA_RATIO_BY_CLASS = {
    "plate": 0.95,
    "paper": 0.40,
    "hole": 0.40,
}

# paper 补回 plate
PAPER_FILL_DILATE_KERNEL = 9
PAPER_FILL_CLOSE_KERNEL = 15
PAPER_FILL_USE_BOX_IF_NO_MASK = True


# =========================
# 基础工具
# =========================
def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def parse_name_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        text = text.replace("，", ",").replace(";", ",")
        return [x.strip() for x in text.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def parse_user_point_ratio(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "false"}:
            return None
        text = text.replace("，", ",").replace(";", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"user_point_ratio 格式错误，应为 'ratio_x,ratio_y'，当前值：{value}")
        rx = float(parts[0])
        ry = float(parts[1])
    else:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"user_point_ratio 格式错误，应为 [ratio_x, ratio_y]，当前值：{value}")
        rx = float(value[0])
        ry = float(value[1])
    return max(0.0, min(1.0, rx)), max(0.0, min(1.0, ry))


def ratio_to_xy(user_point_ratio: Any, image_shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    ratio = parse_user_point_ratio(user_point_ratio)
    if ratio is None:
        return None
    rx, ry = ratio
    h, w = image_shape[:2]
    x = int(round(rx * (w - 1)))
    y = int(round(ry * (h - 1)))
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def load_image_rgb_from_path(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise RuntimeError(f"图片不存在：{path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError(f"不支持的图片格式：{path.suffix}")
    image_pil = Image.open(path)
    image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")
    return np.asarray(image_pil).copy()


def save_image_rgb(image_rgb: np.ndarray, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb.astype(np.uint8)).save(path)
    return str(path)


def save_mask(mask: np.ndarray, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(out).save(path)
    return str(path)


def detected_class_names(result) -> List[str]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return sorted({str(names[int(c)]) for c in classes})


# =========================
# YOLO 工具
# =========================
def run_yolo_on_canonical_image(
    model: YOLO,
    image_rgb: np.ndarray,
    canonical_image_path: str | Path,
    conf: float,
    imgsz: int,
    input_mode: str = "canonical_path",
):
    """
    推荐 canonical_path：YOLO 读标准中间图路径，SAM2 使用同一张中间图读出的 RGB。
    这样避免 YOLO 和 SAM2 图像方向/尺寸/EXIF 不一致导致坐标错位。
    """
    mode = (input_mode or "canonical_path").strip().lower()
    if mode == "canonical_path":
        source = str(canonical_image_path)
    elif mode == "rgb_array":
        source = np.ascontiguousarray(image_rgb.astype(np.uint8))
    elif mode == "bgr_array":
        source = np.ascontiguousarray(cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        raise RuntimeError(f"不支持的 yolo_input_mode：{input_mode}，可选 canonical_path/rgb_array/bgr_array")

    results = model.predict(
        source=source,
        conf=float(conf),
        imgsz=int(imgsz),
        verbose=False,
        retina_masks=True,
    )
    if not results:
        raise RuntimeError("YOLO 没有返回结果")
    return results[0]


def _normalize_box(x1, y1, x2, y2, w, h) -> List[int]:
    x1 = max(0, min(w - 1, int(round(float(x1)))))
    y1 = max(0, min(h - 1, int(round(float(y1)))))
    x2 = max(0, min(w - 1, int(round(float(x2)))))
    y2 = max(0, min(h - 1, int(round(float(y2)))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _get_mask_from_yolo_result(result, index: int, h: int, w: int) -> Optional[np.ndarray]:
    if result.masks is None:
        return None
    mask = result.masks.data[index].cpu().numpy()
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8)


def build_avoid_mask(result, avoid_class_names: Sequence[str], use_box_if_no_mask: bool = True) -> np.ndarray:
    h, w = result.orig_shape
    avoid_mask = np.zeros((h, w), dtype=np.uint8)
    avoid_class_names = set(parse_name_list(avoid_class_names))

    if result.boxes is None or len(result.boxes) == 0 or not avoid_class_names:
        return avoid_mask

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    avoid_class_ids = {cls_id for cls_id, cls_name in names.items() if str(cls_name) in avoid_class_names}
    if not avoid_class_ids:
        return avoid_mask

    for i, cls_id in enumerate(classes):
        if int(cls_id) not in avoid_class_ids:
            continue
        m = _get_mask_from_yolo_result(result, i, h, w) if result.masks is not None else None
        if m is not None:
            avoid_mask = np.maximum(avoid_mask, m.astype(np.uint8))
        elif use_box_if_no_mask:
            x1, y1, x2, y2 = _normalize_box(*boxes[i], w=w, h=h)
            if x2 > x1 and y2 > y1:
                avoid_mask[y1:y2 + 1, x1:x2 + 1] = 1

    if avoid_mask.sum() > 0 and AVOID_MASK_DILATE_KERNEL > 0:
        k = int(AVOID_MASK_DILATE_KERNEL)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        avoid_mask = cv2.dilate(avoid_mask, kernel, iterations=1)
    return (avoid_mask > 0).astype(np.uint8)


def get_largest_yolo_instance_by_classes(result, class_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    class_names_set = set(parse_name_list(class_names))
    if result is None or result.boxes is None or len(result.boxes) == 0 or not class_names_set:
        return None

    h, w = result.orig_shape
    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    target_class_ids = {
        cls_id
        for cls_id, cls_name in names.items()
        if str(cls_name) in class_names_set
    }

    if not target_class_ids:
        return None

    candidates = []

    for i, cls_id in enumerate(classes):
        if int(cls_id) not in target_class_ids:
            continue

        box = _normalize_box(*boxes[i], w=w, h=h)
        mask = _get_mask_from_yolo_result(result, i, h, w) if result.masks is not None else None

        mask_area = int((mask > 0).sum()) if mask is not None else 0
        box_area = max(0, int((box[2] - box[0]) * (box[3] - box[1])))
        area = mask_area if mask_area > 0 else box_area

        candidates.append({
            "index": int(i),
            "class_id": int(cls_id),
            "class_name": str(names[int(cls_id)]),
            "conf": float(confs[i]),
            "box": box,
            "mask": mask,
            "mask_area": int(mask_area),
            "box_area": int(box_area),
            "area": int(area),
        })

    if not candidates:
        return None
    # 按按面积取paper信息
    # return max(candidates, key=lambda item: item["area"])

    # 按置信度取取paper信息
    return max(candidates, key=lambda item: (item["conf"], item["area"]))
def get_target_from_yolo_result(result, target_class_name: str = "plate", avoid_class_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if avoid_class_names is None:
        avoid_class_names = []
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO 没有识别到任何目标")

    h, w = result.orig_shape
    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    target_class_ids = [cls_id for cls_id, cls_name in names.items() if str(cls_name) == target_class_name]
    if not target_class_ids:
        raise RuntimeError(f"YOLO 模型中没有类别：{target_class_name}，当前类别：{names}")
    target_class_id = int(target_class_ids[0])
    avoid_mask = build_avoid_mask(result, avoid_class_names)

    candidates = []
    for i, cls_id in enumerate(classes):
        if int(cls_id) != target_class_id:
            continue
        box = _normalize_box(*boxes[i], w=w, h=h)
        score = float(confs[i])
        target_mask = _get_mask_from_yolo_result(result, i, h, w) if result.masks is not None else None
        mask_area = int((target_mask > 0).sum()) if target_mask is not None else 0
        box_area = max(0, int((box[2] - box[0]) * (box[3] - box[1])))
        area = mask_area if mask_area > 0 else box_area
        candidates.append({
            "box": box,
            "conf": score,
            "area": int(area),
            "box_area": int(box_area),
            "mask_area": int(mask_area),
            "class_name": target_class_name,
            "target_mask": target_mask,
            "avoid_mask": avoid_mask,
            "yolo_index": int(i),
        })

    if not candidates:
        detected = detected_class_names(result)
        raise RuntimeError(f"没有找到目标类别：{target_class_name}，本图识别到：{detected}")
    return max(candidates, key=lambda item: item["area"])


def is_no_plate_detected_error(error: Exception) -> bool:
    msg = str(error)
    return "YOLO 没有识别到任何目标" in msg or "没有找到目标类别：plate" in msg


# =========================
# 点选择：plate
# =========================
def score_point_cleanliness(image_rgb: np.ndarray, x: int, y: int, window_size: int = POINT_CLEAN_WINDOW_SIZE) -> Tuple[float, Dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    x = int(x)
    y = int(y)
    if x < 0 or x >= w or y < 0 or y >= h:
        return -9999.0, {"edge_density": 1.0, "variance": 1.0, "brightness": 0.0}

    k = int(window_size)
    if k % 2 == 0:
        k += 1
    r = k // 2
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r + 1)
    y2 = min(h, y + r + 1)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return -9999.0, {"edge_density": 1.0, "variance": 1.0, "brightness": 0.0}

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)
    edge_density = float(np.mean(edges > 0))
    gray_f = gray.astype(np.float32) / 255.0
    variance = float(np.var(gray_f))
    brightness = float(np.mean(gray_f))
    highlight_penalty = max(0.0, brightness - 0.88) * 1.5
    clean_score = 1.0 - edge_density * 3.0 - variance * 2.0 - highlight_penalty
    return float(clean_score), {
        "edge_density": edge_density,
        "variance": variance,
        "brightness": brightness,
    }


def shrink_box(x1, y1, x2, y2, img_w, img_h, shrink_ratio: float = BOX_SHRINK_RATIO) -> Tuple[int, int, int, int]:
    x1 = float(max(0, min(img_w - 1, x1)))
    y1 = float(max(0, min(img_h - 1, y1)))
    x2 = float(max(0, min(img_w - 1, x2)))
    y2 = float(max(0, min(img_h - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return int(x1), int(y1), int(x2), int(y2)

    bw = x2 - x1
    bh = y2 - y1
    shrink_ratio = max(0.0, min(0.45, float(shrink_ratio)))
    nx1 = x1 + bw * shrink_ratio
    ny1 = y1 + bh * shrink_ratio
    nx2 = x2 - bw * shrink_ratio
    ny2 = y2 - bh * shrink_ratio
    if nx2 <= nx1 or ny2 <= ny1:
        return int(x1), int(y1), int(x2), int(y2)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def generate_points_in_box(x1, y1, x2, y2, grid_size: int = CANDIDATE_GRID_SIZE) -> List[Tuple[int, int]]:
    if x2 <= x1 or y2 <= y1:
        return []
    if grid_size == 3:
        ratios = [
            (0.50, 0.50), (0.35, 0.50), (0.65, 0.50),
            (0.50, 0.35), (0.50, 0.65),
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
        ]
    else:
        vals = np.linspace(0.25, 0.75, grid_size)
        ratios = [(float(rx), float(ry)) for ry in vals for rx in vals]
        ratios.sort(key=lambda p: abs(p[0] - 0.5) + abs(p[1] - 0.5))

    w = x2 - x1
    h = y2 - y1
    points = []
    seen = set()
    for rx, ry in ratios:
        x = int(round(x1 + w * rx))
        y = int(round(y1 + h * ry))
        if (x, y) not in seen:
            seen.add((x, y))
            points.append((x, y))
    return points


def is_point_valid_for_masks(x: int, y: int, image_shape: Sequence[int], target_mask=None, avoid_mask=None) -> bool:
    h, w = image_shape[:2]
    x = int(x)
    y = int(y)
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    if target_mask is not None and target_mask.size > 0 and target_mask[y, x] <= 0:
        return False
    if USE_AVOID_MASK_FOR_POINTS and avoid_mask is not None and avoid_mask.size > 0 and avoid_mask[y, x] > 0:
        return False
    return True


def make_candidate_points_from_yolo_box(image_rgb: np.ndarray, point_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = point_info["box"]
    sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, img_w=w, img_h=h, shrink_ratio=BOX_SHRINK_RATIO)

    target_mask = point_info.get("target_mask")
    avoid_mask = point_info.get("avoid_mask")

    # 如果 YOLO 有 plate mask，优先从 distanceTransform 取最内部点，减少点到字、锈迹、边缘上的概率。
    candidates: List[Dict[str, Any]] = []
    if target_mask is not None and int((target_mask > 0).sum()) > 0:
        valid = (target_mask > 0).astype(np.uint8)
        if avoid_mask is not None and avoid_mask.size > 0:
            valid[avoid_mask > 0] = 0
        if int(valid.sum()) > 0:
            dist = cv2.distanceTransform(valid * 255, cv2.DIST_L2, 5)
            # 取局部最大附近几个点
            for _ in range(MAX_SAM_TRY_POINTS * 3):
                _, max_dist, _, max_loc = cv2.minMaxLoc(dist)
                if max_dist <= 1:
                    break
                x, y = int(max_loc[0]), int(max_loc[1])
                s, detail = score_point_cleanliness(image_rgb, x, y)
                candidates.append({
                    "x": x,
                    "y": y,
                    "point_score": float(s + min(max_dist / 100.0, 0.5)),
                    "point_detail": detail,
                    "stage": "yolo_mask_distance_transform",
                    "inner_dist": float(max_dist),
                })
                cv2.circle(dist, (x, y), 80, 0, -1)

    # 再补充框内网格点，避免 mask 不准时没有候选点。
    raw_points = generate_points_in_box(sx1, sy1, sx2, sy2, grid_size=CANDIDATE_GRID_SIZE)
    for x, y in raw_points:
        if not is_point_valid_for_masks(x, y, image_rgb.shape, target_mask=target_mask, avoid_mask=avoid_mask):
            continue
        s, detail = score_point_cleanliness(image_rgb, x, y)
        candidates.append({
            "x": int(x),
            "y": int(y),
            "point_score": float(s),
            "point_detail": detail,
            "stage": "box_grid_candidate",
        })

    if not candidates:
        # 最后兜底：框中心
        x = int(round((sx1 + sx2) / 2.0))
        y = int(round((sy1 + sy2) / 2.0))
        s, detail = score_point_cleanliness(image_rgb, x, y)
        candidates.append({"x": x, "y": y, "point_score": float(s), "point_detail": detail, "stage": "box_center_fallback"})

    candidates.sort(key=lambda p: p.get("point_score", 0.0), reverse=True)

    # 去重并控制数量
    selected = []
    seen = set()
    for p in candidates:
        key = (int(p["x"]), int(p["y"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
        if len(selected) >= MAX_SAM_TRY_POINTS:
            break

    meta = {
        "shrink_box": [int(sx1), int(sy1), int(sx2), int(sy2)],
        "raw_candidates": candidates[:10],
    }
    return selected, meta


def _dilate_binary_mask(mask: Optional[np.ndarray], kernel_size: int) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return mask
    out = (mask > 0).astype(np.uint8)
    if out.sum() <= 0 or kernel_size <= 0:
        return out
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(out, kernel, iterations=1) > 0).astype(np.uint8)


def build_fallback_avoid_mask(result, image_shape: Sequence[int], avoid_class_names: Optional[Sequence[str]] = None) -> np.ndarray:
    h, w = image_shape[:2]
    avoid_mask = np.zeros((h, w), dtype=np.uint8)
    if avoid_class_names is None:
        avoid_class_names = FALLBACK_AVOID_CLASS_NAMES
    if result is None:
        return avoid_mask
    try:
        avoid_mask = build_avoid_mask(result, avoid_class_names=avoid_class_names, use_box_if_no_mask=True)
    except Exception:
        avoid_mask = np.zeros((h, w), dtype=np.uint8)
    if avoid_mask.shape != (h, w):
        avoid_mask = cv2.resize(avoid_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    avoid_mask = _dilate_binary_mask(avoid_mask, FALLBACK_AVOID_DILATE_KERNEL)
    return (avoid_mask > 0).astype(np.uint8)


def is_point_on_avoid_mask(x: int, y: int, image_shape: Sequence[int], avoid_mask: Optional[np.ndarray] = None) -> bool:
    h, w = image_shape[:2]
    x = int(x)
    y = int(y)
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    if avoid_mask is not None and avoid_mask.size > 0 and avoid_mask[y, x] > 0:
        return True
    return False


def find_nearest_point_outside_avoid_mask(image_rgb: np.ndarray, base_x: int, base_y: int, avoid_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    h, w = image_rgb.shape[:2]
    base_x = int(max(0, min(w - 1, base_x)))
    base_y = int(max(0, min(h - 1, base_y)))
    center_hit_avoid = is_point_on_avoid_mask(base_x, base_y, image_rgb.shape, avoid_mask)
    if not center_hit_avoid:
        score, detail = score_point_cleanliness(image_rgb, base_x, base_y)
        return {"x": base_x, "y": base_y, "point_score": float(score), "point_detail": detail, "stage": "center_fallback"}

    step = max(5, int(FALLBACK_POINT_SEARCH_STEP_PX))
    max_radius = int(min(h, w) * float(FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO))
    angle_count = max(8, int(FALLBACK_POINT_SEARCH_ANGLE_COUNT))
    for radius in range(step, max_radius + step, step):
        ring_candidates = []
        for i in range(angle_count):
            angle = 2.0 * np.pi * i / angle_count
            x = int(round(base_x + np.cos(angle) * radius))
            y = int(round(base_y + np.sin(angle) * radius))
            if is_point_on_avoid_mask(x, y, image_rgb.shape, avoid_mask):
                continue
            score, detail = score_point_cleanliness(image_rgb, x, y)
            ring_candidates.append({"x": x, "y": y, "point_score": float(score), "point_detail": detail, "stage": "center_fallback_avoid_adjusted", "search_radius": int(radius)})
        if ring_candidates:
            ring_candidates.sort(key=lambda p: p["point_score"], reverse=True)
            return ring_candidates[0]

    score, detail = score_point_cleanliness(image_rgb, base_x, base_y)
    return {"x": base_x, "y": base_y, "point_score": float(score), "point_detail": detail, "stage": "center_fallback_force_use_center"}


def make_candidate_point_from_center_fallback(image_rgb: np.ndarray, result=None, target_class_name: str = "plate") -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    base_xy = ratio_to_xy(PLATE_FALLBACK_POINT_RATIO, image_rgb.shape) or (w // 2, h // 2)
    base_x, base_y = base_xy
    avoid_mask = build_fallback_avoid_mask(result, image_rgb.shape, avoid_class_names=FALLBACK_AVOID_CLASS_NAMES)
    p = find_nearest_point_outside_avoid_mask(image_rgb, base_x, base_y, avoid_mask=avoid_mask)
    candidate_points = [{
        "x": int(p["x"]),
        "y": int(p["y"]),
        "point_score": float(p.get("point_score", 1.0)),
        "point_detail": p.get("point_detail", {}),
        "stage": p.get("stage", "center_fallback"),
    }]
    point_info = {
        "box": None,
        "conf": None,
        "area": 0,
        "class_name": target_class_name,
        "target_mask": None,
        "avoid_mask": avoid_mask,
        "x": int(p["x"]),
        "y": int(p["y"]),
        "mode": "center_fallback_no_plate",
        "fallback_base_point": [int(base_x), int(base_y)],
        "avoid_mask_area": int((avoid_mask > 0).sum()) if avoid_mask is not None else 0,
    }
    point_meta = {"raw_candidates": candidate_points}
    return candidate_points, point_info, point_meta


# =========================
# paper：YOLO -> 一个点 -> SAM2
# =========================
def choose_one_point_inside_yolo_instance(instance: Dict[str, Any], image_shape: Sequence[int]) -> Dict[str, Any]:
    h, w = image_shape[:2]
    mask = instance.get("mask")
    if mask is not None and mask.size > 0 and int((mask > 0).sum()) > 0:
        bin_mask = (mask > 0).astype(np.uint8)
        dist = cv2.distanceTransform(bin_mask * 255, cv2.DIST_L2, 5)
        _, max_dist, _, max_loc = cv2.minMaxLoc(dist)
        return {"x": int(max_loc[0]), "y": int(max_loc[1]), "source": "yolo_mask_distance_transform", "inner_dist": float(max_dist)}

    x1, y1, x2, y2 = instance["box"]
    x = int(round((x1 + x2) / 2.0))
    y = int(round((y1 + y2) / 2.0))
    return {"x": max(0, min(w - 1, x)), "y": max(0, min(h - 1, y)), "source": "yolo_box_center", "inner_dist": 0.0}


def make_yolo_instance_mask_or_box(instance: Dict[str, Any], image_shape: Sequence[int]) -> np.ndarray:
    h, w = image_shape[:2]
    mask = instance.get("mask")
    if mask is not None and mask.size > 0 and int((mask > 0).sum()) > 0:
        out = (mask > 0).astype(np.uint8)
    else:
        out = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = instance["box"]
        if x2 > x1 and y2 > y1:
            out[y1:y2 + 1, x1:x2 + 1] = 1

    if out.sum() > 0 and PAPER_FILL_DILATE_KERNEL > 0:
        k = int(PAPER_FILL_DILATE_KERNEL)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)

    if out.sum() > 0 and PAPER_FILL_CLOSE_KERNEL > 0:
        k = int(PAPER_FILL_CLOSE_KERNEL)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return (out > 0).astype(np.uint8)


# =========================
# SAM2
# =========================
def _import_osam():
    try:
        import osam.apis  # type: ignore
        import osam.types  # type: ignore
        return osam.apis, osam.types
    except Exception as e:
        raise RuntimeError("未找到 osam/SAM2 运行环境。请先安装并配置sam/sam2 环境。") from e


def _sam_annotation_to_full_mask(annotation, h: int, w: int) -> Optional[np.ndarray]:
    bbox = annotation.bounding_box
    small_mask = np.asarray(annotation.mask)
    if small_mask.dtype != np.bool_:
        small_mask = small_mask > 0

    full_mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, int(bbox.xmin))
    y1 = max(0, int(bbox.ymin))
    mh, mw = small_mask.shape[:2]
    x2 = min(w, x1 + mw)
    y2 = min(h, y1 + mh)
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return None
    full_mask[y1:y2, x1:x2] = small_mask[:crop_h, :crop_w].astype(np.uint8) * 255
    return full_mask


def run_sam2_masks_by_point(image_rgb: np.ndarray, x: int, y: int, model_name: str = "sam2") -> List[np.ndarray]:
    osam_apis, osam_types = _import_osam()
    h, w = image_rgb.shape[:2]
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))
    request = osam_types.GenerateRequest(
        model=model_name,
        image=image_rgb,
        prompt=osam_types.Prompt(points=[[x, y]], point_labels=[1]),
    )
    response = osam_apis.generate(request=request)
    if not response.annotations:
        return []
    masks = []
    for annotation in response.annotations:
        full_mask = _sam_annotation_to_full_mask(annotation, h, w)
        if full_mask is not None:
            masks.append(full_mask)
    return masks


def run_sam2_masks_by_points(image_rgb: np.ndarray, points: Sequence[Dict[str, Any]], model_name: str = "sam2") -> List[np.ndarray]:
    osam_apis, osam_types = _import_osam()
    if not points:
        return []
    h, w = image_rgb.shape[:2]
    sam_points = []
    for p in points:
        x = int(max(0, min(w - 1, int(p["x"]))))
        y = int(max(0, min(h - 1, int(p["y"]))))
        sam_points.append([x, y])
    request = osam_types.GenerateRequest(
        model=model_name,
        image=image_rgb,
        prompt=osam_types.Prompt(points=sam_points, point_labels=[1] * len(sam_points)),
    )
    response = osam_apis.generate(request=request)
    if not response.annotations:
        return []
    masks = []
    for annotation in response.annotations:
        full_mask = _sam_annotation_to_full_mask(annotation, h, w)
        if full_mask is not None:
            masks.append(full_mask)
    return masks


def get_mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def score_sam_mask_weak(mask: np.ndarray, image_shape: Sequence[int], target_class_name: str, point_score: float = 0.0) -> Tuple[float, Dict[str, Any]]:
    h, w = image_shape[:2]
    image_area = max(1, h * w)
    mask_bin = mask > 0
    area = int(mask_bin.sum())
    area_ratio = area / image_area
    min_ratio = MIN_SAM_MASK_AREA_RATIO_BY_CLASS.get(target_class_name, 0.001)
    max_ratio = MAX_SAM_MASK_AREA_RATIO_BY_CLASS.get(target_class_name, 0.95)

    if area <= 0:
        return -9999.0, {"area": 0, "area_ratio": 0.0, "reason": "empty"}
    if area_ratio < min_ratio:
        return -1000.0 + area_ratio, {"area": area, "area_ratio": float(area_ratio), "reason": "too_small"}
    if area_ratio > max_ratio:
        return -900.0 - area_ratio, {"area": area, "area_ratio": float(area_ratio), "reason": "too_large"}

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
    component_count = max(0, num_labels - 1)
    largest_component_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 0 else 0
    largest_ratio = largest_component_area / max(1, area)
    fragmentation_penalty = max(0.0, 1.0 - largest_ratio)

    bbox = get_mask_bbox(mask)
    bbox_area_ratio = 0.0
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        bbox_area = max(1, (bx2 - bx1 + 1) * (by2 - by1 + 1))
        bbox_area_ratio = area / bbox_area

    if target_class_name == "plate":
        target_area_score = np.log1p(area_ratio * 100.0)
    else:
        target_area_score = np.log1p(area_ratio * 20.0)

    score = target_area_score + 0.35 * float(point_score) + 0.35 * float(bbox_area_ratio) - 1.0 * fragmentation_penalty
    return float(score), {
        "area": area,
        "area_ratio": float(area_ratio),
        "component_count": int(component_count),
        "largest_ratio": float(largest_ratio),
        "bbox_area_ratio": float(bbox_area_ratio),
        "reason": "ok",
    }


def pick_best_mask_from_masks(masks: Sequence[np.ndarray], image_shape: Sequence[int], target_class_name: str, point_score: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    best = None
    for mask_idx, mask in enumerate(masks):
        sam_score, detail = score_sam_mask_weak(mask, image_shape, target_class_name, point_score)
        item = {"mask": mask, "sam_score": float(sam_score), "sam_detail": detail, "mask_index": int(mask_idx)}
        if best is None or item["sam_score"] > best["sam_score"]:
            best = item
    if best is None:
        raise RuntimeError("SAM2 没有得到有效 mask")
    return best["mask"], best


def run_sam2_by_candidate_points(image_rgb: np.ndarray, candidate_points: Sequence[Dict[str, Any]], target_class_name: str, model_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    image_shape = image_rgb.shape[:2]
    best = None
    tried = []
    multi_points = list(candidate_points[:MULTI_POINT_COUNT])

    if len(multi_points) >= 2:
        try:
            masks = run_sam2_masks_by_points(image_rgb=image_rgb, points=multi_points, model_name=model_name)
            avg_point_score = float(np.mean([p.get("point_score", 0.0) for p in multi_points]))
            for mask_idx, mask in enumerate(masks):
                sam_score, detail = score_sam_mask_weak(mask, image_shape, target_class_name, avg_point_score)
                item = {
                    "mask": mask,
                    "x": int(multi_points[0]["x"]),
                    "y": int(multi_points[0]["y"]),
                    "point_score": avg_point_score,
                    "sam_score": float(sam_score),
                    "sam_detail": detail,
                    "mask_index": int(mask_idx),
                    "mode": "multi_points",
                    "used_points": multi_points,
                }
                tried.append({"mode": "multi_points", "points": [[int(p["x"]), int(p["y"])] for p in multi_points], "sam_score": float(sam_score), "sam_detail": detail, "mask_index": int(mask_idx)})
                if best is None or item["sam_score"] > best["sam_score"]:
                    best = item
            if best is not None and best["sam_score"] > -100 and best["sam_detail"].get("reason") == "ok":
                best["tried"] = tried
                return best["mask"], best
        except Exception as e:
            tried.append({"mode": "multi_points", "message": str(e), "sam_score": -9999.0})

    if ENABLE_SINGLE_POINT_FALLBACK:
        for p in candidate_points:
            x = int(p["x"])
            y = int(p["y"])
            point_score = float(p.get("point_score", 0.0))
            try:
                masks = run_sam2_masks_by_point(image_rgb=image_rgb, x=x, y=y, model_name=model_name)
            except Exception as e:
                tried.append({"mode": "single_point", "x": x, "y": y, "point_score": point_score, "sam_score": -9999.0, "message": str(e)})
                continue
            if not masks:
                tried.append({"mode": "single_point", "x": x, "y": y, "point_score": point_score, "sam_score": -9999.0, "message": "no_mask"})
                continue
            for mask_idx, mask in enumerate(masks):
                sam_score, detail = score_sam_mask_weak(mask, image_shape, target_class_name, point_score)
                item = {
                    "mask": mask,
                    "x": x,
                    "y": y,
                    "point_score": point_score,
                    "sam_score": float(sam_score),
                    "sam_detail": detail,
                    "mask_index": int(mask_idx),
                    "mode": "single_point",
                    "used_points": [p],
                }
                tried.append({"mode": "single_point", "x": x, "y": y, "point_score": point_score, "sam_score": float(sam_score), "sam_detail": detail, "mask_index": int(mask_idx)})
                if best is None or item["sam_score"] > best["sam_score"]:
                    best = item
                if sam_score > -100 and detail.get("reason") == "ok":
                    item["tried"] = tried
                    return mask, item

    if best is None or best["sam_score"] < -100:
        raise RuntimeError(f"多个候选点尝试后，SAM2 仍然没有得到合理 mask，tried={tried[:5]}")
    best["tried"] = tried
    return best["mask"], best


def run_sam2_by_user_ratio_point(image_rgb: np.ndarray, user_point_ratio: Any, target_class_name: str, model_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    xy = ratio_to_xy(user_point_ratio, image_rgb.shape)
    if xy is None:
        raise RuntimeError("user_point_ratio 为空，不能走人工点逻辑")
    x, y = xy
    masks = run_sam2_masks_by_point(image_rgb=image_rgb, x=x, y=y, model_name=model_name)
    if not masks:
        raise RuntimeError(f"人工点 SAM2 没有生成任何 mask，point=({x},{y})")
    mask, best = pick_best_mask_from_masks(masks, image_rgb.shape[:2], target_class_name, point_score=1.0)
    ratio = parse_user_point_ratio(user_point_ratio)
    best.update({"x": int(x), "y": int(y), "mode": "user_ratio_point", "user_ratio": ratio, "used_points": [{"x": int(x), "y": int(y), "ratio_x": float(ratio[0]), "ratio_y": float(ratio[1])}]})
    return mask, best


def run_paper_from_yolo_only(
    result,
    image_rgb: np.ndarray,
    paper_class_names: Sequence[str],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    直接使用 YOLO 识别到的 paper mask/box，不再通过 SAM2。
    优先使用 YOLO segmentation mask；如果没有 mask，退回 YOLO box。
    """
    info: Dict[str, Any] = {
        "paper_class_names": parse_name_list(paper_class_names),
        "yolo_detected_classes": detected_class_names(result),
        "yolo_paper_detected": False,
        "paper_point": None,
        "paper_mask_source": None,
        "message": "",
    }

    instance = get_largest_yolo_instance_by_classes(result, paper_class_names)
    if instance is None:
        info["message"] = "YOLO 未识别到 paper。请检查 paper_class_names、yolo_conf 或图片方向。"
        return None, info

    info["yolo_paper_detected"] = True
    info["yolo_paper"] = {
        "class_name": instance["class_name"],
        "conf": float(instance["conf"]),
        "box": instance["box"],
        "mask_area": int(instance["mask_area"]),
        "box_area": int(instance["box_area"]),
        "area": int(instance["area"]),
        "has_yolo_mask": instance.get("mask") is not None and int((instance["mask"] > 0).sum()) > 0,
    }

    # 只是为了 debug_overlay 里还能画 paper 点
    point = choose_one_point_inside_yolo_instance(instance, image_rgb.shape)
    info["paper_point"] = point

    yolo_mask = make_yolo_instance_mask_or_box(instance, image_rgb.shape)
    if yolo_mask is not None and int((yolo_mask > 0).sum()) > 0:
        if info["yolo_paper"]["has_yolo_mask"]:
            info["paper_mask_source"] = "yolo_seg_mask"
            info["message"] = "paper 使用 YOLO segmentation mask"
        else:
            info["paper_mask_source"] = "yolo_box"
            info["message"] = "paper 使用 YOLO box，注意：box 四角不是透视 A4 四角，精度可能较低"

        return yolo_mask.astype(np.uint8) * 255, info

    info["message"] = "YOLO 找到 paper，但没有可用 mask/box"
    return None, info


def run_sam2_for_paper_from_yolo(
    result,
    image_rgb: np.ndarray,
    paper_class_names: Sequence[str],
    model_name: str,
    fallback_to_yolo_mask: bool = True,
    detect_by_sam2: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "paper_class_names": parse_name_list(paper_class_names),
        "yolo_detected_classes": detected_class_names(result),
        "yolo_paper_detected": False,
        "paper_point": None,
        "paper_mask_source": None,
        "message": "",
    }

    instance = get_largest_yolo_instance_by_classes(result, paper_class_names)
    if instance is None:
        info["message"] = "YOLO 未识别到 paper。请检查 paper_class_names、yolo_conf 或图片方向。"
        return None, info

    info["yolo_paper_detected"] = True
    info["yolo_paper"] = {
        "class_name": instance["class_name"],
        "conf": float(instance["conf"]),
        "box": instance["box"],
        "mask_area": int(instance["mask_area"]),
        "box_area": int(instance["box_area"]),
        "area": int(instance["area"]),
    }

    point = choose_one_point_inside_yolo_instance(instance, image_rgb.shape)
    info["paper_point"] = point

    if detect_by_sam2:
        try:
            masks = run_sam2_masks_by_point(image_rgb=image_rgb, x=int(point["x"]), y=int(point["y"]), model_name=model_name)
            if masks:
                paper_mask, best = pick_best_mask_from_masks(masks, image_rgb.shape[:2], "paper", point_score=1.0)
                if best["sam_detail"].get("reason") == "ok" and best["sam_score"] > -100:
                    info["paper_mask_source"] = "sam2_one_point_from_yolo_paper"
                    info["sam_info"] = {"x": int(point["x"]), "y": int(point["y"]), "mode": "paper_one_point", "sam_score": float(best["sam_score"]), "sam_detail": best["sam_detail"], "mask_index": int(best["mask_index"])}
                    info["message"] = "YOLO 已识别 paper，并用单点传给 SAM2 成功生成 paper mask"
                    return paper_mask, info
                info["sam_info"] = {"sam_score": float(best["sam_score"]), "sam_detail": best["sam_detail"], "message": "SAM2 paper mask 评分不合理"}
            else:
                info["sam_info"] = {"message": "SAM2 没有返回 paper mask"}
        except Exception as e:
            info["sam_info"] = {"message": str(e), "traceback": traceback.format_exc(limit=5)}

    if fallback_to_yolo_mask:
        yolo_mask = make_yolo_instance_mask_or_box(instance, image_rgb.shape)
        if yolo_mask is not None and yolo_mask.sum() > 0:
            info["paper_mask_source"] = "yolo_mask_or_box_fallback"
            info["message"] = "SAM2 paper 未成功，已退回 YOLO paper mask/box"
            return yolo_mask.astype(np.uint8) * 255, info

    info["message"] = "YOLO 找到 paper，但 SAM2 没有成功生成 paper mask，且无法使用 YOLO fallback"
    return None, info


def run_sam2_for_plate_from_yolo_or_fallback(
    result,
    image_rgb: np.ndarray,
    plate_class_name: str,
    model_name: str,
    user_point_ratio: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    manual_ratio = parse_user_point_ratio(user_point_ratio)
    if manual_ratio is not None:
        sam_mask, sam_info = run_sam2_by_user_ratio_point(image_rgb, manual_ratio, plate_class_name, model_name)
        point_info = {
            "x": int(sam_info["x"]),
            "y": int(sam_info["y"]),
            "class_name": plate_class_name,
            "mode": "user_ratio_point",
            "user_ratio": manual_ratio,
            "sam_score": float(sam_info["sam_score"]),
            "sam_detail": sam_info["sam_detail"],
            "candidate_points": sam_info.get("used_points", []),
        }
        return sam_mask, point_info

    try:
        avoid_class_names = AVOID_BY_TARGET.get(plate_class_name, [])
        point_info = get_target_from_yolo_result(result, target_class_name=plate_class_name, avoid_class_names=avoid_class_names)
        candidate_points, point_meta = make_candidate_points_from_yolo_box(image_rgb, point_info)
        point_info.update(point_meta)
    except RuntimeError as yolo_target_error:
        if plate_class_name == "plate" and FALLBACK_TO_CENTER_WHEN_NO_PLATE and is_no_plate_detected_error(yolo_target_error):
            candidate_points, point_info, point_meta = make_candidate_point_from_center_fallback(image_rgb, result=result, target_class_name=plate_class_name)
            point_info.update(point_meta)
        else:
            raise

    if point_info.get("mode") == "center_fallback_no_plate":
        fallback_point = candidate_points[0]
        masks = run_sam2_masks_by_point(image_rgb=image_rgb, x=int(fallback_point["x"]), y=int(fallback_point["y"]), model_name=model_name)
        if not masks:
            raise RuntimeError(f"中心兜底点 SAM2 没有生成任何 mask，point=({fallback_point['x']},{fallback_point['y']})")
        sam_mask, sam_info = pick_best_mask_from_masks(masks, image_rgb.shape[:2], target_class_name=plate_class_name, point_score=float(fallback_point.get("point_score", 1.0)))
        sam_info.update({"x": int(fallback_point["x"]), "y": int(fallback_point["y"]), "mode": "center_fallback_no_plate", "used_points": [fallback_point]})
    else:
        sam_mask, sam_info = run_sam2_by_candidate_points(image_rgb=image_rgb, candidate_points=candidate_points, target_class_name=plate_class_name, model_name=model_name)

    point_info["x"] = int(sam_info["x"])
    point_info["y"] = int(sam_info["y"])
    point_info["sam_score"] = float(sam_info["sam_score"])
    point_info["sam_detail"] = sam_info["sam_detail"]
    point_info["candidate_points"] = candidate_points
    point_info["sam_mode"] = sam_info.get("mode")
    point_info["used_points"] = sam_info.get("used_points", [])
    return sam_mask, point_info


# =========================
# 后处理：paper 补回 plate
# =========================
def apply_paper_fill_to_plate_mask(plate_mask: np.ndarray, paper_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if plate_mask is None:
        raise RuntimeError("plate_mask 为空，无法补回 A4纸区域")
    plate_bin = plate_mask > 0
    if paper_mask is None or paper_mask.size == 0 or int((paper_mask > 0).sum()) <= 0:
        return plate_bin.astype(np.uint8) * 255, {"filled": False, "paper_area": 0, "added_area": 0, "reason": "no_paper_mask"}

    paper_bin = paper_mask > 0
    if paper_bin.shape != plate_bin.shape:
        paper_bin = cv2.resize(paper_bin.astype(np.uint8), (plate_bin.shape[1], plate_bin.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    added = paper_bin & (~plate_bin)
    final_bin = plate_bin | paper_bin
    return final_bin.astype(np.uint8) * 255, {"filled": True, "paper_area": int(paper_bin.sum()), "added_area": int(added.sum()), "reason": "ok"}


# =========================
# A4 四角 / Homography / 尺寸 / DXF
# =========================
def _binary_mask(mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def _largest_contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    bin_mask = _binary_mask(mask)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _odd_kernel_size(value: int, min_value: int = 0) -> int:
    k = int(value or 0)
    if k < int(min_value):
        return 0
    if k <= 0:
        return 0
    if k % 2 == 0:
        k += 1
    return k


def _fill_holes_in_binary_mask(bin_mask: np.ndarray) -> np.ndarray:
    src = (_binary_mask(bin_mask) > 0).astype(np.uint8)
    if src.size == 0 or int(src.sum()) <= 0:
        return src
    h, w = src.shape[:2]
    flood = src.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return ((src > 0) | (holes > 0)).astype(np.uint8)


def _keep_largest_component(bin_mask: np.ndarray) -> np.ndarray:
    src = (_binary_mask(bin_mask) > 0).astype(np.uint8)
    if src.size == 0 or int(src.sum()) <= 0:
        return src
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    if num_labels <= 1:
        return src
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    return (labels == largest_label).astype(np.uint8)


def clean_paper_mask_for_rect(paper_mask: np.ndarray, close_kernel_size: int = 15, open_kernel_size: int = 5) -> np.ndarray:
    out = _binary_mask(paper_mask)
    out = _keep_largest_component(out)
    close_k = _odd_kernel_size(close_kernel_size, min_value=1)
    if close_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    open_k = _odd_kernel_size(open_kernel_size, min_value=1)
    if open_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    out = _fill_holes_in_binary_mask(out)
    out = _keep_largest_component(out)
    return (out > 0).astype(np.uint8)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    # 防止重复点，退回角度排序
    if len({(round(float(p[0]), 3), round(float(p[1]), 3)) for p in ordered}) < 4:
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        ordered = pts[np.argsort(angles)]
        start_idx = np.argmin(ordered.sum(axis=1))
        ordered = np.roll(ordered, -start_idx, axis=0).astype(np.float32)
        if ordered[1, 0] < ordered[3, 0]:
            ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered


def parse_points(text: str) -> np.ndarray:
    pts = []
    for part in text.split(";"):
        x, y = part.split(",")
        pts.append([float(x.strip()), float(y.strip())])
    if len(pts) != 4:
        raise ValueError("paper points 必须是4个点，格式：x1,y1;x2,y2;x3,y3;x4,y4")
    return order_quad_points(np.array(pts, dtype=np.float32))


def _score_quad_candidate(quad: np.ndarray, contour: np.ndarray) -> float:
    """
    给 A4 四边形候选打分。分数越小越好。
    主要看：候选四边形面积和 paper mask 轮廓面积是否接近。
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    contour_area = max(1.0, float(abs(cv2.contourArea(cnt.reshape(-1, 1, 2)))))
    quad_area = max(1.0, float(abs(cv2.contourArea(q.reshape(-1, 1, 2)))))
    # quad 面积应该略大于/接近 contour 面积；过大或过小都不好
    area_ratio = quad_area / contour_area
    area_penalty = abs(math.log(max(area_ratio, 1e-6)))
    # 四边形不能太瘦长
    ordered = order_quad_points(q)
    edges = [
        float(np.linalg.norm(ordered[1] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[1])),
        float(np.linalg.norm(ordered[2] - ordered[3])),
        float(np.linalg.norm(ordered[3] - ordered[0])),
    ]
    min_edge = max(1.0, min(edges))
    max_edge = max(edges)
    slender_penalty = 0.0 if max_edge / min_edge < 4.0 else (max_edge / min_edge - 4.0)
    return float(area_penalty + slender_penalty)


def _try_quad_from_contour_perspective(contour: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    从 paper contour 中提取透视四边形。

    关键：不能默认用 minAreaRect。minAreaRect 得到的是“旋转矩形”，会把真实透视下的梯形
    强行变成矩形，这会破坏 Homography，导致钢板尺寸差很多。
    """
    info: Dict[str, Any] = {}
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(cnt) < 4:
        return None, {"quad_source": "none_contour_too_few_points"}

    cnt_i = cnt.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(cnt_i)
    peri = cv2.arcLength(hull, True)
    if peri <= 1:
        return None, {"quad_source": "none_invalid_perimeter"}

    candidates: List[Tuple[float, float, np.ndarray]] = []
    for ratio in np.linspace(0.003, 0.12, 60):
        approx = cv2.approxPolyDP(hull, float(ratio) * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            score = _score_quad_candidate(quad, cnt_i)
            candidates.append((score, float(ratio), quad))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        score, ratio, quad = candidates[0]
        quad = order_quad_points(quad)
        info.update({
            "quad_source": f"convex_hull_approx_poly_{ratio:.4f}",
            "quad_score": float(score),
            "quad_candidate_count": int(len(candidates)),
        })
        return quad, info

    return None, {"quad_source": "none_approx_poly_failed"}


def find_paper_quad_from_mask(paper_mask: np.ndarray, mode: str = "approx_poly") -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    从 paper mask 得到 A4 四角。返回 ordered TL/TR/BR/BL。

    推荐 mode=approx_poly：
      - 先清理 paper mask；
      - 再从凸包拟合真实透视四边形；
      - 失败才退回 minAreaRect。

    不建议默认用 minAreaRect，因为它会把透视梯形变成旋转矩形，尺寸会偏差很大。
    """
    mode = (mode or "approx_poly").strip().lower()
    if mode == "raw":
        mask_for_rect = _binary_mask(paper_mask)
    else:
        mask_for_rect = clean_paper_mask_for_rect(paper_mask, close_kernel_size=15, open_kernel_size=5)

    contour = _largest_contour_from_mask(mask_for_rect)
    if contour is None or len(contour) < 4:
        raise RuntimeError("未能从 paper mask 中提取 A4 轮廓")

    info: Dict[str, Any] = {
        "mode": mode,
        "paper_mask_area_px": int((mask_for_rect > 0).sum()),
    }

    quad = None
    quad_info: Dict[str, Any] = {}

    # robust_fit 现在也优先保留透视四边形，不再直接 minAreaRect。
    if mode in {"approx_poly", "robust_fit", "raw"}:
        quad, quad_info = _try_quad_from_contour_perspective(contour)
        info.update(quad_info)

    if quad is None:
        rect = cv2.minAreaRect(contour.astype(np.float32))
        quad = cv2.boxPoints(rect).astype(np.float32)
        # 不覆盖 approx_poly 失败信息，补充 fallback 信息。
        info["quad_fallback"] = "min_area_rect"
        info["rect_center_px"] = [float(rect[0][0]), float(rect[0][1])]
        info["rect_size_px"] = [float(rect[1][0]), float(rect[1][1])]
        info["rect_angle_deg"] = float(rect[2])
        if "quad_source" not in info:
            info["quad_source"] = "min_area_rect"

    quad = order_quad_points(quad)

    top = float(np.linalg.norm(quad[1] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    info["paper_quad_px_tl_tr_br_bl"] = np.round(quad, 3).tolist()
    info["paper_edge_lengths_px"] = {
        "top": round(top, 3),
        "right": round(right, 3),
        "bottom": round(bottom, 3),
        "left": round(left, 3),
    }
    info["warning"] = None
    if info.get("quad_fallback") == "min_area_rect":
        info["warning"] = "A4四角使用了minAreaRect兜底，透视精度可能下降；建议传 --paper-points 使用更准确的4个A4角点。"
    return quad, info, mask_for_rect

def build_a4_homography(paper_quad_img: np.ndarray, orientation: str = "auto") -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], str]:
    paper_quad_img = order_quad_points(paper_quad_img)
    top = np.linalg.norm(paper_quad_img[1] - paper_quad_img[0])
    bottom = np.linalg.norm(paper_quad_img[2] - paper_quad_img[3])
    left = np.linalg.norm(paper_quad_img[3] - paper_quad_img[0])
    right = np.linalg.norm(paper_quad_img[2] - paper_quad_img[1])
    img_w = (top + bottom) / 2.0
    img_h = (left + right) / 2.0

    orientation = orientation.strip().lower()
    if orientation == "landscape":
        real_w, real_h = 297.0, 210.0
        used_orientation = "landscape"
    elif orientation == "portrait":
        real_w, real_h = 210.0, 297.0
        used_orientation = "portrait"
    elif orientation == "auto":
        if img_w >= img_h:
            real_w, real_h = 297.0, 210.0
            used_orientation = "landscape"
        else:
            real_w, real_h = 210.0, 297.0
            used_orientation = "portrait"
    else:
        raise ValueError("--a4-orientation 只能是 auto / landscape / portrait")

    paper_quad_mm = np.array([[0.0, 0.0], [real_w, 0.0], [real_w, real_h], [0.0, real_h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(paper_quad_img.astype(np.float32), paper_quad_mm.astype(np.float32))
    return H, paper_quad_mm, (real_w, real_h), used_orientation


def transform_points_px_to_mm(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out.astype(np.float32)


def simplify_contour_mm(contour_mm: np.ndarray, epsilon_mm: float) -> np.ndarray:
    pts = np.asarray(contour_mm, dtype=np.float32).reshape(-1, 2)
    if epsilon_mm <= 0:
        return pts
    cnt = pts.reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, float(epsilon_mm), True)
    return approx.reshape(-1, 2).astype(np.float32)


def calc_dimensions(contour_mm: np.ndarray) -> dict:
    pts = np.asarray(contour_mm, dtype=np.float32).reshape(-1, 2)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect)
    (_, _), (rw, rh), angle = rect
    length = max(float(rw), float(rh))
    width = min(float(rw), float(rh))
    area = float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))
    perimeter = float(cv2.arcLength(pts.reshape(-1, 1, 2), True))

    return {
        "length_mm": round(length, 2),
        "width_mm": round(width, 2),
        "min_area_rect_length_mm": round(length, 2),
        "min_area_rect_width_mm": round(width, 2),
        "axis_bbox_width_mm": round(float(bbox_w), 2),
        "axis_bbox_height_mm": round(float(bbox_h), 2),
        "area_mm2": round(area, 2),
        "area_m2": round(area / 1_000_000.0, 6),
        "perimeter_mm": round(perimeter, 2),
        "rect_angle_deg": round(float(angle), 3),
        "rotated_rect_box_points_mm": np.round(box, 3).tolist(),
        "mm_bounds": {
            "x_min": round(x_min, 2),
            "x_max": round(x_max, 2),
            "y_min": round(y_min, 2),
            "y_max": round(y_max, 2),
        },
    }


def write_simple_dxf(path: Path, plate_contour_mm: np.ndarray, offset_to_positive: bool = True) -> None:
    """
    写 DXF，单位 mm。

    注意：这里故意只写钢板外轮廓，不写 A4 纸轮廓。
    A4 只用于标定尺寸，不属于最终 DXF 图形。
    """
    plate = np.asarray(plate_contour_mm, dtype=np.float64).reshape(-1, 2)
    if len(plate) > 1 and np.linalg.norm(plate[0] - plate[-1]) < 1e-6:
        plate = plate[:-1]

    dx = dy = 0.0
    if offset_to_positive and len(plate) > 0:
        min_xy = plate.min(axis=0)
        dx = -min(0.0, float(min_xy[0]))
        dy = -min(0.0, float(min_xy[1]))

    def lwpolyline(layer: str, pts: np.ndarray, closed: bool = True) -> List[str]:
        lines = ["0", "LWPOLYLINE", "8", layer, "90", str(len(pts)), "70", "1" if closed else "0"]
        for x, y in pts:
            lines.extend(["10", f"{x + dx:.3f}", "20", f"{y + dy:.3f}"])
        return lines

    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]
    lines.extend(lwpolyline("PLATE_OUTER", plate, True))
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")



def save_topdown_warp_outputs(
    run_dir: Path,
    image_rgb: np.ndarray,
    final_plate_mask: np.ndarray,
    H_px_to_mm: np.ndarray,
    mm_per_px: float = 2.0,
    padding_mm: float = 50.0,
) -> Dict[str, Any]:
    """
    基于 A4 Homography，把原图和最终钢板 mask 透视展开成俯视图。

    H_px_to_mm:
        原图像素坐标 -> 真实毫米坐标 的矩阵

    mm_per_px:
        输出俯视图中 1 个像素代表多少毫米。
        例如 2.0 表示 1px = 2mm。
        值越小，输出图越大，细节越高。

    padding_mm:
        俯视图四周额外留白，单位 mm。
    """

    plate_outer_contour = _largest_contour_from_mask(final_plate_mask)
    if plate_outer_contour is None or len(plate_outer_contour) < 4:
        raise RuntimeError("无法从 final_plate_mask 提取钢板轮廓，不能生成俯视矫正图")

    plate_contour_px = plate_outer_contour.reshape(-1, 2).astype(np.float32)

    # 原图钢板轮廓 -> 毫米坐标
    plate_contour_mm = transform_points_px_to_mm(plate_contour_px, H_px_to_mm)

    x_min = float(np.min(plate_contour_mm[:, 0]) - padding_mm)
    y_min = float(np.min(plate_contour_mm[:, 1]) - padding_mm)
    x_max = float(np.max(plate_contour_mm[:, 0]) + padding_mm)
    y_max = float(np.max(plate_contour_mm[:, 1]) + padding_mm)

    width_mm = max(1.0, x_max - x_min)
    height_mm = max(1.0, y_max - y_min)

    out_w = int(math.ceil(width_mm / mm_per_px))
    out_h = int(math.ceil(height_mm / mm_per_px))

    # 毫米坐标 -> 俯视图像素坐标
    # x_px = (x_mm - x_min) / mm_per_px
    # y_px = (y_mm - y_min) / mm_per_px
    scale = 1.0 / float(mm_per_px)
    T_mm_to_top_px = np.array([
        [scale, 0.0, -x_min * scale],
        [0.0, scale, -y_min * scale],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 原图像素 -> 毫米 -> 俯视图像素
    H_px_to_top_px = T_mm_to_top_px @ H_px_to_mm.astype(np.float64)

    topdown_image_rgb = cv2.warpPerspective(
        image_rgb,
        H_px_to_top_px,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    topdown_mask = cv2.warpPerspective(
        ((final_plate_mask > 0).astype(np.uint8) * 255),
        H_px_to_top_px,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    topdown_image_path = run_dir / "topdown_image.png"
    topdown_mask_path = run_dir / "topdown_plate_mask.png"
    topdown_overlay_path = run_dir / "topdown_overlay.png"

    save_image_rgb(topdown_image_rgb, topdown_image_path)
    save_mask(topdown_mask, topdown_mask_path)

    # 画俯视 mask 轮廓 overlay
    overlay = topdown_image_rgb.copy()
    contour_top = _largest_contour_from_mask(topdown_mask)
    if contour_top is not None:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, [contour_top], -1, (0, 255, 0), 3)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    save_image_rgb(overlay, topdown_overlay_path)

    return {
        "topdown": {
            "mm_per_px": float(mm_per_px),
            "padding_mm": float(padding_mm),
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "output_width_px": int(out_w),
            "output_height_px": int(out_h),
            "bounds_mm": {
                "x_min": round(x_min, 2),
                "y_min": round(y_min, 2),
                "x_max": round(x_max, 2),
                "y_max": round(y_max, 2),
            },
            "paths": {
                "topdown_image": str(topdown_image_path),
                "topdown_plate_mask": str(topdown_mask_path),
                "topdown_overlay": str(topdown_overlay_path),
            }
        }
    }


def save_mm_preview(path: Path, plate_contour_mm: np.ndarray, paper_quad_mm: np.ndarray, dims: dict, max_px: int = 1800) -> None:
    plate = np.asarray(plate_contour_mm, dtype=np.float32).reshape(-1, 2)
    paper = np.asarray(paper_quad_mm, dtype=np.float32).reshape(-1, 2)
    all_pts = np.vstack([plate, paper])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    size_mm = max_xy - min_xy
    max_dim = max(float(size_mm[0]), float(size_mm[1]), 1.0)
    scale = min(1.0, max_px / max_dim)

    margin = 60
    w = int(math.ceil(size_mm[0] * scale + margin * 2))
    h = int(math.ceil(size_mm[1] * scale + margin * 2))
    w = max(w, 400)
    h = max(h, 300)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    def to_px(pts: np.ndarray) -> np.ndarray:
        out = (pts - min_xy) * scale + np.array([margin, margin], dtype=np.float32)
        return np.round(out).astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(canvas, [to_px(plate)], True, (0, 0, 0), 2)
    cv2.polylines(canvas, [to_px(paper)], True, (0, 0, 255), 2)
    text1 = f"Plate: {dims['length_mm']} x {dims['width_mm']} mm"
    text2 = f"Area: {dims['area_m2']} m2"
    cv2.putText(canvas, text1, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(canvas, text2, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


# =========================
# debug 图
# =========================
def save_debug_overlay(
    path: Path,
    image_rgb: np.ndarray,
    plate_contour_px: Optional[np.ndarray],
    paper_quad_px: Optional[np.ndarray],
    plate_point_info: Optional[Dict[str, Any]] = None,
    paper_info: Optional[Dict[str, Any]] = None,
    dims: Optional[dict] = None,
) -> None:
    bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if plate_contour_px is not None:
        cv2.polylines(bgr, [plate_contour_px.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    if paper_quad_px is not None:
        cv2.polylines(bgr, [paper_quad_px.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(paper_quad_px):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(bgr, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(bgr, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if plate_point_info:
        for idx, p in enumerate(plate_point_info.get("candidate_points", []), start=1):
            x, y = int(p["x"]), int(p["y"])
            cv2.circle(bgr, (x, y), 7, (0, 255, 255), -1)
            cv2.putText(bgr, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if "x" in plate_point_info and "y" in plate_point_info:
            x, y = int(plate_point_info["x"]), int(plate_point_info["y"])
            cv2.circle(bgr, (x, y), 13, (255, 0, 0), 3)
            cv2.putText(bgr, "plate", (x + 14, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if paper_info and paper_info.get("paper_point"):
        x, y = int(paper_info["paper_point"]["x"]), int(paper_info["paper_point"]["y"])
        cv2.circle(bgr, (x, y), 13, (255, 0, 255), 3)
        cv2.putText(bgr, "paper", (x + 14, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if dims:
        text = f"L={dims['length_mm']}mm W={dims['width_mm']}mm"
        cv2.putText(bgr, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)



from app.core.dxf_postprocess import (
    DxfPostProcessConfig,
    postprocess_plate_contour_mm,
    save_postprocess_preview,
)

def fit_circle_mm(points_mm: np.ndarray) -> dict:
    pts = np.asarray(points_mm, dtype=np.float32).reshape(-1, 2)

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y

    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = float(np.sqrt(c + cx * cx + cy * cy))

    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius_std = float(np.std(dists))
    radius_error_ratio = radius_std / max(radius, 1e-6)

    area = float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))
    perimeter = float(cv2.arcLength(pts.reshape(-1, 1, 2), True))
    circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-6)

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "radius": radius,
        "diameter": radius * 2.0,
        "circularity": float(circularity),
        "radius_error_ratio": float(radius_error_ratio),
        "is_circle_like": circularity >= 0.86 and radius_error_ratio <= 0.08,
    }


def write_circle_dxf(output_path: str | Path, circle: dict, offset_to_positive: bool = True) -> None:
    cx = float(circle["center_x"])
    cy = float(circle["center_y"])
    r = float(circle["radius"])

    offset_x = 0.0
    offset_y = 0.0

    if offset_to_positive:
        offset_x = max(0.0, -cx + r + 10.0)
        offset_y = max(0.0, -cy + r + 10.0)

    cx += offset_x
    cy += offset_y

    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
        "0", "CIRCLE",
        "8", "PLATE_OUTER",
        "10", f"{cx:.6f}",
        "20", f"{cy:.6f}",
        "30", "0.0",
        "40", f"{r:.6f}",
        "0", "ENDSEC",
        "0", "EOF",
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")



def make_circle_preview_contour_mm(circle: dict, point_count: int = 240) -> np.ndarray:
    cx = float(circle["center_x"])
    cy = float(circle["center_y"])
    r = float(circle["radius"])

    angles = np.linspace(0.0, 2.0 * np.pi, int(point_count), endpoint=False)
    xs = cx + np.cos(angles) * r
    ys = cy + np.sin(angles) * r

    return np.column_stack([xs, ys]).astype(np.float32)


# =========================
# 主流程
# =========================
def process_one_image(args) -> Dict[str, Any]:
    image_path = Path(args.image)
    if not image_path.exists():
        raise RuntimeError(f"图片不存在：{image_path}")

    out_root = Path(args.out)
    stem = image_path.stem or "image"
    run_dir = out_root / f"{stem}_{uuid.uuid4().hex[:8]}"
    mkdir(run_dir)

    image_rgb = load_image_rgb_from_path(image_path)

    # 关键：保存 canonical 中间图；YOLO 和 SAM2 都使用它。
    canonical_image_path = run_dir / "input_canonical_used_by_yolo_and_sam2.png"
    save_image_rgb(image_rgb, canonical_image_path)
    image_rgb = load_image_rgb_from_path(canonical_image_path)

    model_path = Path(args.model)
    if not model_path.exists():
        raise RuntimeError(f"YOLO 权重文件不存在：{model_path}")

    # HTTP 服务会把已缓存的 YOLO 模型对象放到 args.yolo_model。
    # 控制台模式下没有该字段，则仍按原逻辑从模型文件加载。
    cached_yolo_model = getattr(args, "yolo_model", None)
    model = cached_yolo_model if cached_yolo_model is not None else YOLO(str(model_path))

    result = run_yolo_on_canonical_image(
        model=model,
        image_rgb=image_rgb,
        canonical_image_path=canonical_image_path,
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        input_mode=args.yolo_input_mode,
    )

    h, w = image_rgb.shape[:2]
    result_shape = tuple(int(x) for x in result.orig_shape)
    shape_warning = None
    if result_shape != (h, w):
        shape_warning = f"YOLO result.orig_shape={result_shape} 与 image_rgb.shape={(h, w)} 不一致"

    mask_paths: Dict[str, str] = {
        "canonical_image": str(canonical_image_path)
    }

    # 1. paper：默认直接用 YOLO；可切换成 YOLO -> 点 -> SAM2
    paper_class_names = parse_name_list(args.paper_class) or ["paper"]
    print(f"args.paper_source:{args.paper_source}")
    print(f"args.sam_model:{args.sam_model}")
    if args.paper_source == "yolo":
        paper_mask, paper_info = run_paper_from_yolo_only(
            result=result,
            image_rgb=image_rgb,
            paper_class_names=paper_class_names,
        )
    else:
        paper_mask, paper_info = run_sam2_for_paper_from_yolo(
            result=result,
            image_rgb=image_rgb,
            paper_class_names=paper_class_names,
            model_name=args.sam_model,
            fallback_to_yolo_mask=bool(args.paper_sam2_yolo_fallback),
            detect_by_sam2=True,
        )

    if paper_mask is not None and int((paper_mask > 0).sum()) > 0:
        mask_paths["paper_mask"] = save_mask(paper_mask, run_dir / "paper_mask.png")
    else:
        raise RuntimeError(f"未得到 paper mask，无法基于 A4 计算尺寸。paper_info={paper_info}")

    # 2. plate：YOLO -> 多点 -> SAM2 / 中心兜底
    plate_mask, plate_point_info = run_sam2_for_plate_from_yolo_or_fallback(
        result=result,
        image_rgb=image_rgb,
        plate_class_name=args.plate_class,
        model_name=args.sam_model,
        user_point_ratio=args.user_point_ratio,
    )
    mask_paths["plate_raw_mask"] = save_mask(plate_mask, run_dir / "plate_raw_mask.png")

    # 3. final_plate = plate OR paper
    final_plate_mask, fill_info = apply_paper_fill_to_plate_mask(plate_mask, paper_mask)
    mask_paths["plate_final_with_paper_fill"] = save_mask(
        final_plate_mask,
        run_dir / "plate_final_with_paper_fill.png",
    )

    # 4. 从 paper mask 或用户指定四角中获取 A4 四角
    if args.paper_points:
        paper_quad_px = parse_points(args.paper_points)
        paper_quad_info = {
            "mode": "manual_points",
            "paper_quad_px_tl_tr_br_bl": np.round(paper_quad_px, 3).tolist(),
        }
        paper_cleaned_mask = _binary_mask(paper_mask)
    else:
        paper_quad_px, paper_quad_info, paper_cleaned_mask = find_paper_quad_from_mask(
            paper_mask,
            mode=args.paper_rect_mode,
        )

        # 基于 A4 尺寸做二次矫正
        refine_orientation = args.a4_orientation
        if refine_orientation == "auto":
            top = np.linalg.norm(paper_quad_px[1] - paper_quad_px[0])
            right = np.linalg.norm(paper_quad_px[2] - paper_quad_px[1])
            refine_orientation = "landscape" if top >= right else "portrait"

        paper_quad_px_refined, refine_info = refine_paper_quad_by_a4_rect(
            paper_mask=paper_cleaned_mask,
            paper_quad_px=paper_quad_px,
            orientation=refine_orientation,
            scale_px_per_mm=3.0,
            padding_mm=20.0,
        )

        paper_quad_info["before_refine_quad_px"] = np.round(paper_quad_px, 3).tolist()
        paper_quad_info["refine_info"] = refine_info
        paper_quad_px = paper_quad_px_refined
        paper_quad_info["paper_quad_px_tl_tr_br_bl"] = np.round(paper_quad_px, 3).tolist()

    # 5. 钢板外轮廓 -> mm 坐标
    plate_outer_contour = _largest_contour_from_mask(final_plate_mask)
    if plate_outer_contour is None or len(plate_outer_contour) < 4:
        raise RuntimeError("未能从最终 plate mask 中提取钢板外轮廓")

    plate_contour_px = plate_outer_contour.reshape(-1, 2).astype(np.float32)

    H, paper_quad_mm, a4_size, used_orientation = build_a4_homography(
        paper_quad_px,
        orientation=args.a4_orientation,
    )

    plate_contour_mm_raw = transform_points_px_to_mm(plate_contour_px, H)

    # 先用原始mm轮廓判断是否接近圆形。
    # 注意：圆形检测必须放在 simplify_contour_mm 和 DXF后处理之前。
    circle_info = fit_circle_mm(plate_contour_mm_raw)

    dxf_path = run_dir / "plate_outer.dxf"

    if circle_info["is_circle_like"]:
        # 圆形钢板直接写 DXF CIRCLE，不走折线和后处理。
        write_circle_dxf(
            dxf_path,
            circle_info,
            offset_to_positive=True,
        )

        # 预览图和尺寸计算仍然需要一个轮廓，这里生成一个圆形轮廓用于预览。
        plate_contour_mm = make_circle_preview_contour_mm(circle_info)

        radius = float(circle_info["radius"])
        diameter = float(circle_info["diameter"])
        area_mm2 = float(np.pi * radius * radius)
        perimeter_mm = float(2.0 * np.pi * radius)

        dims = {
            "length_mm": round(diameter, 2),
            "width_mm": round(diameter, 2),
            "min_area_rect_length_mm": round(diameter, 2),
            "min_area_rect_width_mm": round(diameter, 2),
            "axis_bbox_width_mm": round(diameter, 2),
            "axis_bbox_height_mm": round(diameter, 2),
            "area_mm2": round(area_mm2, 2),
            "area_m2": round(area_mm2 / 1_000_000.0, 6),
            "perimeter_mm": round(perimeter_mm, 2),
            "circle_mode": True,
            "circle_center_x_mm": round(float(circle_info["center_x"]), 2),
            "circle_center_y_mm": round(float(circle_info["center_y"]), 2),
            "circle_radius_mm": round(radius, 2),
            "circle_diameter_mm": round(diameter, 2),
            "circle_circularity": round(float(circle_info["circularity"]), 4),
            "circle_radius_error_ratio": round(float(circle_info["radius_error_ratio"]), 4),
        }

        dxf_postprocess_info = {
            "enabled": False,
            "reason": "circle_like_contour_use_dxf_circle",
            "circle_info": json_safe(circle_info),
        }

    else:
        # 非圆形才走原来的轮廓简化和DXF后处理。
        plate_contour_mm_before_postprocess = simplify_contour_mm(
            plate_contour_mm_raw,
            args.simplify_mm,
        )

        dxf_postprocess_config = DxfPostProcessConfig(
            enabled=bool(getattr(args, "dxf_postprocess_enabled", True)),
            notch_fill_enabled=bool(getattr(args, "dxf_notch_fill_enabled", True)),
            notch_fill_max_width_mm=float(getattr(args, "dxf_notch_fill_max_width_mm", 280.0)),
            notch_fill_max_depth_mm=float(getattr(args, "dxf_notch_fill_max_depth_mm", 90.0)),
        )

        plate_contour_mm, dxf_postprocess_info = postprocess_plate_contour_mm(
            contour_mm=plate_contour_mm_before_postprocess,
            config=dxf_postprocess_config,
        )

        dxf_postprocess_preview_path = run_dir / "debug_dxf_postprocess_preview.png"
        save_postprocess_preview(
            before_mm=plate_contour_mm_before_postprocess,
            after_mm=plate_contour_mm,
            output_path=dxf_postprocess_preview_path,
            padding_mm=80.0,
            mm_per_px=2.0,
        )
        mask_paths["debug_dxf_postprocess_preview"] = str(dxf_postprocess_preview_path)

        dims = calc_dimensions(plate_contour_mm)

        write_simple_dxf(
            dxf_path,
            plate_contour_mm,
            offset_to_positive=True,
        )

    ENABLE_TOPDOWN_WARP_OUTPUT = True

    # 8. 透视矫正输出
    if ENABLE_TOPDOWN_WARP_OUTPUT:
        topdown_info = save_topdown_warp_outputs(
            run_dir=run_dir,
            image_rgb=image_rgb,
            final_plate_mask=final_plate_mask,
            H_px_to_mm=H,
            mm_per_px=args.topdown_mm_per_px,
            padding_mm=args.topdown_padding_mm,
        )
        mask_paths.update(topdown_info["topdown"]["paths"])
    else:
        topdown_info = {
            "topdown": {
                "enabled": False,
                "reason": "disabled_by_ENABLE_TOPDOWN_WARP_OUTPUT",
                "paths": {},
            }
        }

    # 9. DXF使用后处理后的轮廓
    dxf_path = run_dir / "plate_outer.dxf"
    write_simple_dxf(
        dxf_path,
        plate_contour_mm,
        offset_to_positive=True,
    )

    debug_overlay_path = run_dir / "debug_overlay.jpg"
    save_debug_overlay(
        debug_overlay_path,
        image_rgb,
        plate_contour_px,
        paper_quad_px,
        plate_point_info,
        paper_info,
        dims,
    )
    mask_paths["debug_overlay"] = str(debug_overlay_path)

    mm_preview_path = run_dir / "debug_mm_preview.png"
    save_mm_preview(
        mm_preview_path,
        plate_contour_mm,
        paper_quad_mm,
        dims,
    )
    mask_paths["debug_mm_preview"] = str(mm_preview_path)

    result_json_path = run_dir / "result.json"

    clean_plate_point_info = dict(plate_point_info)
    clean_plate_point_info.pop("target_mask", None)
    clean_plate_point_info.pop("avoid_mask", None)

    output = {
        "ok": True,
        "run_dir": str(run_dir),
        "input": {
            "image": str(image_path),
            "canonical_image": str(canonical_image_path),
            "image_shape": {
                "height": int(h),
                "width": int(w),
            },
            "shape_warning": shape_warning,
        },
        "model": {
            "yolo_model_path": str(model_path),
            "yolo_model_names": {str(k): str(v) for k, v in model.names.items()},
            "yolo_detected_classes": detected_class_names(result),
            "sam_model": args.sam_model,
        },
        "paper": paper_info,
        "plate": {
            "point_info": json_safe(clean_plate_point_info),
        },
        "fill_paper_to_plate": fill_info,
        "a4": {
            "requested_orientation": args.a4_orientation,
            "used_orientation": used_orientation,
            "used_width_mm": a4_size[0],
            "used_height_mm": a4_size[1],
            "paper_quad_info": paper_quad_info,
        },
        "topdown": topdown_info["topdown"],

        # 新增：DXF后处理信息
        "dxf_postprocess": dxf_postprocess_info,

        # 尺寸基于后处理后的轮廓
        "plate_dimensions": dims,

        "paths": {
            "dxf": str(dxf_path),
            "result_json": str(result_json_path),
            **mask_paths,
        },
        "note": (
            "尺寸基于单张 A4 的平面 Homography 换算。"
            "DXF只包含钢板外轮廓，不包含A4纸轮廓。"
            "DXF轮廓已做后处理：去毛刺、压直线、局部凹陷修复。"
            "A4 与钢板不共面、钢板弯曲、A4角点不准、相机广角畸变都会引入误差。"
            "若尺寸偏差明显，优先传 --paper-points 使用准确A4四角，或指定 --a4-orientation landscape/portrait。"
        ),
    }

    result_json_path.write_text(
        json.dumps(json_safe(output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return json_safe(output)


def refine_paper_quad_by_a4_rect(
    paper_mask: np.ndarray,
    paper_quad_px: np.ndarray,
    orientation: str = "landscape",
    scale_px_per_mm: float = 3.0,
    padding_mm: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    用 A4 已知尺寸，对 SAM2/轮廓得到的 A4 四角做二次矫正。

    思路：
    1. 用初始 paper_quad_px 建立原图 -> A4俯视平面的 Homography
    2. 把 paper_mask warp 到 A4俯视平面
    3. 在俯视平面中重新找 mask 的边界
    4. 强制边界接近 297x210 或 210x297
    5. 再反变换回原图，得到 corrected_quad_px
    """

    paper_quad_px = order_quad_points(paper_quad_px)

    if orientation == "portrait":
        real_w_mm, real_h_mm = 210.0, 297.0
    else:
        real_w_mm, real_h_mm = 297.0, 210.0

    pad_px = int(round(padding_mm * scale_px_per_mm))
    dst_w = int(round(real_w_mm * scale_px_per_mm)) + pad_px * 2
    dst_h = int(round(real_h_mm * scale_px_per_mm)) + pad_px * 2

    dst_quad_px = np.array([
        [pad_px, pad_px],
        [pad_px + real_w_mm * scale_px_per_mm, pad_px],
        [pad_px + real_w_mm * scale_px_per_mm, pad_px + real_h_mm * scale_px_per_mm],
        [pad_px, pad_px + real_h_mm * scale_px_per_mm],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        paper_quad_px.astype(np.float32),
        dst_quad_px.astype(np.float32)
    )

    H_inv = np.linalg.inv(H)

    warped_mask = cv2.warpPerspective(
        ((paper_mask > 0).astype(np.uint8) * 255),
        H,
        (dst_w, dst_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    bin_mask = (warped_mask > 0).astype(np.uint8)

    # 清理一下 mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = _keep_largest_component(bin_mask)

    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return paper_quad_px, {
            "refined": False,
            "reason": "warped paper mask empty"
        }

    # 在俯视平面里重新找 A4 边界
    x_min = float(np.percentile(xs, 1))
    x_max = float(np.percentile(xs, 99))
    y_min = float(np.percentile(ys, 1))
    y_max = float(np.percentile(ys, 99))

    detected_w = x_max - x_min
    detected_h = y_max - y_min

    expected_w = real_w_mm * scale_px_per_mm
    expected_h = real_h_mm * scale_px_per_mm

    # 用检测到的中心 + A4固定尺寸，重新构造标准矩形
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    refined_dst_quad = np.array([
        [cx - expected_w / 2.0, cy - expected_h / 2.0],
        [cx + expected_w / 2.0, cy - expected_h / 2.0],
        [cx + expected_w / 2.0, cy + expected_h / 2.0],
        [cx - expected_w / 2.0, cy + expected_h / 2.0],
    ], dtype=np.float32)

    # 反变换回原图坐标
    corrected_quad = cv2.perspectiveTransform(
        refined_dst_quad.reshape(-1, 1, 2),
        H_inv.astype(np.float32)
    ).reshape(-1, 2)

    corrected_quad = order_quad_points(corrected_quad)

    return corrected_quad.astype(np.float32), {
        "refined": True,
        "orientation": orientation,
        "real_w_mm": real_w_mm,
        "real_h_mm": real_h_mm,
        "scale_px_per_mm": scale_px_per_mm,
        "detected_w_px": round(float(detected_w), 2),
        "detected_h_px": round(float(detected_h), 2),
        "expected_w_px": round(float(expected_w), 2),
        "expected_h_px": round(float(expected_h), 2),
        "old_quad_px": np.round(paper_quad_px, 3).tolist(),
        "corrected_quad_px": np.round(corrected_quad, 3).tolist(),
    }
