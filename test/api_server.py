# -*- coding: utf-8 -*-
"""
YOLO + SAM2 钢板/A4纸识别 HTTP 服务版

核心改动：
1. HTTP 服务：FastAPI 提供 /process-path 和 /process-upload。
2. YOLO 和 SAM2 强制共用同一张“标准中间图”：
   - 原图先统一转成 input_canonical_used_by_yolo_and_sam2.png
   - YOLO 使用 model.predict(source=canonical_image_path, ...)
   - SAM2 使用从 canonical_image_path 读出的 image_rgb
   - debug 图也使用同一个 image_rgb
   这样既保留 YOLO 走图片路径时的稳定识别效果，又保证 YOLO 坐标和 SAM2 图像坐标一致。
3. A4纸 paper 逻辑：
   - YOLO 识别到 paper 后，只从 paper mask/box 取一个内部点给 SAM2
   - SAM2 用这个点识别 paper 范围
   - plate 最终 mask = plate_mask OR paper_mask
   - 如果 SAM2 paper 失败，可选退回 YOLO paper mask/box 作为补洞区域
4. 尺寸和 DXF：
   - 根据 A4 纸 210mm x 297mm 计算 mm_per_px
   - 使用最终 plate mask 提取外轮廓和内轮廓
   - 生成毫米单位 DXF，并返回 dxf_content、dxf_path、mask_paths、dimensions

启动：
    pip install fastapi uvicorn python-multipart ultralytics pillow opencv-python numpy
    # 还需要你原来使用的 osam/sam2 运行环境

    python plate_yolo_sam2_http_service.py \
        --yolo-model D:/models/best2.pt \
        --output-root D:/output \
        --host 0.0.0.0 \
        --port 8000

调用1：处理服务端已有图片路径
    POST http://127.0.0.1:8000/process-path
    Content-Type: application/json
    {
      "image_path": "D:/test/1.jpg",
      "target_class_names": ["plate"],
      "fill_paper_to_plate": true,
      "detect_paper_by_sam2": true
    }

调用2：上传图片
    POST http://127.0.0.1:8000/process-upload
    form-data:
      file: 图片文件
      request_json: {"target_class_names":["plate"]}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

import osam.apis
import osam.types


# =========================
# 默认配置：可通过启动参数或请求参数覆盖
# =========================
DEFAULT_YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "../best2.pt")
DEFAULT_OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "./output_http")
DEFAULT_SAM_MODEL_NAME = os.getenv("SAM_MODEL_NAME", "sam2")

YOLO_CONF = 0.35
YOLO_IMGSZ = 1280

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# plate 自动取点配置
FALLBACK_TO_CENTER_WHEN_NO_PLATE = True
PLATE_FALLBACK_POINT_RATIO = (0.5, 0.5)
FALLBACK_AVOID_CLASS_NAMES = ["paper", "hole"]
FALLBACK_AVOID_DILATE_KERNEL = 35
FALLBACK_POINT_SEARCH_STEP_PX = 40
FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO = 0.45
FALLBACK_POINT_SEARCH_ANGLE_COUNT = 32

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
ENABLE_SINGLE_POINT_FALLBACK = False

ANCHOR_INNER_SEARCH_RADIUS = 180
ANCHOR_INNER_MIN_DIST = 10.0
POINT_AROUND_RADIUS_PX = None
POINT_AROUND_RADIUS_RATIO = 0.18
POINT_AROUND_RADIUS_MIN_PX = 50
POINT_AROUND_RADIUS_MAX_PX = 180
MIN_POINT_DISTANCE_RATIO_FOR_AROUND = 0.70
POINT_CLEAN_WINDOW_SIZE = 51

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

# A4 纸补回钢板
FILL_PAPER_TO_PLATE = True
PAPER_CLASS_NAMES = ["paper"]
PAPER_FILL_DILATE_KERNEL = 9
PAPER_FILL_CLOSE_KERNEL = 15
PAPER_FILL_USE_BOX_IF_NO_MASK = True
SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL = True
DETECT_PAPER_BY_SAM2 = True
PAPER_SAM2_FALLBACK_TO_YOLO_MASK = True


# =========================
# HTTP 请求模型
# =========================
class ProcessRequest(BaseModel):
    image_path: Optional[str] = Field(default=None, description="服务端本地图片路径。/process-path 必填")
    yolo_model_path: Optional[str] = Field(default=None, description="YOLO best2.pt 路径；不传则使用启动参数")
    output_root: Optional[str] = Field(default=None, description="输出目录；不传则使用启动参数")

    target_class_names: List[str] = Field(default_factory=lambda: ["plate"])
    sam_model_name: str = DEFAULT_SAM_MODEL_NAME
    yolo_conf: float = YOLO_CONF
    yolo_imgsz: int = YOLO_IMGSZ
    # 默认使用 canonical_path：先生成标准中间图，再让 YOLO 读这个路径。
    # 这样避免 model.predict(source=image_rgb) 在某些环境下和 model.predict(source=path) 识别效果不一致。
    # 可选：canonical_path / rgb_array / bgr_array
    yolo_input_mode: str = "canonical_path"

    # 人工点模式：例如 "0.5,0.5" 或 [0.5,0.5]
    user_point_ratio: Optional[Any] = None

    # 是否把 A4纸补回 plate
    fill_paper_to_plate: bool = FILL_PAPER_TO_PLATE
    detect_paper_by_sam2: bool = DETECT_PAPER_BY_SAM2
    paper_class_names: List[str] = Field(default_factory=lambda: ["paper"])
    paper_sam2_fallback_to_yolo_mask: bool = PAPER_SAM2_FALLBACK_TO_YOLO_MASK

    # 尺寸与 DXF
    enable_dxf: bool = True
    a4_width_mm: float = 210.0
    a4_height_mm: float = 297.0
    scale_method: str = "mean_edge"  # mean_edge / area / short_edge / long_edge
    contour_epsilon_mm: float = 2.0
    min_contour_area_mm2: float = 100.0
    include_holes_in_dxf: bool = True
    dxf_include_content: bool = True

    # 调试/返回
    # 默认关闭 debug 图，减少一次大图绘制和 JPG 保存；需要排查点位时再传 true。
    save_debug_images: bool = False
    return_mask_base64: bool = False


# 运行时参数，通过 main() 写入
RUNTIME_CONFIG = {
    "yolo_model_path": DEFAULT_YOLO_MODEL_PATH,
    "output_root": DEFAULT_OUTPUT_ROOT,
}


# =========================
# 基础工具
# =========================
def _json_safe(value: Any) -> Any:
    """把 numpy/path 等转换成 JSON 可序列化对象。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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

    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))
    return rx, ry


def ratio_to_xy(user_point_ratio: Any, image_shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    ratio = parse_user_point_ratio(user_point_ratio)
    if ratio is None:
        return None
    rx, ry = ratio
    h, w = image_shape[:2]
    x = int(round(rx * (w - 1)))
    y = int(round(ry * (h - 1)))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return int(x), int(y)


def load_image_rgb_from_path(image_path: str | Path) -> np.ndarray:
    """
    只读取一次图片，并处理 EXIF 旋转。

    关键点：
    - 旋转图片/手机图片常带 EXIF Orientation。
    - 如果 YOLO 从路径读，SAM2 从 PIL 读，可能一个应用了旋转，一个没应用，坐标就错。
    - 这里统一用 ImageOps.exif_transpose 后得到 image_rgb。
    """
    path = Path(image_path)
    if not path.exists():
        raise RuntimeError(f"图片不存在：{path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError(f"不支持的图片格式：{path.suffix}")

    image_pil = Image.open(path)
    image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")
    return np.asarray(image_pil).copy()


def load_image_rgb_from_bytes(data: bytes) -> np.ndarray:
    image_pil = Image.open(io.BytesIO(data))
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


def mask_to_base64_png(mask: np.ndarray) -> str:
    out = (mask > 0).astype(np.uint8) * 255
    buff = io.BytesIO()
    Image.fromarray(out).save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("ascii")


@lru_cache(maxsize=4)
def get_yolo_model(model_path: str) -> YOLO:
    if not model_path:
        raise RuntimeError("未配置 yolo_model_path，请启动时传 --yolo-model 或请求中传 yolo_model_path")
    if not Path(model_path).exists():
        raise RuntimeError(f"YOLO 权重文件不存在：{model_path}")
    return YOLO(model_path)


def run_yolo_on_canonical_image(
    model: YOLO,
    image_rgb: np.ndarray,
    canonical_image_path: str | Path,
    conf: float,
    imgsz: int,
    input_mode: str = "canonical_path",
):
    """
    YOLO 输入策略。

    推荐默认 canonical_path：
      1. 先把已经统一方向/像素的 image_rgb 保存成标准中间图 canonical_image_path。
      2. YOLO 读 canonical_image_path。
      3. SAM2 使用从同一个 canonical_image_path 读出的 image_rgb。

    这样不是“YOLO 用原始 path，SAM2 用另一个 image_rgb”，而是二者共用同一张标准中间图。
    好处：保留 Ultralytics 走 path 时更稳定的预处理链路，同时避免坐标系错位。

    input_mode：
      - canonical_path：推荐，YOLO 读标准中间图路径。
      - rgb_array：YOLO 直接吃 RGB ndarray。仅用于对比调试。
      - bgr_array：YOLO 直接吃 BGR ndarray。某些 Ultralytics 版本/环境下 ndarray 可能按 BGR 解释。
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
    )
    if not results:
        raise RuntimeError("YOLO 没有返回结果")
    return results[0]

def detected_class_names(result) -> List[str]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return sorted({str(names[int(c)]) for c in classes})


# =========================
# YOLO mask/box 工具
# =========================
def _get_mask_from_yolo_result(result, index: int, h: int, w: int) -> Optional[np.ndarray]:
    if result.masks is None:
        return None
    mask = result.masks.data[index].cpu().numpy()
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8)


def _normalize_box(x1, y1, x2, y2, w, h) -> List[int]:
    x1 = max(0, min(w - 1, int(round(float(x1)))))
    y1 = max(0, min(h - 1, int(round(float(y1)))))
    x2 = max(0, min(w - 1, int(round(float(x2)))))
    y2 = max(0, min(h - 1, int(round(float(y2)))))
    return [x1, y1, x2, y2]


def build_class_mask_from_yolo_result(
    result,
    class_names: Sequence[str],
    use_box_if_no_mask: bool = True,
    dilate_kernel_size: int = 0,
    close_kernel_size: int = 0,
) -> np.ndarray:
    h, w = result.orig_shape
    out = np.zeros((h, w), dtype=np.uint8)
    class_names = set(parse_name_list(class_names))

    if result.boxes is None or len(result.boxes) == 0 or not class_names:
        return out

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    target_class_ids = {cls_id for cls_id, cls_name in names.items() if str(cls_name) in class_names}
    if not target_class_ids:
        return out

    for i, cls_id in enumerate(classes):
        if int(cls_id) not in target_class_ids:
            continue

        m = _get_mask_from_yolo_result(result, i, h, w) if result.masks is not None else None
        if m is not None:
            out = np.maximum(out, m.astype(np.uint8))
        elif use_box_if_no_mask:
            x1, y1, x2, y2 = _normalize_box(*boxes[i], w=w, h=h)
            if x2 > x1 and y2 > y1:
                out[y1:y2 + 1, x1:x2 + 1] = 1

    if out.sum() <= 0:
        return out

    if dilate_kernel_size and dilate_kernel_size > 0:
        k = int(dilate_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)

    if close_kernel_size and close_kernel_size > 0:
        k = int(close_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    return (out > 0).astype(np.uint8)


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

        x1, y1, x2, y2 = boxes[i]
        box = _normalize_box(x1, y1, x2, y2, w=w, h=h)
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


def move_anchor_to_inner_point(anchor_x: int, anchor_y: int, target_mask: Optional[np.ndarray], search_radius: int = ANCHOR_INNER_SEARCH_RADIUS, min_dist: float = ANCHOR_INNER_MIN_DIST) -> Tuple[int, int, float, bool]:
    if target_mask is None or target_mask.size == 0:
        return int(anchor_x), int(anchor_y), 0.0, False

    h, w = target_mask.shape[:2]
    ax = int(anchor_x)
    ay = int(anchor_y)
    if ax < 0 or ax >= w or ay < 0 or ay >= h:
        return ax, ay, 0.0, False

    x1 = max(0, ax - int(search_radius))
    y1 = max(0, ay - int(search_radius))
    x2 = min(w, ax + int(search_radius) + 1)
    y2 = min(h, ay + int(search_radius) + 1)
    crop = (target_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if crop.sum() <= 0:
        return ax, ay, 0.0, False

    dist = cv2.distanceTransform(crop * 255, cv2.DIST_L2, 5)
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist)
    if max_dist < float(min_dist):
        return ax, ay, float(max_dist), False
    inner_x = x1 + max_loc[0]
    inner_y = y1 + max_loc[1]
    return int(inner_x), int(inner_y), float(max_dist), True


def calc_point_around_radius(sx1, sy1, sx2, sy2) -> int:
    if POINT_AROUND_RADIUS_PX is not None:
        return max(1, int(POINT_AROUND_RADIUS_PX))
    bw = max(1, int(sx2 - sx1))
    bh = max(1, int(sy2 - sy1))
    r = int(min(bw, bh) * POINT_AROUND_RADIUS_RATIO)
    r = max(POINT_AROUND_RADIUS_MIN_PX, r)
    r = min(POINT_AROUND_RADIUS_MAX_PX, r)
    return int(r)


def generate_points_around_anchor(anchor_x: int, anchor_y: int, image_shape: Sequence[int], target_mask=None, avoid_mask=None, radius: int = 80) -> List[Dict[str, Any]]:
    r = int(radius)
    offsets = [
        (0, 0), (0, -r), (r, 0), (0, r), (-r, 0),
        (-r, -r), (r, -r), (-r, r), (r, r),
        (0, -r // 2), (r // 2, 0), (0, r // 2), (-r // 2, 0),
    ]
    points = []
    seen = set()
    for dx, dy in offsets:
        x = int(anchor_x + dx)
        y = int(anchor_y + dy)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if is_point_valid_for_masks(x, y, image_shape, target_mask=target_mask, avoid_mask=avoid_mask):
            points.append({"x": x, "y": y, "from_anchor": True, "dx": int(dx), "dy": int(dy)})
    return points


def filter_points_by_min_distance(points: List[Dict[str, Any]], min_distance: int, max_points: int) -> List[Dict[str, Any]]:
    if not points:
        return []
    selected = []
    for p in points:
        if len(selected) >= max_points:
            break
        x = int(p["x"])
        y = int(p["y"])
        too_close = False
        for sp in selected:
            sx = int(sp["x"])
            sy = int(sp["y"])
            dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist < float(min_distance):
                too_close = True
                break
        if not too_close:
            selected.append(p)
    if len(selected) < max_points:
        for p in points:
            if len(selected) >= max_points:
                break
            exists = any(int(sp["x"]) == int(p["x"]) and int(sp["y"]) == int(p["y"]) for sp in selected)
            if not exists:
                selected.append(p)
    return selected[:max_points]


def make_candidate_points_from_yolo_box(image_rgb: np.ndarray, point_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = point_info["box"]
    sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, img_w=w, img_h=h, shrink_ratio=BOX_SHRINK_RATIO)
    raw_points = generate_points_in_box(sx1, sy1, sx2, sy2, grid_size=CANDIDATE_GRID_SIZE)

    avoid_mask = point_info.get("avoid_mask")
    target_mask = point_info.get("target_mask")
    filtered_raw_points = []
    for x, y in raw_points:
        if is_point_valid_for_masks(x, y, image_rgb.shape, target_mask=target_mask, avoid_mask=avoid_mask):
            filtered_raw_points.append((x, y))
    if not filtered_raw_points:
        filtered_raw_points = raw_points

    scored_raw = []
    for x, y in filtered_raw_points:
        s, detail = score_point_cleanliness(image_rgb, x, y, window_size=POINT_CLEAN_WINDOW_SIZE)
        scored_raw.append({"x": int(x), "y": int(y), "point_score": float(s), "point_detail": detail, "stage": "raw_anchor_candidate"})
    if not scored_raw:
        raise RuntimeError("没有可用的原始候选点")
    scored_raw.sort(key=lambda p: p["point_score"], reverse=True)

    raw_anchor = scored_raw[0]
    raw_anchor_x = int(raw_anchor["x"])
    raw_anchor_y = int(raw_anchor["y"])
    inner_x, inner_y, inner_dist, moved_to_inner = move_anchor_to_inner_point(raw_anchor_x, raw_anchor_y, target_mask)
    around_radius = calc_point_around_radius(sx1, sy1, sx2, sy2)
    around_points = generate_points_around_anchor(inner_x, inner_y, image_rgb.shape, target_mask=target_mask, avoid_mask=avoid_mask, radius=around_radius)

    if not around_points:
        fallback_x, fallback_y = inner_x, inner_y
        if not is_point_valid_for_masks(fallback_x, fallback_y, image_rgb.shape, target_mask=target_mask, avoid_mask=avoid_mask):
            fallback_x, fallback_y = raw_anchor_x, raw_anchor_y
        around_points = [{"x": int(fallback_x), "y": int(fallback_y), "from_anchor": True, "dx": 0, "dy": 0}]

    scored_around = []
    for p in around_points:
        x = int(p["x"])
        y = int(p["y"])
        s, detail = score_point_cleanliness(image_rgb, x, y, window_size=POINT_CLEAN_WINDOW_SIZE)
        anchor_bonus = 0.15 if int(p.get("dx", 0)) == 0 and int(p.get("dy", 0)) == 0 else 0.0
        scored_around.append({
            "x": x,
            "y": y,
            "point_score": float(s + anchor_bonus),
            "point_detail": detail,
            "stage": "around_inner_anchor",
            "raw_anchor": [raw_anchor_x, raw_anchor_y],
            "inner_anchor": [inner_x, inner_y],
            "moved_to_inner": bool(moved_to_inner),
            "inner_dist": float(inner_dist),
            "around_radius": int(around_radius),
            "dx": int(p.get("dx", 0)),
            "dy": int(p.get("dy", 0)),
        })
    scored_around.sort(key=lambda p: p["point_score"], reverse=True)
    min_point_distance = int(max(1, around_radius * MIN_POINT_DISTANCE_RATIO_FOR_AROUND))
    selected_points = filter_points_by_min_distance(scored_around, min_distance=min_point_distance, max_points=MAX_SAM_TRY_POINTS)
    if not selected_points:
        raise RuntimeError("没有可用候选点")

    meta = {
        "shrink_box": [int(sx1), int(sy1), int(sx2), int(sy2)],
        "raw_anchor": [int(raw_anchor_x), int(raw_anchor_y)],
        "inner_anchor": [int(inner_x), int(inner_y)],
        "moved_to_inner": bool(moved_to_inner),
        "inner_dist": float(inner_dist),
        "around_radius": int(around_radius),
        "min_point_distance": int(min_point_distance),
        "raw_candidates": scored_raw,
    }
    return selected_points, meta


# =========================
# 中心兜底点
# =========================
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
        return {
            "x": base_x,
            "y": base_y,
            "point_score": float(score),
            "point_detail": detail,
            "stage": "center_fallback",
            "base_point": [base_x, base_y],
            "center_hit_avoid": False,
            "avoid_adjusted": False,
            "search_radius": 0,
        }

    step = max(5, int(FALLBACK_POINT_SEARCH_STEP_PX))
    max_radius = int(min(h, w) * float(FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO))
    max_radius = max(step, max_radius)
    angle_count = max(8, int(FALLBACK_POINT_SEARCH_ANGLE_COUNT))

    for radius in range(step, max_radius + step, step):
        ring_candidates = []
        for i in range(angle_count):
            angle = 2.0 * np.pi * i / angle_count
            x = int(round(base_x + np.cos(angle) * radius))
            y = int(round(base_y + np.sin(angle) * radius))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if is_point_on_avoid_mask(x, y, image_rgb.shape, avoid_mask):
                continue
            score, detail = score_point_cleanliness(image_rgb, x, y)
            ring_candidates.append({
                "x": int(x),
                "y": int(y),
                "point_score": float(score),
                "point_detail": detail,
                "stage": "center_fallback_avoid_adjusted",
                "base_point": [base_x, base_y],
                "center_hit_avoid": True,
                "avoid_adjusted": True,
                "search_radius": int(radius),
            })
        if ring_candidates:
            ring_candidates.sort(key=lambda p: p["point_score"], reverse=True)
            return ring_candidates[0]

    score, detail = score_point_cleanliness(image_rgb, base_x, base_y)
    return {
        "x": base_x,
        "y": base_y,
        "point_score": float(score),
        "point_detail": detail,
        "stage": "center_fallback_force_use_center",
        "base_point": [base_x, base_y],
        "center_hit_avoid": True,
        "avoid_adjusted": False,
        "search_radius": -1,
        "message": "no_available_point_outside_avoid_mask",
    }


def make_candidate_point_from_center_fallback(image_rgb: np.ndarray, result=None, target_class_name: str = "plate") -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    base_xy = ratio_to_xy(PLATE_FALLBACK_POINT_RATIO, image_rgb.shape)
    if base_xy is None:
        base_xy = (w // 2, h // 2)
    base_x, base_y = base_xy
    avoid_mask = build_fallback_avoid_mask(result, image_rgb.shape, avoid_class_names=FALLBACK_AVOID_CLASS_NAMES)
    p = find_nearest_point_outside_avoid_mask(image_rgb, base_x, base_y, avoid_mask=avoid_mask)
    candidate_points = [{
        "x": int(p["x"]),
        "y": int(p["y"]),
        "point_score": float(p.get("point_score", 1.0)),
        "point_detail": p.get("point_detail", {}),
        "stage": p.get("stage", "center_fallback"),
        "base_point": p.get("base_point", [base_x, base_y]),
        "center_hit_avoid": bool(p.get("center_hit_avoid", False)),
        "avoid_adjusted": bool(p.get("avoid_adjusted", False)),
        "search_radius": int(p.get("search_radius", 0)),
    }]
    point_info = {
        "box": None,
        "conf": None,
        "area": 0,
        "box_area": 0,
        "mask_area": 0,
        "class_name": target_class_name,
        "target_mask": None,
        "avoid_mask": avoid_mask,
        "x": int(p["x"]),
        "y": int(p["y"]),
        "mode": "center_fallback_no_plate",
        "fallback_ratio": parse_user_point_ratio(PLATE_FALLBACK_POINT_RATIO),
        "fallback_base_point": [int(base_x), int(base_y)],
        "center_hit_avoid": bool(p.get("center_hit_avoid", False)),
        "avoid_adjusted": bool(p.get("avoid_adjusted", False)),
        "avoid_mask_area": int((avoid_mask > 0).sum()) if avoid_mask is not None else 0,
        "search_radius": int(p.get("search_radius", 0)),
    }
    point_meta = {
        "shrink_box": "",
        "raw_anchor": [int(base_x), int(base_y)],
        "inner_anchor": [int(p["x"]), int(p["y"])],
        "moved_to_inner": bool(p.get("avoid_adjusted", False)),
        "inner_dist": 0.0,
        "around_radius": "",
        "min_point_distance": "",
        "raw_candidates": candidate_points,
    }
    return candidate_points, point_info, point_meta


# =========================
# paper：YOLO 识别后只取一个点给 SAM2
# =========================
def get_largest_yolo_instance_by_classes(result, class_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    class_names_set = set(parse_name_list(class_names))
    if result is None or result.boxes is None or len(result.boxes) == 0 or not class_names_set:
        return None

    h, w = result.orig_shape
    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    target_class_ids = {cls_id for cls_id, cls_name in names.items() if str(cls_name) in class_names_set}
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
    return max(candidates, key=lambda item: item["area"])


def choose_one_point_inside_yolo_instance(instance: Dict[str, Any], image_shape: Sequence[int]) -> Dict[str, Any]:
    """
    paper 只取一个点：
    - 优先从 YOLO mask 中用 distanceTransform 取最内部的点。
    - 没有 mask 时取 YOLO box 中心点。
    """
    h, w = image_shape[:2]
    mask = instance.get("mask")
    if mask is not None and mask.size > 0 and int((mask > 0).sum()) > 0:
        bin_mask = (mask > 0).astype(np.uint8)
        dist = cv2.distanceTransform(bin_mask * 255, cv2.DIST_L2, 5)
        _, max_dist, _, max_loc = cv2.minMaxLoc(dist)
        x = int(max_loc[0])
        y = int(max_loc[1])
        return {
            "x": x,
            "y": y,
            "source": "yolo_mask_distance_transform",
            "inner_dist": float(max_dist),
        }

    x1, y1, x2, y2 = instance["box"]
    x = int(round((x1 + x2) / 2.0))
    y = int(round((y1 + y2) / 2.0))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return {
        "x": x,
        "y": y,
        "source": "yolo_box_center",
        "inner_dist": 0.0,
    }


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
    h, w = image_rgb.shape[:2]
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))
    request = osam.types.GenerateRequest(
        model=model_name,
        image=image_rgb,
        prompt=osam.types.Prompt(points=[[x, y]], point_labels=[1]),
    )
    response = osam.apis.generate(request=request)
    if not response.annotations:
        return []
    masks = []
    for annotation in response.annotations:
        full_mask = _sam_annotation_to_full_mask(annotation, h, w)
        if full_mask is not None:
            masks.append(full_mask)
    return masks


def run_sam2_masks_by_points(image_rgb: np.ndarray, points: Sequence[Dict[str, Any]], model_name: str = "sam2") -> List[np.ndarray]:
    if not points:
        return []
    h, w = image_rgb.shape[:2]
    sam_points = []
    for p in points:
        x = int(max(0, min(w - 1, int(p["x"]))))
        y = int(max(0, min(h - 1, int(p["y"]))))
        sam_points.append([x, y])
    request = osam.types.GenerateRequest(
        model=model_name,
        image=image_rgb,
        prompt=osam.types.Prompt(points=sam_points, point_labels=[1] * len(sam_points)),
    )
    response = osam.apis.generate(request=request)
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
    best.update({
        "x": int(x),
        "y": int(y),
        "mode": "user_ratio_point",
        "user_ratio": ratio,
        "used_points": [{"x": int(x), "y": int(y), "ratio_x": float(ratio[0]), "ratio_y": float(ratio[1])}],
    })
    return mask, best


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
                tried.append({
                    "mode": "multi_points",
                    "points": [[int(p["x"]), int(p["y"])] for p in multi_points],
                    "point_score": avg_point_score,
                    "sam_score": float(sam_score),
                    "sam_detail": detail,
                    "mask_index": int(mask_idx),
                })
                if best is None or item["sam_score"] > best["sam_score"]:
                    best = item
            if best is not None and best["sam_score"] > -100 and best["sam_detail"].get("reason") == "ok":
                best["tried"] = tried
                return best["mask"], best
        except Exception as e:
            tried.append({"mode": "multi_points", "message": str(e), "sam_score": -9999.0})

    if not ENABLE_SINGLE_POINT_FALLBACK:
        if best is not None:
            best["tried"] = tried
            return best["mask"], best
        raise RuntimeError(f"多点 SAM2 未得到有效结果，且 ENABLE_SINGLE_POINT_FALLBACK=False，tried={tried[:5]}")

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


def run_sam2_for_paper_from_yolo(
    result,
    image_rgb: np.ndarray,
    paper_class_names: Sequence[str],
    model_name: str,
    fallback_to_yolo_mask: bool = True,
    detect_by_sam2: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    A4纸识别：YOLO 找到 paper -> 只取一个点 -> SAM2 识别 paper 范围。
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
        info["message"] = (
            "YOLO 未识别到 paper。请检查：1）paper_class_names 是否和 model.names 完全一致；"
            "2）yolo_conf 是否过高；3）这张图是否因方向/EXIF导致旧流程坐标错位。"
        )
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
            masks = run_sam2_masks_by_point(
                image_rgb=image_rgb,
                x=int(point["x"]),
                y=int(point["y"]),
                model_name=model_name,
            )
            if masks:
                paper_mask, best = pick_best_mask_from_masks(masks, image_rgb.shape[:2], "paper", point_score=1.0)
                if best["sam_detail"].get("reason") == "ok" and best["sam_score"] > -100:
                    info["paper_mask_source"] = "sam2_one_point_from_yolo_paper"
                    info["sam_info"] = {
                        "x": int(point["x"]),
                        "y": int(point["y"]),
                        "mode": "paper_one_point",
                        "sam_score": float(best["sam_score"]),
                        "sam_detail": best["sam_detail"],
                        "mask_index": int(best["mask_index"]),
                    }
                    info["message"] = "YOLO 已识别 paper，并用单点传给 SAM2 成功生成 paper mask"
                    return paper_mask, info
                info["sam_info"] = {
                    "sam_score": float(best["sam_score"]),
                    "sam_detail": best["sam_detail"],
                    "message": "SAM2 paper mask 评分不合理",
                }
            else:
                info["sam_info"] = {"message": "SAM2 没有返回 paper mask"}
        except Exception as e:
            info["sam_info"] = {"message": str(e), "traceback": traceback.format_exc(limit=5)}

    if fallback_to_yolo_mask:
        yolo_mask = make_yolo_instance_mask_or_box(instance, image_rgb.shape)
        if yolo_mask is not None and yolo_mask.sum() > 0:
            info["paper_mask_source"] = "yolo_mask_or_box_fallback"
            info["message"] = "SAM2 paper 未成功，已退回 YOLO paper mask/box 作为补洞区域"
            return yolo_mask.astype(np.uint8) * 255, info

    info["paper_mask_source"] = None
    info["message"] = "YOLO 找到 paper，但 SAM2 没有成功生成 paper mask，且未启用/无法使用 YOLO fallback"
    return None, info


# =========================
# debug 输出
# =========================
def save_debug_point_image_from_rgb(image_rgb: np.ndarray, point_info: Dict[str, Any], output_debug_path: str | Path) -> str:
    img = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    x = int(point_info["x"])
    y = int(point_info["y"])
    cls_name = point_info.get("class_name", "target")
    conf = point_info.get("conf")
    mode = point_info.get("mode", "auto")

    if point_info.get("box"):
        x1, y1, x2, y2 = point_info["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    if point_info.get("shrink_box"):
        try:
            sx1, sy1, sx2, sy2 = point_info["shrink_box"]
            draw.rectangle([sx1, sy1, sx2, sy2], outline="orange", width=4)
        except Exception:
            pass

    if point_info.get("raw_anchor"):
        ax, ay = point_info["raw_anchor"]
        rr = 9
        draw.ellipse([ax - rr, ay - rr, ax + rr, ay + rr], fill="purple", outline="white", width=2)
        draw.text((ax + 10, ay - 10), "raw", fill="purple")

    if point_info.get("inner_anchor"):
        ix, iy = point_info["inner_anchor"]
        rr = 9
        draw.ellipse([ix - rr, iy - rr, ix + rr, iy + rr], fill="green", outline="white", width=2)
        draw.text((ix + 10, iy - 10), "inner", fill="green")

    for idx, p in enumerate(point_info.get("candidate_points", []), start=1):
        px = int(p["x"])
        py = int(p["y"])
        rr = 7
        draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill="yellow", outline="black", width=2)
        draw.text((px + 8, py - 8), str(idx), fill="yellow")

    r = 13
    draw.ellipse([x - r, y - r, x + r, y + r], fill="blue", outline="white", width=3)

    if mode == "user_ratio_point":
        ratio = point_info.get("user_ratio")
        text = f"{cls_name} manual_ratio={ratio} selected=({x},{y})"
    else:
        text = f"{cls_name} selected=({x},{y})" if conf is None else f"{cls_name} conf={conf:.2f} selected=({x},{y})"
    if "sam_score" in point_info:
        text += f" sam_score={float(point_info['sam_score']):.3f}"
    draw.text((x + 16, y - 16), text, fill="blue")

    path = Path(output_debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)


def save_paper_debug_image_from_rgb(image_rgb: np.ndarray, paper_info: Dict[str, Any], output_debug_path: str | Path) -> Optional[str]:
    point = paper_info.get("paper_point")
    yolo_paper = paper_info.get("yolo_paper")
    if not point and not yolo_paper:
        return None

    img = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    if yolo_paper and yolo_paper.get("box"):
        x1, y1, x2, y2 = yolo_paper["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1 + 8, y1 + 8), f"paper conf={yolo_paper.get('conf', 0):.2f}", fill="red")

    if point:
        x = int(point["x"])
        y = int(point["y"])
        r = 13
        draw.ellipse([x - r, y - r, x + r, y + r], fill="blue", outline="white", width=3)
        draw.text((x + 16, y - 16), f"paper point=({x},{y})", fill="blue")

    path = Path(output_debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)


# =========================
# 后处理：paper 补回 plate
# =========================
def apply_paper_fill_to_plate_mask(plate_mask: np.ndarray, paper_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if plate_mask is None:
        raise RuntimeError("plate_mask 为空，无法补回 A4纸区域")

    plate_bin = plate_mask > 0
    if paper_mask is None or paper_mask.size == 0 or int((paper_mask > 0).sum()) <= 0:
        return plate_bin.astype(np.uint8) * 255, {
            "filled": False,
            "paper_area": 0,
            "added_area": 0,
            "reason": "no_paper_mask",
        }

    paper_bin = paper_mask > 0
    if paper_bin.shape != plate_bin.shape:
        paper_bin = cv2.resize(paper_bin.astype(np.uint8), (plate_bin.shape[1], plate_bin.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    added = paper_bin & (~plate_bin)
    final_bin = plate_bin | paper_bin
    return final_bin.astype(np.uint8) * 255, {
        "filled": True,
        "paper_area": int(paper_bin.sum()),
        "added_area": int(added.sum()),
        "reason": "ok",
    }




# =========================
# 尺寸计算与 DXF 生成
# =========================
def _binary_mask(mask: Optional[np.ndarray]) -> np.ndarray:
    """把任意 mask 统一成 uint8 二值图。"""
    if mask is None or mask.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def _largest_contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """从 mask 中取面积最大的外轮廓。"""
    bin_mask = _binary_mask(mask)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def calculate_a4_scale_from_paper_mask(
    paper_mask: np.ndarray,
    a4_width_mm: float = 210.0,
    a4_height_mm: float = 297.0,
    scale_method: str = "mean_edge",
) -> Dict[str, Any]:
    """
    根据 A4 纸 mask 计算像素到毫米的比例。

    说明：
    - 使用 A4 的最小外接旋转矩形估算短边和长边像素长度。
    - A4 实际短边是 210mm，长边是 297mm。
    - 默认采用短边比例和长边比例的平均值，降低单边分割误差影响。
    """
    contour = _largest_contour_from_mask(paper_mask)
    if contour is None or cv2.contourArea(contour) <= 1:
        raise RuntimeError("无法根据 paper mask 计算 A4 尺寸：paper mask 为空")

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    side1_px = float(rw)
    side2_px = float(rh)
    if side1_px <= 1 or side2_px <= 1:
        raise RuntimeError(f"无法根据 paper mask 计算 A4 尺寸：旋转矩形尺寸异常 side=({side1_px}, {side2_px})")

    short_px = min(side1_px, side2_px)
    long_px = max(side1_px, side2_px)
    short_mm = min(float(a4_width_mm), float(a4_height_mm))
    long_mm = max(float(a4_width_mm), float(a4_height_mm))

    short_edge_mm_per_px = short_mm / short_px
    long_edge_mm_per_px = long_mm / long_px

    contour_area_px = float(cv2.contourArea(contour))
    mask_area_px = int((_binary_mask(paper_mask) > 0).sum())
    a4_area_mm2 = short_mm * long_mm
    area_mm_per_px = float((a4_area_mm2 / max(1.0, contour_area_px)) ** 0.5)

    method = (scale_method or "mean_edge").strip().lower()
    if method == "short_edge":
        mm_per_px = short_edge_mm_per_px
    elif method == "long_edge":
        mm_per_px = long_edge_mm_per_px
    elif method == "area":
        mm_per_px = area_mm_per_px
    elif method == "mean_edge":
        mm_per_px = (short_edge_mm_per_px + long_edge_mm_per_px) / 2.0
    else:
        raise RuntimeError("scale_method 只支持 mean_edge / area / short_edge / long_edge")

    box_points = cv2.boxPoints(rect).astype(np.float32)

    return {
        "a4_width_mm": float(a4_width_mm),
        "a4_height_mm": float(a4_height_mm),
        "a4_short_mm": float(short_mm),
        "a4_long_mm": float(long_mm),
        "paper_short_px": float(short_px),
        "paper_long_px": float(long_px),
        "paper_contour_area_px": float(contour_area_px),
        "paper_mask_area_px": int(mask_area_px),
        "short_edge_mm_per_px": float(short_edge_mm_per_px),
        "long_edge_mm_per_px": float(long_edge_mm_per_px),
        "area_mm_per_px": float(area_mm_per_px),
        "mm_per_px": float(mm_per_px),
        "px_per_mm": float(1.0 / mm_per_px),
        "scale_method": method,
        "paper_min_area_rect": {
            "center": [float(cx), float(cy)],
            "size_px": [float(rw), float(rh)],
            "angle": float(angle),
            "box_points": [[float(x), float(y)] for x, y in box_points],
        },
    }


def extract_plate_contours_for_dxf(
    plate_mask: np.ndarray,
    mm_per_px: float,
    contour_epsilon_mm: float = 2.0,
    min_contour_area_mm2: float = 100.0,
    include_holes: bool = True,
) -> Dict[str, Any]:
    """
    从最终钢板 mask 提取 DXF 轮廓。

    输出只保留最大外轮廓，避免环境噪点进入 DXF。
    如果 include_holes=True，则保留最大外轮廓下面的内轮廓。
    """
    bin_mask = _binary_mask(plate_mask)
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise RuntimeError("无法生成 DXF：plate mask 中没有可用轮廓")

    hierarchy = hierarchy[0]
    min_area_px = float(min_contour_area_mm2) / max(1e-9, float(mm_per_px) ** 2)
    epsilon_px = max(0.5, float(contour_epsilon_mm) / max(1e-9, float(mm_per_px)))

    external_indices = [i for i, h in enumerate(hierarchy) if int(h[3]) == -1]
    if not external_indices:
        raise RuntimeError("无法生成 DXF：未找到外轮廓")

    largest_external_idx = max(external_indices, key=lambda i: cv2.contourArea(contours[i]))
    selected = []

    def add_contour(idx: int, role: str):
        area_px = float(abs(cv2.contourArea(contours[idx])))
        if area_px < min_area_px:
            return
        approx = cv2.approxPolyDP(contours[idx], epsilon_px, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            return
        selected.append({
            "role": role,
            "index": int(idx),
            "area_px": area_px,
            "area_mm2": float(area_px * mm_per_px * mm_per_px),
            "point_count": int(len(pts)),
            "points_px": [[float(x), float(y)] for x, y in pts],
        })

    add_contour(largest_external_idx, "outer")

    if include_holes:
        for i, h in enumerate(hierarchy):
            parent = int(h[3])
            if parent == largest_external_idx:
                add_contour(i, "hole")

    if not selected or selected[0]["role"] != "outer":
        raise RuntimeError("无法生成 DXF：外轮廓面积过小或简化后无效")

    return {
        "contours": selected,
        "epsilon_px": float(epsilon_px),
        "epsilon_mm": float(contour_epsilon_mm),
        "min_contour_area_px": float(min_area_px),
        "min_contour_area_mm2": float(min_contour_area_mm2),
    }


def convert_contours_to_mm_polylines(contour_data: Dict[str, Any], mm_per_px: float) -> Dict[str, Any]:
    """
    将图像像素坐标转换为 DXF 毫米坐标。

    图像坐标 y 轴向下，DXF 坐标 y 轴向上，所以这里会翻转 y 轴。
    """
    all_points = []
    for item in contour_data["contours"]:
        all_points.extend(item["points_px"])
    if not all_points:
        raise RuntimeError("无法转换 DXF 坐标：没有轮廓点")

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x = float(min(xs))
    max_x = float(max(xs))
    min_y = float(min(ys))
    max_y = float(max(ys))

    polylines = []
    for item in contour_data["contours"]:
        pts_mm = []
        for x, y in item["points_px"]:
            x_mm = (float(x) - min_x) * mm_per_px
            y_mm = (max_y - float(y)) * mm_per_px
            pts_mm.append([round(x_mm, 3), round(y_mm, 3)])
        polylines.append({
            "role": item["role"],
            "layer": "PLATE_OUTER" if item["role"] == "outer" else "PLATE_HOLE",
            "points_mm": pts_mm,
            "area_mm2": item["area_mm2"],
            "point_count": item["point_count"],
        })

    width_mm = (max_x - min_x) * mm_per_px
    height_mm = (max_y - min_y) * mm_per_px

    return {
        "polylines": polylines,
        "origin_px": [float(min_x), float(max_y)],
        "source_bbox_px": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "dxf_extents_mm": {
            "width_mm": float(width_mm),
            "height_mm": float(height_mm),
            "length_mm": float(max(width_mm, height_mm)),
            "short_side_mm": float(min(width_mm, height_mm)),
        },
    }


def build_lwpolyline_dxf(polylines: Sequence[Dict[str, Any]]) -> str:
    """生成最小可用 DXF 内容，单位为毫米。"""
    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "TABLES",
        "0", "TABLE",
        "2", "LAYER",
        "70", "2",
        "0", "LAYER", "2", "PLATE_OUTER", "70", "0", "62", "7", "6", "CONTINUOUS",
        "0", "LAYER", "2", "PLATE_HOLE", "70", "0", "62", "1", "6", "CONTINUOUS",
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    for poly in polylines:
        pts = poly.get("points_mm") or []
        if len(pts) < 3:
            continue
        lines.extend([
            "0", "LWPOLYLINE",
            "8", str(poly.get("layer") or "PLATE"),
            "90", str(len(pts)),
            "70", "1",
        ])
        for x, y in pts:
            lines.extend(["10", f"{float(x):.3f}", "20", f"{float(y):.3f}"])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"


def calculate_plate_rotated_rect_dimension(plate_mask: np.ndarray, mm_per_px: float) -> Dict[str, Any]:
    """计算钢板最大外轮廓的旋转矩形尺寸。"""
    contour = _largest_contour_from_mask(plate_mask)
    if contour is None or cv2.contourArea(contour) <= 1:
        raise RuntimeError("无法计算钢板尺寸：plate mask 为空")

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    side1_px = float(rw)
    side2_px = float(rh)
    long_px = max(side1_px, side2_px)
    short_px = min(side1_px, side2_px)
    contour_area_px = float(abs(cv2.contourArea(contour)))
    mask_area_px = int((_binary_mask(plate_mask) > 0).sum())
    box_points = cv2.boxPoints(rect).astype(np.float32)

    return {
        "length_mm": float(long_px * mm_per_px),
        "width_mm": float(short_px * mm_per_px),
        "rotated_rect_side_px": [float(side1_px), float(side2_px)],
        "rotated_rect_angle": float(angle),
        "rotated_rect_center_px": [float(cx), float(cy)],
        "rotated_rect_box_points_px": [[float(x), float(y)] for x, y in box_points],
        "contour_area_mm2": float(contour_area_px * mm_per_px * mm_per_px),
        "mask_area_mm2": float(mask_area_px * mm_per_px * mm_per_px),
        "contour_area_px": float(contour_area_px),
        "mask_area_px": int(mask_area_px),
    }


def generate_dxf_and_dimensions(
    plate_mask: np.ndarray,
    paper_mask: np.ndarray,
    output_dxf_path: str | Path,
    req: ProcessRequest,
) -> Dict[str, Any]:
    """基于 A4 标定结果生成钢板尺寸和 DXF 文件。"""
    a4_scale = calculate_a4_scale_from_paper_mask(
        paper_mask=paper_mask,
        a4_width_mm=float(req.a4_width_mm),
        a4_height_mm=float(req.a4_height_mm),
        scale_method=req.scale_method,
    )
    mm_per_px = float(a4_scale["mm_per_px"])

    contour_data = extract_plate_contours_for_dxf(
        plate_mask=plate_mask,
        mm_per_px=mm_per_px,
        contour_epsilon_mm=float(req.contour_epsilon_mm),
        min_contour_area_mm2=float(req.min_contour_area_mm2),
        include_holes=bool(req.include_holes_in_dxf),
    )
    dxf_geometry = convert_contours_to_mm_polylines(contour_data, mm_per_px=mm_per_px)
    dxf_content = build_lwpolyline_dxf(dxf_geometry["polylines"])

    dxf_path = Path(output_dxf_path)
    dxf_path.parent.mkdir(parents=True, exist_ok=True)
    dxf_path.write_text(dxf_content, encoding="utf-8")

    plate_rect_dimension = calculate_plate_rotated_rect_dimension(plate_mask, mm_per_px=mm_per_px)

    return {
        "dxf_path": str(dxf_path),
        "dxf_content": dxf_content if bool(req.dxf_include_content) else None,
        "dimensions": {
            "a4": a4_scale,
            "plate": {
                **plate_rect_dimension,
                "dxf_extents_mm": dxf_geometry["dxf_extents_mm"],
                "outer_point_count": int(dxf_geometry["polylines"][0]["point_count"]),
                "hole_count": int(sum(1 for p in dxf_geometry["polylines"] if p.get("role") == "hole")),
            },
        },
    }


# =========================
# 单图处理主流程
# =========================
def process_image_rgb(
    image_rgb: np.ndarray,
    req: ProcessRequest,
    original_name: str = "image.jpg",
) -> Dict[str, Any]:
    yolo_model_path = req.yolo_model_path or RUNTIME_CONFIG.get("yolo_model_path") or DEFAULT_YOLO_MODEL_PATH
    output_root = Path(req.output_root or RUNTIME_CONFIG.get("output_root") or DEFAULT_OUTPUT_ROOT)
    request_id = uuid.uuid4().hex[:12]
    stem = Path(original_name).stem or "image"
    run_dir = output_root / f"{stem}_{request_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 关键：不要直接把 ndarray 喂给 YOLO。
    # 先把当前 image_rgb 保存成无 EXIF、无方向歧义的标准中间图，再让 YOLO 读这张中间图路径。
    # SAM2 也重新从这张中间图读取 image_rgb，保证二者看到的是同一张图、同一个尺寸、同一套坐标。
    canonical_image_path = run_dir / "input_canonical_used_by_yolo_and_sam2.png"
    input_saved_path = save_image_rgb(image_rgb, canonical_image_path)
    image_rgb = load_image_rgb_from_path(canonical_image_path)

    model = get_yolo_model(str(yolo_model_path))
    result = run_yolo_on_canonical_image(
        model=model,
        image_rgb=image_rgb,
        canonical_image_path=canonical_image_path,
        conf=req.yolo_conf,
        imgsz=req.yolo_imgsz,
        input_mode=req.yolo_input_mode,
    )

    h, w = image_rgb.shape[:2]
    result_shape = tuple(int(x) for x in result.orig_shape)
    shape_warning = None
    if result_shape != (h, w):
        shape_warning = f"YOLO result.orig_shape={result_shape} 与 image_rgb.shape={(h, w)} 不一致，请检查 ultralytics 输入处理"

    target_class_names = parse_name_list(req.target_class_names) or ["plate"]
    manual_ratio = parse_user_point_ratio(req.user_point_ratio)
    use_manual_point = manual_ratio is not None

    outputs: Dict[str, Any] = {
        "request_id": request_id,
        "run_dir": str(run_dir),
        "input_rgb_path": input_saved_path,
        "image_shape": {"height": int(h), "width": int(w)},
        "yolo_model_path": str(yolo_model_path),
        "yolo_input_mode": req.yolo_input_mode,
        "canonical_image_path": str(canonical_image_path),
        "yolo_model_names": {str(k): str(v) for k, v in model.names.items()},
        "yolo_detected_classes": detected_class_names(result),
        "shape_warning": shape_warning,
        "same_image_rgb": True,
        "targets": {},
        "paper": None,
        "fill_paper_to_plate": None,
    }

    generated_masks: Dict[str, np.ndarray] = {}
    mask_paths: Dict[str, Any] = {
        "canonical_image": str(canonical_image_path),
    }
    final_plate_mask: Optional[np.ndarray] = None

    # 先识别 paper：只用一个 YOLO paper 点跑 SAM2。
    paper_mask = None
    paper_info = None
    if req.fill_paper_to_plate and "plate" in target_class_names:
        paper_mask, paper_info = run_sam2_for_paper_from_yolo(
            result=result,
            image_rgb=image_rgb,
            paper_class_names=req.paper_class_names,
            model_name=req.sam_model_name,
            fallback_to_yolo_mask=req.paper_sam2_fallback_to_yolo_mask,
            detect_by_sam2=req.detect_paper_by_sam2,
        )
        outputs["paper"] = paper_info
        if paper_mask is not None and int((paper_mask > 0).sum()) > 0:
            paper_mask_path = save_mask(paper_mask, run_dir / "paper" / f"{stem}_paper_mask.png")
            outputs["paper"]["mask_path"] = paper_mask_path
            mask_paths["paper_mask"] = paper_mask_path
            if req.return_mask_base64:
                outputs["paper"]["mask_base64_png"] = mask_to_base64_png(paper_mask)
        if req.save_debug_images:
            debug_path = save_paper_debug_image_from_rgb(image_rgb, paper_info or {}, run_dir / "paper" / f"{stem}_paper_debug.jpg")
            if debug_path and outputs.get("paper") is not None:
                outputs["paper"]["debug_path"] = debug_path
                mask_paths["paper_debug"] = debug_path

    for cls_name in target_class_names:
        cls_dir = run_dir / cls_name
        mask_path = cls_dir / f"{stem}_{cls_name}_mask.png"
        debug_path = cls_dir / f"{stem}_{cls_name}_debug.jpg"

        if use_manual_point:
            sam_mask, sam_info = run_sam2_by_user_ratio_point(
                image_rgb=image_rgb,
                user_point_ratio=manual_ratio,
                target_class_name=cls_name,
                model_name=req.sam_model_name,
            )
            point_info = {
                "x": int(sam_info["x"]),
                "y": int(sam_info["y"]),
                "class_name": cls_name,
                "mode": "user_ratio_point",
                "user_ratio": manual_ratio,
                "sam_score": float(sam_info["sam_score"]),
                "sam_detail": sam_info["sam_detail"],
                "candidate_points": sam_info.get("used_points", []),
            }
        else:
            try:
                avoid_class_names = AVOID_BY_TARGET.get(cls_name, [])
                point_info = get_target_from_yolo_result(result, target_class_name=cls_name, avoid_class_names=avoid_class_names)
                candidate_points, point_meta = make_candidate_points_from_yolo_box(image_rgb, point_info)
                point_info.update(point_meta)
            except RuntimeError as yolo_target_error:
                if cls_name == "plate" and FALLBACK_TO_CENTER_WHEN_NO_PLATE and is_no_plate_detected_error(yolo_target_error):
                    candidate_points, point_info, point_meta = make_candidate_point_from_center_fallback(image_rgb, result=result, target_class_name=cls_name)
                    point_info.update(point_meta)
                else:
                    raise

            if point_info.get("mode") == "center_fallback_no_plate":
                fallback_point = candidate_points[0]
                masks = run_sam2_masks_by_point(
                    image_rgb=image_rgb,
                    x=int(fallback_point["x"]),
                    y=int(fallback_point["y"]),
                    model_name=req.sam_model_name,
                )
                if not masks:
                    raise RuntimeError(f"中心兜底点 SAM2 没有生成任何 mask，point=({fallback_point['x']},{fallback_point['y']})")
                sam_mask, sam_info = pick_best_mask_from_masks(
                    masks,
                    image_rgb.shape[:2],
                    target_class_name=cls_name,
                    point_score=float(fallback_point.get("point_score", 1.0)),
                )
                sam_info.update({
                    "x": int(fallback_point["x"]),
                    "y": int(fallback_point["y"]),
                    "mode": "center_fallback_no_plate",
                    "used_points": [fallback_point],
                })
            else:
                sam_mask, sam_info = run_sam2_by_candidate_points(
                    image_rgb=image_rgb,
                    candidate_points=candidate_points,
                    target_class_name=cls_name,
                    model_name=req.sam_model_name,
                )

            point_info["x"] = int(sam_info["x"])
            point_info["y"] = int(sam_info["y"])
            point_info["sam_score"] = float(sam_info["sam_score"])
            point_info["sam_detail"] = sam_info["sam_detail"]
            point_info["candidate_points"] = candidate_points
            point_info["sam_mode"] = sam_info.get("mode")
            point_info["used_points"] = sam_info.get("used_points", [])

        # 保存原始 mask
        raw_mask_path = save_mask(sam_mask, mask_path)
        generated_masks[cls_name] = sam_mask
        mask_paths[f"{cls_name}_mask"] = raw_mask_path

        # debug 图必须使用 image_rgb，而不是重新打开 image_path。
        debug_out = None
        if req.save_debug_images:
            debug_out = save_debug_point_image_from_rgb(image_rgb, point_info, debug_path)
            mask_paths[f"{cls_name}_debug"] = debug_out

        # 去掉无法 JSON 序列化的大数组
        clean_point_info = dict(point_info)
        clean_point_info.pop("target_mask", None)
        clean_point_info.pop("avoid_mask", None)

        outputs["targets"][cls_name] = {
            "mask_path": raw_mask_path,
            "debug_path": debug_out,
            "point_info": _json_safe(clean_point_info),
        }
        if req.return_mask_base64:
            outputs["targets"][cls_name]["mask_base64_png"] = mask_to_base64_png(sam_mask)

    # paper 补回 plate
    if req.fill_paper_to_plate and "plate" in generated_masks:
        raw_plate_path = save_mask(generated_masks["plate"], run_dir / "plate" / f"{stem}_plate_raw_before_paper_fill.png") if SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL else None
        final_plate_mask, fill_info = apply_paper_fill_to_plate_mask(generated_masks["plate"], paper_mask)
        final_path = save_mask(final_plate_mask, run_dir / "plate" / f"{stem}_plate_final_with_paper_fill.png")
        if raw_plate_path:
            mask_paths["plate_raw_before_paper_fill"] = raw_plate_path
        mask_paths["plate_final_with_paper_fill"] = final_path
        outputs["fill_paper_to_plate"] = {
            **fill_info,
            "raw_plate_mask_path": raw_plate_path,
            "final_plate_mask_path": final_path,
            "paper_mask_source": (paper_info or {}).get("paper_mask_source"),
        }
        if req.return_mask_base64:
            outputs["fill_paper_to_plate"]["final_mask_base64_png"] = mask_to_base64_png(final_plate_mask)

    # 尺寸计算与 DXF 生成。优先使用补完 A4 后的 plate mask。
    dxf_content = None
    dxf_path = None
    dimensions = None
    dimension_error = None
    plate_for_dxf = final_plate_mask if final_plate_mask is not None else generated_masks.get("plate")

    if req.enable_dxf and plate_for_dxf is not None:
        try:
            if paper_mask is None or int((paper_mask > 0).sum()) <= 0:
                raise RuntimeError("无法计算尺寸和 DXF：没有可用 paper mask，无法用 A4 标定毫米比例")
            dxf_result = generate_dxf_and_dimensions(
                plate_mask=plate_for_dxf,
                paper_mask=paper_mask,
                output_dxf_path=run_dir / "dxf" / f"{stem}_plate.dxf",
                req=req,
            )
            dxf_path = dxf_result["dxf_path"]
            dxf_content = dxf_result["dxf_content"]
            dimensions = dxf_result["dimensions"]
        except Exception as e:
            dimension_error = str(e)

    outputs["dxf_path"] = dxf_path
    outputs["dimensions"] = dimensions
    outputs["dimension_error"] = dimension_error
    outputs["mask_paths"] = mask_paths

    # 保存完整处理日志，接口只返回精简结果。
    log_path = run_dir / "result.json"
    outputs["result_json_path"] = str(log_path)
    log_path.write_text(json.dumps(_json_safe(outputs), ensure_ascii=False, indent=2), encoding="utf-8")

    response = {
        "dxf_content": dxf_content,
        "dxf_path": dxf_path,
        "mask_paths": mask_paths,
        "dimensions": dimensions,
    }
    if dimension_error:
        response["dimension_error"] = dimension_error
    return _json_safe(response)


# =========================
# FastAPI
# =========================
app = FastAPI(title="YOLO + SAM2 Plate/Paper Service", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_model_path": RUNTIME_CONFIG.get("yolo_model_path"),
        "output_root": RUNTIME_CONFIG.get("output_root"),
    }


@app.post("/process-path")
def process_path(req: ProcessRequest):
    try:
        if not req.image_path:
            raise RuntimeError("/process-path 需要传 image_path")
        image_rgb = load_image_rgb_from_path(req.image_path)
        return process_image_rgb(image_rgb=image_rgb, req=req, original_name=Path(req.image_path).name)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "message": str(e),
            "traceback": traceback.format_exc(limit=20),
        })


@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    request_json: str = Form(default="{}"),
):
    try:
        try:
            payload = json.loads(request_json or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"request_json 不是合法 JSON：{e}")
        req = ProcessRequest(**payload)
        data = await file.read()
        image_rgb = load_image_rgb_from_bytes(data)
        return process_image_rgb(image_rgb=image_rgb, req=req, original_name=file.filename or "upload.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "message": str(e),
            "traceback": traceback.format_exc(limit=20),
        })


# =========================
# 启动入口
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL_PATH, help="YOLO best2.pt 路径")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    RUNTIME_CONFIG["yolo_model_path"] = args.yolo_model
    RUNTIME_CONFIG["output_root"] = args.output_root

    import uvicorn
    if args.reload:
        uvicorn.run(
            f"{Path(__file__).stem}:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
