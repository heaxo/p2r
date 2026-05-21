#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plate_measure_by_a4.py

功能：
1. 使用 YOLO(best2.pt) 检测/分割 plate 和 paper
2. 从 paper mask/bbox 中获取 A4 四角
3. 用 A4 真实尺寸建立像素坐标 -> 毫米坐标的透视矩阵
4. 将钢板轮廓转换到毫米坐标
5. 输出钢板大概长宽、面积、DXF、调试图

前提：
- best2.pt 最好是 YOLO segmentation 模型，例如 yolov8n-seg.pt 训练出来的模型
- 类别名必须包含 plate 和 paper，或者通过参数指定
- A4纸和钢板必须基本在同一平面
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetInstance:
    class_id: int
    class_name: str
    conf: float
    xyxy: np.ndarray
    contour: Optional[np.ndarray]


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_name(s: str) -> str:
    return str(s).strip().lower()


def contour_area(contour: Optional[np.ndarray], xyxy: Optional[np.ndarray] = None) -> float:
    if contour is not None and len(contour) >= 3:
        return float(abs(cv2.contourArea(contour.astype(np.float32))))
    if xyxy is not None:
        x1, y1, x2, y2 = xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    return 0.0


def parse_points(text: str) -> np.ndarray:
    """
    解析格式：
    "x1,y1;x2,y2;x3,y3;x4,y4"
    """
    pts = []
    for part in text.split(";"):
        x, y = part.split(",")
        pts.append([float(x.strip()), float(y.strip())])
    if len(pts) != 4:
        raise ValueError("paper points 必须是4个点，格式：x1,y1;x2,y2;x3,y3;x4,y4")
    return np.array(pts, dtype=np.float32)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    将四个点排序为：
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    # 防止极端角度导致点重复
    if len({(round(float(p[0]), 3), round(float(p[1]), 3)) for p in ordered}) < 4:
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        sorted_pts = pts[np.argsort(angles)]
        start_idx = np.argmin(sorted_pts.sum(axis=1))
        ordered = np.roll(sorted_pts, -start_idx, axis=0).astype(np.float32)

        if ordered[1, 0] < ordered[3, 0]:
            ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return ordered


def quad_from_contour(contour: np.ndarray) -> np.ndarray:
    """
    从 paper contour 估计四边形角点。
    优先使用 approxPolyDP 找真实透视四边形；
    如果找不到，则退化为 minAreaRect。
    """
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(cnt) < 4:
        raise ValueError("paper contour 点数不足，无法提取四角")

    cnt_i = cnt.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(cnt_i)
    peri = cv2.arcLength(hull, True)

    best = None
    for ratio in np.linspace(0.005, 0.08, 30):
        approx = cv2.approxPolyDP(hull, ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2).astype(np.float32)
            break

    if best is None:
        rect = cv2.minAreaRect(cnt.astype(np.float32))
        best = cv2.boxPoints(rect).astype(np.float32)

    return order_quad_points(best)


def bbox_to_quad(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)
    return order_quad_points(pts)


def load_yolo_instances(model_path: str,
                        image_path: str,
                        imgsz: int,
                        conf: float,
                        device: Optional[str]) -> Tuple[List[DetInstance], np.ndarray, dict]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    model = YOLO(model_path)
    kwargs = {
        "source": image_path,
        "imgsz": imgsz,
        "conf": conf,
        "retina_masks": True,
        "verbose": False,
    }
    if device is not None and device.strip():
        kwargs["device"] = device

    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("YOLO 没有返回结果")

    r = results[0]
    names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}

    instances: List[DetInstance] = []
    if r.boxes is None or len(r.boxes) == 0:
        return instances, image_bgr, names

    boxes = r.boxes
    masks_xy = None
    if r.masks is not None and getattr(r.masks, "xy", None) is not None:
        masks_xy = r.masks.xy

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = str(names.get(cls_id, str(cls_id)))
        cf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(np.float32)

        contour = None
        if masks_xy is not None and i < len(masks_xy):
            poly = np.asarray(masks_xy[i], dtype=np.float32)
            if poly.ndim == 2 and len(poly) >= 3:
                contour = poly

        instances.append(DetInstance(
            class_id=cls_id,
            class_name=cls_name,
            conf=cf,
            xyxy=xyxy,
            contour=contour,
        ))

    return instances, image_bgr, names


def select_instance(instances: List[DetInstance], class_name: str, prefer: str = "area") -> DetInstance:
    matched = [it for it in instances if norm_name(it.class_name) == norm_name(class_name)]
    if not matched:
        available = sorted(set(it.class_name for it in instances))
        raise RuntimeError(f"没有检测到类别 {class_name}。当前检测到的类别：{available}")

    if prefer == "conf":
        return max(matched, key=lambda it: it.conf)

    return max(matched, key=lambda it: contour_area(it.contour, it.xyxy) * max(0.001, it.conf))


def build_a4_homography(paper_quad_img: np.ndarray,
                        orientation: str = "auto") -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    返回：
    H: 图片像素 -> A4毫米坐标 的透视矩阵
    paper_quad_mm: A4毫米坐标四角
    real_size: (w_mm, h_mm)
    """
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
    elif orientation == "portrait":
        real_w, real_h = 210.0, 297.0
    elif orientation == "auto":
        if img_w >= img_h:
            real_w, real_h = 297.0, 210.0
        else:
            real_w, real_h = 210.0, 297.0
    else:
        raise ValueError("--a4-orientation 只能是 auto / landscape / portrait")

    paper_quad_mm = np.array([
        [0.0, 0.0],
        [real_w, 0.0],
        [real_w, real_h],
        [0.0, real_h],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        paper_quad_img.astype(np.float32),
        paper_quad_mm.astype(np.float32)
    )
    return H, paper_quad_mm, (real_w, real_h)


def transform_points_px_to_mm(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out.astype(np.float32)


def simplify_contour_mm(contour_mm: np.ndarray, epsilon_mm: float) -> np.ndarray:
    if epsilon_mm <= 0:
        return np.asarray(contour_mm, dtype=np.float32).reshape(-1, 2)

    cnt = np.asarray(contour_mm, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, epsilon_mm, True)
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
    (_, _), (rw, rh), angle = rect

    length = max(float(rw), float(rh))
    width = min(float(rw), float(rh))

    area = float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))
    perimeter = float(cv2.arcLength(pts.reshape(-1, 1, 2), True))

    return {
        "min_area_rect_length_mm": round(length, 2),
        "min_area_rect_width_mm": round(width, 2),
        "axis_bbox_width_mm": round(float(bbox_w), 2),
        "axis_bbox_height_mm": round(float(bbox_h), 2),
        "area_mm2": round(area, 2),
        "area_m2": round(area / 1_000_000.0, 6),
        "perimeter_mm": round(perimeter, 2),
        "rect_angle_deg": round(float(angle), 3),
        "mm_bounds": {
            "x_min": round(x_min, 2),
            "x_max": round(x_max, 2),
            "y_min": round(y_min, 2),
            "y_max": round(y_max, 2),
        }
    }


def write_simple_dxf(path: Path,
                     plate_contour_mm: np.ndarray,
                     paper_quad_mm: np.ndarray,
                     offset_to_positive: bool = True) -> None:
    """
    不依赖 ezdxf，直接写一个简单 DXF。
    单位：mm
    """
    plate = np.asarray(plate_contour_mm, dtype=np.float64).reshape(-1, 2)
    paper = np.asarray(paper_quad_mm, dtype=np.float64).reshape(-1, 2)

    if len(plate) > 1 and np.linalg.norm(plate[0] - plate[-1]) < 1e-6:
        plate = plate[:-1]

    all_pts = np.vstack([plate, paper])
    dx = dy = 0.0
    if offset_to_positive:
        min_xy = all_pts.min(axis=0)
        dx = -min(0.0, float(min_xy[0]))
        dy = -min(0.0, float(min_xy[1]))

    def lwpolyline(layer: str, pts: np.ndarray, closed: bool = True) -> List[str]:
        lines = [
            "0", "LWPOLYLINE",
            "8", layer,
            "90", str(len(pts)),
            "70", "1" if closed else "0",
        ]
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
    lines.extend(lwpolyline("A4_PAPER", paper, True))

    lines.extend([
        "0", "ENDSEC",
        "0", "EOF",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def save_debug_detection(path: Path,
                         image_bgr: np.ndarray,
                         plate_contour: np.ndarray,
                         paper_quad: np.ndarray,
                         dims: Optional[dict] = None) -> None:
    vis = image_bgr.copy()

    cv2.polylines(vis, [plate_contour.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    cv2.polylines(vis, [paper_quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)

    labels = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(paper_quad):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(vis, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if dims:
        text = f"L={dims['min_area_rect_length_mm']}mm  W={dims['min_area_rect_width_mm']}mm"
        cv2.putText(vis, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imwrite(str(path), vis)


def save_mm_preview(path: Path,
                    plate_contour_mm: np.ndarray,
                    paper_quad_mm: np.ndarray,
                    dims: dict,
                    max_px: int = 1800) -> None:
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

    text1 = f"Plate minAreaRect: {dims['min_area_rect_length_mm']} x {dims['min_area_rect_width_mm']} mm"
    text2 = f"Area: {dims['area_m2']} m2"
    cv2.putText(canvas, text1, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(canvas, text2, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    cv2.imwrite(str(path), canvas)


def save_mask_like(path: Path, image_shape: Tuple[int, int], contour: np.ndarray) -> None:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32).reshape(-1, 1, 2)], 255)
    cv2.imwrite(str(path), mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--model", default="best2.pt", help="YOLO模型路径，默认 best2.pt")
    parser.add_argument("--out", default="measure_out", help="输出目录")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="例如 0 / cpu；不填则自动")
    parser.add_argument("--plate-class", default="plate")
    parser.add_argument("--paper-class", default="paper")
    parser.add_argument("--a4-orientation", default="auto", choices=["auto", "landscape", "portrait"])
    parser.add_argument("--paper-points", default=None,
                        help="可选：直接传入A4四角坐标，格式 x1,y1;x2,y2;x3,y3;x4,y4")
    parser.add_argument("--simplify-mm", type=float, default=3.0,
                        help="DXF轮廓简化精度，单位mm。越大点越少，默认3mm")
    args = parser.parse_args()

    image_path = str(Path(args.image))
    out_dir = Path(args.out)
    mkdir(out_dir)

    instances, image_bgr, names = load_yolo_instances(
        model_path=args.model,
        image_path=image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )

    if not instances:
        raise RuntimeError("YOLO没有检测到任何目标")

    plate = select_instance(instances, args.plate_class, prefer="area")
    paper = select_instance(instances, args.paper_class, prefer="area")

    if plate.contour is None:
        print("警告：plate 没有 mask，使用 bbox 兜底，尺寸和DXF会非常粗略。", file=sys.stderr)
        plate_contour_px = bbox_to_quad(plate.xyxy)
    else:
        plate_contour_px = np.asarray(plate.contour, dtype=np.float32).reshape(-1, 2)

    if args.paper_points:
        paper_quad_px = parse_points(args.paper_points)
        paper_quad_px = order_quad_points(paper_quad_px)
    else:
        if paper.contour is not None:
            paper_quad_px = quad_from_contour(paper.contour)
        else:
            print("警告：paper 没有 mask，使用 bbox 作为A4四角，透视精度会下降。", file=sys.stderr)
            paper_quad_px = bbox_to_quad(paper.xyxy)

    H, paper_quad_mm, a4_size = build_a4_homography(
        paper_quad_img=paper_quad_px,
        orientation=args.a4_orientation,
    )

    plate_contour_mm_raw = transform_points_px_to_mm(plate_contour_px, H)
    plate_contour_mm = simplify_contour_mm(plate_contour_mm_raw, args.simplify_mm)

    dims = calc_dimensions(plate_contour_mm)

    dxf_path = out_dir / "plate_outer.dxf"
    json_path = out_dir / "result.json"
    det_img_path = out_dir / "debug_detection.jpg"
    preview_path = out_dir / "debug_mm_preview.png"
    plate_mask_path = out_dir / "plate_mask.png"
    paper_mask_path = out_dir / "paper_mask.png"

    write_simple_dxf(dxf_path, plate_contour_mm, paper_quad_mm, offset_to_positive=True)

    save_debug_detection(det_img_path, image_bgr, plate_contour_px, paper_quad_px, dims)
    save_mm_preview(preview_path, plate_contour_mm, paper_quad_mm, dims)
    save_mask_like(plate_mask_path, image_bgr.shape, plate_contour_px)
    save_mask_like(paper_mask_path, image_bgr.shape, paper_quad_px)

    result = {
        "ok": True,
        "model": str(args.model),
        "image": image_path,
        "classes": {str(k): str(v) for k, v in names.items()},
        "a4": {
            "orientation": args.a4_orientation,
            "used_width_mm": a4_size[0],
            "used_height_mm": a4_size[1],
            "paper_points_px_ordered_tl_tr_br_bl": np.round(paper_quad_px, 3).tolist(),
        },
        "detections": {
            "plate_conf": round(float(plate.conf), 4),
            "paper_conf": round(float(paper.conf), 4),
            "plate_used_mask": plate.contour is not None,
            "paper_used_mask": paper.contour is not None,
        },
        "plate_dimensions": dims,
        "paths": {
            "dxf": str(dxf_path),
            "result_json": str(json_path),
            "debug_detection": str(det_img_path),
            "debug_mm_preview": str(preview_path),
            "plate_mask": str(plate_mask_path),
            "paper_mask": str(paper_mask_path),
        },
        "note": "尺寸基于单张A4纸的平面透视换算。若A4和钢板不共面、钢板弯曲、A4角点不准，会产生明显误差。"
    }

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
