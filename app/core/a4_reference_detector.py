from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


A4_ASPECT_RATIO = 297.0 / 210.0


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    ordered = np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmax(diff)],
            pts[np.argmax(s)],
            pts[np.argmin(diff)],
        ],
        dtype=np.float32,
    )
    if len({(round(float(p[0]), 3), round(float(p[1]), 3)) for p in ordered}) < 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered = pts[np.argsort(angles)]
        start_idx = int(np.argmin(ordered.sum(axis=1)))
        ordered = np.roll(ordered, -start_idx, axis=0).astype(np.float32)
    return ordered


def _quad_edge_lengths(quad: np.ndarray) -> List[float]:
    q = _order_quad_points(quad)
    return [
        float(np.linalg.norm(q[1] - q[0])),
        float(np.linalg.norm(q[2] - q[1])),
        float(np.linalg.norm(q[2] - q[3])),
        float(np.linalg.norm(q[3] - q[0])),
    ]


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    top, right, bottom, left = _quad_edge_lengths(quad)
    width = max(1.0, (top + bottom) / 2.0)
    height = max(1.0, (left + right) / 2.0)
    return max(width, height) / max(1.0, min(width, height))


def _quad_from_contour(contour: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    contour = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    if len(contour) < 4:
        return None, "too_few_points"

    hull = cv2.convexHull(contour.astype(np.int32))
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 1:
        return None, "invalid_perimeter"

    candidates: List[Tuple[float, np.ndarray, str]] = []
    for ratio in np.linspace(0.01, 0.08, 20):
        approx = cv2.approxPolyDP(hull, float(ratio) * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
            aspect = _quad_aspect_ratio(quad)
            aspect_error = abs(math.log(max(aspect, 1e-6) / A4_ASPECT_RATIO))
            candidates.append((aspect_error, quad, f"approx_poly_{ratio:.4f}"))

    rect_quad = _order_quad_points(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))
    rect_aspect = _quad_aspect_ratio(rect_quad)
    rect_error = abs(math.log(max(rect_aspect, 1e-6) / A4_ASPECT_RATIO))
    candidates.append((rect_error + 0.08, rect_quad, "min_area_rect"))

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def detect_a4_paper_opencv(
    image_rgb: np.ndarray,
    *,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.35,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Detect an A4-like white quadrilateral without YOLO/SAM2."""

    image = np.asarray(image_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    image_area = max(1, h * w)
    info: Dict[str, Any] = {
        "enabled": True,
        "method": "opencv_white_a4_quad",
        "paper_mask_source": None,
        "candidate_count": 0,
        "message": "",
    }

    if image.ndim != 3 or image.shape[2] < 3:
        info["message"] = "unsupported image shape"
        return None, info

    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    v_threshold = max(135, int(np.percentile(value, 72)))
    white_mask = ((saturation < 80) & (value >= v_threshold) & (gray >= 120)).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, close_kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict[str, Any]] = []

    for contour in contours:
        contour_area = float(abs(cv2.contourArea(contour)))
        area_ratio = contour_area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        quad, quad_source = _quad_from_contour(contour)
        if quad is None:
            continue

        quad_area = float(abs(cv2.contourArea(quad.reshape(-1, 1, 2))))
        if quad_area <= 1:
            continue
        rectangularity = contour_area / max(quad_area, 1.0)
        aspect = _quad_aspect_ratio(quad)
        aspect_error = abs(math.log(max(aspect, 1e-6) / A4_ASPECT_RATIO))
        if aspect_error > 0.22 or rectangularity < 0.68:
            continue

        candidate_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(candidate_mask, [quad.astype(np.int32).reshape(-1, 1, 2)], -1, 1, thickness=-1)
        mean_s = float(np.mean(saturation[candidate_mask > 0]))
        mean_v = float(np.mean(value[candidate_mask > 0]))
        white_fill_ratio = float(np.sum((white_mask > 0) & (candidate_mask > 0)) / max(1, int(candidate_mask.sum())))
        if mean_v < 145 or mean_s > 95 or white_fill_ratio < 0.45:
            continue

        score = (
            max(0.0, 1.0 - aspect_error / 0.22) * 0.35
            + min(1.0, rectangularity) * 0.25
            + min(1.0, white_fill_ratio) * 0.25
            + min(1.0, mean_v / 255.0) * 0.15
        )
        candidates.append(
            {
                "score": float(score),
                "quad": quad,
                "quad_source": quad_source,
                "area_ratio": float(area_ratio),
                "contour_area_px": float(contour_area),
                "quad_area_px": float(quad_area),
                "rectangularity": float(rectangularity),
                "aspect_ratio": float(aspect),
                "aspect_error": float(aspect_error),
                "mean_saturation": mean_s,
                "mean_value": mean_v,
                "white_fill_ratio": white_fill_ratio,
            }
        )

    info["candidate_count"] = int(len(candidates))
    if not candidates:
        info["message"] = "OpenCV did not find a reliable A4-like white quadrilateral"
        return None, info

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    quad = np.asarray(best["quad"], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [quad.astype(np.int32).reshape(-1, 1, 2)], -1, 255, thickness=-1)

    info.update(
        {
            "paper_mask_source": "opencv_a4_quad",
            "message": "OpenCV detected an A4-like white quadrilateral",
            "selected_score": round(float(best["score"]), 6),
            "paper_quad_px_tl_tr_br_bl": np.round(quad, 3).tolist(),
            "selected_candidate": {
                key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value
                for key, value in best.items()
                if key != "quad"
            },
            "candidate_summaries": [
                {
                    "score": round(float(item["score"]), 6),
                    "quad_source": item["quad_source"],
                    "area_ratio": round(float(item["area_ratio"]), 6),
                    "aspect_ratio": round(float(item["aspect_ratio"]), 6),
                    "rectangularity": round(float(item["rectangularity"]), 6),
                    "white_fill_ratio": round(float(item["white_fill_ratio"]), 6),
                }
                for item in candidates[:5]
            ],
        }
    )
    return mask, info

