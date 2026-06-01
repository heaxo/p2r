from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class DxfPostProcessConfig:
    """
    DXF轮廓后处理配置。
    """

    enabled: bool = True

    # 连续点去重阈值
    dedup_tolerance_mm: float = 1.0

    # 第一轮轮廓简化，主要去小毛刺、小锯齿
    burr_tolerance_mm: float = 5.0

    # 第二轮最终简化
    final_simplify_mm: float = 2.0

    # 近似共线点删除阈值
    collinear_tolerance_mm: float = 6.0

    # 点夹角接近180度时认为近似共线
    collinear_angle_deg: float = 12.0

    # 是否修复局部凹陷
    notch_fill_enabled: bool = True

    # 凹陷宽度范围
    notch_fill_min_width_mm: float = 30.0
    notch_fill_max_width_mm: float = 80.0

    # 凹陷深度范围
    notch_fill_min_depth_mm: float = 10
    notch_fill_max_depth_mm: float = 25.0

    # 凹陷链条长度 / 直连距离，越大越像凹陷
    notch_fill_min_chain_ratio: float = 1.18

    # 一个凹陷最多跨多少个轮廓点
    notch_fill_max_points: int = 10

    # 修复凹陷时，凹陷两侧边与修复直线的最大夹角
    notch_edge_parallel_angle_deg: float = 35.0

    # 最大迭代次数
    max_iterations: int = 3


def postprocess_plate_contour_mm(
    contour_mm: np.ndarray,
    config: DxfPostProcessConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    对钢板mm轮廓做DXF后处理。
    """

    cfg = config or DxfPostProcessConfig()
    original = _as_points(contour_mm)

    if not cfg.enabled:
        return original, {
            "enabled": False,
            "original_points": int(len(original)),
            "processed_points": int(len(original)),
        }

    pts = _remove_closing_duplicate(original)
    pts = _remove_near_duplicate_points(pts, cfg.dedup_tolerance_mm)

    if len(pts) < 4:
        return pts, {
            "enabled": True,
            "warning": "contour points less than 4",
            "original_points": int(len(original)),
            "processed_points": int(len(pts)),
        }

    input_area = _polygon_area_abs(pts)

    # 先做轻量简化，去掉SAM2的小锯齿
    pts = _approx_poly(pts, cfg.burr_tolerance_mm)
    pts = _remove_near_duplicate_points(pts, cfg.dedup_tolerance_mm)

    # 修复夹钳、遮挡、局部识别错误导致的凹陷
    notch_records: List[Dict[str, Any]] = []

    if cfg.notch_fill_enabled:
        for _ in range(max(1, int(cfg.max_iterations))):
            pts, changed, record = _fill_one_local_notch(pts, cfg)

            if not changed:
                break

            notch_records.append(record)
            pts = _remove_near_duplicate_points(pts, cfg.dedup_tolerance_mm)

    # 删除近似共线点，把直边压成直线
    collinear_removed_total = 0

    for _ in range(max(1, int(cfg.max_iterations))):
        pts, removed = _remove_near_collinear_points(pts, cfg)
        collinear_removed_total += removed

        if removed <= 0:
            break

    # 最终轻量简化
    pts = _approx_poly(pts, cfg.final_simplify_mm)
    pts = _remove_near_duplicate_points(pts, cfg.dedup_tolerance_mm)

    output_area = _polygon_area_abs(pts)
    area_change_ratio = 0.0

    if input_area > 1e-6:
        area_change_ratio = abs(output_area - input_area) / input_area

    info = {
        "enabled": True,
        "config": asdict(cfg),
        "original_points": int(len(original)),
        "processed_points": int(len(pts)),
        "input_area_mm2": round(float(input_area), 3),
        "output_area_mm2": round(float(output_area), 3),
        "area_change_ratio": round(float(area_change_ratio), 6),
        "notch_fill_count": int(len(notch_records)),
        "notch_records": notch_records,
        "collinear_removed_total": int(collinear_removed_total),
    }

    return pts.astype(np.float32), info


def save_postprocess_preview(
    before_mm: np.ndarray,
    after_mm: np.ndarray,
    output_path: str | Path,
    padding_mm: float = 80.0,
    mm_per_px: float = 2.0,
    max_preview_px: int = 1800,
) -> str:
    """
    保存后处理对比图。
    黑色：原轮廓
    灰色：后处理轮廓
    """

    before = _as_points(before_mm)
    after = _as_points(after_mm)

    all_pts = np.vstack([before, after])
    if not np.all(np.isfinite(all_pts)):
        raise ValueError("postprocess preview points contain non-finite coordinates")

    x_min, y_min = np.min(all_pts, axis=0) - float(padding_mm)
    x_max, y_max = np.max(all_pts, axis=0) + float(padding_mm)

    width_mm = max(float(x_max - x_min), 1.0)
    height_mm = max(float(y_max - y_min), 1.0)
    max_preview_px = max(100, int(max_preview_px))
    effective_mm_per_px = max(
        float(mm_per_px),
        width_mm / max_preview_px,
        height_mm / max_preview_px,
    )

    w = max(100, int(round(width_mm / effective_mm_per_px)))
    h = max(100, int(round(height_mm / effective_mm_per_px)))

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    def to_px(pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float32)
        out[:, 0] = (pts[:, 0] - x_min) / effective_mm_per_px
        out[:, 1] = (pts[:, 1] - y_min) / effective_mm_per_px
        return np.round(out).astype(np.int32)

    before_px = to_px(before).reshape(-1, 1, 2)
    after_px = to_px(after).reshape(-1, 1, 2)

    cv2.polylines(canvas, [before_px], True, (0, 0, 0), 2)
    cv2.polylines(canvas, [after_px], True, (120, 120, 120), 3)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path), canvas)

    return str(path)


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(pts) == 0:
        raise ValueError("contour_mm is empty")

    return pts


def _remove_closing_duplicate(points: np.ndarray) -> np.ndarray:
    pts = _as_points(points)

    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        return pts[:-1].copy()

    return pts.copy()


def _remove_near_duplicate_points(
    points: np.ndarray,
    tolerance_mm: float,
) -> np.ndarray:
    pts = _remove_closing_duplicate(points)

    if len(pts) <= 2:
        return pts

    tol = max(0.0, float(tolerance_mm))

    if tol <= 0:
        return pts

    kept = [pts[0]]

    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) >= tol:
            kept.append(p)

    if len(kept) >= 2 and np.linalg.norm(kept[0] - kept[-1]) < tol:
        kept.pop()

    return np.asarray(kept, dtype=np.float32)


def _approx_poly(
    points: np.ndarray,
    epsilon_mm: float,
) -> np.ndarray:
    pts = _remove_closing_duplicate(points)

    if len(pts) < 4 or epsilon_mm <= 0:
        return pts

    cnt = pts.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(cnt, float(epsilon_mm), True)
    out = approx.reshape(-1, 2).astype(np.float32)

    if len(out) < 4:
        return pts

    return _remove_closing_duplicate(out)


def _polygon_area_abs(points: np.ndarray) -> float:
    pts = _remove_closing_duplicate(points)

    if len(pts) < 3:
        return 0.0

    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32))))


def _segment_distance(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ab = b - a
    length2 = float(np.dot(ab, ab))

    if length2 <= 1e-9:
        return float(np.linalg.norm(point - a))

    t = float(np.dot(point - a, ab) / length2)
    t = max(0.0, min(1.0, t))

    proj = a + t * ab

    return float(np.linalg.norm(point - proj))


def _signed_distance_to_line(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ab = b - a
    length = float(np.linalg.norm(ab))

    if length <= 1e-9:
        return 0.0

    ap = point - a

    return float((ab[0] * ap[1] - ab[1] * ap[0]) / length)


def _polyline_length(points: np.ndarray) -> float:
    pts = _as_points(points)

    if len(pts) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _angle_between_vectors_deg(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0

    cos_v = float(np.dot(v1, v2) / (n1 * n2))
    cos_v = max(-1.0, min(1.0, cos_v))

    return float(np.degrees(np.arccos(cos_v)))


def _angle_to_line_deg(
    v: np.ndarray,
    line: np.ndarray,
) -> float:
    angle = _angle_between_vectors_deg(v, line)
    return min(angle, abs(180.0 - angle))


def _remove_near_collinear_points(
    points: np.ndarray,
    cfg: DxfPostProcessConfig,
) -> Tuple[np.ndarray, int]:
    pts = _remove_closing_duplicate(points)
    n = len(pts)

    if n < 4:
        return pts, 0

    remove_indices = set()

    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]

        dist = _segment_distance(b, a, c)

        v1 = a - b
        v2 = c - b
        angle = _angle_between_vectors_deg(v1, v2)

        is_collinear = (
            dist <= cfg.collinear_tolerance_mm
            and abs(180.0 - angle) <= cfg.collinear_angle_deg
        )

        if is_collinear:
            remove_indices.add(i)

    if not remove_indices:
        return pts, 0

    kept = [
        p
        for i, p in enumerate(pts)
        if i not in remove_indices
    ]

    if len(kept) < 4:
        return pts, 0

    return np.asarray(kept, dtype=np.float32), len(remove_indices)


def _cyclic_chain(
    points: np.ndarray,
    start: int,
    span: int,
) -> np.ndarray:
    pts = _remove_closing_duplicate(points)
    n = len(pts)

    indices = [
        (start + k) % n
        for k in range(span + 1)
    ]

    return pts[indices]


def _remove_cyclic_middle_points(
    points: np.ndarray,
    start: int,
    span: int,
) -> np.ndarray:
    pts = _remove_closing_duplicate(points)
    n = len(pts)

    remove_indices = {
        (start + k) % n
        for k in range(1, span)
    }

    kept = [
        p
        for i, p in enumerate(pts)
        if i not in remove_indices
    ]

    if len(kept) < 4:
        return pts

    return np.asarray(kept, dtype=np.float32)


def _fill_one_local_notch(
    points: np.ndarray,
    cfg: DxfPostProcessConfig,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    pts = _remove_closing_duplicate(points)
    n = len(pts)

    if n < 6:
        return pts, False, {}

    best = None

    max_span = min(
        max(3, int(cfg.notch_fill_max_points)),
        n - 2,
    )

    for start in range(n):
        for span in range(3, max_span + 1):
            chain = _cyclic_chain(pts, start, span)

            a = chain[0]
            b = chain[-1]
            chord = b - a
            chord_len = float(np.linalg.norm(chord))

            if chord_len < cfg.notch_fill_min_width_mm:
                continue

            if chord_len > cfg.notch_fill_max_width_mm:
                continue

            middle = chain[1:-1]

            if len(middle) <= 0:
                continue

            signed_distances = np.array(
                [
                    _signed_distance_to_line(p, a, b)
                    for p in middle
                ],
                dtype=np.float32,
            )

            max_pos = float(np.max(signed_distances))
            max_neg = float(np.min(signed_distances))
            depth = max(abs(max_pos), abs(max_neg))

            if depth < cfg.notch_fill_min_depth_mm:
                continue

            if depth > cfg.notch_fill_max_depth_mm:
                continue

            # 中间点基本应在直线同一侧，否则更像波浪线，不按凹陷修复
            if (
                max_pos > cfg.collinear_tolerance_mm
                and abs(max_neg) > cfg.collinear_tolerance_mm
            ):
                continue

            chain_len = _polyline_length(chain)
            chain_ratio = chain_len / max(chord_len, 1e-6)

            if chain_ratio < cfg.notch_fill_min_chain_ratio:
                continue

            prev_pt = pts[(start - 1) % n]
            next_pt = pts[(start + span + 1) % n]

            before_vec = a - prev_pt
            after_vec = next_pt - b

            before_angle = _angle_to_line_deg(before_vec, chord)
            after_angle = _angle_to_line_deg(after_vec, chord)

            # 凹陷两侧应该大致属于同一条直边
            if before_angle > cfg.notch_edge_parallel_angle_deg:
                continue

            if after_angle > cfg.notch_edge_parallel_angle_deg:
                continue

            score = depth * 2.0 + chord_len * 0.1 + chain_ratio * 20.0

            candidate = {
                "start_index": int(start),
                "span": int(span),
                "width_mm": round(chord_len, 3),
                "depth_mm": round(depth, 3),
                "chain_length_mm": round(chain_len, 3),
                "chain_ratio": round(chain_ratio, 4),
                "before_angle_deg": round(before_angle, 3),
                "after_angle_deg": round(after_angle, 3),
                "removed_points": int(span - 1),
                "score": round(float(score), 3),
            }

            if best is None or score > best["score_value"]:
                best = {
                    "score_value": score,
                    "candidate": candidate,
                    "start": start,
                    "span": span,
                }

    if best is None:
        return pts, False, {}

    out = _remove_cyclic_middle_points(
        pts,
        best["start"],
        best["span"],
    )

    return out, True, best["candidate"]
