from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class DxfGeometryOptimizeConfig:
    enabled: bool = True
    line_tolerance_mm: float = 2.0
    line_max_tolerance_mm: float = 6.0
    line_rms_tolerance_mm: float = 2.5
    line_tolerance_ratio: float = 0.003
    line_max_tolerance_ratio: float = 0.008
    line_min_length_mm: float = 20.0
    line_endpoint_angle_deg: float = 18.0
    line_merge_angle_deg: float = 3.0
    line_merge_gap_mm: float = 2.0
    arc_tolerance_mm: float = 3.0
    arc_max_tolerance_mm: float = 6.0
    arc_min_points: int = 5
    arc_min_length_mm: float = 20.0
    arc_min_sweep_deg: float = 8.0
    arc_max_sweep_deg: float = 355.0
    arc_max_radial_error_ratio: float = 0.03
    arc_min_monotonic_ratio: float = 0.82
    arc_endpoint_tangent_angle_deg: float = 15.0
    arc_max_edge_length_ratio: float = 8.0
    circle_tolerance_mm: float = 3.0
    circle_max_tolerance_mm: float = 6.0
    circle_min_points: int = 12
    circle_min_circularity: float = 0.94
    circle_max_radial_error_ratio: float = 0.02
    circle_max_angle_gap_deg: float = 60.0
    preprocess_enabled: bool = True
    preprocess_resample_step_mm: float = 3.0
    preprocess_min_points: int = 48
    preprocess_max_points: int = 720
    corner_window_mm: float = 18.0
    corner_min_deflection_deg: float = 24.0
    corner_min_spacing_mm: float = 12.0
    corner_max_points: int = 120
    max_fit_scan_edges: int = 320


def write_optimized_dxf(
    path: str | Path,
    plate_contour_mm: np.ndarray,
    offset_to_positive: bool = True,
    layer: str = "PLATE_OUTER",
    config: DxfGeometryOptimizeConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or DxfGeometryOptimizeConfig()
    plate = _remove_closing_duplicate(_as_points(plate_contour_mm))
    entities, info = optimize_contour_entities(plate, cfg)

    dx, dy = _compute_positive_offset(plate, entities, offset_to_positive)

    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    for entity in entities:
        lines.extend(_entity_to_dxf_lines(entity, dx, dy, layer))

    lines.extend(["0", "ENDSEC", "0", "EOF"])

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    out = dict(info)
    out["offset_x_mm"] = round(float(dx), 6)
    out["offset_y_mm"] = round(float(dy), 6)
    out["path"] = str(output_path)
    return out


def write_optimized_dxf_multi(
    path: str | Path,
    contours: List[Tuple[np.ndarray, str]],
    offset_to_positive: bool = True,
    config: DxfGeometryOptimizeConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or DxfGeometryOptimizeConfig()
    optimized: List[Tuple[Dict[str, Any], str]] = []
    contour_infos: List[Dict[str, Any]] = []
    source_points: List[np.ndarray] = []

    for index, (contour, layer) in enumerate(contours):
        pts = _remove_closing_duplicate(_as_points(contour))
        source_points.append(pts)
        entities, info = optimize_contour_entities(pts, cfg)
        contour_infos.append({
            "index": int(index),
            "layer": str(layer),
            "input_points": int(len(pts)),
            "entity_count": int(len(entities)),
            "entity_type_counts": dict(info.get("entity_type_counts") or {}),
            "mode": info.get("mode"),
        })
        optimized.extend((entity, str(layer)) for entity in entities)

    all_entities = [entity for entity, _ in optimized]
    all_points = np.vstack(source_points) if source_points else np.zeros((0, 2), dtype=np.float64)
    dx, dy = _compute_positive_offset(all_points, all_entities, offset_to_positive)

    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    for entity, layer in optimized:
        lines.extend(_entity_to_dxf_lines(entity, dx, dy, layer))

    lines.extend(["0", "ENDSEC", "0", "EOF"])

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    type_counts: Dict[str, int] = {}
    max_error = 0.0
    for entity, _ in optimized:
        entity_type = str(entity.get("type") or "UNKNOWN")
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        max_error = max(max_error, float(entity.get("max_error_mm", 0.0)))

    return {
        "enabled": bool(cfg.enabled),
        "mode": "multi_contour",
        "config": asdict(cfg),
        "contour_count": int(len(contours)),
        "inner_contour_count": max(0, int(len(contours)) - 1),
        "entity_count": int(len(optimized)),
        "entity_type_counts": type_counts,
        "max_fit_error_mm": round(float(max_error), 6),
        "offset_x_mm": round(float(dx), 6),
        "offset_y_mm": round(float(dy), 6),
        "path": str(output_path),
        "contours": contour_infos,
    }


def optimize_contour_entities(
    contour_mm: np.ndarray,
    config: DxfGeometryOptimizeConfig | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = config or DxfGeometryOptimizeConfig()
    source_pts = _remove_closing_duplicate(_as_points(contour_mm))
    pts, preprocess_info = _prepare_contour_for_geometry(source_pts, cfg)

    if len(source_pts) == 1:
        info = _empty_info(cfg, len(source_pts), "not_enough_points")
        info["preprocess"] = preprocess_info
        return [], info

    if len(source_pts) == 2:
        entity = _line_entity(source_pts[0], source_pts[1], 2, 0.0)
        info = _summary_info(cfg, len(source_pts), [entity], "single_segment")
        info["preprocess"] = preprocess_info
        return [entity], info

    if not cfg.enabled:
        entities = [_polyline_fallback_entity(source_pts)]
        info = _summary_info(cfg, len(source_pts), entities, "disabled")
        info["preprocess"] = preprocess_info
        return entities, info

    circle = _try_fit_full_circle(pts, cfg)
    if circle is not None:
        info = _summary_info(cfg, len(source_pts), [circle], "full_circle")
        info["working_points"] = int(len(pts))
        info["preprocess"] = preprocess_info
        return [circle], info

    rotated, start_index = _rotate_to_best_break(pts)
    ext = np.vstack([rotated, rotated[:1]])
    n = len(rotated)
    entities: List[Dict[str, Any]] = []
    i = 0

    while i < n:
        best = _best_entity_from_index(ext, i, n, cfg)

        if best is None or best.get("edge_count", 0) <= 0:
            best = _line_entity(ext[i], ext[i + 1], 2, 0.0)
            best["reason"] = "fallback_edge"

        entities.append(best)
        i += int(best["edge_count"])

    info = _summary_info(cfg, len(source_pts), entities, "segmented")
    info["working_points"] = int(len(pts))
    info["preprocess"] = preprocess_info
    info["break_start_index"] = int(start_index)
    return entities, info


def _prepare_contour_for_geometry(
    points: np.ndarray,
    cfg: DxfGeometryOptimizeConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pts = _remove_consecutive_duplicates(_remove_closing_duplicate(_as_points(points)))
    info: Dict[str, Any] = {
        "enabled": bool(cfg.preprocess_enabled),
        "source_points": int(len(pts)),
        "working_points": int(len(pts)),
        "resampled": False,
        "corner_points": 0,
    }

    if not bool(cfg.preprocess_enabled) or len(pts) < 4:
        return pts.copy(), info

    perimeter = _perimeter(pts)
    if perimeter <= 1e-6:
        return pts.copy(), info

    max_points = max(12, int(cfg.preprocess_max_points))
    min_points = min(max_points, max(4, int(cfg.preprocess_min_points)))
    step = max(0.5, float(cfg.preprocess_resample_step_mm))
    uniform_count = int(math.ceil(perimeter / step))
    uniform_count = max(min_points, min(max_points, uniform_count))

    corners = _candidate_corner_distances(pts, perimeter, cfg)
    sampled = _resample_closed_contour(pts, perimeter, uniform_count, corners, cfg)
    sampled = _remove_consecutive_duplicates(sampled)

    if len(sampled) < 3:
        return pts.copy(), info

    info.update({
        "working_points": int(len(sampled)),
        "resampled": True,
        "resample_step_mm": round(float(perimeter / max(len(sampled), 1)), 6),
        "perimeter_mm": round(float(perimeter), 6),
        "corner_points": int(len(corners)),
    })
    return sampled, info


def _candidate_corner_distances(
    points: np.ndarray,
    perimeter: float,
    cfg: DxfGeometryOptimizeConfig,
) -> List[float]:
    pts = _remove_closing_duplicate(_as_points(points))
    n = len(pts)
    if n < 4 or perimeter <= 1e-6:
        return []

    closed, seg_lengths, cumulative = _closed_polyline_metrics(pts)
    if len(seg_lengths) == 0:
        return []

    window = max(
        float(cfg.corner_window_mm),
        float(cfg.preprocess_resample_step_mm) * 3.0,
    )
    threshold = float(cfg.corner_min_deflection_deg)
    min_spacing = max(1.0, float(cfg.corner_min_spacing_mm))
    candidates: List[Tuple[float, float]] = []

    for i in range(n):
        distance = float(cumulative[i])
        curr = pts[i]
        prev_pt = _point_at_closed_distance(closed, seg_lengths, cumulative, distance - window, perimeter)
        next_pt = _point_at_closed_distance(closed, seg_lengths, cumulative, distance + window, perimeter)
        v1 = curr - prev_pt
        v2 = next_pt - curr

        if float(np.linalg.norm(v1)) <= 1e-6 or float(np.linalg.norm(v2)) <= 1e-6:
            continue

        deflection = _angle_between_vectors_deg(v1, v2)
        if deflection >= threshold:
            candidates.append((float(deflection), distance))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: List[Tuple[float, float]] = []
    max_corners = max(0, int(cfg.corner_max_points))
    for deflection, distance in candidates:
        if all(_cyclic_distance(distance, kept_distance, perimeter) >= min_spacing for _, kept_distance in kept):
            kept.append((deflection, distance))
            if len(kept) >= max_corners:
                break

    return [distance for _, distance in sorted(kept, key=lambda item: item[1])]


def _resample_closed_contour(
    points: np.ndarray,
    perimeter: float,
    uniform_count: int,
    corner_distances: List[float],
    cfg: DxfGeometryOptimizeConfig,
) -> np.ndarray:
    pts = _remove_closing_duplicate(_as_points(points))
    closed, seg_lengths, cumulative = _closed_polyline_metrics(pts)
    if perimeter <= 1e-6 or len(seg_lengths) == 0:
        return pts.copy()

    max_points = max(12, int(cfg.preprocess_max_points))
    min_spacing = max(1e-4, min(float(cfg.preprocess_resample_step_mm) * 0.4, float(cfg.corner_min_spacing_mm) * 0.35))
    distances: List[float] = []

    for distance in corner_distances:
        d = float(distance % perimeter)
        if not _has_near_distance(d, distances, min_spacing, perimeter):
            distances.append(d)

    remaining_slots = max(0, max_points - len(distances))
    if remaining_slots <= 0:
        uniform_distances = np.asarray([], dtype=np.float64)
    else:
        uniform_samples = min(max(4, int(uniform_count)), remaining_slots)
        uniform_distances = np.linspace(0.0, perimeter, uniform_samples, endpoint=False)

    for distance in uniform_distances:
        d = float(distance % perimeter)
        if not _has_near_distance(d, distances, min_spacing, perimeter):
            distances.append(d)

    distances = sorted(distances)
    sampled = [
        _point_at_closed_distance(closed, seg_lengths, cumulative, distance, perimeter)
        for distance in distances
    ]
    return np.asarray(sampled, dtype=np.float64).reshape(-1, 2)


def _best_entity_from_index(
    ext: np.ndarray,
    start: int,
    n: int,
    cfg: DxfGeometryOptimizeConfig,
) -> Dict[str, Any] | None:
    best_line: Dict[str, Any] | None = None
    best_arc: Dict[str, Any] | None = None
    max_end = min(n, start + max(2, int(cfg.max_fit_scan_edges)))

    for end in range(start + 1, max_end + 1):
        chain = ext[start:end + 1]
        edge_count = end - start

        if edge_count >= 2:
            line = _try_fit_line(chain, cfg)
            if line is not None:
                best_line = line

        if len(chain) >= max(3, int(cfg.arc_min_points)):
            arc = _try_fit_arc(chain, cfg)
            if arc is not None:
                best_arc = arc

    if best_line is None:
        return best_arc

    if best_arc is None:
        return best_line

    line_edges = int(best_line["edge_count"])
    arc_edges = int(best_arc["edge_count"])

    if arc_edges > line_edges:
        return best_arc

    if arc_edges == line_edges:
        arc_error = float(best_arc.get("rms_error_mm", best_arc.get("max_error_mm", 0.0)))
        line_error = float(best_line.get("rms_error_mm", best_line.get("max_error_mm", 0.0)))
        if arc_error + 0.25 < line_error:
            return best_arc

    return best_line


def _try_fit_line(
    points: np.ndarray,
    cfg: DxfGeometryOptimizeConfig,
) -> Dict[str, Any] | None:
    pts = _as_points(points)
    if len(pts) < 3:
        return None

    a = pts[0]
    b = pts[-1]
    ab = b - a
    length = float(np.linalg.norm(ab))

    if length < float(cfg.line_min_length_mm):
        return None

    distances, projections = _line_distances_and_projection(pts, a, b)
    max_error = float(np.max(distances))
    p95_error = float(np.percentile(distances, 95.0))
    rms_error = float(np.sqrt(np.mean(distances ** 2)))
    p95_limit, max_limit, rms_limit = _line_fit_limits(length, cfg)

    if p95_error > p95_limit:
        return None

    if max_error > max_limit:
        return None

    if rms_error > rms_limit:
        return None

    if float(np.min(projections)) < -0.02 or float(np.max(projections)) > 1.02:
        return None

    projection_steps = np.diff(projections)
    if len(projection_steps) and float(np.min(projection_steps)) < -0.05:
        return None

    start_angle = _angle_between_vectors_deg(pts[1] - pts[0], ab)
    end_angle = _angle_between_vectors_deg(pts[-1] - pts[-2], ab)
    start_angle = min(start_angle, abs(180.0 - start_angle))
    end_angle = min(end_angle, abs(180.0 - end_angle))
    if start_angle > float(cfg.line_endpoint_angle_deg):
        return None

    if end_angle > float(cfg.line_endpoint_angle_deg):
        return None

    entity = _line_entity(a, b, len(pts), max_error)
    entity["p95_error_mm"] = float(p95_error)
    entity["rms_error_mm"] = float(rms_error)
    entity["fit_tolerance_mm"] = float(p95_limit)
    entity["fit_max_tolerance_mm"] = float(max_limit)
    return entity


def _try_fit_arc(
    points: np.ndarray,
    cfg: DxfGeometryOptimizeConfig,
) -> Dict[str, Any] | None:
    pts = _as_points(points)
    if len(pts) < max(3, int(cfg.arc_min_points)):
        return None

    a = pts[0]
    b = pts[-1]
    chord = b - a
    chord_length = float(np.linalg.norm(chord))
    if chord_length < float(cfg.arc_min_length_mm):
        return None

    edge_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    median_edge_length = float(np.median(edge_lengths)) if len(edge_lengths) else 0.0
    if median_edge_length <= 1e-9:
        return None

    if float(np.max(edge_lengths)) / median_edge_length > float(cfg.arc_max_edge_length_ratio):
        return None

    line_distances, _ = _line_distances_and_projection(pts, a, b)
    if float(np.max(line_distances)) <= float(cfg.line_tolerance_mm):
        return None

    center_fit, radius_fit = _least_squares_circle(pts)
    if center_fit is None or radius_fit <= 1e-6:
        return None

    center, radius = _project_center_to_endpoint_bisector(center_fit, a, b)
    if radius <= 1e-6:
        return None

    dists = np.linalg.norm(pts - center, axis=1)
    radial_errors = np.abs(dists - radius)
    max_error = float(np.max(radial_errors))
    p95_error = float(np.percentile(radial_errors, 95.0))
    rms_error = float(np.sqrt(np.mean(radial_errors ** 2)))

    arc_p95_limit = max(float(cfg.arc_tolerance_mm), radius * float(cfg.arc_max_radial_error_ratio) * 0.5)
    arc_max_limit = max(float(cfg.arc_max_tolerance_mm), radius * float(cfg.arc_max_radial_error_ratio))

    if p95_error > arc_p95_limit:
        return None

    if max_error > arc_max_limit:
        return None

    if rms_error > max(1.5, float(cfg.arc_tolerance_mm) * 0.5):
        return None

    if max_error / max(radius, 1e-6) > float(cfg.arc_max_radial_error_ratio):
        return None

    angles = np.unwrap(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    sweep_rad = float(angles[-1] - angles[0])
    sweep_deg = float(math.degrees(sweep_rad))
    abs_sweep = abs(sweep_deg)

    if abs_sweep < float(cfg.arc_min_sweep_deg):
        return None

    if abs_sweep > float(cfg.arc_max_sweep_deg):
        return None

    deltas = np.diff(angles)
    sum_abs = float(np.sum(np.abs(deltas)))
    if sum_abs <= 1e-9:
        return None

    monotonic_ratio = abs(sweep_rad) / sum_abs
    if monotonic_ratio < float(cfg.arc_min_monotonic_ratio):
        return None

    start_tangent_angle = _angle_to_circle_tangent_deg(pts[1] - pts[0], pts[0] - center)
    end_tangent_angle = _angle_to_circle_tangent_deg(pts[-1] - pts[-2], pts[-1] - center)
    if start_tangent_angle > float(cfg.arc_endpoint_tangent_angle_deg):
        return None

    if end_tangent_angle > float(cfg.arc_endpoint_tangent_angle_deg):
        return None

    start_angle = _angle_deg(center, a)
    end_angle = _angle_deg(center, b)

    if sweep_deg < 0.0:
        start_angle, end_angle = end_angle, start_angle

    return {
        "type": "ARC",
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "start_angle_deg": float(start_angle),
        "end_angle_deg": float(end_angle),
        "input_sweep_deg": float(sweep_deg),
        "point_count": int(len(pts)),
        "edge_count": int(len(pts) - 1),
        "max_error_mm": float(max_error),
        "p95_error_mm": float(p95_error),
        "rms_error_mm": float(rms_error),
        "monotonic_ratio": float(monotonic_ratio),
        "start_tangent_angle_deg": float(start_tangent_angle),
        "end_tangent_angle_deg": float(end_tangent_angle),
    }


def _try_fit_full_circle(
    points: np.ndarray,
    cfg: DxfGeometryOptimizeConfig,
) -> Dict[str, Any] | None:
    pts = _remove_closing_duplicate(_as_points(points))
    if len(pts) < int(cfg.circle_min_points):
        return None

    center, radius = _least_squares_circle(pts)
    if center is None or radius <= 1e-6:
        return None

    dists = np.linalg.norm(pts - center, axis=1)
    radial_errors = np.abs(dists - radius)
    max_error = float(np.max(radial_errors))
    p95_error = float(np.percentile(radial_errors, 95.0))
    radius_std = float(np.std(dists))
    radius_error_ratio = radius_std / max(radius, 1e-6)

    circle_p95_limit = max(float(cfg.circle_tolerance_mm), radius * float(cfg.circle_max_radial_error_ratio) * 0.5)
    circle_max_limit = max(float(cfg.circle_max_tolerance_mm), radius * float(cfg.circle_max_radial_error_ratio))

    if p95_error > circle_p95_limit:
        return None

    if max_error > circle_max_limit:
        return None

    if radius_error_ratio > float(cfg.circle_max_radial_error_ratio):
        return None

    circularity = _circularity(pts)
    if circularity < float(cfg.circle_min_circularity):
        return None

    raw_angles = np.mod(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]), 2.0 * math.pi)
    sorted_angles = np.sort(raw_angles)
    gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 2.0 * math.pi]))
    max_gap_deg = float(math.degrees(float(np.max(gaps))))
    if max_gap_deg > float(cfg.circle_max_angle_gap_deg):
        return None

    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    bbox_w = float(max_xy[0] - min_xy[0])
    bbox_h = float(max_xy[1] - min_xy[1])
    diameter = float(radius * 2.0)
    bbox_error = max(abs(bbox_w - diameter), abs(bbox_h - diameter))

    if bbox_error > float(cfg.circle_tolerance_mm) * 2.0:
        return None

    return {
        "type": "CIRCLE",
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "diameter": float(diameter),
        "point_count": int(len(pts)),
        "edge_count": int(len(pts)),
        "max_error_mm": float(max_error),
        "p95_error_mm": float(p95_error),
        "radius_std_mm": float(radius_std),
        "radius_error_ratio": float(radius_error_ratio),
        "circularity": float(circularity),
        "max_angle_gap_deg": float(max_gap_deg),
        "bbox_diameter_error_mm": float(bbox_error),
    }


def _least_squares_circle(points: np.ndarray) -> Tuple[np.ndarray | None, float]:
    pts = _as_points(points)
    if len(pts) < 3:
        return None, 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y

    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, 0.0

    radius2 = float(c + cx * cx + cy * cy)
    if radius2 <= 0.0 or not math.isfinite(radius2):
        return None, 0.0

    return np.array([float(cx), float(cy)], dtype=np.float64), float(math.sqrt(radius2))


def _project_center_to_endpoint_bisector(
    center_fit: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    chord = b - a
    length = float(np.linalg.norm(chord))
    if length <= 1e-9:
        return center_fit, 0.0

    midpoint = (a + b) * 0.5
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / length
    signed_offset = float(np.dot(center_fit - midpoint, normal))
    center = midpoint + normal * signed_offset
    radius = float(np.linalg.norm(center - a))
    return center.astype(np.float64), radius


def _line_distances_and_projection(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = _as_points(points)
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length <= 1e-9:
        distances = np.linalg.norm(pts - a, axis=1)
        projections = np.zeros(len(pts), dtype=np.float64)
        return distances, projections

    unit = ab / length
    rel = pts - a
    signed = rel @ unit
    projections = signed / length
    closest = a + np.outer(signed, unit)
    distances = np.linalg.norm(pts - closest, axis=1)
    return distances.astype(np.float64), projections.astype(np.float64)


def _line_fit_limits(
    length: float,
    cfg: DxfGeometryOptimizeConfig,
) -> Tuple[float, float, float]:
    p95_limit = max(float(cfg.line_tolerance_mm), float(length) * float(cfg.line_tolerance_ratio))
    max_limit = max(float(cfg.line_max_tolerance_mm), float(length) * float(cfg.line_max_tolerance_ratio))
    rms_limit = max(float(cfg.line_rms_tolerance_mm), p95_limit * 1.25)

    p95_limit = min(p95_limit, 8.0)
    max_limit = min(max_limit, 18.0)
    rms_limit = min(rms_limit, 8.0)
    return float(p95_limit), float(max_limit), float(rms_limit)


def _closed_polyline_metrics(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = _remove_closing_duplicate(_as_points(points))
    closed = np.vstack([pts, pts[:1]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return closed, seg_lengths.astype(np.float64), cumulative.astype(np.float64)


def _point_at_closed_distance(
    closed: np.ndarray,
    seg_lengths: np.ndarray,
    cumulative: np.ndarray,
    distance: float,
    perimeter: float,
) -> np.ndarray:
    if perimeter <= 1e-9 or len(seg_lengths) == 0:
        return closed[0].copy()

    d = float(distance % perimeter)
    idx = int(np.searchsorted(cumulative, d, side="right") - 1)
    idx = max(0, min(idx, len(seg_lengths) - 1))

    while idx < len(seg_lengths) - 1 and seg_lengths[idx] <= 1e-9:
        idx += 1

    seg_len = float(seg_lengths[idx])
    if seg_len <= 1e-9:
        return closed[idx].copy()

    t = (d - float(cumulative[idx])) / seg_len
    t = max(0.0, min(1.0, float(t)))
    return (closed[idx] * (1.0 - t) + closed[idx + 1] * t).astype(np.float64)


def _has_near_distance(
    distance: float,
    distances: List[float],
    min_spacing: float,
    perimeter: float,
) -> bool:
    return any(_cyclic_distance(distance, existing, perimeter) < min_spacing for existing in distances)


def _cyclic_distance(a: float, b: float, period: float) -> float:
    if period <= 1e-9:
        return abs(float(a) - float(b))
    delta = abs(float(a) - float(b)) % float(period)
    return min(delta, float(period) - delta)


def _rotate_to_best_break(points: np.ndarray) -> Tuple[np.ndarray, int]:
    pts = _remove_closing_duplicate(_as_points(points))
    n = len(pts)
    if n < 4:
        return pts.copy(), 0

    best_index = 0
    best_deflection = -1.0

    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        curr = pts[i]
        next_pt = pts[(i + 1) % n]
        v1 = curr - prev_pt
        v2 = next_pt - curr
        angle = _angle_between_vectors_deg(v1, v2)
        deflection = angle
        if deflection > best_deflection:
            best_deflection = deflection
            best_index = i

    if best_deflection < 6.0:
        return pts.copy(), 0

    return np.roll(pts, -best_index, axis=0), best_index


def _polyline_fallback_entity(points: np.ndarray) -> Dict[str, Any]:
    pts = _remove_closing_duplicate(_as_points(points))
    return {
        "type": "LWPOLYLINE",
        "points": [[float(x), float(y)] for x, y in pts],
        "point_count": int(len(pts)),
        "edge_count": int(len(pts)),
        "max_error_mm": 0.0,
    }


def _line_entity(
    start: np.ndarray,
    end: np.ndarray,
    point_count: int,
    max_error_mm: float,
) -> Dict[str, Any]:
    return {
        "type": "LINE",
        "start": [float(start[0]), float(start[1])],
        "end": [float(end[0]), float(end[1])],
        "point_count": int(point_count),
        "edge_count": max(1, int(point_count) - 1),
        "max_error_mm": float(max_error_mm),
    }


def _entity_to_dxf_lines(
    entity: Dict[str, Any],
    dx: float,
    dy: float,
    layer: str,
) -> List[str]:
    entity_type = entity.get("type")

    if entity_type == "CIRCLE":
        cx, cy = entity["center"]
        return [
            "0", "CIRCLE",
            "8", layer,
            "10", _fmt(float(cx) + dx),
            "20", _fmt(float(cy) + dy),
            "30", "0.0",
            "40", _fmt(float(entity["radius"])),
        ]

    if entity_type == "ARC":
        cx, cy = entity["center"]
        return [
            "0", "ARC",
            "8", layer,
            "10", _fmt(float(cx) + dx),
            "20", _fmt(float(cy) + dy),
            "30", "0.0",
            "40", _fmt(float(entity["radius"])),
            "50", _fmt_angle(float(entity["start_angle_deg"])),
            "51", _fmt_angle(float(entity["end_angle_deg"])),
        ]

    if entity_type == "LWPOLYLINE":
        pts = entity.get("points") or []
        lines = ["0", "LWPOLYLINE", "8", layer, "90", str(len(pts)), "70", "1"]
        for x, y in pts:
            lines.extend(["10", _fmt(float(x) + dx), "20", _fmt(float(y) + dy)])
        return lines

    x1, y1 = entity["start"]
    x2, y2 = entity["end"]
    return [
        "0", "LINE",
        "8", layer,
        "10", _fmt(float(x1) + dx),
        "20", _fmt(float(y1) + dy),
        "30", "0.0",
        "11", _fmt(float(x2) + dx),
        "21", _fmt(float(y2) + dy),
        "31", "0.0",
    ]


def _compute_positive_offset(
    points: np.ndarray,
    entities: List[Dict[str, Any]],
    offset_to_positive: bool,
) -> Tuple[float, float]:
    if not offset_to_positive:
        return 0.0, 0.0

    bounds = _entity_bounds(entities)
    if bounds is None:
        pts = _as_points(points)
        min_xy = np.min(pts, axis=0)
        return -min(0.0, float(min_xy[0])), -min(0.0, float(min_xy[1]))

    min_x, min_y, _, _ = bounds
    return -min(0.0, float(min_x)), -min(0.0, float(min_y))


def _entity_bounds(entities: List[Dict[str, Any]]) -> Tuple[float, float, float, float] | None:
    bounds: List[Tuple[float, float, float, float]] = []

    for entity in entities:
        entity_type = entity.get("type")

        if entity_type == "CIRCLE":
            cx, cy = entity["center"]
            r = float(entity["radius"])
            bounds.append((float(cx) - r, float(cy) - r, float(cx) + r, float(cy) + r))
        elif entity_type == "ARC":
            bounds.append(_arc_bounds(entity))
        elif entity_type == "LWPOLYLINE":
            pts = np.asarray(entity.get("points") or [], dtype=np.float64).reshape(-1, 2)
            if len(pts):
                min_xy = np.min(pts, axis=0)
                max_xy = np.max(pts, axis=0)
                bounds.append((float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])))
        elif entity_type == "LINE":
            pts = np.asarray([entity["start"], entity["end"]], dtype=np.float64)
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            bounds.append((float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])))

    if not bounds:
        return None

    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


def _arc_bounds(entity: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cx, cy = entity["center"]
    r = float(entity["radius"])
    start = _normalize_angle(float(entity["start_angle_deg"]))
    end = _normalize_angle(float(entity["end_angle_deg"]))
    angles = [start, end]

    for cardinal in (0.0, 90.0, 180.0, 270.0):
        if _angle_on_ccw_arc(cardinal, start, end):
            angles.append(cardinal)

    xs = [float(cx) + math.cos(math.radians(a)) * r for a in angles]
    ys = [float(cy) + math.sin(math.radians(a)) * r for a in angles]
    return min(xs), min(ys), max(xs), max(ys)


def _angle_on_ccw_arc(angle: float, start: float, end: float) -> bool:
    angle = _normalize_angle(angle)
    start = _normalize_angle(start)
    end = _normalize_angle(end)

    if end < start:
        end += 360.0

    if angle < start:
        angle += 360.0

    return start <= angle <= end


def _summary_info(
    cfg: DxfGeometryOptimizeConfig,
    input_points: int,
    entities: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    max_error = 0.0
    for entity in entities:
        entity_type = str(entity.get("type", "UNKNOWN"))
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        max_error = max(max_error, float(entity.get("max_error_mm", 0.0)))

    return {
        "enabled": bool(cfg.enabled),
        "mode": mode,
        "config": asdict(cfg),
        "input_points": int(input_points),
        "entity_count": int(len(entities)),
        "entity_type_counts": type_counts,
        "max_fit_error_mm": round(float(max_error), 6),
        "entities": [_entity_info_for_json(e) for e in entities],
    }


def _empty_info(
    cfg: DxfGeometryOptimizeConfig,
    input_points: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "enabled": bool(cfg.enabled),
        "mode": reason,
        "config": asdict(cfg),
        "input_points": int(input_points),
        "entity_count": 0,
        "entity_type_counts": {},
        "max_fit_error_mm": 0.0,
        "entities": [],
    }


def _entity_info_for_json(entity: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "type",
        "point_count",
        "edge_count",
        "max_error_mm",
        "p95_error_mm",
        "rms_error_mm",
        "fit_tolerance_mm",
        "fit_max_tolerance_mm",
        "radius",
        "diameter",
        "input_sweep_deg",
        "start_angle_deg",
        "end_angle_deg",
        "monotonic_ratio",
        "start_tangent_angle_deg",
        "end_tangent_angle_deg",
        "circularity",
        "radius_error_ratio",
        "bbox_diameter_error_mm",
        "reason",
    }
    out = {k: entity[k] for k in keep if k in entity}
    if "center" in entity:
        out["center"] = [round(float(v), 6) for v in entity["center"]]
    return out


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("contour points are empty")
    return pts


def _remove_closing_duplicate(points: np.ndarray) -> np.ndarray:
    pts = _as_points(points)
    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        return pts[:-1].copy()
    return pts.copy()


def _remove_consecutive_duplicates(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    pts = _remove_closing_duplicate(_as_points(points))
    if len(pts) <= 1:
        return pts.copy()

    kept = [pts[0]]
    for point in pts[1:]:
        if float(np.linalg.norm(point - kept[-1])) > float(tolerance):
            kept.append(point)

    if len(kept) >= 2 and float(np.linalg.norm(kept[0] - kept[-1])) <= float(tolerance):
        kept.pop()

    return np.asarray(kept, dtype=np.float64).reshape(-1, 2)


def _polygon_area(points: np.ndarray) -> float:
    pts = _remove_closing_duplicate(points)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _perimeter(points: np.ndarray) -> float:
    pts = _remove_closing_duplicate(points)
    if len(pts) < 2:
        return 0.0
    closed = np.vstack([pts, pts[:1]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def _circularity(points: np.ndarray) -> float:
    area = abs(_polygon_area(points))
    perimeter = _perimeter(points)
    if perimeter <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def _angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 180.0
    cos_v = float(np.dot(v1, v2) / (n1 * n2))
    cos_v = max(-1.0, min(1.0, cos_v))
    return float(math.degrees(math.acos(cos_v)))


def _angle_to_circle_tangent_deg(vector: np.ndarray, radial: np.ndarray) -> float:
    tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
    angle = _angle_between_vectors_deg(vector, tangent)
    return min(angle, abs(180.0 - angle))


def _angle_deg(center: np.ndarray, point: np.ndarray) -> float:
    return _normalize_angle(math.degrees(math.atan2(float(point[1] - center[1]), float(point[0] - center[0]))))


def _normalize_angle(angle: float) -> float:
    out = float(angle) % 360.0
    if out < 0.0:
        out += 360.0
    return out


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _fmt_angle(value: float) -> str:
    return f"{_normalize_angle(float(value)):.6f}"
