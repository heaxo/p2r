from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def create_dxf_preview(dxf_path: str | Path, output_path: str | Path | None = None) -> Path | None:
    """Render the small DXF subset produced by this service into a PNG preview."""

    path = Path(dxf_path)
    if not path.exists():
        return None

    output = Path(output_path) if output_path else path.with_name("dxf_preview.png")
    entities = _read_supported_entities(path)
    polylines = _entities_to_polylines(entities)

    if not polylines:
        return None

    all_points = np.vstack(polylines).astype(np.float64)
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)

    max_w = 1200
    max_h = 860
    margin = 44
    scale = min((max_w - margin * 2) / size[0], (max_h - margin * 2) / size[1])
    scale = max(0.05, float(scale))

    canvas_w = int(max(360, min(max_w, math.ceil(size[0] * scale + margin * 2))))
    canvas_h = int(max(260, min(max_h, math.ceil(size[1] * scale + margin * 2))))
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    def to_px(points: np.ndarray) -> np.ndarray:
        pts = points.astype(np.float64)
        out = np.empty_like(pts)
        out[:, 0] = (pts[:, 0] - min_xy[0]) * scale + margin
        out[:, 1] = canvas_h - ((pts[:, 1] - min_xy[1]) * scale + margin)
        return np.round(out).astype(np.int32)

    cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), (225, 232, 235), 1)

    for points in polylines:
        px = to_px(points).reshape(-1, 1, 2)
        closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1e-6)
        cv2.polylines(canvas, [px], closed, (15, 23, 42), 3, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{size[0]:.0f} x {size[1]:.0f} mm",
        (18, canvas_h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (71, 85, 105),
        1,
        cv2.LINE_AA,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output


def _read_supported_entities(path: Path) -> List[Dict[str, Any]]:
    values = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pairs = []

    for i in range(0, len(values) - 1, 2):
        pairs.append((values[i].strip(), values[i + 1].strip()))

    entities: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for code, value in pairs:
        if code == "0":
            if current:
                entities.append(current)

            if value in {"LINE", "ARC", "CIRCLE", "LWPOLYLINE"}:
                current = {"type": value, "raw": []}
            else:
                current = None
            continue

        if current is not None:
            current["raw"].append((code, value))

    if current:
        entities.append(current)

    return entities


def _entities_to_polylines(entities: List[Dict[str, Any]]) -> List[np.ndarray]:
    polylines: List[np.ndarray] = []

    for entity in entities:
        entity_type = entity.get("type")
        raw = entity.get("raw") or []

        if entity_type == "LINE":
            values = _first_values(raw)
            if {"10", "20", "11", "21"} <= values.keys():
                polylines.append(np.asarray([
                    [values["10"], values["20"]],
                    [values["11"], values["21"]],
                ], dtype=np.float64))

        elif entity_type == "CIRCLE":
            values = _first_values(raw)
            if {"10", "20", "40"} <= values.keys():
                center = np.asarray([values["10"], values["20"]], dtype=np.float64)
                radius = float(values["40"])
                angles = np.linspace(0.0, 2.0 * math.pi, 241)
                pts = np.column_stack([
                    center[0] + np.cos(angles) * radius,
                    center[1] + np.sin(angles) * radius,
                ])
                polylines.append(pts.astype(np.float64))

        elif entity_type == "ARC":
            values = _first_values(raw)
            if {"10", "20", "40", "50", "51"} <= values.keys():
                center = np.asarray([values["10"], values["20"]], dtype=np.float64)
                radius = float(values["40"])
                start = math.radians(float(values["50"]))
                end = math.radians(float(values["51"]))
                if end < start:
                    end += 2.0 * math.pi
                steps = max(24, int(abs(end - start) / (math.pi / 90.0)))
                angles = np.linspace(start, end, steps)
                pts = np.column_stack([
                    center[0] + np.cos(angles) * radius,
                    center[1] + np.sin(angles) * radius,
                ])
                polylines.append(pts.astype(np.float64))

        elif entity_type == "LWPOLYLINE":
            pts = _lwpolyline_points(raw)
            if len(pts) >= 2:
                polylines.append(pts)

    return polylines


def _first_values(raw: List[Tuple[str, str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for code, value in raw:
        if code in out:
            continue
        try:
            out[code] = float(value)
        except ValueError:
            pass
    return out


def _lwpolyline_points(raw: List[Tuple[str, str]]) -> np.ndarray:
    points: List[List[float]] = []
    pending_x: float | None = None
    closed = False

    for code, value in raw:
        if code == "70":
            try:
                closed = bool(int(value) & 1)
            except ValueError:
                closed = False
        elif code == "10":
            try:
                pending_x = float(value)
            except ValueError:
                pending_x = None
        elif code == "20" and pending_x is not None:
            try:
                points.append([pending_x, float(value)])
            except ValueError:
                pass
            pending_x = None

    if closed and len(points) >= 2 and points[0] != points[-1]:
        points.append(points[0])

    return np.asarray(points, dtype=np.float64).reshape(-1, 2)
