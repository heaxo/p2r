from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.reference_detector import ReferenceDetection


CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 10
CHARUCO_SQUARE_LENGTH_MM = 25.0
CHARUCO_MARKER_LENGTH_MM = 18.0
CHARUCO_BOARD_WIDTH_MM = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_MM
CHARUCO_BOARD_HEIGHT_MM = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_MM
CHARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmax(diff)],
            pts[np.argmax(s)],
            pts[np.argmin(diff)],
        ],
        dtype=np.float32,
    )


def create_charuco_board() -> Tuple[Any, Any]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is unavailable; install opencv-contrib-python")

    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY_ID)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LENGTH_MM,
            CHARUCO_MARKER_LENGTH_MM,
            dictionary,
        )
    elif hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            CHARUCO_SQUARES_X,
            CHARUCO_SQUARES_Y,
            CHARUCO_SQUARE_LENGTH_MM,
            CHARUCO_MARKER_LENGTH_MM,
            dictionary,
        )
    else:
        raise RuntimeError("OpenCV CharucoBoard API is unavailable; install opencv-contrib-python")
    return board, dictionary


def _create_detector_parameters() -> Any:
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return None


def _detect_markers(gray: np.ndarray, dictionary: Any) -> Tuple[Sequence[np.ndarray], Optional[np.ndarray], Any]:
    params = _create_detector_parameters()
    if hasattr(cv2.aruco, "detectMarkers"):
        if params is None:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        return corners, ids, rejected

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params) if params is not None else cv2.aruco.ArucoDetector(dictionary)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    raise RuntimeError("OpenCV ArUco marker detection API is unavailable")


def _interpolate_charuco(
    gray: np.ndarray,
    board: Any,
    marker_corners: Sequence[np.ndarray],
    marker_ids: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if marker_ids is None or len(marker_corners) == 0:
        return None, None

    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
        return charuco_corners, charuco_ids

    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(
            gray,
            None,
            None,
            marker_corners,
            marker_ids,
        )
        return charuco_corners, charuco_ids

    raise RuntimeError("OpenCV ChArUco corner interpolation API is unavailable")


def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    raise ValueError("Unsupported image shape for ChArUco detection")


def _valid_homography(H: Optional[np.ndarray]) -> bool:
    if H is None:
        return False
    H = np.asarray(H, dtype=np.float64)
    return H.shape == (3, 3) and np.all(np.isfinite(H)) and abs(float(np.linalg.det(H))) > 1e-12


def _reprojection_error_px(
    H_px_to_mm: np.ndarray,
    image_points_px: np.ndarray,
    object_points_mm: np.ndarray,
    inlier_mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    H_mm_to_px = np.linalg.inv(H_px_to_mm)
    projected_px = cv2.perspectiveTransform(
        object_points_mm.astype(np.float32).reshape(-1, 1, 2),
        H_mm_to_px.astype(np.float64),
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected_px - image_points_px.astype(np.float32), axis=1)
    if inlier_mask is not None:
        mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
        if np.any(mask):
            errors = errors[mask]
    if errors.size == 0:
        return math.inf, math.inf
    return float(np.sqrt(np.mean(errors * errors))), float(np.max(errors))


def detect_charuco_reference(
    image_rgb: np.ndarray,
    *,
    min_charuco_corners: int = 8,
    min_markers: int = 2,
    max_reprojection_rmse_px: float = 8.0,
    ransac_threshold_mm: float = 3.0,
) -> ReferenceDetection:
    info: Dict[str, Any] = {
        "board": {
            "type": "ChArUco Board",
            "squares_x": CHARUCO_SQUARES_X,
            "squares_y": CHARUCO_SQUARES_Y,
            "square_length_mm": CHARUCO_SQUARE_LENGTH_MM,
            "marker_length_mm": CHARUCO_MARKER_LENGTH_MM,
            "dictionary": "DICT_4X4_50",
            "board_width_mm": CHARUCO_BOARD_WIDTH_MM,
            "board_height_mm": CHARUCO_BOARD_HEIGHT_MM,
        },
        "min_charuco_corners": int(min_charuco_corners),
        "min_markers": int(min_markers),
        "max_reprojection_rmse_px": float(max_reprojection_rmse_px),
    }

    try:
        board, dictionary = create_charuco_board()
        gray = _to_gray(image_rgb)
        marker_corners, marker_ids, rejected = _detect_markers(gray, dictionary)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        info["marker_count"] = marker_count
        info["rejected_marker_candidate_count"] = int(len(rejected)) if rejected is not None else 0
        if marker_count < int(min_markers):
            return ReferenceDetection(
                ok=False,
                source="charuco",
                info=info,
                failure_reason=f"detected ArUco markers fewer than {min_markers}",
            )

        charuco_corners, charuco_ids = _interpolate_charuco(gray, board, marker_corners, marker_ids)
        charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
        info["charuco_corner_count"] = charuco_count
        if charuco_count < int(min_charuco_corners):
            return ReferenceDetection(
                ok=False,
                source="charuco",
                info=info,
                failure_reason=f"detected ChArUco corners fewer than {min_charuco_corners}",
            )

        if hasattr(board, "checkCharucoCornersCollinear") and bool(board.checkCharucoCornersCollinear(charuco_ids)):
            return ReferenceDetection(
                ok=False,
                source="charuco",
                info=info,
                failure_reason="detected ChArUco corners are collinear",
            )

        image_points_px = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        corner_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        board_corners_mm = np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
        object_points_mm = board_corners_mm[corner_ids, :2].astype(np.float32)

        H, inlier_mask = cv2.findHomography(
            image_points_px,
            object_points_mm,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold_mm),
        )
        if not _valid_homography(H):
            return ReferenceDetection(
                ok=False,
                source="charuco",
                info=info,
                failure_reason="failed to compute a valid ChArUco homography",
            )

        inlier_count = int(np.asarray(inlier_mask).sum()) if inlier_mask is not None else int(charuco_count)
        info["homography_inlier_count"] = inlier_count
        if inlier_count < int(min_charuco_corners):
            return ReferenceDetection(
                ok=False,
                source="charuco",
                info=info,
                failure_reason=f"ChArUco homography inliers fewer than {min_charuco_corners}",
            )

        rmse_px, max_error_px = _reprojection_error_px(H, image_points_px, object_points_mm, inlier_mask)
        info["reprojection_rmse_px"] = round(float(rmse_px), 4)
        info["reprojection_max_error_px"] = round(float(max_error_px), 4)
        if not math.isfinite(rmse_px) or rmse_px > float(max_reprojection_rmse_px):
            return ReferenceDetection(
                ok=False,
                source="charuco",
                info=info,
                failure_reason=f"ChArUco reprojection RMSE too high: {rmse_px:.3f}px",
            )

        reference_quad_mm = np.array(
            [
                [0.0, 0.0],
                [CHARUCO_BOARD_WIDTH_MM, 0.0],
                [CHARUCO_BOARD_WIDTH_MM, CHARUCO_BOARD_HEIGHT_MM],
                [0.0, CHARUCO_BOARD_HEIGHT_MM],
            ],
            dtype=np.float32,
        )
        reference_quad_px = cv2.perspectiveTransform(
            reference_quad_mm.reshape(-1, 1, 2),
            np.linalg.inv(H).astype(np.float64),
        ).reshape(-1, 2)
        reference_quad_px = _order_quad_points(reference_quad_px)

        info["charuco_ids"] = corner_ids.astype(int).tolist()
        info["charuco_corners_px"] = np.round(image_points_px, 3).tolist()

        return ReferenceDetection(
            ok=True,
            source="charuco",
            H_px_to_mm=H.astype(np.float64),
            reference_quad_px=reference_quad_px.astype(np.float32),
            reference_quad_mm=reference_quad_mm.astype(np.float32),
            size_mm=(CHARUCO_BOARD_WIDTH_MM, CHARUCO_BOARD_HEIGHT_MM),
            info=info,
        )
    except Exception as exc:
        info["exception_type"] = type(exc).__name__
        return ReferenceDetection(
            ok=False,
            source="charuco",
            info=info,
            failure_reason=str(exc),
        )

