from __future__ import annotations

import re
from typing import Any


_ERROR_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"paper_sam2_yolo_fallback", re.IGNORECASE), "paper_recognition_fallback"),
    (re.compile(r"yolo_input_mode", re.IGNORECASE), "recognition_input_mode"),
    (re.compile(r"YOLO_MODEL_PATH", re.IGNORECASE), "MODEL_PATH"),
    (re.compile(r"SAM_MODEL", re.IGNORECASE), "SEGMENTATION_CONFIG"),
    (re.compile(r"sam_model", re.IGNORECASE), "segmentation_config"),
    (re.compile(r"\bYOLO\b", re.IGNORECASE), "识别模块"),
    (re.compile(r"\bSAM2\b", re.IGNORECASE), "分割模块"),
)


def sanitize_public_error_message(message: Any, fallback: str = "处理失败") -> str:
    text = str(message or "").strip()
    if not text:
        return fallback

    for pattern, replacement in _ERROR_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_public_error_detail(detail: Any) -> Any:
    if isinstance(detail, str):
        return sanitize_public_error_message(detail)
    if isinstance(detail, list):
        return [sanitize_public_error_detail(item) for item in detail]
    if isinstance(detail, tuple):
        return [sanitize_public_error_detail(item) for item in detail]
    if isinstance(detail, dict):
        return {
            sanitize_public_error_message(key, fallback=str(key)): sanitize_public_error_detail(value)
            for key, value in detail.items()
        }
    return detail


def public_error_message(exc: BaseException, fallback: str = "处理失败") -> str:
    return sanitize_public_error_message(str(exc), fallback=fallback)
