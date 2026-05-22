from __future__ import annotations

from pathlib import Path
from threading import Lock

from loguru import logger
from ultralytics import YOLO


class YoloModelCache:
    """Lazy cache for YOLO models keyed by model file path.

    The original console program loaded YOLO for every run. HTTP mode usually
    handles repeated calls, so caching avoids the repeated model load cost.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._models: dict[str, YOLO] = {}

    def get(self, model_path: str | Path) -> YOLO:
        resolved = str(Path(model_path).resolve())
        with self._lock:
            model = self._models.get(resolved)
            if model is None:
                logger.info("Loading YOLO model: path={}", resolved)
                model = YOLO(resolved)
                self._models[resolved] = model
                logger.info("YOLO model loaded and cached: path={}", resolved)
            else:
                logger.debug("Using cached YOLO model: path={}", resolved)
            return model


model_cache = YoloModelCache()
