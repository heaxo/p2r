from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from typing import Annotated
from fastapi import Form

from fastapi import UploadFile

from app.config import Settings
from app.core.algorithm import IMAGE_EXTS, json_safe, process_one_image
from app.services.model_cache import model_cache


class MeasureService:
    """Application service that adapts HTTP input to the original algorithm."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.output_root.mkdir(parents=True, exist_ok=True)
        self._process_lock = threading.Lock()

    def _safe_suffix(self, filename: str) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise ValueError(f"不支持的图片格式：{suffix or 'unknown'}")
        return suffix

    async def save_upload(self, image: UploadFile) -> Path:
        """Persist the uploaded image under the output root.

        The generated result directory is also created under the same root, so
        returned paths can be exposed through the static file mount.
        """

        suffix = self._safe_suffix(image.filename or "image.jpg")
        upload_dir = self.settings.output_root / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / f"{Path(image.filename or 'image').stem}_{uuid.uuid4().hex[:8]}{suffix}"

        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with target.open("wb") as f:
            while True:
                chunk = await image.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    target.unlink(missing_ok=True)
                    raise ValueError(f"上传图片超过限制：{self.settings.max_upload_mb}MB")
                f.write(chunk)
        await image.close()
        return target

    def build_args(
            self,
            *,
            image_path: Path,
            model_path: str | None,
            sam_model: str | None,
            imgsz: int | None,
            conf: float | None,
            yolo_input_mode: str,
            plate_class: str,
            paper_class: str,
            user_point_ratio: str | None,
            paper_source: str,
            paper_sam2_yolo_fallback: bool,
            a4_orientation: str,
            paper_points: str | None,
            paper_rect_mode: str,
            simplify_mm: float,
            topdown_mm_per_px: float,
            topdown_padding_mm: float,
            enabled: Annotated[bool, Form(description="是否启用DXF后处理")],
            dxf_notch_fill_enabled: Annotated[bool, Form(description="是否启用夹钳凹陷修复")],
            dxf_notch_fill_max_width_mm: Annotated[float, Form(description="夹钳凹陷最大宽度mm")],
            dxf_notch_fill_max_depth_mm: Annotated[float, Form(description="夹钳凹陷最大深度mm")],
    ) -> SimpleNamespace:
        """Create an args object compatible with process_one_image(args)."""

        selected_model_path = Path(model_path) if model_path else self.settings.model_path
        yolo_model = model_cache.get(selected_model_path)

        return SimpleNamespace(
            image=str(image_path),
            model=str(selected_model_path),
            out=str(self.settings.output_root),
            sam_model=sam_model or self.settings.sam_model,
            imgsz=int(imgsz or self.settings.default_imgsz),
            conf=float(conf if conf is not None else self.settings.default_conf),
            yolo_input_mode=yolo_input_mode,
            plate_class=plate_class,
            paper_class=paper_class,
            user_point_ratio=user_point_ratio,
            paper_source=paper_source,
            paper_sam2_yolo_fallback=paper_sam2_yolo_fallback,
            a4_orientation=a4_orientation,
            paper_points=paper_points,
            paper_rect_mode=paper_rect_mode,
            simplify_mm=float(simplify_mm),
            topdown_mm_per_px=float(topdown_mm_per_px),
            topdown_padding_mm=float(topdown_padding_mm),
            enabled=enabled,
            dxf_notch_fill_enabled=dxf_notch_fill_enabled,
            dxf_notch_fill_max_width_mm=dxf_notch_fill_max_width_mm,
            dxf_notch_fill_max_depth_mm=dxf_notch_fill_max_depth_mm,
            yolo_model=yolo_model,
        )

    def _path_to_url(self, path: str) -> str | None:
        """Convert an output file path to a static URL when it is under output_root."""

        try:
            output_root = self.settings.output_root.resolve()
            resolved = Path(path).resolve()
            rel = resolved.relative_to(output_root).as_posix()
            return f"{self.settings.static_url_prefix.rstrip('/')}/{rel}"
        except Exception:
            return None

    def compact_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return only the fields useful to the HTTP caller."""

        paths = dict(result.get("paths") or {})
        urls = {key: url for key, value in paths.items() if (url := self._path_to_url(value))}
        return {
            "ok": True,
            "run_dir": result.get("run_dir"),
            "paths": paths,
            "urls": urls,
            "plate_dimensions": result.get("plate_dimensions", {}),
            "a4": result.get("a4", {}),
            "topdown": result.get("topdown", {}),
            "input": result.get("input", {}),
            "model": result.get("model", {}),
            "paper": result.get("paper", {}),
            "plate": result.get("plate", {}),
            "fill_paper_to_plate": result.get("fill_paper_to_plate", {}),
            "result_json": paths.get("result_json"),
        }

    def measure(self, args: SimpleNamespace) -> Dict[str, Any]:
        """Run the original measuring algorithm.

        A lock is enabled by default because YOLO and SAM2 inference can be GPU
        heavy. Disable SERIALIZE_PROCESSING only after confirming your runtime
        supports concurrent inference safely.
        """

        if self.settings.serialize_processing:
            with self._process_lock:
                return self.compact_result(process_one_image(args))
        return self.compact_result(process_one_image(args))
