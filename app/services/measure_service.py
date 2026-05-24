from __future__ import annotations

import shutil
import time
import threading
import uuid
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from typing import Annotated
from fastapi import Form

from fastapi import UploadFile
from loguru import logger

from app.config import Settings
from app.core.algorithm import IMAGE_EXTS, json_safe, process_one_image
from app.services.dxf_preview import create_dxf_preview
from app.services.model_cache import model_cache
from app.services.task_store import TaskStore


class MeasureService:
    """Application service that adapts HTTP input to the original algorithm."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.output_root.mkdir(parents=True, exist_ok=True)
        self._process_lock = threading.Lock()
        self.task_store = TaskStore(self.settings.task_db_path)
        self.task_store.mark_interrupted_tasks_failed()
        self._task_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plate-batch-task")
        logger.info(
            "MeasureService initialized: output_root={}, max_upload_mb={}, serialize_processing={}, task_db={}",
            self.settings.output_root,
            self.settings.max_upload_mb,
            self.settings.serialize_processing,
            self.settings.task_db_path,
        )

    def _safe_suffix(self, filename: str) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in IMAGE_EXTS:
            logger.warning("Unsupported image suffix: filename={}, suffix={}", filename, suffix or "unknown")
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
                    logger.warning(
                        "Upload exceeded size limit: filename={}, written_bytes={}, max_mb={}",
                        image.filename,
                        written,
                        self.settings.max_upload_mb,
                    )
                    raise ValueError(f"上传图片超过限制：{self.settings.max_upload_mb}MB")
                f.write(chunk)
        await image.close()
        logger.info("Upload persisted: source_filename={}, target={}, size_bytes={}", image.filename, target, written)
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
            perspective_source: str,
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
        logger.info(
            "Building measure args: image_path={}, model_path={}, sam_model={}, imgsz={}, conf={}, "
            "paper_source={}, a4_orientation={}, perspective_source={}, paper_rect_mode={}",
            image_path,
            selected_model_path,
            sam_model or self.settings.sam_model,
            imgsz or self.settings.default_imgsz,
            conf if conf is not None else self.settings.default_conf,
            paper_source,
            a4_orientation,
            perspective_source,
            paper_rect_mode,
        )
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
            perspective_source=perspective_source,
            paper_points=paper_points,
            paper_rect_mode=paper_rect_mode,
            simplify_mm=float(simplify_mm),
            topdown_mm_per_px=float(topdown_mm_per_px),
            topdown_padding_mm=float(topdown_padding_mm),
            enabled=enabled,
            dxf_postprocess_enabled=enabled,
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
        dxf_path = paths.get("dxf")
        if dxf_path:
            try:
                preview_path = create_dxf_preview(dxf_path)
                if preview_path:
                    paths["dxf_preview"] = str(preview_path)
            except Exception as exc:
                logger.warning("DXF preview generation failed: dxf_path={}, error={}", dxf_path, exc)

        urls = {key: url for key, value in paths.items() if (url := self._path_to_url(value))}
        logger.debug(
            "Compacting measure result: run_dir={}, path_count={}, url_count={}",
            result.get("run_dir"),
            len(paths),
            len(urls),
        )
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
            "dxf_postprocess": result.get("dxf_postprocess", {}),
            "dxf_geometry": result.get("dxf_geometry", {}),
            "result_json": paths.get("result_json"),
        }

    def measure(self, args: SimpleNamespace) -> Dict[str, Any]:
        """Run the original measuring algorithm.

        A lock is enabled by default because YOLO and SAM2 inference can be GPU
        heavy. Disable SERIALIZE_PROCESSING only after confirming your runtime
        supports concurrent inference safely.
        """

        if self.settings.serialize_processing:
            logger.debug("Waiting for serialized processing lock: image={}", args.image)
            with self._process_lock:
                logger.info("Processing started with serialized lock: image={}", args.image)
                started = time.perf_counter()
                result = self.compact_result(process_one_image(args))
                logger.info(
                    "Processing finished with serialized lock: image={}, elapsed_sec={:.3f}, run_dir={}",
                    args.image,
                    time.perf_counter() - started,
                    result.get("run_dir"),
                )
                return result
        logger.info("Processing started without serialized lock: image={}", args.image)
        started = time.perf_counter()
        result = self.compact_result(process_one_image(args))
        logger.info(
            "Processing finished without serialized lock: image={}, elapsed_sec={:.3f}, run_dir={}",
            args.image,
            time.perf_counter() - started,
            result.get("run_dir"),
        )
        return result

    def enqueue_task(
        self,
        *,
        client_id: str,
        image_path: Path,
        original_filename: str,
        file_size_bytes: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        task = self.task_store.create_task(
            task_id=task_id,
            client_id=client_id,
            original_filename=original_filename,
            stored_path=str(image_path),
            file_size_bytes=file_size_bytes,
            params_json=json.dumps(json_safe(params), ensure_ascii=False),
        )
        self._task_executor.submit(self._run_task, task_id)
        logger.info(
            "Batch task queued: task_id={}, client_id={}, filename={}, size_bytes={}",
            task_id,
            client_id,
            original_filename,
            file_size_bytes,
        )
        return self.task_to_response(task)

    def _run_task(self, task_id: str) -> None:
        task = self.task_store.get_task(task_id)
        if task is None:
            return

        filename = task.get("original_filename")
        try:
            self.task_store.update_task(
                task_id,
                status="processing",
                progress=20,
                message="正在识别",
            )
            params = json.loads(task.get("params_json") or "{}")
            args = self.build_args(image_path=Path(task["stored_path"]), **params)
            self.task_store.update_task(task_id, progress=45, message="正在生成 DXF")
            result = self.measure(args)
            self.task_store.update_task(
                task_id,
                status="completed",
                progress=100,
                message="识别完成",
                result_json=json.dumps(json_safe(result), ensure_ascii=False),
                run_dir=str(result.get("run_dir") or ""),
            )
            logger.info("Batch task completed: task_id={}, filename={}", task_id, filename)
        except Exception as exc:
            print(traceback.format_exc())
            logger.exception("Batch task failed: task_id={}, filename={}, error={}", task_id, filename, exc)
            self.task_store.update_task(
                task_id,
                status="failed",
                progress=100,
                message=str(exc),
            )

    def list_client_tasks(self, client_id: str) -> list[Dict[str, Any]]:
        return [self.task_to_response(row) for row in self.task_store.list_tasks(client_id)]

    def get_client_task(self, client_id: str, task_id: str) -> Dict[str, Any] | None:
        row = self.task_store.get_task(task_id, client_id)
        return self.task_to_response(row) if row else None

    def clear_client_tasks(self, client_id: str) -> int:
        rows = self.task_store.list_tasks(client_id)
        deleted = self.task_store.delete_client_tasks(client_id)

        for row in rows:
            self._safe_delete_output_path(row.get("stored_path"))
            self._safe_delete_output_path(row.get("run_dir"))

        logger.info("Client tasks cleared: client_id={}, deleted={}", client_id, deleted)
        return deleted

    def task_to_response(self, row: Dict[str, Any]) -> Dict[str, Any]:
        result = self._load_task_result(row)
        params = self._load_task_params(row)
        urls = dict((result or {}).get("urls") or {})
        paths = dict((result or {}).get("paths") or {})
        upload_url = self._path_to_url(str(row.get("stored_path") or ""))
        if upload_url:
            urls["upload"] = upload_url

        return {
            "id": row.get("id"),
            "client_id": row.get("client_id"),
            "filename": row.get("original_filename"),
            "size_bytes": int(row.get("file_size_bytes") or 0),
            "status": row.get("status"),
            "progress": int(row.get("progress") or 0),
            "message": row.get("message") or "",
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "run_dir": row.get("run_dir"),
            "urls": urls,
            "paths": paths,
            "params": params,
            "plate_dimensions": (result or {}).get("plate_dimensions", {}),
            "dxf_geometry": (result or {}).get("dxf_geometry", {}),
            "dxf_postprocess": (result or {}).get("dxf_postprocess", {}),
            "result": result,
        }

    def _load_task_params(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raw = row.get("params_json")
        if not raw:
            return {}
        try:
            params = json.loads(raw)
            return params if isinstance(params, dict) else {}
        except Exception:
            logger.warning("Failed to parse task params_json: task_id={}", row.get("id"))
            return {}

    def _load_task_result(self, row: Dict[str, Any]) -> Dict[str, Any] | None:
        raw = row.get("result_json")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Failed to parse task result_json: task_id={}", row.get("id"))
            return None

    def _safe_delete_output_path(self, path_value: Any) -> None:
        if not path_value:
            return

        try:
            root = self.settings.output_root.resolve()
            path = Path(str(path_value)).resolve()
            path.relative_to(root)
        except Exception:
            logger.warning("Skip deleting path outside output_root: {}", path_value)
            return

        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to delete output path: path={}, error={}", path, exc)
