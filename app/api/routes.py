from __future__ import annotations

import traceback
import time
import re
import json
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.errors import public_error_message
from app.schemas import MeasureResponse
from app.security import require_token
from app.services.measure_service import MeasureService

router = APIRouter()
service = MeasureService(get_settings())
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,96}$")


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} 参数不正确",
        )
    return value


def _require_client_id(value: str | None) -> str:
    client_id = (value or "").strip()
    if not client_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少客户端标识")

    if not CLIENT_ID_RE.match(client_id):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="客户端标识格式不正确")

    return client_id


def _validate_measure_options(
    *,
    yolo_input_mode: str,
    paper_source: str,
    a4_orientation: str,
    perspective_source: str,
    paper_rect_mode: str,
) -> None:
    _validate_choice(yolo_input_mode, {"canonical_path", "rgb_array", "bgr_array"}, "识别输入模式")
    _validate_choice(paper_source, {"yolo", "sam2"}, "纸张轮廓来源")
    _validate_choice(a4_orientation, {"auto", "landscape", "portrait"}, "A4方向")
    _validate_choice(perspective_source, {"a4", "plate"}, "透视矫正基准")
    _validate_choice(paper_rect_mode, {"robust_fit", "approx_poly", "min_area_rect", "raw"}, "纸张四角拟合方式")


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""

    logger.debug("Health check requested")
    return {"ok": "true", "service": "plate-measure-http"}


@router.post("/tasks/upload", dependencies=[Depends(require_token)])
async def upload_tasks(
    images: Annotated[list[UploadFile], File(description="待识别图片，可一次上传多张")],
    x_client_id: Annotated[str | None, Header(alias="X-Client-Id")] = None,
    model_path: Annotated[str | None, Form(description="可选，YOLO权重路径；不传则读取YOLO_MODEL_PATH")] = None,
    sam_model: Annotated[str | None, Form(description="可选，SAM2模型名；不传则读取SAM_MODEL")] = None,
    imgsz: Annotated[int | None, Form(description="YOLO推理尺寸")] = None,
    conf: Annotated[float | None, Form(description="YOLO置信度")] = None,
    yolo_input_mode: Annotated[str, Form(description="YOLO输入模式")] = "canonical_path",
    plate_class: Annotated[str, Form(description="钢板类别名")] = "plate",
    paper_class: Annotated[str, Form(description="A4纸类别名，支持逗号分隔")] = "paper",
    user_point_ratio: Annotated[str | None, Form(description="可选，手动指定plate点，例如0.5,0.5")] = None,
    paper_source: Annotated[str, Form(description="paper mask来源：yolo或sam2")] = "yolo",
    paper_sam2_yolo_fallback: Annotated[bool, Form(description="paper_source=sam2失败时是否回退YOLO")] = False,
    a4_orientation: Annotated[str, Form(description="A4方向：auto、landscape、portrait")] = "auto",
    perspective_source: Annotated[str, Form(description="透视矫正基准：a4或plate")] = "a4",
    paper_points: Annotated[str | None, Form(description="可选，A4四角坐标 x1,y1;x2,y2;x3,y3;x4,y4")] = None,
    paper_rect_mode: Annotated[str, Form(description="paper四角拟合方式")] = "approx_poly",
    simplify_mm: Annotated[float, Form(description="DXF轮廓简化精度，单位mm")] = 3.0,
    topdown_mm_per_px: Annotated[float, Form(description="俯视图比例，1px代表多少mm")] = 2.0,
    topdown_padding_mm: Annotated[float, Form(description="俯视图四周留白，单位mm")] = 50.0,
    dxf_postprocess_enabled: Annotated[bool, Form(description="是否启用DXF后处理")] = True,
    dxf_notch_fill_enabled: Annotated[bool, Form(description="是否启用夹钳凹陷修复")] = False,
    dxf_notch_fill_max_width_mm: Annotated[float, Form(description="夹钳凹陷最大宽度mm")] = 80.0,
    dxf_notch_fill_max_depth_mm: Annotated[float, Form(description="夹钳凹陷最大深度mm")] = 25.0,
    per_file_options: Annotated[str | None, Form(description="可选，每个文件的独立参数 JSON 数组")] = None,
) -> dict:
    client_id = _require_client_id(x_client_id)
    if not images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择至少一张图片")

    if len(images) > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="单次最多上传100张图片")

    _validate_measure_options(
        yolo_input_mode=yolo_input_mode,
        paper_source=paper_source,
        a4_orientation=a4_orientation,
        perspective_source=perspective_source,
        paper_rect_mode=paper_rect_mode,
    )

    params = {
        "model_path": model_path,
        "sam_model": sam_model,
        "imgsz": imgsz,
        "conf": conf,
        "yolo_input_mode": yolo_input_mode,
        "plate_class": plate_class,
        "paper_class": paper_class,
        "user_point_ratio": user_point_ratio,
        "paper_source": paper_source,
        "paper_sam2_yolo_fallback": paper_sam2_yolo_fallback,
        "a4_orientation": a4_orientation,
        "perspective_source": perspective_source,
        "paper_points": paper_points,
        "paper_rect_mode": paper_rect_mode,
        "simplify_mm": simplify_mm,
        "topdown_mm_per_px": topdown_mm_per_px,
        "topdown_padding_mm": topdown_padding_mm,
        "enabled": dxf_postprocess_enabled,
        "dxf_notch_fill_enabled": dxf_notch_fill_enabled,
        "dxf_notch_fill_max_width_mm": dxf_notch_fill_max_width_mm,
        "dxf_notch_fill_max_depth_mm": dxf_notch_fill_max_depth_mm,
    }

    file_options: list[dict] = []
    if per_file_options:
        try:
            parsed_options = json.loads(per_file_options)
            if not isinstance(parsed_options, list):
                raise ValueError("per_file_options must be a list")
            file_options = [item if isinstance(item, dict) else {} for item in parsed_options]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="per_file_options 格式不正确",
            ) from exc

    tasks = []
    for image in images:
        index = len(tasks)
        original_filename = image.filename or "image"
        try:
            image_path = await service.save_upload(image)
            task_params = dict(params)
            option = file_options[index] if index < len(file_options) else {}

            task_params["perspective_source"] = "plate" if bool(option.get("use_plate_perspective")) else "a4"
            if "dxf_notch_fill_enabled" in option:
                task_params["dxf_notch_fill_enabled"] = bool(option.get("dxf_notch_fill_enabled"))
            if "dxf_notch_fill_max_width_mm" in option:
                task_params["dxf_notch_fill_max_width_mm"] = float(option.get("dxf_notch_fill_max_width_mm") or 80.0)
            if "dxf_notch_fill_max_depth_mm" in option:
                task_params["dxf_notch_fill_max_depth_mm"] = float(option.get("dxf_notch_fill_max_depth_mm") or 25.0)

            tasks.append(service.enqueue_task(
                client_id=client_id,
                image_path=image_path,
                original_filename=original_filename,
                file_size_bytes=image_path.stat().st_size,
                params=task_params,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=public_error_message(exc)) from exc

    return {"ok": True, "client_id": client_id, "tasks": tasks}


@router.get("/tasks", dependencies=[Depends(require_token)])
async def list_tasks(
    x_client_id: Annotated[str | None, Header(alias="X-Client-Id")] = None,
) -> dict:
    client_id = _require_client_id(x_client_id)
    return {"ok": True, "client_id": client_id, "tasks": service.list_client_tasks(client_id)}


@router.get("/tasks/download-dxf", dependencies=[Depends(require_token)])
async def download_client_dxf(
    x_client_id: Annotated[str | None, Header(alias="X-Client-Id")] = None,
) -> StreamingResponse:
    client_id = _require_client_id(x_client_id)
    tasks = service.list_client_tasks(client_id)

    zip_buffer = io.BytesIO()
    added = 0
    used_names: set[str] = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for task in tasks:
            if task.get("status") != "completed":
                continue

            dxf_value = (task.get("paths") or {}).get("dxf")
            if not dxf_value:
                continue

            dxf_path = Path(dxf_value)
            if not dxf_path.exists() or not dxf_path.is_file():
                logger.warning(
                    "Skipping missing DXF during batch download: client_id={}, task_id={}, path={}",
                    client_id,
                    task.get("id"),
                    dxf_path,
                )
                continue

            base_name = Path(str(task.get("filename") or task.get("id") or "plate")).stem
            safe_name = re.sub(r'[\\/:*?"<>|]+', "_", base_name).strip(" .") or str(task.get("id") or "plate")
            archive_name = f"{safe_name}.dxf"
            suffix = 2
            while archive_name in used_names:
                archive_name = f"{safe_name}_{suffix}.dxf"
                suffix += 1

            used_names.add(archive_name)
            archive.write(dxf_path, archive_name)
            added += 1

    if added == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No completed DXF files for this client")

    zip_buffer.seek(0)
    filename = f"plate_dxf_{client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-DXF-File-Count": str(added),
    }
    return StreamingResponse(iter([zip_buffer.getvalue()]), media_type="application/zip", headers=headers)


@router.get("/tasks/{task_id}", dependencies=[Depends(require_token)])
async def get_task(
    task_id: str,
    x_client_id: Annotated[str | None, Header(alias="X-Client-Id")] = None,
) -> dict:
    client_id = _require_client_id(x_client_id)
    task = service.get_client_task(client_id, task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return {"ok": True, "client_id": client_id, "task": task}


@router.delete("/tasks", dependencies=[Depends(require_token)])
async def clear_tasks(
    x_client_id: Annotated[str | None, Header(alias="X-Client-Id")] = None,
) -> dict:
    client_id = _require_client_id(x_client_id)
    deleted = service.clear_client_tasks(client_id)
    return {"ok": True, "client_id": client_id, "deleted": deleted}


@router.post("/measure", response_model=MeasureResponse, dependencies=[Depends(require_token)])
async def measure(
    image: Annotated[UploadFile, File(description="待识别图片")],
    model_path: Annotated[str | None, Form(description="可选，YOLO权重路径；不传则读取YOLO_MODEL_PATH")]=None,
    sam_model: Annotated[str | None, Form(description="可选，SAM2模型名；不传则读取SAM_MODEL")]=None,
    imgsz: Annotated[int | None, Form(description="YOLO推理尺寸")]=None,
    conf: Annotated[float | None, Form(description="YOLO置信度")]=None,
    yolo_input_mode: Annotated[str, Form(description="YOLO输入模式")]= "canonical_path",
    plate_class: Annotated[str, Form(description="钢板类别名")]= "plate",
    paper_class: Annotated[str, Form(description="A4纸类别名，支持逗号分隔")]= "paper",
    user_point_ratio: Annotated[str | None, Form(description="可选，手动指定plate点，例如0.5,0.5")]=None,
    paper_source: Annotated[str, Form(description="paper mask来源：yolo或sam2")]= "yolo",
    paper_sam2_yolo_fallback: Annotated[bool, Form(description="paper_source=sam2失败时是否退回YOLO")]=False,
    a4_orientation: Annotated[str, Form(description="A4方向：auto、landscape、portrait")]= "auto",
    perspective_source: Annotated[str, Form(description="透视矫正基准：a4 或 plate")]= "a4",
    paper_points: Annotated[str | None, Form(description="可选，A4四角坐标 x1,y1;x2,y2;x3,y3;x4,y4")]=None,
    paper_rect_mode: Annotated[str, Form(description="paper四角拟合方式")]= "approx_poly",
    simplify_mm: Annotated[float, Form(description="DXF轮廓简化精度，单位mm")]=3.0,
    topdown_mm_per_px: Annotated[float, Form(description="俯视图比例，1px代表多少mm")]=2.0,
    topdown_padding_mm: Annotated[float, Form(description="俯视图四周留白，单位mm")]=50.0,
        dxf_postprocess_enabled: Annotated[bool, Form(description="是否启用DXF后处理")] = True,
        dxf_notch_fill_enabled: Annotated[bool, Form(description="是否启用夹钳凹陷修复")] = True,
        dxf_notch_fill_max_width_mm: Annotated[float, Form(description="夹钳凹陷最大宽度mm")] = 130,
        dxf_notch_fill_max_depth_mm: Annotated[float, Form(description="夹钳凹陷最大深度mm")] = 60,
) -> MeasureResponse:
    """Measure one uploaded image and return generated file paths."""

    started = time.perf_counter()
    original_filename = image.filename
    logger.info(
        "Measure request received: filename={}, model_path={}, sam_model={}, imgsz={}, conf={}, "
        "yolo_input_mode={}, plate_class={}, paper_class={}, paper_source={}, a4_orientation={}, "
        "perspective_source={}, paper_rect_mode={}, simplify_mm={}, dxf_postprocess_enabled={}",
        original_filename,
        model_path,
        sam_model,
        imgsz,
        conf,
        yolo_input_mode,
        plate_class,
        paper_class,
        paper_source,
        a4_orientation,
        perspective_source,
        paper_rect_mode,
        simplify_mm,
        dxf_postprocess_enabled,
    )

    _validate_measure_options(
        yolo_input_mode=yolo_input_mode,
        paper_source=paper_source,
        a4_orientation=a4_orientation,
        perspective_source=perspective_source,
        paper_rect_mode=paper_rect_mode,
    )

    try:
        image_path = await service.save_upload(image)
        logger.info("Upload saved: filename={}, path={}", original_filename, image_path)
        args = await run_in_threadpool(
            service.build_args,
            image_path=image_path,
            model_path=model_path,
            sam_model=sam_model,
            imgsz=imgsz,
            conf=conf,
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
            simplify_mm=simplify_mm,
            topdown_mm_per_px=topdown_mm_per_px,
            topdown_padding_mm=topdown_padding_mm,
            enabled=dxf_postprocess_enabled,
            dxf_notch_fill_enabled=dxf_notch_fill_enabled,
            dxf_notch_fill_max_width_mm=dxf_notch_fill_max_width_mm,
            dxf_notch_fill_max_depth_mm=dxf_notch_fill_max_depth_mm,
        )
        result = await run_in_threadpool(service.measure, args)
        elapsed = time.perf_counter() - started
        logger.info(
            "Measure request completed: filename={}, run_dir={}, elapsed_sec={:.3f}, dxf={}",
            original_filename,
            result.get("run_dir"),
            elapsed,
            (result.get("paths") or {}).get("dxf"),
        )
        return MeasureResponse(**result)
    except HTTPException:
        logger.warning("Measure request rejected with HTTPException: filename={}", original_filename)
        raise
    except ValueError as exc:
        logger.warning("Measure request failed with validation error: filename={}, error={}", original_filename, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=public_error_message(exc)) from exc
    except Exception as exc:
        # Keep the HTTP response compact. Full traceback is printed to server log.
        print(traceback.format_exc())
        logger.exception("Measure request failed: filename={}, error={}", original_filename, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=public_error_message(exc)) from exc
