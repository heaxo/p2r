from __future__ import annotations

import traceback
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.schemas import MeasureResponse
from app.security import require_token
from app.services.measure_service import MeasureService

router = APIRouter()
service = MeasureService(get_settings())


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} 只能是：{', '.join(sorted(allowed))}",
        )
    return value


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"ok": "true", "service": "plate-measure-http"}


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

    _validate_choice(yolo_input_mode, {"canonical_path", "rgb_array", "bgr_array"}, "yolo_input_mode")
    _validate_choice(paper_source, {"yolo", "sam2"}, "paper_source")
    _validate_choice(a4_orientation, {"auto", "landscape", "portrait"}, "a4_orientation")
    _validate_choice(paper_rect_mode, {"robust_fit", "approx_poly", "min_area_rect", "raw"}, "paper_rect_mode")

    try:
        image_path = await service.save_upload(image)
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
        return MeasureResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        # Keep the HTTP response compact. Full traceback is printed to server log.
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
