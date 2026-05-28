from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.errors import public_error_message, sanitize_public_error_detail
from app.logging_config import setup_logging

settings = get_settings()
setup_logging(settings)
settings.output_root.mkdir(parents=True, exist_ok=True)
logger.info(
    "Starting application: app_name={}, output_root={}, static_url_prefix={}, serialize_processing={}",
    settings.app_name,
    settings.output_root,
    settings.static_url_prefix,
    settings.serialize_processing,
)

from app.api.routes import router

app = FastAPI(
    title="Pic2Remnant HTTP Service",
    version="1.2.0",
    description="Pic2Remnant service. It returns generated DXF, mask, image and JSON paths.",
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": sanitize_public_error_detail(exc.detail)},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Request validation failed: path={}, errors={}", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": sanitize_public_error_detail(exc.errors())},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: path={}, error={}", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": public_error_message(exc, fallback="服务处理失败")},
    )


app.include_router(router)

# 输出文件静态访问地址，例如 /files/xxx/plate_outer.dxf。
app.mount(settings.static_url_prefix, StaticFiles(directory=str(settings.output_root)), name="files")
logger.info("Mounted output files: url_prefix={}, directory={}", settings.static_url_prefix, settings.output_root)

# 前端页面。用户页访问 /ui/，调试参数页访问 /ui/debug.html。
frontend_dir = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend"))
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
    logger.info("Mounted frontend UI: directory={}", frontend_dir)
else:
    logger.warning("Frontend directory not found: {}", frontend_dir)


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    """Redirect browser users to the built-in upload page."""

    logger.debug("Redirecting root request to /ui/")
    return RedirectResponse(url="/ui/")
