from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings

settings = get_settings()
settings.output_root.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Plate Measure HTTP Service",
    version="1.2.0",
    description="YOLO + SAM2 plate measurement service. It returns generated DXF, mask, image and JSON paths.",
)

app.include_router(router)

# 输出文件静态访问地址，例如 /files/xxx/plate_outer.dxf。
app.mount(settings.static_url_prefix, StaticFiles(directory=str(settings.output_root)), name="files")

# 前端页面。用户页访问 /ui/，调试参数页访问 /ui/debug.html。
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    """Redirect browser users to the built-in upload page."""

    return RedirectResponse(url="/ui/")
