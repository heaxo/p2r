from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _set_default_env() -> None:
    root = _bundle_root()

    frontend_dir = root / "frontend"
    if frontend_dir.exists():
        os.environ.setdefault("FRONTEND_DIR", str(frontend_dir))

    model_path = root / "best2.pt"
    if model_path.exists():
        os.environ.setdefault("YOLO_MODEL_PATH", str(model_path))


def main() -> None:
    _set_default_env()
    root = _bundle_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.main import app

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
