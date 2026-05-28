from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = "pic2remnant-http"
    auth_token: str = "tk_c2VjcmV0LXJhbmRvbS10b2tlbi0xMjM0NTY3OA"
    output_root: Path = Path("measure_out")
    model_path: Path = Path("best2.pt")
    sam_model: str = "sam2"
    default_imgsz: int = 1280
    default_conf: float = 0.35
    max_upload_mb: int = 80
    static_url_prefix: str = "/files"
    serialize_processing: bool = True
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    task_db_path: Path = Path("data/tasks.sqlite3")


def get_settings() -> Settings:
    """Build settings from environment variables.

    Environment variables are intentionally simple so the service can run in a
    customer's intranet without extra configuration libraries.
    """

    output_root = Path(os.getenv("OUTPUT_ROOT", "measure_out"))

    return Settings(
        auth_token=os.getenv("APP_TOKEN", "tk_c2VjcmV0LXJhbmRvbS10b2tlbi0xMjM0NTY3OA"),
        output_root=output_root,
        model_path=Path(os.getenv("YOLO_MODEL_PATH", "best2.pt")),
        sam_model=os.getenv("SAM_MODEL", "sam2"),
        default_imgsz=int(os.getenv("YOLO_IMGSZ", "1280")),
        default_conf=float(os.getenv("YOLO_CONF", "0.35")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "80")),
        static_url_prefix=os.getenv("STATIC_URL_PREFIX", "/files"),
        serialize_processing=os.getenv("SERIALIZE_PROCESSING", "true").lower() in {"1", "true", "yes", "y"},
        log_dir=Path(os.getenv("LOG_DIR", "logs")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        task_db_path=Path(os.getenv("TASK_DB_PATH", "data/tasks.sqlite3")),
    )
