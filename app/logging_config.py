from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure application logging once for HTTP and CLI entry points."""

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.info(
        "Logging initialized: dir={}, level={}, rotation=10 MB, retention=30 days, compression=zip",
        log_dir,
        settings.log_level,
    )
