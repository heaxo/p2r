from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MeasureResponse(BaseModel):
    """Compact response returned by the HTTP endpoint."""

    ok: bool = True
    run_dir: str
    paths: Dict[str, str]
    urls: Dict[str, str] = Field(default_factory=dict)
    plate_dimensions: Dict[str, Any]
    a4: Dict[str, Any]
    topdown: Dict[str, Any]
    input: Dict[str, Any]
    model: Dict[str, Any]
    paper: Dict[str, Any]
    plate: Dict[str, Any]
    fill_paper_to_plate: Dict[str, Any]
    result_json: Optional[str] = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
