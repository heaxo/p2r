from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ReferenceDetection:
    """Result for a planar measurement reference."""

    ok: bool
    source: str
    H_px_to_mm: Optional[np.ndarray] = None
    reference_quad_px: Optional[np.ndarray] = None
    reference_quad_mm: Optional[np.ndarray] = None
    size_mm: Optional[Tuple[float, float]] = None
    info: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ok": bool(self.ok),
            "source": self.source,
            "failure_reason": self.failure_reason,
            "info": self.info,
        }
        if self.size_mm is not None:
            data["size_mm"] = [float(self.size_mm[0]), float(self.size_mm[1])]
        if self.reference_quad_px is not None:
            data["reference_quad_px_tl_tr_br_bl"] = np.round(self.reference_quad_px, 3).tolist()
        if self.reference_quad_mm is not None:
            data["reference_quad_mm_tl_tr_br_bl"] = np.round(self.reference_quad_mm, 3).tolist()
        return data

